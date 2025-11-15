import argparse
import asyncio
import os
import time
import logging
import ray
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Tuple
import gc
import uuid

# from weight_sharding_manager import WeightShardingManager # This is now handled by UnifiedWorker

@ray.remote
class LogProbFSDPWorker:
    """
    A Ray Actor that acts as the public interface for a group of UnifiedWorkers
    performing log probability calculations. It preserves the efficient, stateful
    batching logic of the original async-grpo LogprobWorker and is analogous to
    the GenerationVLLMWorker controller.
    """
    def __init__(
        self,
        group_id: str,
        worker_handles: list,
        fsdp_group_ranks: List[int],
        trainer_controller_handle: ray.actor.ActorHandle,
        max_tokens: int,
    ):
        self.group_id = group_id
        self.worker_handles = worker_handles
        self.fsdp_group_ranks = fsdp_group_ranks
        self.max_tokens_per_gpu = max_tokens
        self.world_size = len(worker_handles)
        self.batching_queue = asyncio.Queue()
        
        # This loop pulls requests from the queue, batches them, and processes them.
        self.centralizer_loop = asyncio.create_task(self._centralize_inference_requests())

    def get_worker_handles(self) -> List[ray.actor.ActorHandle]:
        """Returns the handles to the UnifiedWorker actors it manages."""
        return self.worker_handles

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """Triggers weight synchronization for all managed workers."""
        if not self.worker_handles:
            pass
            return

        sync_futures = [
            worker.sync_weights.remote(group_name, source_rank_in_collective_group)
            for worker in self.worker_handles
        ]
        await asyncio.gather(*sync_futures)

    async def initialize_fsdp(self, state_dict_to_load: bytes = None):
        """
        Initializes the FSDP model structure on all managed workers.
        If state_dict_to_load is provided, it also loads the weights.
        """
        if not self.worker_handles:
            return

        # First, create the FSDP shells on all workers.
        init_futures = [
            worker.init_model.remote(for_computation=False, fsdp_group_ranks=self.fsdp_group_ranks) 
            for worker in self.worker_handles
        ]
        await asyncio.gather(*init_futures)

        if state_dict_to_load:
            await self.load_initial_weights(state_dict_to_load)
        
    async def load_initial_weights(self, state_dict_to_load: bytes):
        """Loads the provided state_dict into all managed workers."""
        load_futures = [worker._load_initial_weights.remote(state_dict_to_load) for worker in self.worker_handles]
        await asyncio.gather(*load_futures)
        
        worker_ranks = await asyncio.gather(*[w.get_global_rank.remote() for w in self.worker_handles])
        
    async def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Processes a finalized batch of requests by dispatching the computation
        to the UnifiedWorker pool. Now uses sample_ids directly from generation worker
        instead of tokenizing again.
        """
        world_size = len(self.worker_handles)
        if not batch:
            return []
            
        # Simple scatter operation
        chunks = [[] for _ in range(world_size)]
        for i, sample in enumerate(batch):
            chunks[i % world_size].append(sample)
        
        # Create dummy sample for empty chunks to ensure FSDP synchronization
        dummy_sample = {'sample_ids': [1], 'dummy': True, 'request_id': 'dummy_fsdp_sync'}
        
        processed_samples_map = {}
        # Send raw samples to each worker; let workers build tensors on-GPU
        futures = []
        worker_chunks = []
        for i, worker in enumerate(self.worker_handles):
            chunk_to_process = chunks[i] if chunks[i] else [dummy_sample]
            worker_chunks.append(chunk_to_process)
            futures.append(worker.compute_log_probs_from_samples.remote(chunk_to_process))
        worker_results = await asyncio.gather(*futures)

        # Assign results back to samples (now dicts with logprobs/entropies)
        for chunk_samples, chunk_results in zip(worker_chunks, worker_results):
            for s, res in zip(chunk_samples, chunk_results):
                if s.get('dummy'):
                    continue
                if isinstance(res, dict):
                    s["sample_logprobs"] = res.get("logprobs")
                    s["sample_entropies"] = res.get("entropies")
                else:
                    # backward-compat: list means only logprobs
                    s["sample_logprobs"] = res
                    s["sample_entropies"] = None
                # Preserve alignment id if present
                try:
                    s["__uid__"] = s.get("__uid__", s.get("uid", s.get("id", -1)))
                except Exception:
                    pass
                processed_samples_map[tuple(s["sample_ids"])] = s

        # Re-order the results to match the original input 'batch' order
        final_results = [
            processed_samples_map.get(tuple(s["sample_ids"])) for s in batch
        ]

        return [res for res in final_results if res is not None]

    async def inference(self, sample: Dict, **kwargs) -> Dict:
        """Asynchronous interface for external callers."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.batching_queue.put((future, sample))
        result = await future
        return result

    async def _centralize_inference_requests(self):
        """
        An asyncio loop that pulls requests from a queue, groups them into
        batches based on token count, and processes them.
        Now uses sample_ids length instead of re-tokenizing.
        """
        pending_request = None
        while True:
            try:
                if pending_request is None:
                    try:
                        # debug wait log suppressed
                        current_request = await asyncio.wait_for(self.batching_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                else:
                    current_request = pending_request
                    pending_request = None

                future, sample = current_request
                
                inference_requests: List[Tuple[asyncio.Future, Dict]] = [(future, sample)]
                # Use sample_ids length instead of tokenizing
                total_length = len(sample.get("sample_ids", []))

                # Dynamically fill a batch with a short timeout.
                BATCHING_TIMEOUT = 0.01  # 10ms
                try:
                    while total_length < self.max_tokens_per_gpu:
                        timeout = BATCHING_TIMEOUT if inference_requests else None # Wait forever for the first request
                        next_request = await asyncio.wait_for(
                            self.batching_queue.get(), timeout=timeout
                        )
                        next_len = len(next_request[1].get("sample_ids", []))

                        if total_length + next_len > self.max_tokens_per_gpu:
                            pending_request = next_request
                            break

                        inference_requests.append(next_request)
                        total_length += next_len
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    pass # Batching finished

                
                # Process the finalized batch.
                futures, samples_to_process = zip(*inference_requests)
                try:
                    samples_with_logprobs = await self._process_batch(list(samples_to_process))
                    
                    # Create a map from a unique sample identifier to its result
                    # Use sample_ids as the unique identifier
                    result_map = {tuple(s['sample_ids']): s for s in samples_with_logprobs}

                    for fut, samp in zip(futures, samples_to_process):
                        if not fut.done():
                            result = result_map.get(tuple(samp['sample_ids']))
                            if result:
                                fut.set_result(result)
                            else:
                                fut.set_exception(Exception(f"Result for sample not found in processed batch."))
                except Exception as e:
                    for future_to_set in futures:
                        if not future_to_set.done():
                            future_to_set.set_exception(e)
            except Exception as e:
                # Ensure pending requests get a response
                if 'inference_requests' in locals():
                    for fut, _ in inference_requests:
                        if not fut.done():
                            fut.set_exception(e)
                pending_request = None
                await asyncio.sleep(1) # Avoid tight loop on error

    async def get_full_cpu_state_dict(self):
        """
        Retrieves the full CPU state dict from the FSDP group.
        IMPORTANT: For FSDP world_size > 1 this must be invoked on ALL ranks
        concurrently because the underlying API performs collective ops.
        We therefore call get_cpu_state_dict on every worker and return the
        non-empty bytes (rank 0 in the group will provide it).
        """
        if not self.worker_handles:
            return None

        try:
            # Trigger the collective on all workers
            futures = [w.get_cpu_state_dict.remote() for w in self.worker_handles]
            results = await asyncio.gather(*futures)

            # Only rank 0 will return the pickled bytes; others return None
            for res in results:
                if res:
                    return res

            return None
        except Exception as e:
            return None

    async def load_pretrained_weights_for_testing(self, model_name: str):
        """
        Testing-only method: Load pretrained weights on all managed workers for standalone tests.
        This bypasses the normal orchestrated weight synchronization flow.
        """
        if not self.worker_handles:
            return
        
        # Call the testing method on all GPU workers
        load_futures = [
            worker.load_pretrained_weights_for_testing.remote(model_name) 
            for worker in self.worker_handles
        ]
        await asyncio.gather(*load_futures)
        
