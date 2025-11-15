
import argparse
from copy import deepcopy
from functools import partial
from hashlib import sha256
import json
import os
import random
import logging
import time
import asyncio
import ray
import logging
import atexit
import uuid
import torch
import numpy as np
from numba import njit
from vllm import AsyncEngineArgs, SamplingParams, TokensPrompt
from vllm.outputs import CompletionOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.outputs import CompletionOutput
from transformers import AutoTokenizer
import wandb # Import wandb
from typing import List

from verifier_pool import get_or_create_verifier_pool
from reward_registry import RewardType
# WeightShardingManager will be kept but its internal logic will be adapted later.
from external_process_executor import ExternalProcessExecutor


import numpy as np
from numba import njit
import torch.distributed as dist
from datetime import timedelta

logging.getLogger("numba").setLevel(logging.WARNING)

# Honor global logging config from launcher/workers; avoid resetting here
logger = logging.getLogger(__name__)


delimiter = "\n\n"

def _task_with_index(idx: int, coro: "asyncio.Task"):
    async def wrapper():
        try:
            res = await coro
        except Exception:
            res = ""
        return idx, res
    return wrapper()

async def _as_completed_iter(tasks: list["asyncio.Task"]):
    indexed = [asyncio.create_task(_task_with_index(i, t)) for i, t in enumerate(tasks)]
    for fut in asyncio.as_completed(indexed):
        yield await fut

def get_indices_of_delimiter(response, delimiter):
    indices = []
    start = 0
    while True:
        index = response.find(delimiter, start)
        if index == -1:
            break
        indices.append(index)
        start = index + len(delimiter)
    return indices

def insert_phrase(response, delimiter, special_phrases, eos_str):
    return response, False

async def rewrite_with_insert_phrase(sample, tokenizer):
    return sample

@njit
def normalize_rewards(rewards):
    """
    Normalize rewards within each group to compute advantages.

    Parameters:
        rewards : np.ndarray (1D)
            Array of rewards for each sample of shape (n_samples,).

    Returns:
        np.ndarray (1D)
            Normalized rewards of shape (n_samples,).
    """
    mean = np.mean(rewards)
    std = np.std(rewards) + 1e-4
    return (rewards - mean) / std

def _get_model_runner_workers(group_id: str, namespace: str, init_ray: bool = True):
    """
    Finds and retrieves handles to the RolloutWorker actors based on the group_id
    and namespace. This is adapted from ExternalProcessExecutor.
    """
    if not group_id:
        raise ValueError("group_id must be provided.")
    
    if init_ray and not ray.is_initialized():
        ray.init(namespace=namespace)
    
    # Worker names are expected to be f"{group_id}_rank-{j}"
    actor_names = [
        actor["name"] for actor in ray.util.list_named_actors(all_namespaces=True)
        if actor["name"].startswith(group_id) and actor["namespace"] == namespace
    ]
    
    # Sort actors by their rank, which is the number at the end of the name
    def get_rank_from_name(name):
        try:
            return int(name.split('-')[-1])
        except (ValueError, IndexError):
            return -1

    sorted_actor_names = sorted(actor_names, key=get_rank_from_name)
    
    if not sorted_actor_names:
        raise RuntimeError(f"No actors found for group_id '{group_id}' in namespace '{namespace}'.")

    # logger.info(f"[{group_id}] Discovered workers via _get_model_runner_workers: {sorted_actor_names}")
    workers = [ray.get_actor(name, namespace=namespace) for name in sorted_actor_names]
    return workers


@ray.remote(num_cpus=1) # Controller is a lightweight CPU process, it does not require a GPU.
class GenerationVLLMWorker:
    """
    This class is the Controller for a group of RolloutWorkers. It initializes
    and owns the AsyncLLM engine, which in turn controls the remote workers.
    This mirrors the role of verl's AsyncvLLMServer.
    """
    def __init__(
        self,
        model_path: str,
        group_id: str,
        tensor_parallel_size: int,
        max_num_seqs: int,
        worker_handles: List[ray.actor.ActorHandle], # Added worker_handles
        global_num_verifiers: int = 4,
        write_failed: bool = False,
        overhead_seqs: int = 8,
        enable_prefix_caching: bool = True,
        reward_fns: list[RewardType] = [RewardType.MATHD, RewardType.SYMPY],
        namespace: str = "",
        engine_preset: str = "eager", # "eager" or "throughput"
    ):
        
        self.group_id = group_id
        self.tensor_parallel_size = tensor_parallel_size
        self.model_path = model_path
        self.overhead_seqs = overhead_seqs
        self.max_num_seqs = max_num_seqs
        self.llm: AsyncLLM = None
        self.namespace = namespace
        self.worker_handles = worker_handles # Store the provided handles
        self.max_model_len = None
        self.engine_preset = engine_preset if engine_preset in ("eager", "throughput") else "eager"

        self.tokenizer = None # Initialize as None
        
        reward_enum_list = [RewardType(fn) for fn in reward_fns]
        self.verifier_pool = get_or_create_verifier_pool(
            global_num_verifiers, write_failed, reward_enum_list
        )
        self.enable_prefix_caching = enable_prefix_caching
        
        # self.worker_handles = [] # This controller doesn't need to know its workers beforehand.

        # Ensure third-party library logs are not overly verbose
        try:
            import logging as _logging
            _logging.getLogger("vllm").setLevel(_logging.WARNING)
        except Exception:
            pass

        # logger.info(f"[{group_id}] GenerationVLLMWorker initialization complete with {len(self.worker_handles)} workers. Awaiting engine init.")

    async def get_worker_handles(self) -> List[ray.actor.ActorHandle]:
        """Returns the handles to the RolloutGPUWorker actors."""
        return self.worker_handles

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """Triggers weight synchronization for all managed vLLM workers."""
        # logger.info(f"[{self.group_id}] ENTERING: sync_weights")
        # logger.info(f"[{self.group_id}] Triggering weight sync for vLLM workers...")
        
        workers_to_sync = self.worker_handles
        if not workers_to_sync:
            # logger.warning(f"[{self.group_id}] No vLLM workers discovered to sync weights for.")
            return

        sync_futures = [
            worker.sync_weights.remote(group_name, source_rank_in_collective_group)
            for worker in workers_to_sync
        ]
        await asyncio.gather(*sync_futures)
        # logger.info(f"[{self.group_id}] Weight sync complete for all vLLM workers.")
        # logger.info(f"[{self.group_id}] LEAVING: sync_weights")

    async def init_engine(self):
        """
        Initializes the AsyncLLM engine. This is a collective operation.
        The engine, via its ExternalProcessExecutor, will find and initialize
        the remote RolloutWorker actors.
        """
        # logger.info(f"[{self.group_id}] ENTERING: init_engine")
        # logger.info(f"[{self.group_id}] Initializing vLLM Engine...")
        
        # Load the tokenizer here, just before the engine needs it.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(self.model_path)
        self.max_model_len = hf_config.max_position_embeddings

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.90,
            dtype="bfloat16",
            enable_prefix_caching=self.enable_prefix_caching,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            distributed_executor_backend=ExternalProcessExecutor,
            load_format="auto",
            disable_custom_all_reduce=True,
            disable_log_stats=True,
            trust_remote_code=True,
            seed=0,
            # enforce_eager=True is not a standard vLLM argument and might cause issues.
            # It is safer to rely on the PlacementGroup setup for correctness.
        )

        vllm_config = engine_args.create_engine_config()
        # Log vLLM version and engine seed for reproducibility diagnostics
        try:
            import vllm as _vllm
            _seed = None
            try:
                _seed = getattr(vllm_config.model_config, "seed", None)
            except Exception:
                pass
            # logger.info(f"[{self.group_id}] vLLM version={getattr(_vllm, '__version__', 'unknown')} seed={_seed} tp={self.tensor_parallel_size} max_num_seqs={self.max_num_seqs} preset={self.engine_preset}")
        except Exception as _e:
            logger.warning(f"[{self.group_id}] Failed to log vLLM version/seed: {_e}")
        
        # Inject the namespace for the executor's subprocess to connect to Ray
        vllm_config.ray_namespace = self.namespace
        # Set the instance_id. This is a critical signal for vLLM to use the
        # external executor. We format it to contain info our executor can parse.
        vllm_config.instance_id = f"{self.namespace}:{self.group_id}"
        
        # Apply binary preset to control compile and CUDA graph behavior
        try:
            cc = getattr(vllm_config, "compilation_config", None)
            if cc is not None:
                if self.engine_preset == "eager":
                    # Fast startup: disable compile and cudagraph
                    if hasattr(cc, "level"):
                        cc.level = 0
                    if hasattr(cc, "use_cudagraph"):
                        cc.use_cudagraph = False
                    if hasattr(cc, "cudagraph_num_of_warmups"):
                        cc.cudagraph_num_of_warmups = 0
                    # Optionally shrink capture sizes to minimize any residual warmup
                    if hasattr(cc, "cudagraph_capture_sizes"):
                        cc.cudagraph_capture_sizes = [16, 8, 4, 2, 1]
                else:
                    # Throughput mode: aggressive compile and cudagraph enabled
                    if hasattr(cc, "level"):
                        cc.level = 3
                    if hasattr(cc, "use_cudagraph"):
                        cc.use_cudagraph = True
                    # keep defaults for warmups/capture sizes
        except Exception as _e:
            logger.warning(f"[{self.group_id}] Failed to apply engine preset '{self.engine_preset}': {_e}")
        
        # logger.info(f"[{self.group_id}] Calling AsyncLLM.from_vllm_config with instance_id='{vllm_config.instance_id}'...")
        self.llm = AsyncLLM.from_vllm_config(vllm_config)
        # logger.info(f"[{self.group_id}] ✅ AsyncLLM Engine initialization complete.")
        # logger.info(f"[{self.group_id}] LEAVING: init_engine")

    async def generate_with_logprobs(self, sample: dict, n: int = 1, temperature: float = 0.7, max_tokens: int = 128):
        """Generate with chosen-token logprobs and return tokens+logprobs per sample.
        Returns (outputs, gen_metrics). Each output has: input_token_ids, output_token_ids, output_logprobs.
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")
        # Sampling params: request per-position candidate logprobs (K>1 improves chance of including chosen)
        sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, logprobs=5)

        # Build prompt
        if "input_token_ids" in sample:
            prompt = TokensPrompt(prompt_token_ids=sample["input_token_ids"])
            in_ids = sample["input_token_ids"]
        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            in_ids = self.tokenizer.encode(sample.get("input") or sample.get("prompt") or "")
            prompt = TokensPrompt(prompt_token_ids=in_ids)

        request_id = sample.get("request_id", f"probe-{uuid.uuid4().hex}")

        # vLLM AsyncLLM.generate returns an async generator; iterate to final outputs
        outputs = []
        total = 0
        final_results: dict[int, "CompletionOutput"] = {}
        async for req_out in self.llm.generate(prompt=prompt, sampling_params=sp, request_id=request_id):
            for out in req_out.outputs:
                final_results[out.index] = out

        for idx in sorted(final_results.keys()):
            o = final_results[idx]
            toks = list(o.token_ids)
            total += len(toks)
            # chosen token logprobs for vLLM v1 with robust fallbacks
            lps = []
            try:
                entries = getattr(o, "logprobs", None)
                if entries is None:
                    # v0 compatibility
                    entries = getattr(o, "top_logprobs", None)
                for j, tid in enumerate(toks):
                    lp_j = 0.0
                    try:
                        entry = entries[j] if (entries is not None and j < len(entries)) else None
                        if entry is not None:
                            # direct numeric
                            if isinstance(entry, (int, float)):
                                lp_j = float(entry)
                            else:
                                # preferred attributes
                                val = getattr(entry, "chosen_logprob", None)
                                if val is None:
                                    val = getattr(entry, "selected_logprob", None)
                                if val is None:
                                    val = getattr(entry, "logprob", None)
                                if val is None and isinstance(entry, dict):
                                    val = entry.get("chosen_logprob") or entry.get("selected_logprob") or entry.get("logprob")
                                if val is not None:
                                    lp_j = float(val)
                                elif isinstance(entry, dict):
                                    # dict keyed by chosen token id
                                    v = entry.get(tid)
                                    if v is None:
                                        v = entry.get(str(tid))
                                    if v is not None:
                                        if isinstance(v, (int, float)):
                                            lp_j = float(v)
                                        elif hasattr(v, "logprob"):
                                            lp_j = float(getattr(v, "logprob"))
                                        elif isinstance(v, dict):
                                            lp = v.get("logprob") or v.get("logp") or v.get("score")
                                            if lp is not None:
                                                lp_j = float(lp)
                                if lp_j == 0.0:
                                    # fallback to candidate lists
                                    cand_list = getattr(entry, "top_logprobs", None) or getattr(entry, "candidates", None)
                                    if isinstance(cand_list, (list, tuple)):
                                        for cand in cand_list:
                                            if cand is None:
                                                continue
                                            if isinstance(cand, (list, tuple)) and len(cand) >= 2 and int(cand[0]) == int(tid):
                                                lp_j = float(cand[1]); break
                                            if hasattr(cand, "token_id") and hasattr(cand, "logprob") and int(cand.token_id) == int(tid):
                                                lp_j = float(cand.logprob); break
                                            if hasattr(cand, "id") and hasattr(cand, "logprob") and int(cand.id) == int(tid):
                                                lp_j = float(cand.logprob); break
                                            if isinstance(cand, dict):
                                                _cid = cand.get("token_id") or cand.get("id") or cand.get("token")
                                                if _cid is not None and int(_cid) == int(tid):
                                                    lp = cand.get("logprob") or cand.get("logp") or cand.get("score")
                                                    if lp is not None:
                                                        lp_j = float(lp)
                                                        break
                        else:
                            lp_j = 0.0
                    except Exception:
                        lp_j = 0.0
                    lps.append(lp_j)
            except Exception:
                lps = [0.0] * len(toks)
            outputs.append({
                "input_token_ids": list(in_ids),
                "output_token_ids": toks,
                "output_logprobs": lps,
            })
        metrics = {"generation/total_tokens_generated": total, "generation/num_samples_generated": len(outputs)}
        return outputs, metrics

    def get_max_tokens(self, sample: dict, max_tokens=None) -> int:
        if max_tokens is None:
            max_tokens = self.llm.engine.model_config.max_model_len
        
        max_tokens = max_tokens - len(sample["input_token_ids"]) - 1
        if max_tokens <= 0:
            max_tokens = 1
        return max_tokens

    def _get_gen_kwargs(self, sample: dict, **kwargs) -> dict:
        """Helper to construct the generation kwargs for SamplingParams, mirroring old working code."""
        # Start with some sane defaults, similar to the old code.
        gen_kwargs = {
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 0.7),
            # Ensure deterministic sampling by default (overridden if caller provides seed)
            "seed": kwargs.get("seed", 0),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", True),
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
        }
        # Explicitly update with any other kwargs passed in.
        gen_kwargs.update(kwargs)
        
        # Adjust max_tokens based on input length.
        gen_kwargs["max_tokens"] = self.get_max_tokens(sample, kwargs.get("max_tokens"))
        
        return gen_kwargs


    async def generate_for_evaluation(self, prompts: list[str], max_tokens: int, temperature: float) -> list[str]:
        # logger.info(f"[{self.group_id}] ENTERING: generate_for_evaluation (num_prompts={len(prompts)})")
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=0,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

        prompt_token_ids = self.tokenizer(prompts, add_special_tokens=False).input_ids

        # Concurrency bound to engine capacity to avoid overload/cancellations
        concurrency = max(1, min(self.max_num_seqs, 32))
        sem = asyncio.Semaphore(concurrency)

        async def generate_single(tokens: list[int], req_id: str) -> str:
            async with sem:
                try:
                    generator = self.llm.generate(
                        prompt=TokensPrompt(prompt_token_ids=tokens),
                        sampling_params=sampling_params,
                        request_id=req_id,
                    )
                    final_results: dict[int, "CompletionOutput"] = {}
                    async for output in generator:
                        for out in output.outputs:
                            final_results[out.index] = out
                    out0 = final_results.get(0)
                    if out0 is None:
                        return ""
                    decoded = self.tokenizer.decode(list(out0.token_ids), skip_special_tokens=True).strip()
                    if decoded:
                        return decoded
                    raw_text = (out0.text or "").strip()
                    if raw_text:
                        return raw_text
                    # Fallback once with include_stop_str_in_output
                    sp2 = SamplingParams(
                        n=1,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        skip_special_tokens=False,
                        include_stop_str_in_output=True,
                    )
                    gen2 = self.llm.generate(
                        prompt=TokensPrompt(prompt_token_ids=tokens),
                        sampling_params=sp2,
                        request_id=f"{req_id}-retry",
                    )
                    final_results2: dict[int, "CompletionOutput"] = {}
                    async for o2 in gen2:
                        for o in o2.outputs:
                            final_results2[o.index] = o
                    o = final_results2.get(0)
                    if o is not None:
                        return self.tokenizer.decode(list(o.token_ids), skip_special_tokens=True)
                    return ""
                except Exception as e:
                    logger.warning(f"[{self.group_id}] Generation failed for request {req_id}: {e}")
                    return ""

        tasks = [
            asyncio.create_task(generate_single(token_ids, f"eval-{uuid.uuid4()}"))
            for token_ids in prompt_token_ids
        ]

        start_time = time.time()
        total = len(tasks)
        # logger.info(f"[{self.group_id}] Eval generation start: total_requests={total} max_tokens={max_tokens} T={temperature}")
        results: list[str] = [""] * total
        done_count = 0
        try:
            async for fut in _as_completed_iter(tasks):
                try:
                    idx, text = fut
                except Exception:
                    # Fallback in case helper yields plain text
                    idx, text = None, ""
                if isinstance(idx, int) and 0 <= idx < total:
                    results[idx] = text or ""
                else:
                    # If index not tracked, place in first empty slot
                    try:
                        empty_idx = results.index("")
                        results[empty_idx] = text or ""
                    except ValueError:
                        pass
                done_count += 1
                if done_count % 5 == 0 or done_count == total:
                    elapsed = time.time() - start_time
                    logger.info(f"[{self.group_id}] Eval progress: done={done_count}/{total} elapsed={elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"[{self.group_id}] Eval loop interrupted: {e}")

        # finished_count = len([r for r in results if r.strip()])
        elapsed = time.time() - start_time
        logger.info(f"[{self.group_id}] Eval generation done: finished={finished_count}/{total} elapsed={elapsed:.1f}s")
        return results

    async def batch_inference_for_evaluation(
        self,
        prompts: list[str],
        answers: list[str] | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        suppress_eval_logs: bool = True,
        progress_interval: int = 10,
    ) -> list[dict]:
        """
        Lightweight evaluation path that generates completions for a batch of prompts
        and computes rewards inside the worker. Returns a list of dicts with
        {prompt, completion, reward, answer}.
        """
        if self.llm is None:
            raise RuntimeError("vLLM engine not initialized")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

        # Tokenize all prompts at once
        prompt_token_ids: list[list[int]] = self.tokenizer(prompts, add_special_tokens=False).input_ids

        # Maximize engine throughput: allow up to max_num_seqs concurrent requests
        concurrency = max(1, int(self.max_num_seqs))
        try:
            logger.info(f"[{self.group_id}] Eval batch start: total={len(prompts)} max_tokens={max_tokens} T={temperature} concurrency={concurrency}")
        except Exception:
            pass
        sem = asyncio.Semaphore(concurrency)

        async def gen_one(tokens: list[int], req_id: str) -> tuple[list[int], str]:
            async with sem:
                try:
                    generator = self.llm.generate(
                        prompt=TokensPrompt(prompt_token_ids=tokens),
                        sampling_params=sampling_params,
                        request_id=req_id,
                    )
                    final_results: dict[int, "CompletionOutput"] = {}
                    async for output in generator:
                        for out in output.outputs:
                            final_results[out.index] = out
                    out0 = final_results.get(0)
                    if out0 is None:
                        return [], ""
                    out_ids = list(out0.token_ids)
                    # Prefer decoded text; fallback to raw
                    decoded = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                    if decoded:
                        return out_ids, decoded
                    raw_text = (out0.text or "").strip()
                    return out_ids, raw_text
                except Exception as e:
                    logger.warning(f"[{self.group_id}] batch eval generation failed: {e}")
                    return [], ""

        tasks = [
            asyncio.create_task(gen_one(toks, f"eval-batch-{uuid.uuid4()}"))
            for toks in prompt_token_ids
        ]

        # Gather completions in-order
        results_token_out: list[tuple[list[int], str]] = [([], "")] * len(tasks)
        idx_done = 0
        async for fut in _as_completed_iter(tasks):
            try:
                i, val = fut
            except Exception:
                i, val = None, ([], "")
            if isinstance(i, int) and 0 <= i < len(tasks):
                results_token_out[i] = val  # type: ignore
            else:
                # place into first empty
                try:
                    empty_idx = results_token_out.index(([], ""))
                    results_token_out[empty_idx] = val  # type: ignore
                except ValueError:
                    pass
            idx_done += 1
            # Progress logging (always on; adv logs are controlled separately)
            try:
                if progress_interval > 0 and (idx_done % int(progress_interval) == 0 or idx_done == len(tasks)):
                    logger.info(f"[{self.group_id}] Eval batch progress: done={idx_done}/{len(tasks)}")
            except Exception:
                pass

        # Build samples and compute rewards (lightweight: skip labels/adv)
        answers = answers or [""] * len(prompts)
        verify_tasks = []
        samples_meta: list[dict] = []
        for (out_ids, completion_text), prompt_text, ans, in_ids in zip(
            results_token_out, prompts, answers, prompt_token_ids
        ):
            try:
                sample_ids = list(in_ids) + list(out_ids)
                sample_text = self.tokenizer.decode(sample_ids)
                smp = {
                    "input": prompt_text,
                    "input_token_ids": list(in_ids),
                    "output_token_ids": list(out_ids),
                    "sample_ids": sample_ids,
                    "sample_text": sample_text,
                    "answer": ans,
                    "output_len": int(len(out_ids)),
                    "num_non_masked_output_tokens": int(len(out_ids)),
                }
                samples_meta.append({
                    "prompt": prompt_text,
                    "completion": completion_text or sample_text[len(prompt_text):],
                    "answer": ans,
                })
                verify_tasks.append(self.verifier_pool.verify.remote(smp, max_gen_length=max_tokens))
            except Exception:
                samples_meta.append({"prompt": prompt_text, "completion": "", "answer": ans})
                verify_tasks.append(asyncio.sleep(0, result={"reward": 0.0}))

        rewards = await asyncio.gather(*verify_tasks)

        packed: list[dict] = []
        for meta, r in zip(samples_meta, rewards):
            try:
                rew = float(r.get("reward", 0.0)) if isinstance(r, dict) else 0.0
            except Exception:
                rew = 0.0
            packed.append({
                "prompt": meta["prompt"],
                "completion": meta.get("completion", ""),
                "answer": meta.get("answer", ""),
                "reward": rew,
            })

        try:
            logger.info(f"[{self.group_id}] Eval batch done: finished={len(packed)}/{len(prompts)}")
        except Exception:
            pass
        return packed

    async def inference(self, sample: dict, **kwargs) -> tuple[list[dict], dict]:
        # logger.info(f"[{self.group_id}] ENTERING: inference (input_len={len(sample.get('input_token_ids', []))})")
        # Template verification and boundary logging (once tokenizer is ready)
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            # Ensure input_token_ids exist
            if 'input_token_ids' in sample:
                itoks = sample['input_token_ids']
                # Decode last ~20 tokens of prompt to check assistant header boundary
                tail = itoks[-20:] if len(itoks) > 0 else []
                decoded_tail = self.tokenizer.decode(tail, skip_special_tokens=False)
                _san_tail = decoded_tail.replace("\n", "\\n")
                # print(f"[TEMPLATE-CHECK] prompt_tail='{_san_tail}' len={len(itoks)}")
        except Exception as _e:
            pass
        
        # The chat template is already applied during preprocessing.
        # We should directly use the input_token_ids from the sample.
        # logger.debug(f"[{self.group_id}] sample: {sample}")
        # logger.debug(f"[{self.group_id}] kwargs: {kwargs}")
        try:
            input_ids = sample["input_token_ids"]
            prompt_text = self.tokenizer.decode(input_ids)
            # logger.debug(f"[{self.group_id}] Generating for pre-formatted prompt: \"{prompt_text[:200].replace(os.linesep, ' ')}...\"")
        except Exception as e:
            # logger.error(f"[{self.group_id}] Failed to decode prompt tokens for logging: {e}")
            input_ids = []

        # Construct generation kwargs using the helper method
        generation_kwargs = self._get_gen_kwargs(sample, **kwargs)
        
        # logger.debug(f"[{self.group_id}] Creating SamplingParams with kwargs: {generation_kwargs}")
        
        request_out = None
        
        start_time = time.time()
        
        # input_ids is already set above from the sample
        sampling_params = SamplingParams(**generation_kwargs)
        try:
            _sp_seed = getattr(sampling_params, "seed", None)
            logger.info(f"[{self.group_id}] Inference SamplingParams: seed={_sp_seed} n={sampling_params.n} max_tokens={sampling_params.max_tokens}")
        except Exception:
            pass
        # Build a deterministic request_id to keep RNG streams stable across runs
        try:
            import hashlib as _hashlib
            _uid = sample.get("__uid__")
            if _uid is None:
                # Fallback: derive from prompt token ids
                _uid = _hashlib.sha256(bytes(sample.get("input_token_ids", []))).hexdigest()
            request_id = f"inference-{str(_uid)}"
        except Exception:
            request_id = f"inference"
        
        # logger.debug(f"[{self.group_id}] sampling_params: {sampling_params}")
        
        # Use the initialized self.llm engine
        generator = self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=input_ids),
            sampling_params=sampling_params, request_id=request_id)
        
        final_results: dict[int, "CompletionOutput"] = {}
        async for out in generator:
            for output in out.outputs:
                final_results[output.index] = output
            request_out = out

        # Re-construct the final list of outputs from the dictionary
        # to ensure we have all samples, even if some finished early.
        if request_out is not None:
            request_out.outputs = [final_results[i] for i in sorted(final_results.keys())]
            
        duration = time.time() - start_time
        
        if request_out is None:
            return [], {}
        
        if "input_len" not in sample:
            sample["input_len"] = len(sample["input_token_ids"])
        
        samples = [deepcopy(sample) for _ in range(len(request_out.outputs))]
        # logger.debug(f"[{self.group_id}] len(samples): {len(samples)}")
        
        total_tokens_generated = 0
        
        sample_rewards_futures = []
        for i, (sample, out) in enumerate(zip(samples, request_out.outputs)):
            total_tokens_generated += len(out.token_ids)
            sample["modified_reward"] = None
            sample["delimiter_not_found"] = False
            sample["output_token_ids"] = list(out.token_ids)
            sample["output_len"] = len(sample["output_token_ids"])
            sample["sample_ids"] = sample["input_token_ids"] + sample["output_token_ids"]
            sample["sample_text"] = self.tokenizer.decode(sample["sample_ids"])
            sample["sample_position_ids"] = list(range(len(sample["sample_ids"])))
            sample["truncated_sample"] = (sample["sample_ids"][-1] != self.tokenizer.eos_token_id)
            labels = [-100] * len(sample["sample_ids"])
            # Label all generated tokens regardless of truncation; EOS is not required for loss.
            for i in range(sample["output_len"]):
                pos = sample["input_len"] + i
                labels[pos] = sample["sample_ids"][pos]
            sample["labels"] = labels
            sample["num_non_masked_output_tokens"] = sum(1 for label in labels if label != -100)
            ## Token-level diagnostic for the first output tokens (policy expectation boundary)
            # try:
            #     first20 = sample["output_token_ids"][:20]
            #     decoded_first20 = self.tokenizer.decode(first20, skip_special_tokens=False)
            #     _san_first20 = decoded_first20.replace("\n", "\\n")
            #     print(f"[GEN-FIRST20] first_output_tokens='{_san_first20}'")
            # except Exception:
            #     pass
            sample_rewards_futures.append(
                self.verifier_pool.verify.remote(sample, max_gen_length=kwargs.get("max_tokens", self.max_model_len)))

        final_samples = await asyncio.gather(*sample_rewards_futures)
        
        group_rewards = np.array([s["reward"] for s in final_samples])
        max_reward = np.max(group_rewards).item()
        # Compute pre-normalization stats (centered values) and normalized advantages
        mean_r = float(np.mean(group_rewards))
        std_r = float(np.std(group_rewards) + 1e-4)
        adv_centered = group_rewards - mean_r
        group_advantages = adv_centered / std_r
        for i, (sample_, advantage) in enumerate(zip(final_samples, group_advantages)):
            sample_["advantage"] = float(advantage)
        
        gen_metrics = {
            "generation/total_tokens_generated": total_tokens_generated,
            "generation/inference_duration_sec": duration,
            "generation/num_samples_generated": len(samples),
        }

        # logger.info(f"[{self.group_id}] LEAVING: inference (num_final_samples={len(final_samples)})")
        return final_samples, gen_metrics
