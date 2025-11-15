import asyncio
import logging
from typing import Any, Dict, List
import pickle
import ray

logger = logging.getLogger(__name__)


@ray.remote
class TrainerWorker:
    """
    This class is the Controller for a group of UnifiedWorkers dedicated to training.
    It receives batches from the Orchestrator, distributes them to the workers,
    and orchestrates the training step, including gradient computation and weight updates.
    """

    def __init__(
        self,
        group_id: str,
        worker_handles: List[ray.actor.ActorHandle],
        fsdp_group_ranks: List[int],
    ):
        """
        Initializes the TrainerWorker controller.

        Args:
            group_id (str): A unique identifier for this group of training workers.
            worker_handles (List[ray.actor.ActorHandle]): A list of Ray actor handles
                for the UnifiedWorker instances that this controller will manage.
            fsdp_group_ranks (List[int]): The global ranks of the workers in this FSDP group.
        """
        # logger.info(f"[{group_id}] Initializing TrainerWorker (Controller)...") 
        self.group_id = group_id
        self.worker_handles = worker_handles
        self.fsdp_group_ranks = fsdp_group_ranks
        self.world_size = len(worker_handles)
        # logger.info(
        #     f"[{self.group_id}] TrainerWorker initialization complete. Managing {self.world_size} workers with ranks {self.fsdp_group_ranks}."
        # )

    def get_worker_handles(self) -> List[ray.actor.ActorHandle]:
        """Returns the handles to the UnifiedWorker actors it manages."""
        return self.worker_handles

    async def initialize_fsdp(self):
        """
        Initializes the FSDP model on all managed workers.
        This involves creating the FSDP shell and then loading the initial
        weights, which are sourced from the lead worker (global rank 0).
        """
        # logger.debug(f"[{self.group_id}] ENTERING: initialize_fsdp")
        if not self.worker_handles:
            logger.warning(f"[{self.group_id}] No worker handles found. Skipping FSDP initialization.")
            return

        # The controller already knows the ranks, so it passes them to the workers.
        # logger.debug(f"[{self.group_id}] FSDP group will be created with ranks: {self.fsdp_group_ranks}")

        # Step 1: Create the FSDP shells on all workers. This is a collective call.
        init_futures = [
            worker.init_model.remote(for_computation=True, fsdp_group_ranks=self.fsdp_group_ranks)
            for worker in self.worker_handles
        ]
        await asyncio.gather(*init_futures)
        
        # Step 2: Tell the workers to collectively load weights.
        # The lead worker (rank 0 of the FSDP group) will act as the source for its loaded cpu_state_dict.
        load_futures = [worker.load_own_initial_weights.remote() for worker in self.worker_handles]
        await asyncio.gather(*load_futures)

        worker_ranks = await asyncio.gather(*[w.get_global_rank.remote() for w in self.worker_handles])
        # logger.info(f"[{self.group_id}] FSDP init + initial weight load complete. ranks={worker_ranks}")
        # logger.debug(f"[{self.group_id}] LEAVING: initialize_fsdp")

    async def get_full_cpu_state_dict(self) -> Dict[str, Any]:
        """
        Orchestrates the gathering of the full state dict from the FSDP group.
        It calls `get_cpu_state_dict` on all workers concurrently, which is required
        for the FSDP collective to work. Only the rank 0 worker of the group
        will return the actual state dict.
        """
        # logger.debug(f"[{self.group_id}] ENTERING: get_full_cpu_state_dict")
        # logger.debug(f"[{self.group_id}] Fetching full CPU state_dict from FSDP group...")
        if not self.worker_handles:
            logger.warning(f"[{self.group_id}] No workers to fetch state_dict from.")
            return {}

        # This must be called on all workers in the group to trigger the collective.
        futures = [w.get_cpu_state_dict.remote() for w in self.worker_handles]
        results = await asyncio.gather(*futures)

        # The state_dict will only be on the rank 0 worker of the FSDP group.
        # Other workers will return None. We find the pickled bytes.
        state_dict_bytes = None
        for res in results:
            if res: # Find the non-empty bytes
                state_dict_bytes = res
                break
        
        if state_dict_bytes:
            # logger.info(f"[{self.group_id}] Successfully received state_dict bytes from worker.")
            # logger.debug(f"[{self.group_id}] LEAVING: get_full_cpu_state_dict (with data)")
            return state_dict_bytes
        else:
            logger.warning(f"[{self.group_id}] Did not receive a state_dict from any worker.")
            # logger.debug(f"[{self.group_id}] LEAVING: get_full_cpu_state_dict (without data)")
            return None

    async def release_cpu_state_dict(self):
        """Signals all managed workers to release their CPU state_dict."""
        # logger.info(f"[{self.group_id}] Releasing CPU state_dict on all workers.")
        release_futures = [w.release_cpu_state_dict.remote() for w in self.worker_handles]
        await asyncio.gather(*release_futures)

    async def set_mode(self, mode: str):
        """Sets the mode (train or eval) on all managed workers."""
        # logger.debug(f"[{self.group_id}] ENTERING: set_mode (mode={mode})")
        # logger.info(f"[{self.group_id}] Setting mode to '{mode}' on all workers.")
        set_mode_futures = [worker.set_mode.remote(mode) for worker in self.worker_handles]
        await asyncio.gather(*set_mode_futures)
        # logger.info(f"[{self.group_id}] Mode set to '{mode}' on all workers.")
        # logger.debug(f"[{self.group_id}] LEAVING: set_mode")

    async def save_checkpoint(self, save_directory: str):
        """
        Orchestrates saving a checkpoint by instructing the rank 0 worker
        of the FSDP group to save the model.
        """
        # logger.info(f"[{self.group_id}] Saving checkpoint to {save_directory}")
        if not self.worker_handles:
            logger.warning(f"[{self.group_id}] No workers to save checkpoint from.")
            return

        # Instruct all workers to enter the save_model method.
        # The internal logic of save_model ensures only rank 0 performs the save.
        save_futures = [
            worker.save_model.remote(save_directory) for worker in self.worker_handles
        ]
        await asyncio.gather(*save_futures)
        # logger.info(f"[{self.group_id}] Checkpoint saving process completed by workers.")
        # logger.info(f"[{self.group_id}] Checkpoint save finished")

    async def save_optimizer_state(self, save_directory: str):
        """
        Orchestrates saving optimizer and scheduler state by instructing all workers.
        """
        # logger.info(f"[{self.group_id}] Saving optimizer/scheduler state to {save_directory}")
        if not self.worker_handles:
            logger.warning(f"[{self.group_id}] No workers to save optimizer state from.")
            return

        # Instruct all workers to save optimizer and scheduler state
        save_futures = [
            worker.save_optimizer_state.remote(save_directory) for worker in self.worker_handles
        ]
        await asyncio.gather(*save_futures)
        # logger.info(f"[{self.group_id}] Optimizer state saving process completed by workers.")
        # logger.info(f"[{self.group_id}] Optimizer/scheduler state save finished")

    async def load_optimizer_state(self, load_directory: str):
        """
        Orchestrates loading optimizer and scheduler state by instructing all workers.
        This should be called after FSDP initialization and weight loading.
        """
        # logger.info(f"[{self.group_id}] Loading optimizer/scheduler state from {load_directory}")
        if not self.worker_handles:
            logger.warning(f"[{self.group_id}] No workers to load optimizer state from.")
            return

        # Instruct all workers to load optimizer and scheduler state
        load_futures = [
            worker.load_optimizer_state.remote(load_directory) for worker in self.worker_handles
        ]
        await asyncio.gather(*load_futures)
        # logger.info(f"[{self.group_id}] Optimizer state loading process completed by workers.")
        # logger.info(f"[{self.group_id}] Optimizer/scheduler state load finished")

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """Triggers weight synchronization for all managed workers."""
        # logger.debug(f"[{self.group_id}] ENTERING: sync_weights")
        # logger.info(f"[{self.group_id}] Triggering weight sync for workers in group '{group_name}' from source rank {source_rank_in_collective_group}.")
        sync_futures = [
            worker.sync_weights.remote(group_name, source_rank_in_collective_group)
            for worker in self.worker_handles
        ]
        await asyncio.gather(*sync_futures)
        # logger.info(f"[{self.group_id}] Weight sync complete for all workers.") 
        # logger.debug(f"[{self.group_id}] LEAVING: sync_weights") 

    async def ping_all(self) -> Dict[int, bool]:
        """Ping all trainer GPU workers to check liveness.

        Returns a map of local rank -> is_alive.
        """
        results: Dict[int, bool] = {}
        tasks = []
        for idx, w in enumerate(self.worker_handles):
            tasks.append((idx, w.is_ready.remote()))
        for idx, obj in tasks:
            try:
                await obj
                results[idx] = True
            except Exception:
                results[idx] = False
        return results