import logging
import os
import ray
from ray.util.placement_group import placement_group, PlacementGroup
from typing import List, Any, Dict
import uuid
import time
from collections import Counter

logger = logging.getLogger(__name__)


class RayClassWithInitArgs:
    """A helper class to hold a Ray Actor class and its initialization arguments."""

    def __init__(self, cls: Any, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._options = {}

    def options(self, **kwargs):
        """Sets Ray actor options."""
        self._options = kwargs
        return self

    def remote(self):
        """Creates the remote actor."""
        return self.cls.options(**self._options).remote(*self.args, **self.kwargs)


class RayResourcePool:
    """Defines and manages resource allocation using Ray PlacementGroups."""

    def __init__(self, world_size: int, num_gpus_per_node: int, name_prefix: str = None):
        if world_size <= 0 or num_gpus_per_node <= 0:
            raise ValueError("world_size and num_gpus_per_node must be positive.")
        if world_size % num_gpus_per_node != 0:
            logger.warning(
                f"world_size ({world_size}) is not perfectly divisible by num_gpus_per_node "
                f"({num_gpus_per_node}). This might lead to underutilization of the last node."
            )

        self.num_nodes = (world_size + num_gpus_per_node - 1) // num_gpus_per_node
        self.num_gpus_per_node = num_gpus_per_node
        self.world_size = world_size
        self.pgs: List[PlacementGroup] = []
        self.name_prefix = name_prefix or f"grpo_pool_{uuid.uuid4().hex[:6]}"

    def _check_resources_available(self, bundles_per_pg: List[int]):
        """
        Checks if the required GPU resources for all placement groups can be satisfied by the current Ray cluster.
        This is a pre-flight check to avoid deadlocks when creating placement groups.
        """
        num_pgs_needed = len(bundles_per_pg)
        # logger.info(
        #     f"Checking resource availability for pool '{self.name_prefix}': "
        #     f"requesting {num_pgs_needed} nodes/PGs with GPU counts {bundles_per_pg}."
        # )

        try:
            node_resources = ray.state.available_resources_per_node()
        except Exception as e:
            logger.warning(f"Could not query ray.state.available_resources_per_node(): {e}. Skipping resource check.")
            return

        # Use a mutable copy of available GPUs per node
        node_available_gpus = {
            node_info["NodeID"]: node_info.get("Resources", {}).get("GPU", 0) for node_info in ray.nodes() if node_info.get("Alive")
        }

        total_required_gpus = sum(bundles_per_pg)
        total_available_gpus = sum(node_available_gpus.values())

        if total_available_gpus < total_required_gpus:
            raise RuntimeError(
                f"Insufficient total GPU resources for pool '{self.name_prefix}'. "
                f"Required: {total_required_gpus}, Available: {total_available_gpus}. "
                f"Available GPUs per node: {node_available_gpus}"
            )

        pg_requirements = sorted(bundles_per_pg, reverse=True)
        
        # Simple greedy allocation check
        for required_gpus in pg_requirements:
            found_node = False
            # Find a node that can satisfy this PG requirement
            for node_id, available_gpus in sorted(node_available_gpus.items(), key=lambda item: item[1], reverse=True):
                if available_gpus >= required_gpus:
                    node_available_gpus[node_id] -= required_gpus
                    found_node = True
                    break
            
            if not found_node:
                raise RuntimeError(
                    f"Could not satisfy GPU requirement of {required_gpus} for pool '{self.name_prefix}'. "
                    f"Not enough nodes with at least {required_gpus} GPUs available simultaneously. "
                    f"Initial available GPUs per node: { {node_info['NodeID']: node_info.get('Resources', {}).get('GPU', 0) for node_info in ray.nodes() if node_info.get('Alive')} }"
                )
        
        # logger.info(f"Resource availability check passed for pool '{self.name_prefix}'.")


    def get_placement_groups(self, strategy="STRICT_PACK", timeout_seconds=180) -> List[PlacementGroup]:
        """Creates and returns placement groups based on the resource definition."""
        if self.pgs:
            return self.pgs

        # logger.info(
        #     f"Creating {self.num_nodes} placement group(s) for a total of {self.world_size} GPUs "
        #     f"with prefix '{self.name_prefix}'."
        # )

        remaining_gpus = self.world_size
        bundles_per_pg = []
        for _ in range(self.num_nodes):
            gpus_on_this_node = min(remaining_gpus, self.num_gpus_per_node)
            bundles_per_pg.append(gpus_on_this_node)
            remaining_gpus -= gpus_on_this_node
        
        # Pre-flight check for resource availability
        self._check_resources_available(bundles_per_pg)

        for i, gpus_on_this_node in enumerate(bundles_per_pg):
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(gpus_on_this_node)]
            pg_name = f"{self.name_prefix}_pg_{i}"
            pg = placement_group(bundles=bundles, strategy=strategy, name=pg_name)
            self.pgs.append(pg)

        # Wait for all placement groups to be ready.
        logger.info(f"Waiting for all placement groups to be ready (timeout: {timeout_seconds}s)...")
        ready_pgs, remaining_pgs = ray.wait([pg.ready() for pg in self.pgs], timeout=timeout_seconds)
        
        if remaining_pgs:
            raise TimeoutError(
                f"Placement group(s) for '{self.name_prefix}' failed to become ready within {timeout_seconds} seconds. "
                f"This may be due to resource fragmentation or insufficient resources in the cluster. "
                f"Please check the Ray dashboard for more details."
            )

        # logger.info("All placement groups are ready.")
        return self.pgs


class RayWorkerGroup:
    """Manages a group of Ray actors within PlacementGroups."""

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, master_addr: str, master_port: str):
        self.resource_pool = resource_pool
        self.ray_cls_with_init = ray_cls_with_init
        self.master_addr = master_addr
        self.master_port = master_port
        self.workers: List[ray.actor.ActorHandle] = []
        self._worker_names: List[str] = []

        self._init_workers()

    def _init_workers(self):
        """Initializes workers within the allocated placement groups."""
        pgs = self.resource_pool.get_placement_groups()
        world_size = self.resource_pool.world_size

        rank = 0
        for pg_idx, pg in enumerate(pgs):
            num_workers_on_node = len(pg.bundle_specs)
            for bundle_index in range(num_workers_on_node):
                if rank >= world_size:
                    break

                # Each worker is in its own bundle with one GPU, so its local rank is always 0.
                # The CUDA_VISIBLE_DEVICES is set by Ray, making the single assigned GPU appear as device 0.
                local_rank = 0

                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(local_rank),
                    "RAY_LOCAL_RANK": str(local_rank), # For compatibility with Verl/Ray's expectations
                    "MASTER_ADDR": self.master_addr,
                    "MASTER_PORT": self.master_port,
                }
                print(f"[DEBUG][RayWorkerGroup] Launching worker rank {rank} with env_vars={env_vars}")

                actor_name = f"{self.resource_pool.name_prefix}_worker_rank_{rank}_bundle_{bundle_index}"

                actor_handle = self.ray_cls_with_init.options(
                    name=actor_name,
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index, # Target the specific GPU bundle
                    runtime_env={"env_vars": env_vars},
                ).remote()

                self.workers.append(actor_handle)
                self._worker_names.append(actor_name)
                logger.info(f"Launched worker '{actor_name}' with rank {rank} and assigned local_rank {local_rank} to bundle {bundle_index}.")

                rank += 1

    @property
    def world_size(self) -> int:
        return len(self.workers)

    def get_workers(self) -> List[ray.actor.ActorHandle]:
        """Returns the list of worker actor handles."""
        return self.workers

    def execute_all_async(self, method_name: str, *args, **kwargs) -> List[ray.ObjectRef]:
        """
        Executes a method on all workers asynchronously.
        If args/kwargs are lists of the same length as the number of workers,
        each element is passed to the corresponding worker.
        """
        num_workers = len(self.workers)

        # Check if we should distribute arguments
        is_distributable = (
            all(isinstance(arg, list) and len(arg) == num_workers for arg in args)
            and all(isinstance(v, list) and len(v) == num_workers for v in kwargs.values())
        )

        if is_distributable:
            futures = []
            for i in range(num_workers):
                worker_args = [arg[i] for arg in args]
                worker_kwargs = {k: v[i] for k, v in kwargs.items()}
                futures.append(getattr(self.workers[i], method_name).remote(*worker_args, **worker_kwargs))
            return futures
        else:
            # Broadcast the same arguments to all workers
            return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self.workers]
