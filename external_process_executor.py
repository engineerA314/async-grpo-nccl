from typing import Callable, Optional, Any
from vllm.v1.executor.abstract import Executor
import ray
import logging
import pickle

logger = logging.getLogger(__name__)

def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    if not hasattr(vllm_config, "instance_id") or not vllm_config.instance_id:
        raise ValueError("vllm_config must have an 'instance_id' attribute.")

    fields = vllm_config.instance_id.split(":")
    if len(fields) != 2:
        raise ValueError(f"instance_id must be in the format <namespace>:<group_id>.")
    namespace, group_id = fields[0], fields[1]

    if init_ray and not ray.is_initialized():
        ray.init(address="auto", namespace=namespace)

    # Discover actors by name (following RayWorkerGroup's naming convention)
    # Find all actors that start with the group_id (e.g., "generation_group_0")
    actor_infos = ray.util.list_named_actors(all_namespaces=True)
    
    target_actors = [
        info for info in actor_infos
        if info["name"].startswith(group_id)
        and info["namespace"] == namespace
        and ("_worker_rank_" in info["name"])  # only pick real rollout workers
    ]
    
    if not target_actors:
        raise RuntimeError(f"No rollout workers found for group_id '{group_id}' in namespace '{namespace}'.")

    # Sort the discovered actors by their rank
    def get_rank_from_name(actor_info):
        name = actor_info["name"]
        try:
            # Example name: generation_group_0_worker_rank_0_local_0
            return int(name.split('_rank_')[-1].split('_')[0])
        except (ValueError, IndexError):
            return -1

    sorted_actors = sorted(target_actors, key=get_rank_from_name)
    sorted_actor_names = [actor["name"] for actor in sorted_actors]

    workers = [ray.get_actor(name, namespace=namespace) for name in sorted_actor_names]
    return workers

class ExternalProcessExecutor(Executor):
    """
    An executor that connects the vLLM engine to a set of pre-existing
    `RolloutWorker` Ray actors using Ray's native RPC, mirroring the
    design of verl's `ExternalRayDistributedExecutor`.
    """
    uses_ray: bool = False

    def _init_executor(self) -> None:
        """
        Initializes the executor by:
        1. Discovering the remote `RolloutGPUWorker` actors.
        2. Calling `init_vllm_worker` on each to create the vLLM worker instance.
        3. Calling `init_device` to set the GPU.
        4. Calling `load_model` to load weights.
        """
        
        # The primary method of getting workers should be the injected handles.
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        # The vLLM engine expects a list of kwargs, one for each worker.
        all_kwargs = []
        for rank, _ in enumerate(self.workers):
            kwargs = {
                "vllm_config": self.vllm_config,
                "local_rank": 0,  # Ray PlacementGroup assigns 1 GPU per actor, so local_rank is always 0
                "rank": rank,
                "distributed_init_method": "env://",
                "is_driver_worker": True,  # This seems to be required by vLLM
            }
            all_kwargs.append(kwargs)

        # 1. Initialize the vLLM Worker class inside our custom Ray actor
        self.collective_rpc("init_vllm_worker", all_kwargs)

        # 2. Initialize the device for each worker.
        self.collective_rpc("init_device")

        # 3. Load the model weights.
        self.collective_rpc("load_model")

    def collective_rpc(self, method: str, *args, **kwargs) -> list[Any]:
        """
        A simplified version of Ray's collective RPC that sends a command
        """
        futures = [
            worker.execute_method.remote(method, *args, **kwargs)
            for worker in self.workers
        ]
        return ray.get(futures)

    def check_health(self):
        # Health check logic can be added here if needed.
        pass
