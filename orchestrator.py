import asyncio
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import shutil
import re

import wandb
import ray
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import time
from tqdm.auto import tqdm
# REMOVED: import torch.distributed as dist

from ray.util.collective import collective

# Import physical worker classes with aliases to avoid name collisions
from gpu_process_worker import TrainerGPUWorker
from gpu_process_worker import LogProbGPUWorker
from gpu_process_worker import RolloutGPUWorker

# Import controller classes
from trainer_worker import TrainerWorker
from logprob_worker import LogProbFSDPWorker
from vllm_worker import GenerationVLLMWorker

from reward_registry import get_reward_adapter, RewardType

from dataclasses import dataclass
from typing import Optional, Any

from ray_utils import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup

# Use the canonical Message/MessageType from gpu_process_worker to avoid enum mismatches
from gpu_process_worker import MessageType, Message


logger = logging.getLogger(__name__)


async def get_experience_and_ref_logprobs(
    sample,
    num_samples,
    actor_registry_handle: ray.actor.ActorHandle,
    reference_registry_handles: List[ray.actor.ActorHandle], # Changed name for clarity
    temperature=1.0,
    max_tokens=8192,
):
    request_id = sample.get("request_id", "unknown_request")
    gen_start_time = time.time()
    gen_result = await actor_registry_handle.inference_balanced.remote(
        sample,
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    gen_duration = time.time() - gen_start_time
    samples, gen_metrics = gen_result
    # logger.debug(
    #     f"[Request {request_id}] Generated {len(samples)} samples. Now getting reference logprobs."
    # )

    logprob_start_time = time.time()
    
    # Simple round-robin load balancing for logprob requests
    tasks = []
    for i, s in enumerate(samples):
        controller_idx = i % len(reference_registry_handles)
        tasks.append(reference_registry_handles[controller_idx].inference_balanced.remote(s))

    samples_with_ref_logprobs = await asyncio.gather(*tasks)
    logprob_duration = time.time() - logprob_start_time
    # logger.debug(
    #     f"[Request {request_id}] Got reference logprobs for {len(samples_with_ref_logprobs)} samples."
    # )
    
    timing_metrics = {
        "gen_duration": gen_duration,
        "logprob_duration": logprob_duration,
    }
    
    return samples_with_ref_logprobs, gen_metrics, timing_metrics


class OrchestratorConfig:
    """
    Consolidates all configuration parameters for the training run,
    mirroring the arguments previously handled by Typer in trainer_core.py.
    This structure is inspired by the OmegaConf structure used in `verl`.
    """
    def __init__(self, **kwargs: Any):
        # Default values are set here
        self.model: dict = {
            "path": kwargs.get("model_name_or_path", "/path/to/model"),
            "lora_rank": 0,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "trust_remote_code": True,
        }
        self.trainer: dict = {
            "optim": {
                "lr": kwargs.get("learning_rate", 1e-6),
                "weight_decay": 0.0
            },
            "fsdp": {
                "param_offload": False,
                "optimizer_offload": False
            }
        }
        self.rollout: dict = {
            "name": "vllm",
            "mode": "async",
            "free_cache_engine": True
        }
        self.data_path: str = kwargs.get("data_path", "/path/to/data.jsonl")
        self.output_dir: str = kwargs.get("output_dir", "/path/to/output")
        self.test_data_path: str | None = kwargs.get("test_data_path")
        self.resume_from_path: str | None = kwargs.get("resume_from_path")
        self.resume_state: bool = bool(kwargs.get("resume_state", False))
        self.num_epochs: int = kwargs.get("num_epochs", 3)
        self.batch_size: int = kwargs.get("batch_size", 128)
        self.lr_scheduler: str = kwargs.get("lr_scheduler", "constant_with_warmup")
        self.num_warmup_steps: int = kwargs.get("num_warmup_steps", 10)
        self.clip_low: float = kwargs.get("clip_low", 0.2)
        self.clip_high: float = kwargs.get("clip_high", 0.28)
        self.clip_ratio_c: float = kwargs.get("clip_ratio_c", 3.0)
        self.samples_per_question: int = kwargs.get("samples_per_question", 32)
        self.temperature: float = kwargs.get("temperature", 1.0)
        self.max_generation_tokens: int = kwargs.get("max_generation_tokens", 4096)
        self.max_num_seqs: int = kwargs.get("max_num_seqs", 512)
        self.global_num_verifiers: int = kwargs.get("global_num_verifiers", 4)
        self.write_failed_generation_samples: bool = kwargs.get("write_failed_generation_samples", False)
        self.overhead_seqs: int = kwargs.get("overhead_seqs", 8)
        self.enable_prefix_caching: bool = kwargs.get("enable_prefix_caching", True)
        self.reward_fns: list[str] = kwargs.get("reward_fns", ["ksat_math"])
        self.num_training_gpu_workers: int = kwargs.get("num_training_gpu_workers", 4)
        self.num_generation_gpu_workers: int = kwargs.get("num_generation_gpu_workers", 1)
        self.num_logprob_gpu_workers: int = kwargs.get("num_logprob_gpu_workers", 1)
        self.generation_tp_size: int = kwargs.get("generation_tp_size", 1)
        self.logprob_fsdp_size: int = kwargs.get("logprob_fsdp_size", 1)
        self.experience_batcher_name: str = kwargs.get("experience_batcher_name", "experience_batcher")
        self.train_minibatch_sample_size: int = kwargs.get("train_minibatch_sample_size", 4000)
        self.max_tokens_per_gpu: int = kwargs.get("max_tokens_per_gpu", 36000)
        self.infinite_sampler_seed: int = kwargs.get("infinite_sampler_seed", 42)
        # Loss/compile control aligned with async_grpo
        # Entropy bonus controls (default align with VERL configs: 0.0 unless overridden)
        self.apply_entropy_bonus: bool = bool(kwargs.get("apply_entropy_bonus", True))
        # VERL defaults to 0 in most configs; allow override via runner kwargs
        self.entropy_coeff: float = float(kwargs.get("entropy_coeff", 0.0))
        self.use_torch_compile: bool = bool(kwargs.get("use_torch_compile", True))
        self.loss_chunksize: int | None = kwargs.get("loss_chunksize", None)
        self.wandb_project: str | None = kwargs.get("wandb_project")
        self.wandb_entity: str | None = kwargs.get("wandb_entity")
        self.wandb_run_name: str = kwargs.get("wandb_run_name", "async-grpo-run")
        self.job_id: str | None = kwargs.get("job_id")
        self.eval_every_n_macro_batches: int = kwargs.get("eval_every_n_macro_batches", 20)
        self.min_samples_per_checkpoint: int = kwargs.get("min_samples_per_checkpoint", 30000)
        self.keep_last_n_checkpoints: int = kwargs.get("keep_last_n_checkpoints", 2)
        # Packing control for logprob workers (default True to preserve behavior)
        # Manual system metrics collection (multi-node compatible, W&B-independent)
        self.enable_manual_system_metrics: bool = kwargs.get("enable_manual_system_metrics", True)
        self.system_metrics_interval_sec: int = kwargs.get("system_metrics_interval_sec", 1)

        # Control whether to run expensive weight sync verification during init
        # Default to False to speed up startup
        self.verify_weight_sync_during_init: bool = bool(kwargs.get("verify_weight_sync_during_init", False))

        # vLLM engine preset: "eager" (fast init) or "throughput" (slow init, faster run)
        self.vllm_engine_preset: str = str(kwargs.get("vllm_engine_preset", "eager"))

        # Control whether to run an evaluation at the very start of training
        self.eval_at_start: bool = bool(kwargs.get("eval_at_start", True))

        # Update model path from the top-level kwargs if provided
        if "model_name_or_path" in kwargs:
            self.model["path"] = kwargs["model_name_or_path"]

        # Inject LR scheduler settings for trainer optimizer (values passed from runner take precedence)
        warmup_steps = kwargs.get("lr_warmup_steps", 0)
        lr_scheduler = kwargs.get("lr_scheduler", "constant")
        total_steps = int(kwargs.get("total_training_steps", kwargs.get("estimated_total_optimizer_steps", 0)))
        self.trainer["optim"].update({
            "lr_warmup_steps": warmup_steps,
            "lr_scheduler": lr_scheduler,
            "total_training_steps": total_steps,
        })


@ray.remote
class Orchestrator:
    """
    The central coordinator for the entire async-grpo pipeline.
    It manages worker creation, data flow, training loops, evaluation, and checkpointing.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        # Ensure sane log levels inside the actor process (since launcher settings don't propagate)
        try:
            import logging as _logging
            _logging.getLogger().setLevel(_logging.INFO)
            for _name in (
                "urllib3",
                "urllib3.connectionpool",
                "filelock",
                "huggingface_hub",
                "httpx",
                "httpcore",
                "hpack",
                "h2",
                "grpclib",
            ):
                _logging.getLogger(_name).setLevel(_logging.WARNING)
            try:
                from transformers.utils import logging as _hf_logging  # type: ignore
                _hf_logging.set_verbosity_warning()
                _logging.getLogger("transformers").setLevel(_logging.WARNING)
            except Exception:
                pass
        except Exception:
            pass

        self.config = OrchestratorConfig(**config_dict)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model["path"], trust_remote_code=self.config.model["trust_remote_code"])
        self.global_step = 0
        self.total_samples_seen = 0
        self.macro_batch_count = 0
        self.last_checkpoint_samples_seen = 0
        self.last_eval_samples_seen = 0
        self._minibatch_global_step = 0

        # Handles for controller actors
        # REMOVED: self.comm_manager = None
        self.trainer_controller: ray.actor.ActorHandle = None
        self.logprob_controllers: List[ray.actor.ActorHandle] = []
        self.generation_controllers: List[ray.actor.ActorHandle] = []
        
        # Orchestrator-side experience manager actor (pull-based messaging API)
        self.experience_manager = ExperienceManager.remote(self.config)
        # Removed manual system metrics collector
        
        # Batch completion tracking
        self.expected_workers_per_batch = 0

    async def run(self):
        """
        The main entrypoint for the orchestration process.
        It sets up the distributed environment, spawns all workers and controllers,
        and starts the training loop.
        """
        if getattr(self.config, "resume_state", False):
            self._load_training_state()
        physical_workers = await self._spawn_physical_workers()
        await self._spawn_controllers(physical_workers)
        await self._initialize_workers(physical_workers)
        await self._run_training_loop()

    def _get_training_state_path(self) -> Path:
        """Returns the path to the training state file."""
        return Path(self.config.output_dir) / "training_state.json"

    def _load_training_state(self):
        """Loads the training state from a file if it exists."""
        state_path = self._get_training_state_path()
        if state_path.exists():
            logger.info(f"Found training state file at {state_path}. Loading state.")
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.total_samples_seen = state.get("total_samples_seen", 0)
                self.macro_batch_count = state.get("macro_batch_count", 0)
                self.last_checkpoint_samples_seen = self.total_samples_seen
                self.last_eval_samples_seen = state.get("last_eval_samples_seen", 0)
                logger.info(f"Resuming from global_step={self.global_step}, total_samples_seen={self.total_samples_seen}, macro_batch_count={self.macro_batch_count}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load training state from {state_path}: {e}. Starting from scratch.")
        else:
            logger.info("No training state file found. Starting from scratch.")

    def _save_training_state(self):
        """Saves the current training state to a file."""
        state = {
            "global_step": self.global_step,
            "total_samples_seen": self.total_samples_seen,
            "macro_batch_count": self.macro_batch_count,
            "last_eval_samples_seen": self.last_eval_samples_seen,
        }
        state_path = self._get_training_state_path()
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=4)
            logger.info(f"Training state saved to {state_path}")
        except IOError as e:
            logger.error(f"Failed to save training state to {state_path}: {e}")

    async def _spawn_physical_workers(self) -> Dict[str, List[ray.actor.ActorHandle]]:
        """Spawns all physical GPU workers using a consistent PlacementGroup strategy."""
        logger.info("Spawning all physical GPU workers using PlacementGroups...")
        workers = {"trainer": [], "logprob": [], "rollout": []}

        # This orchestrator is running on the head node of the Ray cluster.
        # We get its IP address to use as the master address for all distributed groups.
        master_addr = ray.util.get_node_ip_address()
        base_port = 29500
        logger.info(f"Determined master address for all groups: {master_addr}")

        # --- Spawn Trainer Workers (FSDP in a PlacementGroup) ---
        trainer_world_size = self.config.num_training_gpu_workers
        if trainer_world_size > 0:
            logger.info(f"Spawning {trainer_world_size} trainer workers in a dedicated PlacementGroup...")
            pool = RayResourcePool(
                world_size=trainer_world_size, 
                num_gpus_per_node=trainer_world_size,
                name_prefix="trainer_pool"
            )
            actor_cls = RayClassWithInitArgs(
                TrainerGPUWorker,
                config=self.config,
                resume_from_path=self.config.resume_from_path,
            )
            trainer_group = RayWorkerGroup(
                resource_pool=pool,
                ray_cls_with_init=actor_cls,
                master_addr=master_addr,
                master_port=str(base_port + 1)
            )
            self.trainer_worker_group = trainer_group
            workers["trainer"] = trainer_group.get_workers()

        # --- Spawn LogProb Workers (FSDP groups in PlacementGroups) ---
        num_logprob_gpu_workers = self.config.num_logprob_gpu_workers
        logprob_fsdp_size = self.config.logprob_fsdp_size
        if num_logprob_gpu_workers > 0:
            if num_logprob_gpu_workers % logprob_fsdp_size != 0:
                raise ValueError(f"num_logprob_gpu_workers ({num_logprob_gpu_workers}) must be divisible by logprob_fsdp_size ({logprob_fsdp_size})")
            
            num_logprob_groups = num_logprob_gpu_workers // logprob_fsdp_size
            self.logprob_worker_groups: List[RayWorkerGroup] = []
            for i in range(num_logprob_groups):
                logger.info(f"Spawning logprob group {i} with FSDP size {logprob_fsdp_size} in a PlacementGroup...")
                pool = RayResourcePool(
                    world_size=logprob_fsdp_size, 
                    num_gpus_per_node=logprob_fsdp_size,
                    name_prefix=f"logprob_pool_{i}"
                )
                actor_cls = RayClassWithInitArgs(
                    LogProbGPUWorker,
                    config=self.config,
                )
                worker_group = RayWorkerGroup(
                    resource_pool=pool,
                    ray_cls_with_init=actor_cls,
                    master_addr=master_addr,
                    master_port=str(base_port + 2 + i)
                )
                self.logprob_worker_groups.append(worker_group)
                workers["logprob"].extend(worker_group.get_workers())

        # --- Spawn Rollout Workers (vLLM - Tensor Parallel via PlacementGroup) ---
        total_rollout_gpus = self.config.num_generation_gpu_workers
        tp_size = self.config.generation_tp_size
        if total_rollout_gpus > 0:
            if total_rollout_gpus % tp_size != 0:
                raise ValueError(f"num_generation_gpu_workers ({total_rollout_gpus}) must be divisible by generation_tp_size ({tp_size})")
            
            num_generation_groups = total_rollout_gpus // tp_size
            self.rollout_worker_groups: List[RayWorkerGroup] = []
            for i in range(num_generation_groups):
                group_id_prefix = f"generation_group_{i}"
                logger.info(f"Setting up RayResourcePool for {group_id_prefix}...")
                pool = RayResourcePool(
                    world_size=tp_size, 
                    num_gpus_per_node=tp_size,
                    name_prefix=group_id_prefix
                )
                actor_cls = RayClassWithInitArgs(RolloutGPUWorker, config=self.config)
                worker_group = RayWorkerGroup(
                    resource_pool=pool,
                    ray_cls_with_init=actor_cls,
                    master_addr=master_addr,
                    master_port=str(base_port + 10 + i)
                )
                self.rollout_worker_groups.append(worker_group)
                workers["rollout"].extend(worker_group.get_workers())

        logger.info(f"Spawned {len(workers['trainer'])} trainer, {len(workers['logprob'])} logprob, and {len(workers['rollout'])} physical workers.")

        # ... (rest of the function remains the same)
        self.physical_workers = workers
        return workers

    async def _spawn_controllers(self, physical_workers: Dict[str, List[ray.actor.ActorHandle]]):
        """Spawns the controller actors, passing them the handles of the physical workers."""
        logger.info("Spawning controller actors...")

        # --- Spawn Trainer Controller ---
        if hasattr(self, 'trainer_worker_group'):
            trainer_handles = self.trainer_worker_group.get_workers()
            trainer_fsdp_ranks = list(range(len(trainer_handles)))
            self.trainer_controller = TrainerWorker.remote(
                group_id="trainer_group", 
                worker_handles=trainer_handles,
                fsdp_group_ranks=trainer_fsdp_ranks,
            )
        
        # --- Spawn LogProb Controllers ---
        if hasattr(self, 'logprob_worker_groups'):
            for i, worker_group in enumerate(self.logprob_worker_groups):
                group_handles = worker_group.get_workers()
                group_ranks = list(range(len(group_handles)))
                logprob_controller = LogProbFSDPWorker.remote(
                    group_id=f"logprob_group_{i}",
                    worker_handles=group_handles,
                    fsdp_group_ranks=group_ranks,
                    trainer_controller_handle=self.trainer_controller,
                    max_tokens=self.config.max_tokens_per_gpu,
                )
                self.logprob_controllers.append(logprob_controller)
        logger.info(f"Spawned {len(self.logprob_controllers)} LogProbFSDPWorker controllers.")

        # --- Spawn GenerationVLLMWorker controllers ---
        self.generation_controllers = []
        if hasattr(self, 'rollout_worker_groups'):
            # One controller per `RayWorkerGroup` we created.
            for i, worker_group in enumerate(self.rollout_worker_groups):
                group_id = f"generation_group_{i}" # Use the same consistent ID
                controller = GenerationVLLMWorker.remote(
                    model_path=self.config.model["path"],
                    worker_handles=worker_group.get_workers(),
                    group_id=group_id, # Pass the consistent ID
                    tensor_parallel_size=worker_group.world_size,
                    max_num_seqs=self.config.max_num_seqs,
                    global_num_verifiers=self.config.global_num_verifiers,
                    write_failed=self.config.write_failed_generation_samples,
                    overhead_seqs=self.config.overhead_seqs,
                    enable_prefix_caching=self.config.enable_prefix_caching,
                    reward_fns=self.config.reward_fns,
                    namespace=ray.get_runtime_context().namespace,
                    engine_preset=self.config.vllm_engine_preset,
                )
                self.generation_controllers.append(controller)
        logger.info(f"Spawned {len(self.generation_controllers)} GenerationVLLMWorker controllers.")

    async def _initialize_workers(self, physical_workers: Dict[str, List[ray.actor.ActorHandle]]):
        """Initializes all worker groups through their controllers, in the correct order."""
        logger.info("--- Starting Worker Initialization Sequence ---")
        master_addr = os.environ.get("MASTER_ADDR", "localhost")

        # Stage 0: Initialize Distributed Environment for ALL workers.
        # Since all workers are now in placement groups, we need to set up their
        # dist init method. `RayWorkerGroup` already set RANK, WORLD_SIZE, LOCAL_RANK.
        logger.info("Stage 0: Initializing distributed environment for all workers...")
        
        # We need to set a unique port for each distinct distributed group.
        init_tasks = []
        
        # Trainer group
        if hasattr(self, 'trainer_worker_group'):
            port = 29501
            init_method = f"tcp://{master_addr}:{port}"
            for worker in self.trainer_worker_group.get_workers():
                # Pass the init_method directly to the initialization function.
                init_tasks.append(worker.init_distributed_env.remote(init_method))

        # LogProb groups
        if hasattr(self, 'logprob_worker_groups'):
            for i, group in enumerate(self.logprob_worker_groups):
                port = 29502 + i
                init_method = f"tcp://{master_addr}:{port}"
                for worker in group.get_workers():
                    # Pass the init_method directly to the initialization function.
                    init_tasks.append(worker.init_distributed_env.remote(init_method))

        # vLLM/Rollout workers handle their own distributed init inside the engine,
        # which uses the environment variables we've set. So we don't call it here.

        if init_tasks:
            await asyncio.gather(*init_tasks)
        logger.info("✅ Stage 0 Complete: FSDP workers have initialized torch.distributed.")
        
        # Stage 1 & 2: Init Trainer FSDP and get its weights
        logger.info("Stage 1 & 2: Initializing TrainerWorkers and fetching their weights...")
        await self.trainer_controller.initialize_fsdp.remote()
        logger.info("✅ Stage 1 Complete: TrainerWorkers initialized.")
        state_dict_bytes = await self.trainer_controller.get_full_cpu_state_dict.remote()
        if not state_dict_bytes:
            raise RuntimeError("Failed to get initial state_dict from the trainer controller.")
        logger.info("✅ Stage 2 Complete: State dict fetched.")

        # Stage 3: Init LogProb workers with the trainer's weights
        logger.info("Stage 3: Initializing LogProbWorkers with trainer weights...")
        logprob_init_tasks = [
            controller.initialize_fsdp.remote(state_dict_to_load=state_dict_bytes)
            for controller in self.logprob_controllers
        ]
        await asyncio.gather(*logprob_init_tasks)
        logger.info("✅ Stage 3 Complete.")
        
        # Stage 3.5: Load optimizer and scheduler state if resuming from a checkpoint
        if self.config.resume_from_path:
            logger.info("Stage 3.5: Loading optimizer and scheduler state from checkpoint...")
            await self.trainer_controller.load_optimizer_state.remote(self.config.resume_from_path)
            logger.info("✅ Stage 3.5 Complete: Optimizer and scheduler state loaded.")
        
        # --- Parallel Initialization Block ---
        init_tasks = []

        # Task Group 1: Initialize vLLM engines
        logger.info("Stage 4: Initializing vLLM engines on GenerationControllers (in parallel)...")
        init_engine_tasks = [controller.init_engine.remote() for controller in self.generation_controllers]
        init_tasks.extend(init_engine_tasks)

        # Task Group 2: Distribute weight metadata
        logger.info("Stage 5: Distributing weight metadata (in parallel)...")
        trainer_workers = await self.trainer_controller.get_worker_handles.remote()
        # Rank 0 of the FSDP group is the source of truth
        weights_info = await trainer_workers[0].get_trainer_weights_info.remote()

        all_workers_to_set = (
            self.physical_workers["trainer"][1:] +  # Exclude rank 0 as it already has the info
            self.physical_workers["logprob"] +
            self.physical_workers["rollout"]
        )
        set_info_tasks = [worker.set_trainer_weights_info.remote(weights_info) for worker in all_workers_to_set]
        init_tasks.extend(set_info_tasks)

        # Run vLLM init and weight info distribution concurrently
        await asyncio.gather(*init_tasks)
        logger.info("✅ Stages 4 & 5 Complete: vLLM engines initialized and weight info distributed.")

        # --- Synchronization Block ---

        # Stage 6: Create the collective group to bridge all workers for synchronization.
        await self._create_collective_group(self.physical_workers)
        logger.info("✅ Stage 6 Complete: Collective sync group created.")

        # Stage 7: Sync initial weights to all workers, including the now-ready RolloutWorkers.
        logger.info("Stage 7: Syncing initial weights to all workers...")
        await self.sync_all_weights()
        # Optionally verify sync (expensive); default is disabled
        if getattr(self.config, "verify_weight_sync_during_init", False):
            await self._verify_trainer_logprob_sync()
        
        logger.info("--- ✅ All workers initialized successfully ---")

    async def _create_collective_group(self, physical_workers: Dict[str, List[ray.actor.ActorHandle]]):
        """Creates a single NCCL group for all GPU workers using ray.util.collective."""
        logger.info("Creating unified synchronization group using ray.util.collective...")
        
        all_gpu_workers = (
            physical_workers["trainer"] + 
            physical_workers["logprob"] + 
            physical_workers["rollout"]
        )
        
        if not all_gpu_workers:
            logger.warning("No GPU workers found to create a sync group.")
            return

        world_size = len(all_gpu_workers)
        group_name = "sync_group"

        # This one line creates the NCCL group across all specified actors.
        # This is a BLOCKING call, not an awaitable coroutine.
        collective.create_collective_group(
            all_gpu_workers, world_size, ranks=list(range(world_size)), backend="nccl", group_name=group_name
        )
        
        # Store the global ranks list in the order they were passed to create_collective_group
        # This is crucial for determining the source rank for broadcasts.
        self.collective_group_ranks = await asyncio.gather(*[w.get_global_rank.remote() for w in all_gpu_workers])

        logger.info(f"✅ Collective group '{group_name}' created with {world_size} workers.")

    async def _initialize_wandb_for_all_workers(self):
        """Deprecated: worker-side W&B shared init removed. No-op."""
        return

    # Removed: _poll_and_log_system_metrics (manual system/GPU metric logging)

    async def sync_all_weights(self):
        """Triggers weight synchronization from the trainer to all other worker groups via their controllers."""
        logger.info("Orchestrating weight synchronization via controllers...")
        
        if not self.trainer_controller:
            logger.error("Cannot sync weights, trainer controller not found.")
            return
        
        # The global rank of the FSDP group's leader (rank 0).
        fsdp_leader_actor_list = await self.trainer_controller.get_worker_handles.remote()
        if not fsdp_leader_actor_list:
            logger.error("Cannot sync, trainer controller has no workers.")
            return
        fsdp_leader_actor = fsdp_leader_actor_list[0]
        fsdp_leader_global_rank = -1
        
        # Find the actor handle in the list to get its global rank from the pre-computed list
        all_workers = (
            self.physical_workers["trainer"] + 
            self.physical_workers["logprob"] + 
            self.physical_workers["rollout"]
        )
        
        try:
            actor_index = -1
            for i, worker in enumerate(all_workers):
                if worker == fsdp_leader_actor:
                    actor_index = i
                    break
            if actor_index == -1:
                raise ValueError("FSDP leader actor not found in the list of all physical workers.")
                
            source_rank_in_collective_group = actor_index
            fsdp_leader_global_rank = self.collective_group_ranks[actor_index]
        except (ValueError, IndexError) as e:
            logger.error(f"Could not determine the source rank for broadcast: {e}")
            return

        logger.info(f"Determined source for broadcast. FSDP leader global rank: {fsdp_leader_global_rank}, Index in collective group: {source_rank_in_collective_group}")

        sync_tasks = [
            self.trainer_controller.sync_weights.remote("sync_group", source_rank_in_collective_group),
        ]
        # Add sync tasks for all logprob controllers
        for controller in self.logprob_controllers:
            sync_tasks.append(controller.sync_weights.remote("sync_group", source_rank_in_collective_group))
        
        # Add sync tasks for all generation controllers
        for controller in self.generation_controllers:
            sync_tasks.append(controller.sync_weights.remote("sync_group", source_rank_in_collective_group))
        # logger.warning("Temporarily disabling weight synchronization for GenerationVLLMWorker to debug weight corruption issues.")

        await asyncio.gather(*sync_tasks)
        logger.info(f"✅ Weight synchronization complete from collective group source rank {source_rank_in_collective_group}.")

    async def _verify_trainer_logprob_sync(self):
        """
        Verifies that the weights of the LogProb worker match the Trainer worker.
        This is for debugging and ensuring sync works correctly.
        """
        logger.info("--- Starting Weight Sync Verification: Trainer vs. LogProb ---")
        
        if not self.trainer_controller or not self.logprob_controllers:
            logger.warning("Verification skipped: Trainer or LogProb controllers not available.")
            return

        try:
            # Get state dict from Trainer (Rank 0 of the FSDP group)
            logger.info("Fetching state dict from Trainer controller...")
            trainer_state_dict_bytes = await self.trainer_controller.get_full_cpu_state_dict.remote()
            if not trainer_state_dict_bytes:
                logger.error("Failed to get state dict from trainer. Verification aborted.")
                return
            trainer_state_dict = pickle.loads(trainer_state_dict_bytes)
            logger.info(f"Trainer state dict received. Num keys: {len(trainer_state_dict)}.")

            # Get state dict from the first LogProb controller (Rank 0 of its FSDP group)
            logprob_controller_to_check = self.logprob_controllers[0]
            logger.info(f"Fetching state dict from LogProb controller {logprob_controller_to_check}...")
            logprob_state_dict_bytes = await logprob_controller_to_check.get_full_cpu_state_dict.remote()
            if not logprob_state_dict_bytes:
                logger.error("Failed to get state dict from logprob worker. Verification aborted.")
                return
            logprob_state_dict = pickle.loads(logprob_state_dict_bytes)
            logger.info(f"LogProb state dict received. Num keys: {len(logprob_state_dict)}.")

            # --- Comparison Logic ---
            mismatched_keys = []
            keys = set(trainer_state_dict.keys()) | set(logprob_state_dict.keys())

            for key in sorted(list(keys)):
                if key not in logprob_state_dict:
                    mismatched_keys.append((key, "missing from logprob"))
                    continue
                if key not in trainer_state_dict:
                    mismatched_keys.append((key, "missing from trainer"))
                    continue
                
                trainer_tensor = trainer_state_dict[key]
                logprob_tensor = logprob_state_dict[key]

                if trainer_tensor.shape != logprob_tensor.shape:
                    mismatched_keys.append((key, f"tensor shapes differ. trainer: {trainer_tensor.shape}, logprob: {logprob_tensor.shape}"))
                    continue

                if not torch.equal(trainer_tensor, logprob_tensor):
                    diff = torch.abs(trainer_tensor.to(torch.float32) - logprob_tensor.to(torch.float32)).sum()
                    mismatched_keys.append((key, f"tensor values differ, sum of abs diff: {diff}"))

            if not mismatched_keys:
                logger.info("✅✅✅ Verification SUCCESS: All keys and tensor values match between Trainer and LogProb worker.")
            else:
                logger.error(f"🚨🚨🚨 Verification FAILED: Found {len(mismatched_keys)} mismatched keys/tensors.")
                for key, reason in mismatched_keys[:20]: # Log first 20 mismatches
                    logger.error(f"  - Mismatch in key '{key}': {reason}")
        
        except Exception as e:
            logger.error(f"An exception occurred during verification: {e}", exc_info=True)
        
        logger.info("--- Finished Weight Sync Verification ---")

    async def _run_training_loop(self):
        """
        # Main loop integrating logic from trainer_core.py and vllm_experience_batcher.py.
        """
        # --- WandB Initialization ---
        if self.config.wandb_project:
            config = self.config.__dict__
            tags = ["async-grpo-orchestrator"]
            group_name = self.config.wandb_run_name
            if self.config.job_id:
                config['job_id'] = self.config.job_id
                tags.append(self.config.job_id)
                group_name = self.config.job_id
            
            # Probe GPU visibility before WandB init
            try:
                import subprocess as _subprocess
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES")
                try:
                    _smi = _subprocess.run(["nvidia-smi", "-L"], stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True, check=False)
                    logger.info(f"[SYS] nvidia-smi -L:\n{_smi.stdout.strip()}")
                except Exception as _e:
                    logger.warning(f"[SYS] nvidia-smi -L failed: {_e}")
            except Exception as _e:
                logger.warning(f"[SYS] GPU env probe failed: {_e}")

            # Initialize WandB in standard mode (worker-side shared init is removed)
            wandb.init(
                id=self.config.job_id,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.wandb_run_name,
                config=config,
                group=group_name,
                tags=tags,
                resume="allow",
                settings=wandb.Settings(sync_tensorboard=False),
            )
            
            # All metrics will use global_step as x-axis
            logger.info(f"✅ WandB initialized for run: {self.config.wandb_run_name}")
            try:
                wandb.define_metric("trainer_step")
                wandb.define_metric("minibatch_step")
                wandb.define_metric("training/*", step_metric="trainer_step")
                wandb.define_metric("generation/*", step_metric="trainer_step")
                wandb.define_metric("timing/*", step_metric="trainer_step")
                wandb.define_metric("eval/*", step_metric="trainer_step")
                wandb.define_metric("minibatch/*", step_metric="minibatch_step")
            except Exception as _e:
                logger.warning(f"Failed to define WandB metrics: {_e}")

        # Register TrainerGPUWorkers directly with the ExperienceManager
        trainer_worker_handles = await self.trainer_controller.get_worker_handles.remote()
        
        # Set up expected workers count for batch completion tracking
        self.expected_workers_per_batch = len(trainer_worker_handles)
        
        # Register each TrainerGPUWorker with ExperienceManager actor and start their training loops
        for gpu_rank, worker_handle in enumerate(trainer_worker_handles):
            await self.experience_manager.register_training_process.remote(
                gpu_rank,
                self.config.max_tokens_per_gpu,
                self.config.train_minibatch_sample_size,
            )
            await worker_handle.set_orchestrator.remote(self.experience_manager, gpu_rank)
            
            # WandB initialization is now done in _initialize_workers stage
            
            # Start the training loop on the TrainerGPUWorker
            # Ray's remote() returns ObjectRef, not coroutine, so we don't use create_task
            worker_handle.start_training_loop.remote()
        
        dataset = load_dataset("json", data_files=self.config.data_path, split="train")
        # Attach deterministic dataset indices for cross-framework alignment checks
        try:
            if "__uid__" not in dataset.column_names:
                dataset = dataset.map(lambda ex, idx: {"__uid__": int(idx)}, with_indices=True)
                logger.info("Added __uid__ column to dataset for alignment tracing.")
        except Exception:
            pass
        # Pass a custom collate_fn to handle variable-length sequences.
        # The lambda function simply returns the list of samples as-is,
        # preventing the DataLoader from trying to stack them into a tensor,
        # which would fail due to different sequence lengths.
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=lambda x: x)
        
        # Optional: run an initial evaluation at step 0 before any training
        if self.config.eval_at_start and self.global_step == 0:
            try:
                await self.run_evaluation()
            except Exception as e:
                logger.warning(f"Initial evaluation failed: {e}")

        # Calculate training overview
        total_batches_per_epoch = len(dataloader)
        total_batches = total_batches_per_epoch * self.config.num_epochs
        total_questions = len(dataset) * self.config.num_epochs
        expected_total_samples = total_questions * self.config.samples_per_question
        
        logger.info("=" * 80)
        logger.info("🚀 TRAINING OVERVIEW")
        logger.info("=" * 80)
        logger.info(f"📊 Dataset: {len(dataset)} questions")
        logger.info(f"🔄 Epochs: {self.config.num_epochs}")
        logger.info(f"📦 Batch size: {self.config.batch_size}")
        logger.info(f"🎯 Samples per question: {self.config.samples_per_question}")
        logger.info(f"📈 Batches per epoch: {total_batches_per_epoch}")
        logger.info(f"📊 Total batches: {total_batches}")
        logger.info(f"🔬 Expected total samples: {expected_total_samples:,}")
        logger.info(f"⚡ Train minibatch sample size: {self.config.train_minibatch_sample_size}")
        logger.info("=" * 80)
        
        logger.info("Starting main training loop...")
        
        # --- Resume-aware offsets ---
        # Use saved macro_batch_count to compute epoch/batch offsets for resume
        done_macro_batches = int(getattr(self, "macro_batch_count", 0) or 0)
        start_epoch = 0
        start_batch_offset = 0
        if done_macro_batches > 0 and total_batches_per_epoch > 0:
            start_epoch = min(done_macro_batches // total_batches_per_epoch, max(0, self.config.num_epochs - 1))
            start_batch_offset = done_macro_batches % total_batches_per_epoch
            logger.info(
                f"[RESUME] Resuming from saved state: global_step={self.global_step}, "
                f"macro_batch_count={self.macro_batch_count} -> start_epoch={start_epoch+1}, "
                f"start_batch_offset={start_batch_offset}/{total_batches_per_epoch}"
            )

        # Create overall epoch progress bar (resume-aware)
        epoch_pbar = tqdm(
            total=self.config.num_epochs,
            desc="Overall Progress",
            unit="epoch",
            position=0,
            initial=start_epoch,
        )
        
        try:
            for epoch in range(start_epoch, self.config.num_epochs):
                logger.info(f"--- Starting Epoch {epoch + 1}/{self.config.num_epochs} ---")
                
                # Create progress bar for this epoch (manual update; resume-aware initial)
                pbar = tqdm(
                    total=total_batches_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                    unit="batch",
                    position=1,
                    leave=False,
                    initial=(start_batch_offset if epoch == start_epoch else 0),
                )
                
                epoch_start_time = time.time()
                
                for batch_idx, batch in enumerate(dataloader):
                    # Skip already-finished batches in the first epoch after resume
                    if epoch == start_epoch and start_batch_offset > 0 and batch_idx < start_batch_offset:
                        continue
                    # --- Timing Setup ---
                    batch_start_time = time.time()
                    
                    self.macro_batch_count += 1
                    logger.info(f"Processing macro-batch #{self.macro_batch_count} for global step {self.global_step}...")
                    
                    step_metrics = [] # Reset metrics list for each macro-batch
                    
                    # dist.barrier() # This actor runs on a single process, so a barrier here will cause a deadlock.

                    # 1. Asynchronously request experience generation (this function returns immediately)
                    gen_controllers = self.generation_controllers
                    logprob_controllers = self.logprob_controllers
                    # Stamp the same uids through pipeline for per-sample tracing + quick logging
                    try:
                        tagged = []
                        uids = []
                        for s in batch:
                            s = dict(s)
                            uid_val = s.get("__uid__", None)
                            if uid_val is None:
                                # Derive from input text as fallback
                                import hashlib as _hashlib
                                _inp = (s.get("input", "") or "").encode("utf-8")
                                uid_val = _hashlib.sha256(_inp).hexdigest()
                            else:
                                uid_val = str(uid_val)
                            s["__uid__"] = uid_val
                            uids.append(uid_val)
                            tagged.append(s)
                        # lightweight batch UID preview for cross-framework alignment
                        await self.experience_manager.generate_experience.remote(
                            tagged, gen_controllers, logprob_controllers
                        )
                    except Exception:
                        await self.experience_manager.generate_experience.remote(
                            batch, gen_controllers, logprob_controllers
                        )

                    # 2. Trigger batch creation on the ExperienceManager actor
                    # Trigger batch creation on the ExperienceManager actor
                    await self.experience_manager.start_creating_batches.remote()

                    # 3. Wait for all TrainerGPUWorkers to complete the batch
                    batch_start_time = time.time()
                    
                    logger.info(f"Batch processing initiated for global step {self.global_step}")
                    
                    # Wait for all TrainerGPUWorkers to complete the batch using dedicated completion queues
                    training_metrics_list = []
                    batch_metrics_list = []
                    all_batch_data = []
                    minibatch_logs = []

                    for rank in range(self.expected_workers_per_batch):
                        try:
                            completion_data = await asyncio.wait_for(
                                self.experience_manager.get_completion.remote(rank),
                                timeout=300,
                            )
                            gpu_rank = completion_data.get("gpu_rank", rank)
                            metrics = completion_data.get("metrics", {})
                            batch_metrics = completion_data.get("batch_metrics", {})

                            # logger.info(f"Received batch completion from GPU rank {gpu_rank}: {metrics}")

                            if metrics:
                                training_metrics_list.append(metrics)
                            if batch_metrics:
                                batch_metrics_list.append(batch_metrics)
                            all_batch_data.append(completion_data)
                            minibatches = completion_data.get("minibatch_summaries") or []
                            for entry in minibatches:
                                entry_with_meta = dict(entry)
                                entry_with_meta["trainer_rank"] = gpu_rank
                                minibatch_logs.append(entry_with_meta)
                        except asyncio.TimeoutError:
                            logger.error(
                                f"Timeout waiting for batch completion for rank {rank}. Received {len(all_batch_data)}/{self.expected_workers_per_batch}."
                            )
                            # Check liveness of trainer workers; abort early if any dead
                            try:
                                liveness = await self.trainer_controller.ping_all.remote()
                                dead = [r for r, ok in liveness.items() if not ok]
                                if dead:
                                    raise RuntimeError(f"Trainer workers not alive: {dead}")
                            except Exception as e:
                                logger.error(f"Detected trainer failure during batch wait: {e}")
                                raise
                            break
                    
                    # Aggregate training metrics (token-weighted across workers)
                    final_training_metrics = {}
                    if training_metrics_list:
                        # Read token_count from each worker's metrics payload
                        token_counts = [m.get("token_count", 0) for m in training_metrics_list]
                        total_tokens = sum(token_counts) if token_counts and sum(token_counts) > 0 else 0

                        def _tw_mean(k: str) -> float:
                            if total_tokens <= 0:
                                return float(sum(m.get(k, 0.0) for m in training_metrics_list) / max(1, len(training_metrics_list)))
                            s = 0.0
                            for i, m in enumerate(training_metrics_list):
                                s += float(m.get(k, 0.0)) * float(token_counts[i])
                            return s / float(total_tokens)

                        # Core metrics
                        for k in ("loss", "loss_per_token", "kl_div", "entropy"):
                            final_training_metrics[k] = _tw_mean(k)

                        # Fractions / rates -> token-weighted mean
                        for k in ("clipfrac_upper", "clipfrac_lower", "adv_pos_frac", "adv_zero_frac"):
                            if any(k in m for m in training_metrics_list):
                                final_training_metrics[k] = _tw_mean(k)

                        # Means/Std: combine via second-moment identity
                        def _tw_mean_std(mean_key: str, std_key: str) -> None:
                            if not any(mean_key in m for m in training_metrics_list):
                                return
                            if total_tokens <= 0:
                                mu_vals = [float(m.get(mean_key, 0.0)) for m in training_metrics_list]
                                sd_vals = [float(m.get(std_key, 0.0)) for m in training_metrics_list]
                                mu = sum(mu_vals) / max(1, len(mu_vals))
                                # naive combine for no-token case
                                final_training_metrics[mean_key] = mu
                                final_training_metrics[std_key] = sum(sd_vals) / max(1, len(sd_vals))
                                return
                            s_mu = 0.0
                            s_ex2 = 0.0
                            for i, m in enumerate(training_metrics_list):
                                tok = float(token_counts[i])
                                mu_i = float(m.get(mean_key, 0.0))
                                sd_i = float(m.get(std_key, 0.0))
                                s_mu += mu_i * tok
                                s_ex2 += ((sd_i * sd_i) + (mu_i * mu_i)) * tok
                            mu = s_mu / float(total_tokens)
                            ex2 = s_ex2 / float(total_tokens)
                            var = max(ex2 - mu * mu, 0.0)
                            final_training_metrics[mean_key] = mu
                            final_training_metrics[std_key] = var ** 0.5

                        _tw_mean_std("log_ratio_mean", "log_ratio_std")
                        _tw_mean_std("adv_mean", "adv_std")
                        # Grad norm: simple mean across workers that reported it
                        try:
                            gn_vals = [float(m.get("grad_norm", float("nan"))) for m in training_metrics_list if "grad_norm" in m]
                            gn_vals = [v for v in gn_vals if v == v]
                            if gn_vals:
                                final_training_metrics["grad_norm"] = sum(gn_vals) / float(len(gn_vals))
                        except Exception:
                            pass
                    
                    # Aggregate batch metrics (prefer ExperienceManager aggregates)
                    final_batch_metrics = {}
                    # Prefer ExperienceManager's single aggregated snapshot if available
                    exp0 = (all_batch_data[0].get("exp_batch_metrics") or {}) if all_batch_data else {}
                    if exp0:
                        final_batch_metrics = {
                            "total_samples": int(sum(bm.get("total_samples", 0) for bm in batch_metrics_list)) if batch_metrics_list else int(exp0.get("total_samples", 0)),
                            "total_reward": float(sum(bm.get("total_reward", 0) for bm in batch_metrics_list)) if batch_metrics_list else float(exp0.get("total_reward", 0.0)),
                            "avg_reward": float(exp0.get("avg_reward", 0.0)),
                            "reward_mean": float(exp0.get("reward_mean", exp0.get("avg_reward", 0.0))),
                            "reward_std": float(exp0.get("reward_std", 0.0)),
                            "reward_min": float(exp0.get("reward_min", 0.0)),
                            "reward_max": float(exp0.get("reward_max", 0.0)),
                            "group_reward_std_mean": float(exp0.get("group_reward_std_mean", 0.0)),
                            "group_reward_std_zero_frac": float(exp0.get("group_reward_std_zero_frac", 0.0)),
                            "nonzero_reward_frac": float(exp0.get("nonzero_reward_frac", 0.0)),
                            "avg_completion_length": float(exp0.get("avg_completion_length", 0.0)),
                        }
                    elif batch_metrics_list:
                        total_samples = sum(bm.get("total_samples", 0) for bm in batch_metrics_list)
                        total_reward = sum(bm.get("total_reward", 0) for bm in batch_metrics_list)
                        final_batch_metrics = {
                            "total_samples": total_samples,
                            "total_reward": total_reward,
                            "avg_reward": total_reward / total_samples if total_samples > 0 else 0,
                        }
                    
                    self.global_step += 1
                    total_batch_time = time.time() - batch_start_time
                    
                    # Update total samples seen
                    if final_batch_metrics:
                        self.total_samples_seen += final_batch_metrics.get("total_samples", 0)
                    
                    # Calculate throughput metrics
                    samples_per_second = final_batch_metrics.get("total_samples", 0) / total_batch_time if total_batch_time > 0 else 0
                    
                    # Get generation and logprob timing from ExperienceManager
                    # Note: This would need to be passed from BATCH_DONE message data
                    gen_logprob_metrics = all_batch_data[0].get("gen_logprob_metrics", {}) if all_batch_data else {}
                    gen_duration = gen_logprob_metrics.get("gen_duration", 0)
                    logprob_duration = gen_logprob_metrics.get("logprob_duration", 0)
                    gen_logprob_wall_time = gen_logprob_metrics.get("gen_logprob_wall_time", 0)
                    
                    # Calculate overhead and training time
                    trainer_active_time = total_batch_time - gen_logprob_wall_time  # Approximate
                    total_overhead = max(0, total_batch_time - gen_duration - logprob_duration - trainer_active_time)
                    
                    # Update progress bar with comprehensive metrics
                    final_metrics_for_pbar = {
                        "step": self.global_step,
                        "batch_time": f"{total_batch_time:.1f}s",
                        "samples": self.total_samples_seen,
                        "samples_sec": f"{samples_per_second:.2f}"
                    }
                    
                    if final_training_metrics:
                        final_metrics_for_pbar.update({
                            "loss": f"{final_training_metrics.get('loss', 0):.4f}",
                            "grad_norm": f"{final_training_metrics.get('grad_norm', 0):.4f}"
                        })
                    
                    if final_batch_metrics:
                        final_metrics_for_pbar.update({
                            "reward": f"{final_batch_metrics.get('avg_reward', 0):.4f}"
                        })
                    
                    pbar.set_postfix(final_metrics_for_pbar)
                    pbar.update(1)
                    
                    # Comprehensive WandB logging (similar to original trainer_core.py)
                    if self.config.wandb_project and wandb.run:
                        if minibatch_logs:
                            minibatch_count = len(minibatch_logs)
                            base_step = self._minibatch_global_step
                            for idx, mb in enumerate(minibatch_logs):
                                step_id = base_step + idx
                                minibatch_log = {
                                    "minibatch/loss": mb.get("loss", 0.0),
                                    "minibatch/kl_div": mb.get("kl_div", 0.0),
                                    "minibatch/entropy": mb.get("entropy", 0.0),
                                    "minibatch/log_ratio_mean": mb.get("log_ratio_mean", 0.0),
                                    "minibatch/log_ratio_std": mb.get("log_ratio_std", 0.0),
                                    "minibatch/clipfrac_upper": mb.get("clipfrac_upper", 0.0),
                                    "minibatch/clipfrac_lower": mb.get("clipfrac_lower", 0.0),
                                    "minibatch/adv_mean": mb.get("adv_mean", 0.0),
                                    "minibatch/adv_std": mb.get("adv_std", 0.0),
                                    "minibatch/adv_pos_frac": mb.get("adv_pos_frac", 0.0),
                                    "minibatch/adv_zero_frac": mb.get("adv_zero_frac", 0.0),
                                    "minibatch/avg_reward": mb.get("avg_reward", 0.0),
                                    "minibatch/total_reward": mb.get("total_reward", 0.0),
                                    "minibatch/total_samples": mb.get("total_samples", 0),
                                    "minibatch/tokens": mb.get("tokens", 0),
                                    "minibatch/index_within_batch": mb.get("minibatch_index", -1),
                                    "minibatch/macro_step": self.global_step,
                                    "minibatch/trainer_rank": mb.get("trainer_rank", -1),
                                }
                                minibatch_log["minibatch_step"] = step_id
                                minibatch_log["trainer_step"] = self.global_step
                                wandb.log(minibatch_log, step=step_id)
                            self._minibatch_global_step += minibatch_count

                        # 1. Core training metrics
                        training_log = {}
                        if final_training_metrics:
                            training_log.update({
                                "training/avg_loss": final_training_metrics.get('loss', 0),
                                # Prefer approx_kl if available; fall back to kl_div
                                "training/avg_kl_div": final_training_metrics.get('approx_kl', final_training_metrics.get('kl_div', 0)),
                                "training/avg_entropy": final_training_metrics.get('entropy', 0),
                                "training/grad_norm": final_training_metrics.get('grad_norm', 0),
                                "epoch": epoch,
                            })
                            # Optional diagnostics propagated from workers (if present)
                            for k in [
                                "log_ratio_mean", "log_ratio_std",
                                "clipfrac_upper", "clipfrac_lower",
                                "adv_mean", "adv_std", "adv_pos_frac", "adv_zero_frac",
                                "loss_per_token", "old_policy_entropy"
                            ]:
                                if k in final_training_metrics:
                                    training_log[f"training/{k}"] = final_training_metrics.get(k)
                        
                        # 2. Batch performance metrics  
                        if final_batch_metrics:
                            training_log.update({
                                # Unify prefix: move former batch/* under training/*
                                "training/batch_avg_reward": final_batch_metrics.get('avg_reward', 0),
                                "training/batch_total_samples": final_batch_metrics.get('total_samples', 0),
                                "training/samples_per_second": samples_per_second,
                            })
                        
                        # 3. Timing metrics (similar to original structure)
                        training_log.update({
                            "timing/1_total_batch_time": total_batch_time,
                            "timing/2_gen_logprob_wall_time": gen_logprob_wall_time,
                            "timing/2a_generation_time": gen_duration,
                            "timing/2b_logprob_time": logprob_duration, 
                            "timing/3_trainer_active_time": trainer_active_time,
                            "timing/4_total_overhead": total_overhead,
                        })
                        # 3.1 Optimizer step metrics (from worker completions)
                        try:
                            opt_steps = sum(d.get("optimizer_steps_in_batch", 0) for d in all_batch_data)
                            training_log.update({
                                "training/optimizer_steps_in_batch": opt_steps,
                            })
                        except Exception:
                            pass
                        
                        # 4. Generation metrics (if available)
                        gen_metrics = gen_logprob_metrics.get("gen_metrics", {})
                        if gen_metrics:
                            total_duration = gen_metrics.get("inference_duration_sec", 0)
                            total_tokens = gen_metrics.get("total_tokens_generated", 0)
                            throughput = total_tokens / total_duration if total_duration > 0 else 0

                            training_log.update({
                                "generation/inference_throughput_tok_per_sec": throughput,
                                "generation/total_tokens_generated": total_tokens,
                                "generation/inference_duration_sec": total_duration,
                                "generation/num_samples_generated": gen_metrics.get("num_samples_generated", 0),
                            })
                        
                        # Log all metrics together
                        # Log using the current global_step. Using -1 can make charts look stale/zeroed if steps collide.
                        training_log["trainer_step"] = self.global_step
                        wandb.log(training_log, step=self.global_step)

                    # Ensure latest trainer weights are visible to generation/logprob workers
                    # so the next macro-batch and any evaluation use up-to-date parameters.
                    try:
                        await self.sync_all_weights()
                    except Exception as _e:
                        logger.warning(f"Skipping weight sync after batch due to error: {_e}")

                    # --- Evaluation Step (post-batch, based on global_step) ---
                    try:
                        if self.config.eval_every_n_macro_batches and \
                           (self.global_step % self.config.eval_every_n_macro_batches == 0):
                            pbar.set_description(f"Epoch {epoch + 1} - Evaluating...")
                            await self.run_evaluation()
                            pbar.set_description(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                    except Exception as _e:
                        logger.warning(f"Evaluation after batch skipped due to error: {_e}")
                    
                    # --- Checkpointing Step ---
                    if self.total_samples_seen - self.last_checkpoint_samples_seen >= self.config.min_samples_per_checkpoint:
                        logger.info(f"Checkpoint condition met. Total samples seen: {self.total_samples_seen}")
                        pbar.set_description(f"Epoch {epoch + 1} - Saving checkpoint...")
                        await self.save_checkpoint()
                        pbar.set_description(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                        self.last_checkpoint_samples_seen = self.total_samples_seen
                
                # Epoch completed - update progress and show summary
                epoch_duration = time.time() - epoch_start_time
                pbar.close()  # Close the batch progress bar
                
                # Update epoch progress bar
                epoch_pbar.update(1)
                epoch_summary = f"Epoch {epoch + 1} completed in {epoch_duration:.1f}s"
                if final_metrics_for_pbar:
                    epoch_summary += f" | Loss: {final_metrics_for_pbar.get('loss', 'N/A')} | Reward: {final_metrics_for_pbar.get('reward', 'N/A')}"
                epoch_pbar.set_postfix_str(epoch_summary)
                
                logger.info(f"✅ Epoch {epoch + 1}/{self.config.num_epochs} completed in {epoch_duration:.2f}s")
                logger.info(f"📊 Total samples processed so far: {self.total_samples_seen:,}")
        
        except Exception as e:
            # Clean up progress bars on exception
            try:
                pbar.close()
                epoch_pbar.close()
            except:
                pass
            logger.error(f"🚨 Training interrupted by exception: {e}")
            raise
        
        # Close epoch progress bar
        epoch_pbar.close()
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"✅ Total epochs: {self.config.num_epochs}")
        logger.info(f"📊 Total samples processed: {self.total_samples_seen:,}")
        logger.info(f"🔄 Total gradient steps: {self.global_step}")
        logger.info("=" * 80)
        
        logger.info("All epochs completed.")
        if self.config.wandb_project:
            wandb.finish()
        # Manual metrics collector removed

    async def run_evaluation(self):
        """
        Runs evaluation on the test set using the generation workers.
        This logic is ported from trainer_core.py.
        """
        logger.info("=" * 80)
        logger.info(f"🚀 Running evaluation at step {self.global_step}...")
        # Evaluation runs on vLLM generation workers or trainer fallback; no trainer mode toggle

        if not self.config.test_data_path:
            logger.warning("No test_data_path configured. Skipping evaluation.")
            return

        try:
            eval_dataset = load_dataset("json", data_files=self.config.test_data_path, split="train")
            eval_pairs = []
            for item in eval_dataset:
                prompt_text = item.get('prompt') or item.get('input')
                if prompt_text is None:
                    raise KeyError("Neither 'prompt' nor 'input' found in evaluation sample.")
                answer_text = item.get('answer') or item.get('label') or ""
                eval_pairs.append({"prompt": prompt_text, "answer": answer_text})
        except Exception as e:
            logger.error(f"🚨 Failed to load evaluation dataset. Error: {e}")
            return


        # Distributed evaluation across all vLLM generation controllers with worker-side reward
        if not self.generation_controllers:
            logger.error("🚨 No generation workers found for evaluation.")
            return

        controllers = self.generation_controllers
        num_ctrl = len(controllers)

        # Concurrency control similar to async-grpo
        try:
            import os as _os
            concurrency = max(1, int(_os.environ.get("ASYNC_GRPO_EVAL_CONCURRENCY", "16")))
        except Exception:
            concurrency = 16

        sem = asyncio.Semaphore(concurrency)

        # Split eval set into controller-sized shards and call batch RPC per controller
        prompts = [p["prompt"] for p in eval_pairs]
        answers = [p["answer"] for p in eval_pairs]

        # Simple round-robin partition to all controllers
        shards: list[tuple[list[str], list[str]]] = [([], []) for _ in range(num_ctrl)]
        for i, (pr, ans) in enumerate(zip(prompts, answers)):
            idx = i % num_ctrl
            shards[idx][0].append(pr)
            shards[idx][1].append(ans)

        async def run_batch(ctrl_idx: int):
            ctrl = controllers[ctrl_idx]
            shard_prompts, shard_answers = shards[ctrl_idx]
            if not shard_prompts:
                return []
            async with sem:
                try:
                    return await ctrl.batch_inference_for_evaluation.remote(
                        prompts=shard_prompts,
                        answers=shard_answers,
                        max_tokens=self.config.max_generation_tokens,
                        temperature=0.0,
                        suppress_eval_logs=True,
                        progress_interval=max(1, len(shard_prompts)//10),
                    )
                except Exception as _e:
                    logger.warning(f"[EVAL] batch call failed on controller {ctrl_idx}: {_e}")
                    return []

        # Kick all controller batch calls
        ctrl_tasks = [asyncio.create_task(run_batch(ci)) for ci in range(num_ctrl)]

        # Minimal progress logging while gathering
        results = []
        completed_ctrl = 0
        total_ctrl = len(ctrl_tasks)
        try:
            for fut in asyncio.as_completed(ctrl_tasks):
                try:
                    out = await fut
                except Exception:
                    out = []
                if out:
                    results.extend(out)
                completed_ctrl += 1
                # Controller-level shard progress
                try:
                    shard_sz = len(shards[completed_ctrl-1][0]) if (completed_ctrl-1) < len(shards) else 0
                    logger.info(f"[EVAL] Controller progress: {completed_ctrl}/{total_ctrl} shards done (shard_size={shard_sz})")
                except Exception:
                    logger.info(f"[EVAL] Controller progress: {completed_ctrl}/{total_ctrl} shards done")
        except Exception as e:
            logger.warning(f"[EVAL] Gathering controller shards interrupted: {e}")

        total_reward = sum(r.get("reward", 0.0) for r in results)
        count = len(results)
        mean_accuracy = total_reward / count if count > 0 else 0.0
        logger.info(f"✅ Evaluation complete. Mean Accuracy: {mean_accuracy:.4f} ({count}/{len(eval_pairs)} succeeded)")

        if self.config.wandb_project and wandb.run:
            try:
                wandb.log({
                    "eval/accuracy": mean_accuracy,
                    "eval/num_samples": len(eval_pairs),
                    "trainer_step": self.global_step,
                }, step=self.global_step)
            except Exception:
                pass

        logger.info("=" * 80)
        # No trainer mode toggle required after evaluation

    async def _quiesce_trainers(self, timeout_sec: float = 300.0):
        """Ask all TrainerGPUWorkers to acknowledge idle state before a mode switch.

        Sends a QUIESCE control message and waits for an idle ACK from each rank.
        This reduces the chance of collective-phase mismatches causing NCCL timeouts.
        """
        try:
            trainer_worker_handles = await self.trainer_controller.get_worker_handles.remote()
            expected = len(trainer_worker_handles)
        except Exception as e:
            logger.warning(f"_quiesce_trainers: failed to get trainer handles: {e}")
            return

        # No longer needed: eval does not toggle trainer mode; keep as no-op for compatibility
        logger.info("_quiesce_trainers: skipped (no trainer mode toggle during eval)")

    # Removed NVML/nvidia-smi diagnostics and manual system metrics collection


    async def save_checkpoint(self):
        """
        Orchestrates saving a checkpoint by requesting the TrainerWorker to handle it,
        saving the training state, and managing old checkpoints.
        """
        checkpoint_time_str = f"step_{self.global_step}_samples_{self.total_samples_seen}"
        logger.info(f"Orchestrating checkpoint save for {checkpoint_time_str}...")
        
        save_directory = Path(self.config.output_dir) / "hf_format" / checkpoint_time_str
        
        # 1. Save the model weights via the trainer controller
        await self.trainer_controller.save_checkpoint.remote(str(save_directory))
        
        # 2. Save the optimizer and scheduler state
        await self.trainer_controller.save_optimizer_state.remote(str(save_directory))
        
        # 3. Save the orchestrator's training state
        self._save_training_state()

        # 4. Clean up old checkpoints
        try:
            checkpoint_base_dir = Path(self.config.output_dir) / "hf_format"
            if checkpoint_base_dir.exists():
                # Regex to find both step and sample counts from directory names
                def get_dir_stats(d):
                    match = re.search(r"step_(\d+)_samples_(\d+)", d.name)
                    if match:
                        return int(match.group(2)) # Sort by samples seen
                    return -1
                
                sample_dirs = sorted(
                    [d for d in checkpoint_base_dir.iterdir() if d.is_dir() and "samples" in d.name],
                    key=get_dir_stats,
                    reverse=True
                )
                
                if len(sample_dirs) > self.config.keep_last_n_checkpoints:
                    dirs_to_remove = sample_dirs[self.config.keep_last_n_checkpoints:]
                    logger.info(f"Removing {len(dirs_to_remove)} old checkpoint(s)...")
                    for dir_to_remove in dirs_to_remove:
                        logger.info(f"  - Removing {dir_to_remove}")
                        shutil.rmtree(dir_to_remove)
        except Exception as e:
            logger.error(f"🚨 Failed during checkpoint cleanup. Error: {e}")

        logger.info(f"✅ Checkpoint orchestration for {checkpoint_time_str} complete.")


async def main():
    # Configuration would be loaded from a file or command line here.
    # We pass it as a dict to the remote actor.
    config_dict = OrchestratorConfig().__dict__
    
    if not ray.is_initialized():
        ray.init(address="auto", namespace="test")
    
    orchestrator = Orchestrator.options(name="orchestrator", namespace="test", lifetime="detached").remote(config_dict)
    
    # This call is now non-blocking
    orchestrator.run.remote()

    # Keep the script alive to allow the orchestrator to run in the background
    logger.info("Orchestrator has been launched. It will run in the background.")
    while True:
        try:
            await asyncio.sleep(60)
            # You could add logic here to check the status of the orchestrator
            # status = await orchestrator.get_status.remote()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            break

@ray.remote
class ExperienceManager:
    """
    Manages experience generation, batching, and dispatching to trainer workers,
    replicating the logic from `vllm_experience_batcher.py`.
    """
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.experience_queue = []
        self.ready_experience_samples = []
        
        # Will map worker_handle -> asyncio.Queue (control/command messages: MICROBATCH/GRADIENT_STEP/BATCH_DONE control)
        self.training_processes_queues = {}
        # Separate queues for trainer completion payloads (one per rank)
        self.completion_queues = {}
        # Removed quiesce idle ACK queue (no trainer mode toggling during eval)
        # Will map worker_handle -> list_of_samples
        self.training_batches = {}
        # Will map worker_handle -> total_tokens
        self.training_batches_lengths = {}
        
        self.dispatched_since_last_grad_step = 0
        self.logger = logging.getLogger(__name__)
        self.lock = asyncio.Lock()
        # Batch sequencing and per-batch meta
        self.batch_seq_counter = 0
        self.current_batch_seq = 0
        self.current_num_questions = 0
        
        # Enhanced load balancing with thresholds and locks
        self.gen_controller_load = {}
        self.logprob_controller_load = {}
        self.gen_controller_thresholds = {}
        self.logprob_controller_thresholds = {}
        
        # Separate locks for each controller type to reduce contention
        self.gen_balancer_lock = asyncio.Condition(asyncio.Lock())
        self.logprob_balancer_lock = asyncio.Condition(asyncio.Lock())
        
        self._reset_metric_accumulators()


    # ... (metric accumulation methods remain the same) ...
    def _reset_metric_accumulators(self):
        """Resets all metric accumulators for a new macro-batch."""
        self.gen_metrics_accumulator = {
            "total_tokens_generated": 0,
            "inference_duration_sec": 0,
            "num_samples_generated": 0,
        }
        self.timing_metrics_accumulator = {
            "gen_duration": 0,
            "logprob_duration": 0,
        }
        self.batch_metrics_accumulator = {
            "total_reward": 0,
            "total_samples": 0,
            "total_completion_tokens": 0,
        }
        self.batch_processing_start_time = None
        # Reward distribution accumulators
        self._reward_values = []
        self._group_std_values = []
        self._nonzero_reward = 0

    def _accumulate_metrics(self, gen_metrics, timing_metrics):
        """Accumulates metrics from a single experience generation task."""
        # self.logger.info(f"[ExperienceManager] Accumulating metrics: gen_metrics={gen_metrics}, timing_metrics={timing_metrics}")
        
        if gen_metrics:
            tokens_gen = gen_metrics.get("generation/total_tokens_generated", 0)
            duration = gen_metrics.get("generation/inference_duration_sec", 0)  
            samples_gen = gen_metrics.get("generation/num_samples_generated", 0)
            
            self.gen_metrics_accumulator["total_tokens_generated"] += tokens_gen
            self.gen_metrics_accumulator["inference_duration_sec"] += duration
            self.gen_metrics_accumulator["num_samples_generated"] += samples_gen
            
            # self.logger.info(f"[ExperienceManager] Updated accumulator: tokens +{tokens_gen}, duration +{duration}, samples +{samples_gen}")
            # self.logger.info(f"[ExperienceManager] Current accumulator: {self.gen_metrics_accumulator}")
        # else:
        #     self.logger.warning(f"[ExperienceManager] No gen_metrics to accumulate")
        
        if timing_metrics:
            self.timing_metrics_accumulator["gen_duration"] += timing_metrics.get("gen_duration", 0)
            self.timing_metrics_accumulator["logprob_duration"] += timing_metrics.get("logprob_duration", 0)


    def _get_aggregated_batch_metrics(self):
        """Returns the aggregated metrics for the completed batch and resets them."""
        # Calculate batch-level averages
        total_samples = self.batch_metrics_accumulator["total_samples"]
        avg_reward = self.batch_metrics_accumulator["total_reward"] / total_samples if total_samples > 0 else 0
        avg_completion_length = self.batch_metrics_accumulator["total_completion_tokens"] / total_samples if total_samples > 0 else 0
        # Reward stats
        rv = self._reward_values
        gv = self._group_std_values
        def _safe_mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0
        def _safe_std(xs):
            m = _safe_mean(xs)
            return float((sum((x - m) * (x - m) for x in xs) / max(1, len(xs))) ** 0.5) if xs else 0.0
        reward_mean = _safe_mean(rv)
        reward_std = _safe_std(rv)
        reward_min = float(min(rv)) if rv else 0.0
        reward_max = float(max(rv)) if rv else 0.0
        group_std_mean = _safe_mean(gv)
        group_std_zero_frac = float(sum(1 for x in gv if abs(x) < 1e-9)) / float(len(gv)) if gv else 0.0
        nonzero_reward_frac = float(self._nonzero_reward) / float(total_samples) if total_samples > 0 else 0.0
        
        metrics_to_return = {
            "gen_metrics": self.gen_metrics_accumulator,
            "timing_metrics": self.timing_metrics_accumulator,
            "batch_metrics": {
                "avg_reward": avg_reward,
                "total_samples": total_samples,
                "avg_completion_length": avg_completion_length,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "group_reward_std_mean": group_std_mean,
                "group_reward_std_zero_frac": group_std_zero_frac,
                "nonzero_reward_frac": nonzero_reward_frac,
            },
            "gen_logprob_wall_time": time.time() - self.batch_processing_start_time if self.batch_processing_start_time else 0,
        }
        self._reset_metric_accumulators()
        return metrics_to_return

    def _initialize_controller_thresholds(self, generation_controllers: List[ray.actor.ActorHandle], logprob_controllers: List[ray.actor.ActorHandle]):
        """Initialize controller load tracking and thresholds."""
        # Set reasonable defaults based on controller types
        gen_threshold = max(4, self.config.samples_per_question)  # Allow at least samples_per_question concurrent requests
        logprob_threshold = max(8, self.config.samples_per_question * 2)  # LogProb can handle more concurrent requests
        
        for controller in generation_controllers:
            self.gen_controller_load[controller] = 0
            self.gen_controller_thresholds[controller] = gen_threshold
            
        for controller in logprob_controllers:
            self.logprob_controller_load[controller] = 0
            self.logprob_controller_thresholds[controller] = logprob_threshold
            
        self.logger.info(f"Initialized {len(generation_controllers)} generation controllers with threshold {gen_threshold}")
        self.logger.info(f"Initialized {len(logprob_controllers)} logprob controllers with threshold {logprob_threshold}")

    async def _acquire_controller(self, controller_type: str, num_requests: int = 1) -> ray.actor.ActorHandle:
        """
        Safely acquire a controller with load balancing and threshold protection.
        
        Args:
            controller_type: "generation" or "logprob"
            num_requests: Number of requests to reserve on the controller
            
        Returns:
            Controller handle or None if acquisition fails
        """
        if controller_type == "generation":
            load_dict = self.gen_controller_load
            threshold_dict = self.gen_controller_thresholds
            lock = self.gen_balancer_lock
        elif controller_type == "logprob":
            load_dict = self.logprob_controller_load
            threshold_dict = self.logprob_controller_thresholds
            lock = self.logprob_balancer_lock
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
            
        async with lock:
            # Wait until a controller becomes available
            while True:
                available_controllers = [
                    controller for controller in load_dict
                    if load_dict[controller] + num_requests <= threshold_dict[controller]
                ]
                
                if available_controllers:
                    # Choose the least loaded available controller
                    chosen_controller = min(available_controllers, key=lambda c: load_dict[c])
                    load_dict[chosen_controller] += num_requests
                    
                    self.logger.debug(f"Acquired {controller_type} controller {chosen_controller} "
                                    f"(load: {load_dict[chosen_controller]}/{threshold_dict[chosen_controller]})")
                    return chosen_controller
                
                # No available controllers, wait for one to be released
                self.logger.debug(f"No available {controller_type} controllers. Waiting...")
                await lock.wait()

    async def _release_controller(self, controller_type: str, controller: ray.actor.ActorHandle, num_requests: int = 1):
        """
        Release a controller and notify waiting tasks.
        
        Args:
            controller_type: "generation" or "logprob"
            controller: Controller handle to release
            num_requests: Number of requests to release
        """
        if controller_type == "generation":
            load_dict = self.gen_controller_load
            lock = self.gen_balancer_lock
        elif controller_type == "logprob":
            load_dict = self.logprob_controller_load
            lock = self.logprob_balancer_lock
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
            
        async with lock:
            if controller in load_dict:
                load_dict[controller] = max(0, load_dict[controller] - num_requests)
                self.logger.debug(f"Released {controller_type} controller {controller} "
                                f"(load: {load_dict[controller]})")
                lock.notify_all()  # Wake up waiting tasks

    def register_training_process(self, gpu_rank: int, max_tokens_per_gpu: int, train_minibatch_sample_size: int):
        """Registers a training process with its GPU rank and parameters."""
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.train_minibatch_sample_size = train_minibatch_sample_size
        self.training_processes_queues[gpu_rank] = asyncio.Queue()
        self.completion_queues[gpu_rank] = asyncio.Queue()
        self.training_batches[gpu_rank] = []
        self.training_batches_lengths[gpu_rank] = 0
        return True
    
    # Removed: initialize_trainer_workers - no longer needed since queues are passed directly

    # ... (generate_experience and _process_single_prompt remain the same) ...
    async def generate_experience(self, questions: list, generation_controllers: List[ray.actor.ActorHandle], logprob_controllers: List[ray.actor.ActorHandle]):
        """Create asynchronous tasks for experience generation and add them to the internal queue."""
        self.logger.info(f"Generating experience for {len(questions)} questions.")

        # Reset accumulators and start timer for the new macro-batch
        self._reset_metric_accumulators()
        self.batch_processing_start_time = time.time()
        # Meta for this macro-batch
        self.batch_seq_counter += 1
        self.current_batch_seq = self.batch_seq_counter
        try:
            self.current_num_questions = int(len(questions))
        except Exception:
            self.current_num_questions = 0

        # Initialize load trackers and thresholds if they are new
        if not self.gen_controller_load:
            self._initialize_controller_thresholds(generation_controllers, logprob_controllers)

        async def run_with_timeout(coro, timeout=1200):
            try:
                return await asyncio.wait_for(coro, timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Experience generation task for a prompt timed out after {timeout}s.")
                return None, None, None # Match return signature of _process_single_prompt

        tasks = [
            run_with_timeout(self._process_single_prompt(
                question,
                generation_controllers,
                logprob_controllers
            )) for question in questions
        ]
        
        async with self.lock:
            self.experience_queue.extend(tasks)
        self.logger.debug(f"Added {len(tasks)} tasks to experience queue. New size: {len(self.experience_queue)}")

    async def _process_single_prompt(self, question: Dict, gen_controllers: List[ray.actor.ActorHandle], logprob_controllers: List[ray.actor.ActorHandle]):
        """Processes a single prompt: generation followed by logprob calculation with safe load balancing."""
        
        # --- Generation Step (Safe) ---
        gen_start_t = time.time()
        gen_controller = await self._acquire_controller("generation", 1)
        if gen_controller is None: 
            self.logger.warning("Failed to acquire generation controller")
            return None, None, None
        
        try:
            gen_result = await gen_controller.inference.remote(
                question,
                n=self.config.samples_per_question,
                temperature=self.config.temperature,
                max_tokens=self.config.max_generation_tokens,
            )
            gen_duration = time.time() - gen_start_t
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return None, None, None
        finally:
            await self._release_controller("generation", gen_controller, 1)

        samples, gen_metrics = gen_result
        if not samples:
            return None, None, None

        # --- Logprob Step (Safe) ---
        logprob_start_t = time.time()
        logprob_tasks = []
        assigned_controllers = []
        
        for i, sample in enumerate(samples):
            try:
                logprob_controller = await self._acquire_controller("logprob", 1)
                if logprob_controller is None:
                    self.logger.warning(f"Failed to acquire logprob controller for sample {i}")
                    continue
                
                self.logger.debug(f"[ExperienceManager] Sample {i} to LogProb: sample_ids length={len(sample.get('sample_ids', []))}")
                assigned_controllers.append(logprob_controller)
                logprob_tasks.append(logprob_controller.inference.remote(sample))
                
            except Exception as e:
                self.logger.error(f"Failed to acquire logprob controller for sample {i}: {e}")
                continue
            
        # Execute all logprob tasks in parallel
        try:
            samples_with_logprobs = await asyncio.gather(*logprob_tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_samples = []
            for i, result in enumerate(samples_with_logprobs):
                if isinstance(result, Exception):
                    self.logger.error(f"LogProb task {i} failed: {result}")
                elif result:
                    log_probs_list = result.get('sample_logprobs')
                    shape_info = 'N/A'
                    if log_probs_list is not None:
                        try:
                            shape_info = len(log_probs_list)
                        except TypeError:
                            shape_info = 'Not a sequence'
                    
                    self.logger.debug(f"[ExperienceManager] Sample {i} from LogProb: sample_logprobs length={shape_info}")
                    valid_samples.append(result)
                    
        except Exception as e:
            self.logger.error(f"LogProb batch execution failed: {e}")
            valid_samples = []
            
        finally:
            # Release all acquired controllers
            release_tasks = []
            for controller in assigned_controllers:
                release_tasks.append(self._release_controller("logprob", controller, 1))
            
            if release_tasks:
                await asyncio.gather(*release_tasks, return_exceptions=True)
        
        logprob_duration = time.time() - logprob_start_t
        timing_metrics = {
            "gen_duration": gen_duration,
            "logprob_duration": logprob_duration,
        }
        return valid_samples, gen_metrics, timing_metrics


    def start_creating_batches(self):
        """Kicks off the asynchronous batch creation loop."""
        self.logger.info("[ExperienceManager] Starting batch creation loop...")
        asyncio.create_task(self._create_batches())

    async def _create_batches(self):
        """Continuously consumes tasks from the experience_queue and processes them."""
        async with self.lock:

            for task in asyncio.as_completed(self.experience_queue):
                # self.logger.debug(f"Experience queue length in _create_batches: {len(self.experience_queue)}")
                result = await task
                if result is None:  # underlying coroutine timed out
                    continue
                
                # Unpack the result tuple
                samples_with_logprobs, gen_metrics, timing_metrics = result
                if samples_with_logprobs is None:
                    continue
                    
                # Accumulate metrics
                self._accumulate_metrics(gen_metrics, timing_metrics)
                
                for sample_with_logprob in samples_with_logprobs:
                    if sample_with_logprob:
                        
                        # Update batch metrics accumulator
                        self.batch_metrics_accumulator["total_reward"] += sample_with_logprob.get("reward", 0)
                        self.batch_metrics_accumulator["total_samples"] += 1
                        self.batch_metrics_accumulator["total_completion_tokens"] += sample_with_logprob.get("output_len", 0)
                        
                        # Original async buffering by sample count
                        self.ready_experience_samples.append(sample_with_logprob)
                if len(self.ready_experience_samples) >= self.train_minibatch_sample_size:
                            await self._flush_ready_samples()
                # If inline replay applied in this round, emit summary and prevent batch-end injection
                            
            self.logger.debug(f"Experience queue length in _create_batches after processing: {len(self.experience_queue)}")
            # After processing all tasks, flush any remaining samples
            if self.ready_experience_samples:
                await self._flush_ready_samples()
            # Clear the experience queue
            self.experience_queue = []

            num_dispatched = await self.dispatch_batches()
        if num_dispatched > 0:
            await self.dispatch_signal(MessageType.GRADIENT_STEP)
        
            # Get aggregated metrics for BATCH_DONE signal
        aggregated_metrics = self._get_aggregated_batch_metrics()

        # _get_aggregated_batch_metrics() resets internal accumulators before returning,
        # so we must take values from the returned dict (not from the accumulators).
        agg_gen_metrics = (aggregated_metrics.get("gen_metrics") or {}).copy()
        agg_timing_metrics = (aggregated_metrics.get("timing_metrics") or {}).copy()
        agg_wall_time = aggregated_metrics.get("gen_logprob_wall_time", 0)

        # Include generation/logprob timing metrics based on the aggregated snapshot
        aggregated_metrics["gen_logprob_metrics"] = {
            "gen_duration": agg_timing_metrics.get("gen_duration", 0),
            "logprob_duration": agg_timing_metrics.get("logprob_duration", 0),
            "gen_logprob_wall_time": agg_wall_time,
            "gen_metrics": agg_gen_metrics,
            # Add batch meta for worker-side logging
            "batch_seq": getattr(self, "current_batch_seq", 0),
            "num_questions": getattr(self, "current_num_questions", 0),
            "samples_per_question": self.config.samples_per_question,
        }

        await self.dispatch_signal_with_data(MessageType.BATCH_DONE, aggregated_metrics)
        # self.logger.debug("Batch done dispatched")

    async def _flush_ready_samples(self):
        """Attach total_non_masked_output_tokens and forward samples to add_sample_to_batches."""
        # Compute total non-masked output tokens for current minibatch
        total_tokens = sum(sample_with_logprob['num_non_masked_output_tokens'] for sample_with_logprob in self.ready_experience_samples)
        for sample_with_logprob in self.ready_experience_samples:
            sample_with_logprob['total_non_masked_output_tokens'] = total_tokens
            await self.add_sample_to_batches(sample_with_logprob)
        # Clear the minibatch buffer
        self.ready_experience_samples.clear()

    async def add_sample_to_batches(self, sample_with_logprob):
        """
        Add a single sample with logprob into per-GPU token-limited batches, dispatching microbatches as needed.

        Responsibilities:
        - Compute the token length of the sample and choose the least-loaded GPU batch by token count.
        - If adding the sample exceeds max_tokens_per_gpu, dispatch existing microbatches first via dispatch_batches().
        - Append the sample to the chosen GPU batch and update its token-length counter.
        - Increment dispatched_since_last_grad_step by the number of samples just sent.
        - When the cumulative number of samples dispatched equals train_minibatch_sample_size, dispatch all batches and emit a GRADIENT_STEP signal.
        - dispatch_batches() handles sending MICROBATCH messages and resetting per-GPU buffers.
        """
        # Compute sample length and pick the batch to fill
        sample_len = sample_with_logprob['input_len'] + sample_with_logprob['output_len']
        least_loaded_gpu_rank = min(self.training_batches_lengths, key=self.training_batches_lengths.get)
        # Handle oversize sample: if a single sample exceeds the token limit, dispatch what we have and send it alone
        if sample_len > self.max_tokens_per_gpu:
            # Dispatch existing batches if any
            if any(self.training_batches.values()):
                num_dispatched = await self.dispatch_batches()
                if num_dispatched:
                    self.dispatched_since_last_grad_step += num_dispatched
            # Place oversize sample and dispatch immediately
            target_rank = least_loaded_gpu_rank
            self.training_batches[target_rank].append(sample_with_logprob)
            self.training_batches_lengths[target_rank] += sample_len
            num_dispatched = await self.dispatch_batches()
            if num_dispatched:
                self.dispatched_since_last_grad_step += num_dispatched
            return
        # If adding would overflow tokens, flush existing batches first
        if self.training_batches_lengths[least_loaded_gpu_rank] + sample_len > self.max_tokens_per_gpu:
            num_dispatched = await self.dispatch_batches()
            # If nothing to dispatch (empty), accept the sample anyway to avoid deadlock
            if num_dispatched == 0:
                pass
                # try:
                #     self.logger.warning(f"[TOK-LIMIT] empty-batch overflow: accepting sample_len={sample_len} despite limit {self.max_tokens_per_gpu}.")
                # except Exception:
                #     pass
            else:
                self.dispatched_since_last_grad_step += num_dispatched
        # Add the new sample
        self.training_batches[least_loaded_gpu_rank].append(sample_with_logprob)
        self.training_batches_lengths[least_loaded_gpu_rank] += sample_len
        # If we've exactly hit the train_minibatch_sample_size, flush and trigger gradient
        if self.dispatched_since_last_grad_step + sum([len(batch) for batch in self.training_batches.values()]) == self.train_minibatch_sample_size:
            num_dispatched = await self.dispatch_batches()
            if not num_dispatched:
                raise Exception("Reached minibatch size with no batches to dispatch")
            await self.dispatch_signal(MessageType.GRADIENT_STEP)
            self.dispatched_since_last_grad_step = 0

    async def reset_batches(self):
        for gpu_rank in self.training_batches:
            self.training_batches[gpu_rank] = []
            self.training_batches_lengths[gpu_rank] = 0
    
    async def dispatch_batches(self):
        """
        Send exactly one MICROBATCH to each GPU rank. If a rank has no real
        samples, send a dummy sample to ensure all ranks participate in the
        training step (prevents FSDP collectives from mismatching).
        Returns number of real samples dispatched.
        """
        # Count real samples across all ranks
        num_real_samples = sum(len(batch) for batch in self.training_batches.values() if batch)
        if num_real_samples == 0:
            return 0

        # Prepare a dummy placeholder for ranks without real samples
        dummy_sample = {"dummy": True, "total_non_masked_output_tokens": 0}

        # Send each GPU rank its specific batch (or a dummy if empty)
        dispatch_tasks = []
        # # Lightweight logging summary: dummy?, number of samples, total token length
        # try:
        #     for gpu_rank, batch in self.training_batches.items():
        #         is_dummy = not bool(batch)
        #         if is_dummy:
        #             total_len = 0
        #             num_s = 0
        #         else:
        #             num_s = len(batch)
        #             total_len = 0
        #             for s in batch:
        #                 try:
        #                     total_len += int(s.get("input_len", 0)) + int(s.get("output_len", 0))
        #                 except Exception:
        #                     pass
        #         print(f"[MICROBATCH] rank={gpu_rank} dummy={is_dummy} samples={num_s} total_len={total_len}")
        # except Exception:
        #     pass
        for gpu_rank, batch in self.training_batches.items():
            if gpu_rank not in self.training_processes_queues:
                continue
            payload = batch if batch else [dummy_sample]
            dispatch_tasks.append(
                self.training_processes_queues[gpu_rank].put(Message(MessageType.MICROBATCH, payload))
            )
        
        await asyncio.gather(*dispatch_tasks)
        await self.reset_batches()
        return num_real_samples

    async def dispatch_signal(self, signal_type: MessageType):
        """Send a signal to all workers."""
        for i, queue in enumerate(self.training_processes_queues.values()):
            await queue.put(Message(signal_type))
            # print(f"\033[1;38;2;255;20;147mDispatched signal:\033[0m {signal_type} \033[1;38;2;255;20;147mGPU Rank:\033[0m {i}")
    
    async def dispatch_signal_with_data(self, signal_type: MessageType, data: Any = None):
        """Send a signal with data to all workers (for BATCH_DONE signals)."""
        # Send to all GPU ranks
        for i, queue in enumerate(self.training_processes_queues.values()):
            await queue.put(Message(signal_type, data))
            # print(f"\033[1;38;2;255;20;147mDispatched signal with data:\033[0m {signal_type} \033[1;38;2;255;20;147mGPU Rank:\033[0m {i}")

    async def get_batch(self, global_rank: int):
        return await self.training_processes_queues[global_rank].get()

    async def get_completion(self, global_rank: int):
        """Retrieve trainer completion payload for the given rank.

        Returns a dict with keys like 'gpu_rank', 'metrics', 'batch_metrics', etc.
        """
        return await self.completion_queues[global_rank].get()

    async def put_completion(self, global_rank: int, completion_data: Any):
        # Put completion data into a dedicated completion queue to avoid racing with control messages
        await self.completion_queues[global_rank].put(completion_data)

    # Removed QUIESCE idle handshake APIs

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    asyncio.run(main()) 