import os
import re
import logging
import warnings
import torch
import torch.distributed as dist
from datetime import timedelta
import ray
import asyncio
import pickle
import socket
import threading
from filelock import FileLock
from copy import deepcopy
from types import MethodType
from typing import Any, Callable, List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict

from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecisionPolicy, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions, get_optimizer_state_dict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from vllm.worker.worker import Worker
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.model_executor.sampling_metadata import SamplingMetadata

from sample_processing_utils import post_process_batch
from grpo_loss import compute_dual_clip_grpo_loss, make_grpo_forward
from transformers import PreTrainedModel
from torch.optim.lr_scheduler import LambdaLR

from ray.util.collective import collective

# Import MessageType from orchestrator
from enum import Enum

from contextlib import nullcontext
def record_function(name):
    return nullcontext()

class MessageType(Enum):
    MICROBATCH = "microbatch"
    GRADIENT_STEP = "gradient_step"
    BATCH_DONE = "batch_done"

@dataclass
class Message:
    type: MessageType
    data: Optional[Any] = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("GRPO_LOGGING_LEVEL", "INFO"))

def _configure_global_logging():
    """Configure logging levels inside each Ray worker process.
    Suppresses noisy third-party DEBUG logs (urllib3, filelock, HF hub, HTTP/2 stacks).
    """
    try:
        # Root baseline
        logging.getLogger().setLevel(logging.INFO)
        # Noisy libs → WARNING
        for name in (
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
            logging.getLogger(name).setLevel(logging.WARNING)
        # Transformers verbosity (reduce HF info logs like 'loading file ...')
        try:
            from transformers.utils import logging as hf_logging  # type: ignore
            hf_logging.set_verbosity_warning()
            logging.getLogger("transformers").setLevel(logging.WARNING)
        except Exception:
            pass
    except Exception:
        pass

"""Local explicit FA2 varlen wrapper (unpad via cu_seqlens, with GQA KV repeat).
This patch does NOT depend on VERL. It installs a wrapper on Transformers'
_flash_attention_forward so that B==1 packed 1D inputs with segment resets in
position_ids are routed to flash_attn varlen kernels explicitly.
"""
try:
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as _fa2_varlen_qkv,
    )  # type: ignore
    _FA2_VARLEN_OK = True
except Exception:
    _fa2_varlen_qkv = None
    _FA2_VARLEN_OK = False

def _install_explicit_fa2_varlen_wrapper(model: PreTrainedModel | None = None) -> bool:
    if not _FA2_VARLEN_OK:
        logger.warning("FA2 varlen kernel not available; skipping varlen wrapper.")
        return False
    # Locate transformers FA2 forward shim
    try:
        import transformers.modeling_flash_attention_utils as _tf_fa  # type: ignore
    except Exception:
        try:
            from transformers.integrations import flash_attention as _tf_fa  # type: ignore
        except Exception:
            logger.warning("Unable to locate transformers flash_attention module for varlen wrapper.")
            return False
    _orig = getattr(_tf_fa, "_flash_attention_forward", None)
    if _orig is None:
        logger.warning("Transformers _flash_attention_forward not found.")
        return False
    if getattr(_orig, "__grpo_explicit_varlen__", False):
        return True

    # Log binding target for verification
    try:
        _mod_path = getattr(_tf_fa, "__file__", "<unknown>")
        logger.info(f"Binding varlen wrapper to module={getattr(_tf_fa, '__name__', str(_tf_fa))} path={_mod_path} orig_id={id(_orig)}")
    except Exception:
        pass

    import torch as _t

    # Introspect installed FA2 varlen function signature/version once for debugging
    try:
        import flash_attn as _fa
        import inspect as _ins
        _ver = getattr(_fa, "__version__", None)
        try:
            _sig = str(_ins.signature(_fa2_varlen_qkv))
        except Exception as _e_sig:
            _sig = f"<inspect.failed: {_e_sig}>"
        _ts = getattr(_fa2_varlen_qkv, "__text_signature__", None)
        logger.warning(f"[FA2-INTROSPECT] version={_ver} sig={_sig} text_sig_present={bool(_ts)}")
    except Exception as _e_introspect:
        try:
            logger.warning(f"[FA2-INTROSPECT] failed: {_e_introspect}")
        except Exception:
            pass

    def _repeat_kv_for_gqa(x: _t.Tensor, repeats: int) -> _t.Tensor:
        # x: [B,L,Hkv,D] -> repeat along head dim to Hq
        if repeats <= 1:
            return x
        B, L, H, D = x.shape
        x = x.unsqueeze(3).expand(B, L, H, repeats, D)
        return x.reshape(B, L, H * repeats, D)

    def _fa2_forward_wrapper(query_states: _t.Tensor, key_states: _t.Tensor, value_states: _t.Tensor, *args, **kwargs):
        """
        VERL-style wrapper with explicit varlen routing using prepare_fa2_from_position_ids.
        This ensures we actually use flash_attn_varlen_func instead of relying on Transformers' auto-detection.
        """
        try:
            # Log gating for varlen-path traces
            try:
                import os as _os_log
                _log_varlen = _os_log.getenv("GRPO_LOG_VARLEN", "0") == "1"
            except Exception:
                _log_varlen = False
            # # Hook entry counter for verifying actual usage
            # try:
            #     _cnt = getattr(_fa2_forward_wrapper, "__grpo_varlen_hook_entered__", 0) + 1
            #     setattr(_fa2_forward_wrapper, "__grpo_varlen_hook_entered__", _cnt)
            #     if _log_varlen and _cnt <= 10:
            #         _msg = (
            #             f"[VARLEN-HOOK] enter={_cnt} q={tuple(query_states.shape)} k={tuple(key_states.shape)} v={tuple(value_states.shape)}"
            #         )
            #         logger.warning(_msg); print(_msg, flush=True)
            # except Exception:
            #     pass
            
            # Check for position_ids and varlen conditions
            pos = kwargs.get("position_ids", None)
            attention_mask = kwargs.get("attention_mask", None)
            query_length = kwargs.get("query_length", None)
            
            # Log position_ids presence for debugging
            should_use_varlen = False
            if pos is not None and isinstance(pos, _t.Tensor) and pos.dim() == 2 and pos.size(0) == 1:
                try:
                    c = getattr(_fa2_forward_wrapper, "__grpo_varlen_pos_log_cnt__", 0)
                    if _log_varlen and c < 5:
                        B, L = int(pos.size(0)), int(pos.size(1))
                        pos0 = pos[0]
                        resets = int((pos0 == 0).sum().item())
                        # _msg = f"[VARLEN-POS-IDS] shape=({B},{L}) resets={resets} pos[0,:10]={pos0[:10].tolist()}"
                        # logger.warning(_msg); print(_msg, flush=True)
                        setattr(_fa2_forward_wrapper, "__grpo_varlen_pos_log_cnt__", c + 1)
                    # Use varlen if we have multi-segment 1D (resets > 0)
                    if resets > 0 and query_states.size(0) == 1:
                        should_use_varlen = True
                except Exception:
                    pass
            
            # GQA head repeat if needed (VERL does this in their wrapper too)
            Hq = int(query_states.size(2))
            Hkv = int(key_states.size(2))
            if Hq > Hkv and Hq % Hkv == 0:
                repeats = Hq // Hkv
                if repeats > 1:
                    key_states = _repeat_kv_for_gqa(key_states, repeats)
                    value_states = _repeat_kv_for_gqa(value_states, repeats)
                    try:
                        c = getattr(_fa2_forward_wrapper, "__grpo_gqa_log_cnt__", 0)
                        if _log_varlen and c < 5:
                            # _msg = f"[VARLEN-GQA] Hq={Hq} Hkv={Hkv} repeats={repeats}"
                            # logger.warning(_msg); print(_msg, flush=True)
                            setattr(_fa2_forward_wrapper, "__grpo_gqa_log_cnt__", c + 1)
                    except Exception:
                        pass
            
            # Explicit varlen routing (VERL's prepare_fa2_from_position_ids logic)
            if should_use_varlen:
                try:
                    # Import FA2 functions
                    from flash_attn import flash_attn_varlen_func
                    
                    batch_size = query_states.size(0)
                    seqlen = query_states.size(1)
                    
                    # VERL-style cu_seqlens generation from position_ids (NOT from attention_mask)
                    # position_ids shape: (1, total_tokens) with resets at segment boundaries
                    position_ids_flat = pos[0]  # (total_tokens,)
                    
                    # Find segment boundaries: indices where position_ids == 0
                    indices_q = _t.arange(position_ids_flat.size(0), device=position_ids_flat.device, dtype=_t.int32)
                    segment_starts = indices_q[position_ids_flat == 0]  # e.g. [0, 380, 665, ...]
                    
                    # Build cu_seqlens: [segment_starts..., total_length]
                    cu_seqlens = _t.cat([
                        segment_starts,
                        _t.tensor([position_ids_flat.size(0)], device=position_ids_flat.device, dtype=_t.int32)
                    ])
                    
                    # Max sequence length from cu_seqlens
                    max_seqlen = int(cu_seqlens.diff().max().item())
                    
                    # Reshape to (total_tokens, nheads, headdim) for varlen kernel
                    query_states_varlen = query_states.view(-1, query_states.size(-2), query_states.size(-1))
                    key_states_varlen = key_states.view(-1, key_states.size(-2), key_states.size(-1))
                    value_states_varlen = value_states.view(-1, value_states.size(-2), value_states.size(-1))
                    
                    # Extract kwargs
                    dropout_p = kwargs.get("dropout_p", kwargs.get("dropout", 0.0))
                    softmax_scale = kwargs.get("softmax_scale", None)
                    is_causal = kwargs.get("is_causal", kwargs.get("causal", True))
                    
                    # Call flash_attn_varlen_func with proper cu_seqlens
                    attn_output_varlen = flash_attn_varlen_func(
                        query_states_varlen,
                        key_states_varlen,
                        value_states_varlen,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        dropout_p=dropout_p,
                        softmax_scale=softmax_scale,
                        causal=is_causal,
                    )
                    
                    # Reshape back to (batch, seqlen, nheads, headdim)
                    attn_output = attn_output_varlen.view(batch_size, seqlen, attn_output_varlen.size(-2), attn_output_varlen.size(-1))
                    
                    try:
                        c = getattr(_fa2_forward_wrapper, "__grpo_varlen_explicit_log_cnt__", 0)
                        if _log_varlen and c < 5:
                            # _msg = f"[VARLEN-EXPLICIT] cu_seqlens={cu_seqlens.tolist()} max_seqlen={max_seqlen}"
                            # logger.warning(_msg); print(_msg, flush=True)
                            setattr(_fa2_forward_wrapper, "__grpo_varlen_explicit_log_cnt__", c + 1)
                    except Exception:
                        pass
                    
                    return attn_output
                    
                except Exception as _e_varlen:
                    # Fallback to original if explicit varlen fails
                    try:
                        c = getattr(_fa2_forward_wrapper, "__grpo_varlen_fallback_log_cnt__", 0)
                        if _log_varlen and c < 5:
                            # _msg = f"[VARLEN-FALLBACK] {type(_e_varlen).__name__}: {_e_varlen}"
                            # logger.warning(_msg); print(_msg, flush=True)
                            setattr(_fa2_forward_wrapper, "__grpo_varlen_fallback_log_cnt__", c + 1)
                    except Exception:
                        pass
                    # Fall through to delegate to _orig
            
            # Delegate to Transformers' built-in logic as fallback
            return _orig(query_states, key_states, value_states, *args, **kwargs)
            
        except Exception as _e_var:
            try:
                c = getattr(_fa2_forward_wrapper, "__grpo_varlen_exc_log_cnt__", 0)
                if _log_varlen and c < 5:
                    # _msg = f"[VARLEN-EXC] {type(_e_var).__name__}: {_e_var}"
                    # logger.warning(_msg); print(_msg, flush=True)
                    setattr(_fa2_forward_wrapper, "__grpo_varlen_exc_log_cnt__", c + 1)
            except Exception:
                pass
            # Fallback to original on any error
            return _orig(query_states, key_states, value_states, *args, **kwargs)

    setattr(_fa2_forward_wrapper, "__grpo_explicit_varlen__", True)

    # Patch multiple alias modules where models may have bound the symbol
    patched_targets = []

    try:
        # Primary location
        _tf_fa._flash_attention_forward = _fa2_forward_wrapper  # type: ignore
        patched_targets.append((getattr(_tf_fa, "__name__", str(_tf_fa)), "_flash_attention_forward"))
    except Exception:
        pass

    # Also patch transformers.integrations.flash_attention if distinct
    try:
        import transformers.integrations.flash_attention as _tf_int_fa  # type: ignore
        try:
            _orig_alias = getattr(_tf_int_fa, "_flash_attention_forward", None)
            if _orig_alias is not None and not getattr(_orig_alias, "__grpo_explicit_varlen__", False):
                _tf_int_fa._flash_attention_forward = _fa2_forward_wrapper  # type: ignore
                patched_targets.append((getattr(_tf_int_fa, "__name__", str(_tf_int_fa)), "_flash_attention_forward"))
        except Exception:
            pass
    except Exception:
        pass

    # Patch the model module (where the symbol may have been imported into the namespace)
    try:
        import sys as _sys
        _mod = None
        if model is not None:
            _mod = _sys.modules.get(model.__module__)
        if _mod is not None and hasattr(_mod, "_flash_attention_forward"):
            try:
                _orig_mod = getattr(_mod, "_flash_attention_forward")
            except Exception:
                _orig_mod = None
            if _orig_mod is not None and not getattr(_orig_mod, "__grpo_explicit_varlen__", False):
                setattr(_mod, "_flash_attention_forward", _fa2_forward_wrapper)
                patched_targets.append((getattr(_mod, "__name__", str(_mod)), "_flash_attention_forward"))
    except Exception:
        pass

    # Patch well-known model modules by model_type (e.g., qwen2)
    try:
        _mt = getattr(getattr(model, "config", object()), "model_type", None) if model is not None else None
    except Exception:
        _mt = None
    if _mt is not None and str(_mt).startswith("qwen2"):
        try:
            import transformers.models.qwen2.modeling_qwen2 as _qwen2_mod  # type: ignore
            if hasattr(_qwen2_mod, "_flash_attention_forward"):
                _orig_q = getattr(_qwen2_mod, "_flash_attention_forward", None)
                if _orig_q is not None and not getattr(_orig_q, "__grpo_explicit_varlen__", False):
                    setattr(_qwen2_mod, "_flash_attention_forward", _fa2_forward_wrapper)
                    patched_targets.append((getattr(_qwen2_mod, "__name__", str(_qwen2_mod)), "_flash_attention_forward"))
        except Exception:
            pass

    # Log summary of patched alias targets
    try:
        if patched_targets:
            for name, attr in patched_targets[:8]:
                logger.info(f"Installed explicit FA2 varlen wrapper on {name}.{attr}")
        else:
            logger.warning("Explicit FA2 varlen wrapper installed on zero alias targets — wrapper may not be invoked.")
    except Exception:
        pass

    return bool(patched_targets)

# Removed local FA2 varlen patch: rely on HF Transformers' built-in varlen path


def _apply_fsdp_wrapping(model: torch.nn.Module, fsdp_kwargs: Dict[str, Any]):
    """
    Applies FSDP wrapping to the model, mirroring verl's apply_fsdp2 logic.
    It wraps specified transformer blocks and embeddings first, then the entire model.
    """
    from transformers.trainer_pt_utils import get_module_class_from_name
    
    if not hasattr(model, "_no_split_modules") or not model._no_split_modules:
        raise ValueError("Model does not have _no_split_modules attribute, cannot determine wrap policy.")
    
    block_class_name = model._no_split_modules[0]
    block_class = get_module_class_from_name(model, block_class_name)
    if block_class is None:
        raise ValueError(f"Could not find module class named {block_class_name}")

    # Collect modules to wrap: only transformer blocks (do not shard embeddings)
    modules_to_wrap = []
    for module in model.modules():
        if isinstance(module, block_class):
            modules_to_wrap.append(module)

    # Wrap the child modules first
    for module in modules_to_wrap:
        fully_shard(module, **fsdp_kwargs)

    # Finally, wrap the root model itself
    fully_shard(model, **fsdp_kwargs)


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model: PreTrainedModel):
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            # Early exit of the loop
            if n_replace > 0:
                break

        original_weights[key] = value

    return original_weights


# --- Utility Functions (largely unchanged) ---

def get_device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_nccl_backend():
    return "nccl"

def get_torch_device():
    return torch.device(get_device_name())

def get_device_id():
    return torch.cuda.current_device()

def update_model_config(config, override_kwargs):
    for key, value in override_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Config override failed: key '{key}' not found in model config.")

def _monkey_patch_compute_logits(model, vocab_size: int):
    # This might be needed by vLLM engine, so we keep it.
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)

# --- Base Worker Definition ---

class BaseWorker(object):
    """
    A base class holding common initialization logic for all worker types,
    mirroring verl's `Worker` from `single_controller`.
    """
    def __init__(self):
        # Rank, world_size, and dist_init_method are now sourced from environment variables
        self.nccl_groups = {}
        # These will be set in init_distributed_env
        self.rank = -1                # Local rank within FSDP/TP group
        self.global_rank = -1         # Global rank in collective group
        self.world_size = -1
        self.config = None # Will be set by child classes
        # dist.init_process_group is MOVED to a separate method.
        self.dist_init_method = None
        self.worker_role = "unknown"  # Role: trainer/logprob/rollout
        # Debug/instrumentation flags (enabled via env)
        self._collective_patch_installed = False

    async def init_distributed_env(self, dist_init_method: str):
        """Initializes the distributed environment using an explicit init_method."""
        _configure_global_logging()
        # logger.info(f"[{os.getpid()}] ENTERING: init_distributed_env")

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        if not dist_init_method:
            raise RuntimeError("Distributed init_method not provided. It must be passed as an argument.")

        self.dist_init_method = dist_init_method # Store for reference if needed

        # Verl-style device assignment for robustness.
        # This expects LOCAL_RANK or RAY_LOCAL_RANK to be set in the environment.
        local_rank_str = os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", "-1"))
        
        if torch.cuda.is_available():
            if local_rank_str != "-1":
                device_id = int(local_rank_str)
            else:
                # Fallback for systems where local rank is not set, assuming 1 process per GPU.
                # This works when Ray sets CUDA_VISIBLE_DEVICES for an actor with num_gpus=1.
                device_id = 0 
                logger.warning(f"LOCAL_RANK not set, falling back to device_id = 0, assuming Ray GPU isolation.")
            torch.cuda.set_device(device_id)
        else:
            device_id = -1
        
        # Add detailed logging for verification
        hostname = socket.gethostname()
        gpu_name = "N/A"
        if torch.cuda.is_available():
             gpu_name = torch.cuda.get_device_name(device_id)
        
        # logger.info(
        #     f"[{self.rank}] VERIFY_GPU_SETUP | "
        #     f"Hostname: {hostname} | "
        #     f"Process PID: {os.getpid()} | "
        #     f"Assigned Group Rank: {self.rank} (World Size: {self.world_size}) | "
        #     f"Assigned Local Device ID: {device_id} ({gpu_name})"
        # )

        if not dist.is_initialized():
            # Honor configurable process-group timeout (default 600s). Set TORCH_DIST_TIMEOUT_S to override.
            timeout_s = int(os.environ.get("TORCH_DIST_TIMEOUT_S", "600"))
            dist.init_process_group(
                backend=get_nccl_backend(),
                rank=self.rank,
                world_size=self.world_size,
                init_method=dist_init_method,
                timeout=timedelta(seconds=timeout_s),
            )

        # logger.info(f"[{self.rank}] LEAVING: init_distributed_env. Distributed environment initialized on device {device_id}.")

        # Optionally patch python-level collectives for call counting/logging

    def _install_collective_wrappers(self):
        """Wrap common torch.distributed ops to log invocation calls per rank."""
        import torch.distributed as _dist
        rank = self.rank

        def _wrap(fn_name):
            if not hasattr(_dist, fn_name):
                return
            original = getattr(_dist, fn_name)
            # Avoid double wrapping
            if getattr(original, "__grpo_wrapped__", False):
                return

            def wrapped(*args, **kwargs):
                try:
                    print(f"[COLL-DEBUG][rank={rank}] {fn_name} called")
                except Exception:
                    pass
                return original(*args, **kwargs)

            wrapped.__grpo_wrapped__ = True  # type: ignore
            setattr(_dist, fn_name, wrapped)

        for name in (
            "all_reduce",
            "reduce",
            "broadcast",
            "barrier",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "all_gather",
            "all_gather_into_tensor",
            "gather",
            "scatter",
        ):
            _wrap(name)

    # Removed: init_wandb_shared — manual system metrics replaces worker-side W&B init

    def get_global_rank(self) -> int:
        """
        Returns the worker's rank within its own initialized distributed group.
        This name is more descriptive in the new architecture.
        """
        if self.rank == -1:
             # For RolloutGPUWorker which initializes later
             self.rank = int(os.environ.get("RANK", -1))
        return self.rank
    
    def is_ready(self) -> bool:
        return True
# --- FSDP Worker Definitions ---

class BaseFSDPWorker(BaseWorker):
    """
    Handles all FSDP-related initialization and model setup.
    TrainerWorker and LogProbWorker will inherit from this.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_trainer = False
        self._is_logprob = False
        self.tokenizer = None
        self.actor_module = None
        self.actor_module_fsdp = None
        self.actor_optimizer = None
        self.actor_lr_scheduler = None
        self.actor_model_config = None
        self._is_lora = False
        self.fsdp_group = None
        self.fsdp_world_size = 0
        # Track whether weights were already materialized via from_pretrained
        self._weights_already_loaded = False

    def init_model(self, for_computation: bool = False, fsdp_group_ranks: List[int] = None):
        """
        Initializes the meta-model, tokenizer, and conditionally FSDP-wraps the model.
        Initial weights are NOT loaded here; that is a separate, explicit step.
        """
        # logger.debug(f"[{self.rank}] ENTERING: init_model (for_computation={for_computation})")
        # Ranks are now local to the FSDP group.
        self.fsdp_group_ranks = fsdp_group_ranks or list(range(self.world_size))
        self.fsdp_world_size = len(self.fsdp_group_ranks)

        # 1. Build Tokenizer and HF Model Config
        model_config_data = self.config.model
        model_path = model_config_data["path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2")
        update_model_config(hf_model_config, {"bos_token_id": self.tokenizer.bos_token_id, "eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, **model_config_data.get("override_config", {})})
        self.actor_model_config = hf_model_config

        # 2. Instantiate a fully-initialized HF model (VERL-style) via from_pretrained
        #    This ensures both parameters and buffers are correctly initialized.
        self._is_lora = model_config_data.get("lora_rank", 0) > 0
        self.actor_module = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=model_config_data.get("trust_remote_code", True),
            config=self.actor_model_config,
            torch_dtype=torch.bfloat16,
        )
        self._weights_already_loaded = True
        # Install explicit varlen wrapper if flatten packing is requested
        try:
            use_flatten_any = bool(getattr(self, "_is_trainer", False) or getattr(self, "_is_logprob", False))
        except Exception:
            use_flatten_any = False
        if use_flatten_any:
            try:
                ok = _install_explicit_fa2_varlen_wrapper()
                logger.info(f"[{self.rank}] explicit varlen wrapper install attempted -> ok={ok}")
            except Exception as _e_wrap:
                logger.warning(f"[{self.rank}] explicit varlen wrapper install failed; continuing. err={_e_wrap}")
        else:
            logger.info(f"[{self.rank}] skip varlen wrapper install: is_trainer={getattr(self,'_is_trainer',None)} is_logprob={getattr(self,'_is_logprob',None)}")

        # Rely on HF internal FA2 varlen path; no local patching

        # 3. Initialize Model on Device (with or without FSDP)
        # Monkey-patch the forward pass for GRPO BEFORE FSDP wrapping.
        # This is critical and mirrors the logic in `setup_model.py`.
        if self._is_trainer:
            make_grpo_forward(
                self.actor_module,
                temperature=float(getattr(self.config, "temperature", 1.0)),
                mode='training',
                use_torch_compile=bool(getattr(self.config, "use_torch_compile", True)),
                loss_chunksize=getattr(self.config, "loss_chunksize", None),
            )
        elif self._is_logprob:
            make_grpo_forward(
                self.actor_module,
                temperature=float(getattr(self.config, "temperature", 1.0)),
                mode='eval',
                use_torch_compile=bool(getattr(self.config, "use_torch_compile", True)),
                loss_chunksize=getattr(self.config, "loss_chunksize", None),
            )

        # 3. Initialize Model on Device (with or without FSDP)
        if self.fsdp_world_size > 1:
            # logger.info(f"[{self.rank}] Initializing model with FSDP (world_size={self.fsdp_world_size}).")
            self.fsdp_group = dist.new_group(ranks=self.fsdp_group_ranks)

            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=True,
            )
            # Keep parameters materialized; wrap with FSDP2 (fully_shard) so weights are retained and sharded
            fsdp_kwargs = {"reshard_after_forward": True, "mp_policy": mp_policy}
            _apply_fsdp_wrapping(self.actor_module, fsdp_kwargs)
            self.actor_module_fsdp = self.actor_module
        else:
            # logger.info(f"[{self.rank}] FSDP world size is 1. Initializing a standard model (not FSDP-wrapped).")
            self.actor_module.to(get_torch_device())
            self.actor_module_fsdp = self.actor_module

        

        # 4. Create Optimizer/Scheduler if needed
        if for_computation:
            # logger.info(f"[{self.rank}] Creating optimizer and scheduler.")
            from torch import optim
            optim_config = self.config.trainer["optim"]
            # Split optimizer vs scheduler kwargs to avoid passing scheduler keys to AdamW
            _optimizer_allowed_keys = (
                "lr", "betas", "eps", "weight_decay", "amsgrad",
                "foreach", "capturable", "maximize", "differentiable", "fused",
            )
            optimizer_kwargs = {k: v for k, v in optim_config.items() if k in _optimizer_allowed_keys}
            self.actor_optimizer = optim.AdamW(self.actor_module_fsdp.parameters(), **optimizer_kwargs)
            # Build simple constant-with-warmup or cosine-with-warmup scheduler
            total_steps = int(optim_config.get("total_training_steps", 0))
            warmup_steps = int(optim_config.get("lr_warmup_steps", 0))
            scheduler_type = optim_config.get("lr_scheduler", "constant")

            def lr_lambda(step: int):
                if warmup_steps > 0 and step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                if scheduler_type == "cosine" and total_steps > 0:
                    import math
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                return 1.0

            self.actor_lr_scheduler = LambdaLR(self.actor_optimizer, lr_lambda)
        # logger.debug(f"[{self.rank}] LEAVING: init_model")
        
    def _load_initial_weights(self, state_dict_to_load: bytes):
        """Loads weights into the model from a serialized state_dict."""
        # logger.info(f"[{self.rank}] ENTERING: _load_initial_weights")
        if self.actor_module_fsdp is None:
            raise RuntimeError("Model not initialized. Call init_model first.")
            
        active_state_dict = pickle.loads(state_dict_to_load)

        if self.fsdp_world_size > 1:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
            options = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False)
            
            # All ranks have the full state dict now from the broadcast, so all should pass it.
            # FSDP will handle sharding it correctly from the full dict.
            set_model_state_dict(
                self.actor_module_fsdp,
                active_state_dict,
                options=options,
            )
        else:
            self.actor_module_fsdp.load_state_dict(active_state_dict)

        # A barrier is only needed after a collective operation like FSDP's set_model_state_dict.
        # For a local load_state_dict, no barrier is necessary.
        if self.fsdp_world_size > 1 and self.fsdp_group:
            dist.barrier(group=self.fsdp_group)
        
        # logger.info(f"[{self.rank}] LEAVING: _load_initial_weights. Successfully loaded initial weights into model.")

    def get_cpu_state_dict(self):
        """Gets the full state dict, offloaded to CPU."""
        # logger.info(f"[{self.rank}] ENTERING: get_cpu_state_dict")
        if self.fsdp_world_size > 1:
            # Use the modern, unified API for getting the model state dict.
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            # This API handles rank0_only logic internally when full_state_dict and cpu_offload are True.
            # It returns the full state_dict on the designated rank (rank 0 in the default pg) and an empty dict on other ranks.
            cpu_state_dict = get_model_state_dict(
                self.actor_module_fsdp,
                options=options,
            )

            if cpu_state_dict:  # This will be true only on the rank that gets the state_dict
                # logger.info(f"[{self.rank}] LEAVING: get_cpu_state_dict (with data)")
                return pickle.dumps(cpu_state_dict)

            # logger.info(f"[{self.rank}] LEAVING: get_cpu_state_dict (without data)")
            return None
        else:
            cpu_state_dict = {k: v.cpu() for k, v in self.actor_module_fsdp.state_dict().items()}
            # logger.info(f"[{self.rank}] LEAVING: get_cpu_state_dict (with data, non-fsdp)")
            return pickle.dumps(cpu_state_dict)

@ray.remote(num_gpus=1)
class TrainerGPUWorker(BaseFSDPWorker):
    """The worker responsible for FSDP training."""
    def __init__(self, config: DictConfig, resume_from_path: str = None):
        super().__init__()
        self.config = config
        self._is_trainer = True
        self.worker_role = "trainer"  # Set role for WandB labeling
        self.cpu_state_dict = None
        # Store resume_from_path for loading later, but don't load weights here.
        self.resume_from_path = resume_from_path
        
        # Training loop related attributes
        self.gpu_rank = -1  # Local rank within trainer group (deprecated)
        self.is_training_loop_running = False
        # Orchestrator actor handle for pull-based messaging
        self.orchestrator = None
        
        # Metrics tracking
        self.current_batch_metrics = {}
        self.current_training_metrics = {}
        # Compact per-minibatch summaries (reset each BATCH_DONE)
        # Each entry: {"tokens": int, optional metrics with keys used below}
        self._mb_summaries = []
        # Counter for exact micro-batch averaging across MICROBATCH messages
        self._mb_count_since_last_step = 0
        # Backward-scalar diagnostics per step
        self._bwd_scalar_sum_step: float = 0.0
        self._mb_backward_calls_step: int = 0

    def _aggregate_minibatch_summaries(self, summaries: list[dict]) -> dict:
        """Aggregate a list of per-microbatch summaries into a single optimizer-step summary."""
        summary: dict[str, float | int] = {}
        if not summaries:
            summary.update({
                "tokens": 0,
                "total_samples": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
                "loss": 0.0,
                "kl_div": 0.0,
                "entropy": 0.0,
            })
            return summary

        total_tokens = float(sum(m.get("tokens", 0) for m in summaries))
        total_samples = float(sum(m.get("total_samples", 0) for m in summaries))
        total_reward = float(sum(m.get("total_reward", 0.0) for m in summaries))

        summary["tokens"] = total_tokens
        summary["total_samples"] = total_samples
        summary["total_reward"] = total_reward
        summary["avg_reward"] = total_reward / total_samples if total_samples > 0 else 0.0

        if total_tokens > 0:
            summary["loss"] = sum(float(m.get("loss_sum", 0.0)) for m in summaries) / total_tokens
            summary["kl_div"] = sum(float(m.get("kl_mean", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["entropy"] = sum(float(m.get("entropy_mean", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["log_ratio_mean"] = sum(float(m.get("log_ratio_mean", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["log_ratio_std"] = sum(float(m.get("log_ratio_std", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["clipfrac_upper"] = sum(float(m.get("clipfrac_upper", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["clipfrac_lower"] = sum(float(m.get("clipfrac_lower", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["adv_mean"] = sum(float(m.get("adv_mean", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["adv_std"] = sum(float(m.get("adv_std", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["adv_pos_frac"] = sum(float(m.get("adv_pos_frac", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
            summary["adv_zero_frac"] = sum(float(m.get("adv_zero_frac", 0.0)) * float(m.get("tokens", 0)) for m in summaries) / total_tokens
        else:
            summary["loss"] = 0.0
            summary["kl_div"] = 0.0
            summary["entropy"] = 0.0
            summary["log_ratio_mean"] = 0.0
            summary["log_ratio_std"] = 0.0
            summary["clipfrac_upper"] = 0.0
            summary["clipfrac_lower"] = 0.0
            summary["adv_mean"] = 0.0
            summary["adv_std"] = 0.0
            summary["adv_pos_frac"] = 0.0
            summary["adv_zero_frac"] = 0.0

        return summary

    def get_global_rank(self) -> int:
        return self.rank

    async def init_distributed_env(self, dist_init_method: str):
        """Initializes the distributed environment."""
        await super().init_distributed_env(dist_init_method)
        self.global_rank = self.rank # Set after super call
        # Ensure model is in training mode initially
        # self.actor_module_fsdp.train()
        # logger.info(f"✅ [Rank {self.global_rank}] TrainerGPUWorker-specific init_distributed_env finished.")

    def set_mode(self, mode: str):
        """Sets the model to training or evaluation mode."""
        if mode == "train":
            self.actor_module_fsdp.train()
            # logger.info(f"[Rank {self.global_rank}] Set model to TRAIN mode.")
        elif mode == "eval":
            self.actor_module_fsdp.eval()
            # logger.info(f"[Rank {self.global_rank}] Set model to EVAL mode.")
        else:
            logger.warning(f"[Rank {self.global_rank}] Unknown mode '{mode}' requested; keeping previous state.")

    async def set_orchestrator(self, orchestrator_handle, gpu_rank: int):
        """Registers the orchestrator actor handle that provides get_batch/put_completion APIs."""
        self.orchestrator = orchestrator_handle
        self.gpu_rank = gpu_rank
        logger.info(f"[{self.rank}] Experience batcher handle set for GPU rank {gpu_rank}")

    def load_own_initial_weights(self):
        """
        Loads the weights from its own cpu_state_dict into its FSDP model.
        The lead worker (rank 0 in the FSDP group) loads from disk/HF and broadcasts.
        """
        # logger.debug(f"[{self.rank}] ENTERING: load_own_initial_weights")
        # If the model was already materialized via from_pretrained in init_model,
        # skip any further weight loading to preserve properly initialized buffers.
        if getattr(self, "_weights_already_loaded", False):
            # logger.debug(f"[{self.rank}] Weights already loaded via from_pretrained; skipping load_own_initial_weights.")
            return

        # If we are not in an FSDP setup (world size is 1), load locally and skip all communication.
        if self.fsdp_world_size == 1:
            logger.info(f"[{self.rank}] FSDP world size is 1. Loading weights locally without broadcasting.")
            model_path_to_load = self.resume_from_path if self.resume_from_path else self.config.model["path"]
            full_model = AutoModelForCausalLM.from_pretrained(
                model_path_to_load, 
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )
            # Use the internal method to load the weights directly into the model
            self._load_initial_weights(pickle.dumps(full_model.state_dict()))
            del full_model
            # logger.debug(f"[{self.rank}] LEAVING: load_own_initial_weights (local load complete)")
            return

        # --- FSDP logic for world size > 1 ---
        state_dict_bytes_list = [None]
        
        # Rank 0 of the FSDP group is responsible for loading weights from disk/HF
        # and then broadcasting them to other ranks in the same FSDP group.
        if dist.get_rank(self.fsdp_group) == 0:
            logger.info(f"[{self.rank}] Loading initial model weights on trainer rank 0 (CPU) ...")
            model_path_to_load = self.resume_from_path if self.resume_from_path else self.config.model["path"]
            full_model = AutoModelForCausalLM.from_pretrained(
                model_path_to_load, 
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )
            cpu_state_dict = full_model.state_dict()
            del full_model
            logger.info(f"[{self.rank}] Initial weights loaded into CPU memory. Pickling...")
            state_dict_bytes_list = [pickle.dumps(cpu_state_dict)]
            logger.info(f"[{self.rank}] Pickling complete.")

        # logger.debug(f"[{self.rank}] Broadcasting weights within FSDP group...")
        dist.broadcast_object_list(state_dict_bytes_list, src=0, group=self.fsdp_group)
        # logger.debug(f"[{self.rank}] Broadcast complete.")

        # All ranks now have the state_dict bytes, so they can load them.
        self._load_initial_weights(state_dict_bytes_list[0])
        
        logger.info(f"[{self.rank}] Initial weights loaded and distributed (FSDP)")
    
    def compute_gradients(self, worker_batch: list, total_non_masked_tokens: int):
        """Performs a forward and backward pass to compute gradients."""
        # logger.info(f"[{self.rank}] ENTERING: compute_gradients | minibatch_len={len(worker_batch)} total_non_masked_tokens={total_non_masked_tokens}")
        self.actor_module_fsdp.train()
        # Semantics:
        #   --use-flatten-packing -> TRUE 1D varlen (unpad concat, no attention_mask)
        #   --no-use-flatten-packing -> padded BxL path
        processed_batch = post_process_batch(
            worker_batch,
            get_device_id(),
        )

        # Dummy batch handling:
        # - Multi-GPU(FSDP>1): do NOT skip. All ranks must run forward/backward to keep collectives aligned.
        #   Labels are fully masked, so the scalar loss is 0 and grads are 0 -> harmless.
        # - Single-GPU: safe to skip to save compute.
        try:
            non_masked = processed_batch.get("num_output_tokens_non_masked", torch.tensor(0, device=get_device_id())).item()
            if self.fsdp_world_size == 1 and non_masked == 0:
                logger.info(f"[rank={self.rank}] Single-GPU dummy batch detected. Skipping forward/backward.")
                return {"loss": 0.0, "pg_clip": 0, "kl_div": 0.0, "entropy": 0.0}
        except Exception:
            pass

        # Ensure kernels like FlashAttention run with supported dtype
        if torch.cuda.is_available():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()

        with autocast_ctx:
            # logger.info(f"[rank={self.rank}] compute_gradients: BEFORE_FORWARD")
            loss, metrics = compute_dual_clip_grpo_loss(
                self.actor_module_fsdp, processed_batch,
                self.config.clip_low, self.config.clip_high,
                self.config.clip_ratio_c,
                compute_entropy=bool(getattr(self.config, "compute_entropy", True)),
                apply_entropy_bonus=bool(getattr(self.config, "apply_entropy_bonus", True)),
                entropy_coeff=float(getattr(self.config, "entropy_coeff", 0.0)),
            )
            # logger.info(f"[rank={self.rank}] compute_gradients: AFTER_FORWARD | loss={float(loss.detach()) if hasattr(loss,'detach') else loss}")

        # VERL token-mean semantics: loss is already mean over masked tokens.
        # Apply sample-weight scaling per micro-batch before backward: scale by (num_real_samples / train_minibatch_sample_size)
        try:
            num_real_samples = sum(1 for s in worker_batch if not s.get("dummy", False))
        except Exception:
            num_real_samples = len(worker_batch)
        try:
            local_target = int(getattr(self.config, "train_minibatch_sample_size", 0) or 0)
            if self.fsdp_world_size and self.fsdp_world_size > 0:
                local_target = max(1, local_target // self.fsdp_world_size)  # example: 128//2=64
            _denom = float(local_target)
        except Exception:
            _denom = 0.0
        if _denom > 0.0:
            _scale = float(num_real_samples) / _denom
        else:
            _scale = 1.0
        try:
            if _scale != 1.0:
                loss = loss * float(_scale)
            # Log micro-batch backward scalar after scaling
            try:
                mb_tok = int(processed_batch.get("num_output_tokens_non_masked", 0))
            except Exception:
                mb_tok = 0
            try:
                scalar_val = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
            except Exception:
                scalar_val = 0.0
            self._bwd_scalar_sum_step += scalar_val
            self._mb_backward_calls_step += 1
            mb_idx = int(getattr(self, "_mb_backward_calls_step", 0))
        except Exception:
            pass
        # Accumulate grads
        loss.backward()

        # Count this minibatch for exact micro-batch averaging at optimizer step
        try:
            self._mb_count_since_last_step = int(getattr(self, "_mb_count_since_last_step", 0)) + 1
        except Exception:
            self._mb_count_since_last_step = 1
        
        # logger.info(f"[rank={self.rank}] LEAVING: compute_gradients")
        return metrics

    def apply_gradients(self):
        """Clips gradients and performs an optimizer step."""
        # logger.debug(f"[{self.rank}] ENTERING: apply_gradients")
        # Prefer FSDP-native gradient clipping to ensure correct global norm
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            grad_norm = FSDP.clip_grad_norm_(self.actor_module_fsdp, 1.0)
        except Exception:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), 1.0)
        if hasattr(grad_norm, "full_tensor"):
            try:
                grad_norm = grad_norm.full_tensor()
            except Exception:
                pass
        try:
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()
            self.actor_optimizer.zero_grad()
            # logger.debug(f"[{self.rank}] LEAVING: apply_gradients")
            return float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
        except Exception:
            pass
        # Fallback path (no param snapshot)
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()
        self.actor_optimizer.zero_grad()
        # logger.debug(f"[{self.rank}] LEAVING: apply_gradients")
        return grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

    def save_model(self, save_directory: str):
        """
        Saves the FSDP model to the specified directory in HuggingFace format.
        This method should only be executed by the rank 0 worker in the FSDP group.
        """
        # logger.info(f"[{self.rank}] Saving model to {save_directory}")
        
        # Use new checkpoint APIs compatible with recent torch/FSDP
        # All ranks must enter to keep collectives aligned
        try:
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            cpu_state_dict = get_model_state_dict(self.actor_module_fsdp, options=options)
        except Exception as e:
            logger.error(f"[{self.rank}] get_model_state_dict failed: {e}", exc_info=True)
            # Fallback to standard state_dict() as best effort (may be memory heavy)
            cpu_state_dict = self.actor_module_fsdp.state_dict()

        if dist.get_rank(self.fsdp_group) == 0:
            os.makedirs(save_directory, exist_ok=True)
            
            # self.actor_module is on the meta device. save_pretrained can be used with it.
            self.actor_module.save_pretrained(save_directory, state_dict=cpu_state_dict)
            self.tokenizer.save_pretrained(save_directory)
            # logger.info(f"[{self.rank}] Model and tokenizer saved to {save_directory}")

        # Barrier to ensure all processes wait until rank 0 has finished saving.
        if self.fsdp_group:
            dist.barrier(group=self.fsdp_group)
            
        # logger.info(f"[{self.rank}] Model save complete")

    def save_optimizer_state(self, save_directory: str):
        """
        Saves the optimizer and scheduler state to the specified directory.
        This method should be called after save_model.
        """
        # logger.info(f"[{self.rank}] Saving optimizer/scheduler state to {save_directory}")
        
        if self.fsdp_world_size > 1:
            # PyTorch 2.7+ path: use distributed.checkpoint state_dict APIs
            from torch.distributed.checkpoint import FileSystemWriter, save
            writer = FileSystemWriter(save_directory)
            try:
                # PyTorch 2.7: get_optimizer_state_dict signature does not take optim_state_dict_config.
                # Sharding/offload behavior is controlled by the module's FSDP config.
                opt_sd = get_optimizer_state_dict(
                    self.actor_module_fsdp,
                    self.actor_optimizer,
                )
                save({"optimizer": opt_sd}, storage_writer=writer)
            except Exception as e:
                logger.error(f"[{self.rank}] get_optimizer_state_dict/save failed: {e}")
                raise

            # Save scheduler state separately (scheduler state is not distributed) from rank 0 only
            if dist.get_rank(self.fsdp_group) == 0 and self.actor_lr_scheduler is not None:
                scheduler_state_path = os.path.join(save_directory, "scheduler_state.pt")
                torch.save(self.actor_lr_scheduler.state_dict(), scheduler_state_path)
                # logger.info(f"[{self.rank}] Scheduler state saved to {scheduler_state_path}")

            # logger.info(f"[{self.rank}] Optimizer state saved to {save_directory}")
        else:
            # For single GPU, standard save
            if self.actor_optimizer is not None:
                optimizer_state_path = os.path.join(save_directory, "optimizer_state.pt")
                torch.save(self.actor_optimizer.state_dict(), optimizer_state_path)
                # logger.info(f"[{self.rank}] Optimizer state saved to {optimizer_state_path}")
            
            if self.actor_lr_scheduler is not None:
                scheduler_state_path = os.path.join(save_directory, "scheduler_state.pt")
                torch.save(self.actor_lr_scheduler.state_dict(), scheduler_state_path)
                # logger.info(f"[{self.rank}] Scheduler state saved to {scheduler_state_path}")

        # Barrier to ensure all processes wait
        if self.fsdp_group:
            dist.barrier(group=self.fsdp_group)
            
        # logger.info(f"[{self.rank}] Optimizer/scheduler state save complete")

    def load_optimizer_state(self, load_directory: str):
        """
        Loads the optimizer and scheduler state from the specified directory.
        This method should be called after the model weights are loaded.
        """
        # logger.info(f"[{self.rank}] Loading optimizer/scheduler state from {load_directory}")
        
        if self.fsdp_world_size > 1:
            # For FSDP, use the distributed checkpoint API
            from torch.distributed.checkpoint import FileSystemReader, load_sharded_optimizer_state_dict
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            try:
                reader = FileSystemReader(load_directory)
                # Extract model state dict
                model_state = self.actor_module_fsdp.state_dict()
                # Load the sharded optimizer state
                loaded = load_sharded_optimizer_state_dict(
                    model_state,
                    optimizer_key="optimizer",
                    storage_reader=reader
                )
                # Apply the optimizer state via FSDP API
                FSDP.optim_state_dict_to_load(
                    self.actor_module_fsdp,
                    self.actor_optimizer,
                    loaded["optimizer"]
                )
                # logger.info(f"[{self.rank}] Optimizer state loaded from {load_directory}")
                
                # Load scheduler state separately
                if self.actor_lr_scheduler is not None:
                    scheduler_state_path = os.path.join(load_directory, "scheduler_state.pt")
                    if os.path.exists(scheduler_state_path):
                        scheduler_state = torch.load(scheduler_state_path, map_location="cpu")
                        self.actor_lr_scheduler.load_state_dict(scheduler_state)
                        # logger.info(f"[{self.rank}] Scheduler state loaded from {scheduler_state_path}")
            except Exception as e:
                logger.warning(f"[{self.rank}] Failed to load optimizer/scheduler state: {e}. Continuing with fresh state.")
        else:
            # For single GPU, standard load
            try:
                if self.actor_optimizer is not None:
                    optimizer_state_path = os.path.join(load_directory, "optimizer_state.pt")
                    if os.path.exists(optimizer_state_path):
                        optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
                        self.actor_optimizer.load_state_dict(optimizer_state)
                        # logger.info(f"[{self.rank}] Optimizer state loaded from {optimizer_state_path}")
                
                if self.actor_lr_scheduler is not None:
                    scheduler_state_path = os.path.join(load_directory, "scheduler_state.pt")
                    if os.path.exists(scheduler_state_path):
                        scheduler_state = torch.load(scheduler_state_path, map_location="cpu")
                        self.actor_lr_scheduler.load_state_dict(scheduler_state)
                        # logger.info(f"[{self.rank}] Scheduler state loaded from {scheduler_state_path}")
                        
            except Exception as e:
                logger.warning(f"[{self.rank}] Failed to load optimizer/scheduler state: {e}. Continuing with fresh state.")

        # Barrier to ensure all processes wait
        if self.fsdp_group:
            dist.barrier(group=self.fsdp_group)
            
        # logger.info(f"[{self.rank}] Optimizer/scheduler state load complete")

    def barrier_fsdp_group(self):
        """Synchronize all ranks in this worker's FSDP group (no-op if single GPU)."""
        try:
            if self.fsdp_group is not None:
                dist.barrier(group=self.fsdp_group)
        except Exception:
            pass

    async def start_training_loop(self):
        """Start the main training loop similar to the original trainer_core.py"""
        if self.is_training_loop_running:
            logger.warning(f"[{self.rank}] Training loop already running")
            return
            
        # Ensure orchestrator (ExperienceManager) handle is set
        if not self.orchestrator:
            logger.error(f"[{self.rank}] Orchestrator handle not set. Cannot start training loop.")
            return
            
        self.is_training_loop_running = True
        # logger.info(f"[{self.rank}] Starting training loop for GPU rank {self.gpu_rank}")
        
        try:
            # Initialize metrics tracking
            total_samples_accumulated = 0
            iteration = 0
            grad_steps_this_batch = 0
            clipped_steps_this_batch = 0
            # Step-level accumulators (reset at each GRADIENT_STEP)
            self._step_reward_values = []
            self._step_adv_values = []
            self._step_nonzero_reward = 0
            self._step_total_samples = 0
            self._q_to_sample_count_in_step = {}
            self._mb_summaries_step = []
            self._optimizer_step_summaries = []
            # Accumulators for identical-reward question counts since last optimizer step
            self._eq_all1_since_last_step = 0
            self._eq_all0_since_last_step = 0
            self._eq_total_q_since_last_step = 0
            # Track unique question keys across all minibatches until the next optimizer.step
            self._seen_qkeys_in_step = set()
            self._grad_norms_in_batch = []
            self._minibatch_counter_in_batch = 0

            
            while self.is_training_loop_running:
                # Get message from orchestrator actor (pull-based messaging)
                msg = await self.orchestrator.get_batch.remote(self.gpu_rank)
                
                if msg.type == MessageType.MICROBATCH:
                    # logger.info(f"[{self.rank}] Received MICROBATCH with {len(msg.data)} samples")
                    
                    # Process the minibatch and store metrics
                    total_tokens = msg.data[0].get('total_non_masked_output_tokens', 0) if msg.data else 0
                    with record_function("trainer.compute_gradients"):
                        metrics = self.compute_gradients(msg.data, total_tokens)

                    # Note: compute_gradients internally rebuilds processed_batch; we cannot access it here.
                    # Build a compact per-minibatch summary; aggregate later at BATCH_DONE.
                    # Prefer the non-masked token count propagated from ExperienceManager.
                    try:
                        mb_tokens = int(msg.data[0].get("num_non_masked_output_tokens", 0))
                    except Exception:
                        mb_tokens = 0
                    if mb_tokens <= 0:
                        # Final fallback: use total_non_masked_output_tokens passed into compute_gradients
                        try:
                            mb_tokens = int(total_tokens) if total_tokens is not None else 0
                        except Exception:
                            mb_tokens = 0

                    # Summarize this minibatch for later aggregation
                    # Loss is MEAN over masked tokens (VERL mean mode); convert to weighted sum for aggregation
                    mb_loss_sum = float(metrics.get("loss", 0.0)) * float(mb_tokens)
                    mb_kl_mean = float(metrics.get("kl_div", 0.0))
                    mb_entropy_mean = float(metrics.get("entropy", 0.0))
                    mb_summary = {
                        "tokens": mb_tokens,
                        "loss_sum": mb_loss_sum,
                        "kl_mean": mb_kl_mean,
                        "entropy_mean": mb_entropy_mean,
                    }
                    mb_summary["minibatch_index"] = int(self._minibatch_counter_in_batch)
                    self._minibatch_counter_in_batch += 1
                    # Optional diagnostics
                    try:
                        mb_summary["log_ratio_mean"] = float(metrics.get("log_ratio_mean", 0.0))
                        mb_summary["log_ratio_std"] = float(metrics.get("log_ratio_std", 0.0))
                    except Exception:
                        pass
                    try:
                        mb_summary["clipfrac_upper"] = float(metrics.get("clipfrac_upper", 0.0))
                        mb_summary["clipfrac_lower"] = float(metrics.get("clipfrac_lower", 0.0))
                    except Exception:
                        pass
                    try:
                        mb_summary["adv_mean"] = float(metrics.get("adv_mean", 0.0))
                        mb_summary["adv_std"] = float(metrics.get("adv_std", 0.0))
                        mb_summary["adv_pos_frac"] = float(metrics.get("adv_pos_frac", 0.0))
                        mb_summary["adv_zero_frac"] = float(metrics.get("adv_zero_frac", 0.0))
                    except Exception:
                        pass
                    try:
                        real_samples = [s for s in msg.data if not s.get("dummy", False)]
                    except Exception:
                        real_samples = []
                    reward_values = [float(s.get("reward", 0.0)) for s in real_samples]
                    if reward_values:
                        avg_reward = sum(reward_values) / float(len(reward_values))
                        total_reward = sum(reward_values)
                    else:
                        avg_reward = 0.0
                        total_reward = 0.0
                    mb_summary["avg_reward"] = avg_reward
                    mb_summary["total_reward"] = total_reward
                    mb_summary["total_samples"] = len(reward_values)
                    self._mb_summaries.append(mb_summary)
                    self._mb_summaries_step.append(mb_summary)
                    
                elif msg.type == MessageType.GRADIENT_STEP:
                    # logger.info(f"[{self.rank}] Received GRADIENT_STEP")
                    
                    # Take gradient step and store grad norm
                    with record_function("trainer.apply_gradients"):
                        grad_norm = self.apply_gradients()
                    self._grad_norms_in_batch.append(float(grad_norm))

                    grad_steps_this_batch += 1

                    # Extended step-level summary: reward/adv/spq and ratio/clip aggregates
                    try:
                        # Samples-per-question stats
                        spq_vals = list(self._q_to_sample_count_in_step.values())
                        spq_min = min(spq_vals) if spq_vals else 0
                        spq_max = max(spq_vals) if spq_vals else 0
                        spq_med = (sorted(spq_vals)[len(spq_vals)//2] if spq_vals else 0)
                        # Reward stats (sample-level)
                        rv = self._step_reward_values
                        if rv:
                            r_mu = sum(rv) / float(len(rv))
                            r_ex2 = sum(x*x for x in rv) / float(len(rv))
                            r_sd = max(r_ex2 - r_mu * r_mu, 0.0) ** 0.5
                            r_min = min(rv)
                            r_max = max(rv)
                            nz_frac = float(self._step_nonzero_reward) / float(max(1, self._step_total_samples))
                        else:
                            r_mu = r_sd = r_min = r_max = nz_frac = 0.0
                        # Advantage stats (sample-level)
                        av = self._step_adv_values
                        if av:
                            a_mu = sum(av) / float(len(av))
                            a_ex2 = sum(x*x for x in av) / float(len(av))
                            a_sd = max(a_ex2 - a_mu * a_mu, 0.0) ** 0.5
                            a_zf = float(sum(1 for x in av if abs(x) < 1e-12)) / float(len(av))
                        else:
                            a_mu = a_sd = a_zf = 0.0
                        # Ratio/clip aggregates from step-local mb summaries
                        denom = float(sum(m.get('tokens', 0) for m in self._mb_summaries_step))
                        if denom > 0:
                            try:
                                s_rm = sum(float(m.get('log_ratio_mean', 0.0)) * float(m.get('tokens', 0)) for m in self._mb_summaries_step)
                                s_rs = sum(float(m.get('log_ratio_std', 0.0)) * float(m.get('tokens', 0)) for m in self._mb_summaries_step)
                                s_up = sum(float(m.get('clipfrac_upper', 0.0)) * float(m.get('tokens', 0)) for m in self._mb_summaries_step)
                                s_lo = sum(float(m.get('clipfrac_lower', 0.0)) * float(m.get('tokens', 0)) for m in self._mb_summaries_step)
                                rr_mu = s_rm / denom
                                rr_sd = s_rs / denom
                                cf_up = s_up / denom
                                cf_lo = s_lo / denom
                            except Exception:
                                rr_mu = rr_sd = cf_up = cf_lo = float('nan')
                        else:
                            rr_mu = rr_sd = cf_up = cf_lo = float('nan')
                        spq_all_equal = (spq_min == spq_max)
                    except Exception:
                        pass
                    # Reset counters after the step
                    self._eq_all1_since_last_step = 0
                    self._eq_all0_since_last_step = 0
                    self._eq_total_q_since_last_step = 0
                    self._seen_qkeys_in_step = set()
                    # Reset step-level accumulators
                    self._step_reward_values = []
                    self._step_adv_values = []
                    self._step_nonzero_reward = 0
                    self._step_total_samples = 0
                    self._q_to_sample_count_in_step = {}
                    self._mb_summaries_step = []
                    
                    logger.info(f"[{self.rank}] Applied gradients, grad_norm: {grad_norm}")
                    step_summary = self._aggregate_minibatch_summaries(self._mb_summaries_step)
                    step_summary["grad_norm"] = grad_norm
                    step_summary["trainer_rank"] = self.gpu_rank
                    step_summary["minibatch_index"] = grad_steps_this_batch - 1
                    step_summary["optimizer_steps_in_batch"] = grad_steps_this_batch
                    self._optimizer_step_summaries.append(step_summary)
                    iteration += 1
                    
                elif msg.type == MessageType.BATCH_DONE:
                    # logger.info(f"[{self.rank}] Received BATCH_DONE")
                    
                    # Extract generation/logprob metrics and ExperienceManager batch metrics from BATCH_DONE message
                    batch_done_data = msg.data or {}
                    gen_logprob_metrics = batch_done_data.get("gen_logprob_metrics", {})
                    exp_batch_metrics = batch_done_data.get("batch_metrics", {})
                    
                    if self._grad_norms_in_batch:
                        avg_grad_norm = sum(self._grad_norms_in_batch) / float(len(self._grad_norms_in_batch))
                    else:
                        avg_grad_norm = 0.0
                    self.current_training_metrics['grad_norm'] = avg_grad_norm

                    # Finalize aggregated training metrics across MICROBATCH-es
                    denom = sum(m.get("tokens", 0) for m in self._mb_summaries)
                    if denom <= 0:
                        # No valid tokens in this worker's batch
                        self.current_training_metrics["loss"] = 0.0
                        self.current_training_metrics["loss_per_token"] = 0.0
                        self.current_training_metrics["kl_div"] = 0.0
                        self.current_training_metrics["entropy"] = 0.0
                        self.current_training_metrics["token_count"] = 0
                    else:
                        # Token-weighted aggregates
                        loss_sum = sum(m.get("loss_sum", 0.0) for m in self._mb_summaries)
                        self.current_training_metrics["loss"] = loss_sum / float(denom)
                        self.current_training_metrics["loss_per_token"] = self.current_training_metrics["loss"]
                        kl_sum = sum(m.get("kl_mean", 0.0) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                        ent_sum = sum(m.get("entropy_mean", 0.0) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                        self.current_training_metrics["kl_div"] = kl_sum / float(denom)
                        self.current_training_metrics["entropy"] = ent_sum / float(denom)
                        # Means/Std via second moment when available
                        try:
                            s_mu = 0.0
                            s_ex2 = 0.0
                            for m in self._mb_summaries:
                                tok = float(m.get("tokens", 0))
                                mu_i = float(m.get("log_ratio_mean", 0.0))
                                sd_i = float(m.get("log_ratio_std", 0.0))
                                s_mu += mu_i * tok
                                s_ex2 += ((sd_i * sd_i) + (mu_i * mu_i)) * tok
                            mu = s_mu / float(denom)
                            ex2 = s_ex2 / float(denom)
                            var = max(ex2 - mu * mu, 0.0)
                            self.current_training_metrics["log_ratio_mean"] = mu
                            self.current_training_metrics["log_ratio_std"] = var ** 0.5
                        except Exception:
                            pass
                        try:
                            up = sum(float(m.get("clipfrac_upper", 0.0)) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                            lo = sum(float(m.get("clipfrac_lower", 0.0)) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                            self.current_training_metrics["clipfrac_upper"] = up / float(denom)
                            self.current_training_metrics["clipfrac_lower"] = lo / float(denom)
                        except Exception:
                            pass
                        try:
                            s_mu = 0.0
                            s_ex2 = 0.0
                            for m in self._mb_summaries:
                                tok = float(m.get("tokens", 0))
                                mu_i = float(m.get("adv_mean", 0.0))
                                sd_i = float(m.get("adv_std", 0.0))
                                s_mu += mu_i * tok
                                s_ex2 += ((sd_i * sd_i) + (mu_i * mu_i)) * tok
                            mu = s_mu / float(denom)
                            ex2 = s_ex2 / float(denom)
                            var = max(ex2 - mu * mu, 0.0)
                            self.current_training_metrics["adv_mean"] = mu
                            self.current_training_metrics["adv_std"] = var ** 0.5
                            pos = sum(float(m.get("adv_pos_frac", 0.0)) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                            zf = sum(float(m.get("adv_zero_frac", 0.0)) * float(m.get("tokens", 0)) for m in self._mb_summaries)
                            self.current_training_metrics["adv_pos_frac"] = pos / float(denom)
                            self.current_training_metrics["adv_zero_frac"] = zf / float(denom)
                        except Exception:
                            pass
                        # Provide the token denominator used for aggregation to orchestrator
                        self.current_training_metrics["token_count"] = int(denom)
                    # Merge ExperienceManager diagnostics if provided by BATCH_DONE payload later
                    # Merge additional reward distribution stats if provided by ExperienceManager
                    if isinstance(exp_batch_metrics, dict) and exp_batch_metrics:
                        try:
                            self.current_batch_metrics.update({
                                "reward_mean": exp_batch_metrics.get("reward_mean", self.current_batch_metrics.get("avg_reward", 0.0)),
                                "reward_std": exp_batch_metrics.get("reward_std", 0.0),
                                "reward_min": exp_batch_metrics.get("reward_min", 0.0),
                                "reward_max": exp_batch_metrics.get("reward_max", 0.0),
                                "group_reward_std_mean": exp_batch_metrics.get("group_reward_std_mean", 0.0),
                                "group_reward_std_zero_frac": exp_batch_metrics.get("group_reward_std_zero_frac", 0.0),
                                "nonzero_reward_frac": exp_batch_metrics.get("nonzero_reward_frac", 0.0),
                                "avg_completion_length": exp_batch_metrics.get("avg_completion_length", self.current_batch_metrics.get("avg_completion_length", 0.0)),
                            })
                        except Exception:
                            pass

                    # Prepare completion data to send to Orchestrator
                    minibatch_payload = [summary.copy() for summary in self._optimizer_step_summaries]

                    completion_data = {
                        "gpu_rank": self.gpu_rank,
                        "metrics": self.current_training_metrics.copy(),
                        "batch_metrics": self.current_batch_metrics.copy(),
                        "gen_logprob_metrics": gen_logprob_metrics,  # Pass through gen/logprob timing
                        "iteration": iteration,
                        # New: how many optimizer steps (and clipped steps) happened within this batch
                        "optimizer_steps_in_batch": grad_steps_this_batch,
                        "grad_clip_steps_in_batch": clipped_steps_this_batch,
                        "grad_clip_ratio_in_batch": (clipped_steps_this_batch / grad_steps_this_batch) if grad_steps_this_batch > 0 else 0.0,
                        # Echo ExperienceManager batch diagnostics explicitly as well (optional)
                        "exp_batch_metrics": exp_batch_metrics,
                        "minibatch_summaries": minibatch_payload,
                    }

                    # Send completion notification to Orchestrator actor
                    await self.orchestrator.put_completion.remote(self.gpu_rank, completion_data)
                    logger.info(f"[{self.rank}] Sent batch completion notification to Orchestrator")

                    # Removed BATCH_DONE-phase per-UID accuracy logging; now logged earlier when rewards arrive.

                    # Reset metrics for next batch
                    self.current_training_metrics = {}
                    self.current_batch_metrics = {}
                    grad_steps_this_batch = 0
                    clipped_steps_this_batch = 0
                    self._mb_summaries = []
                    self._grad_norms_in_batch = []
                    self._minibatch_counter_in_batch = 0
                    self._optimizer_step_summaries = []
                    # Reset batch samples accumulator
                    if hasattr(self, '_batch_samples'):
                        self._batch_samples = []
                    
                    # logger.info(f"[{self.rank}] Batch done, iteration: {iteration}")
                    
                else:
                    logger.warning(f"[{self.rank}] Unknown message type: {msg.type}")
                    
        except Exception as e:
            logger.error(f"[{self.rank}] Error in training loop: {e}", exc_info=True)
        finally:
            self.is_training_loop_running = False
            logger.info(f"[{self.rank}] Training loop stopped")

    def stop_training_loop(self):
        """Stop the training loop."""
        self.is_training_loop_running = False
        logger.info(f"[{self.rank}] Training loop stop requested")

    def _get_actor_params(self):
        """
        Retrieve the FSDP state_dict and convert its keys for vLLM compatibility.
        This uses Hugging Face's _checkpoint_conversion_mapping via convert_weight_keys.
        """
        # Aligned with verl's implementation: directly call state_dict().
        params = self.actor_module_fsdp.state_dict()
        # Convert keys using unwrapped model
        unwrapped = getattr(self.actor_module_fsdp, '_fsdp_wrapped_module', self.actor_module_fsdp)
        converted = convert_weight_keys(params, unwrapped)
        return converted

    def get_trainer_weights_info(self):
        """
        Creates and returns weight metadata with standard HuggingFace keys.
        This is the single source of truth for synchronization across all workers.
        NOTE: This is designed to be called only on the rank 0 worker of the FSDP group.
        """
        # logger.info(f"[{self.rank}] ENTERING: get_trainer_weights_info")
        
        if hasattr(self, "weights_info"):
            return self.weights_info
            
        # Aligned with verl: get params and then create the metadata list.
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        
        self.weights_info = ret
        # logger.info(f"[{self.rank}] LEAVING: get_trainer_weights_info, created info for {len(self.weights_info)} tensors.")
        return ret

    def set_trainer_weights_info(self, weights_info: List[tuple]):
        """Stores the weight metadata received from the Trainer."""
        # logger.info(f"[{self.rank}] Storing weights info.")
        self.weights_info = weights_info

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """Broadcasts its weights to other workers using ray.util.collective."""
        # logger.info(f"[{self.rank}] ENTERING: sync_weights (as source group participant)")
        
        if not hasattr(self, 'weights_info') or not self.weights_info:
            raise RuntimeError("weights_info not set on TrainerGPUWorker before syncing weights.")

        # Aligned with verl's implementation.
        params = self._get_actor_params()
        
        for key, shape, dtype in self.weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device())

            # This block is for the source of the broadcast (the trainer).
            # It prepares the tensor that will be sent.
            assert key in params, f"Key {key} not found in trainer parameters."
            origin_data = params[key]
            
            # If the parameter is a ShardedTensor, get the full tensor.
            if hasattr(origin_data, "full_tensor"):
                origin_data = origin_data.full_tensor()
            
            # The FSDP group's rank 0 is responsible for copying the data.
            # We assume this rank is the source_rank_in_collective_group.
            my_rank_in_fsdp_group = dist.get_rank(self.fsdp_group)
            if my_rank_in_fsdp_group == 0:
                tensor.copy_(origin_data)

            collective.broadcast(tensor, src_rank=source_rank_in_collective_group, group_name=group_name)
            
        # await collective.barrier(group_name=group_name)
        # logger.info(f"[{self.rank}] LEAVING: sync_weights (as source group participant)")

    

@ray.remote(num_gpus=1)
class LogProbGPUWorker(BaseFSDPWorker):
    """The worker responsible for log probability calculations using FSDP."""
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self._is_logprob = True
        self.worker_role = "logprob"  # Set role for WandB labeling

        # Optional, run-once verification flag for mixing check
        self._verify_done = False
        # This worker needs the FSDP model but doesn't train it.
        # self.init_model(for_computation=False) -> This is now called by the controller.
        # It will need to have its weights loaded by the orchestrator.
    
    def get_global_rank(self) -> int:
        return self.rank
    
    @torch.no_grad()
    def compute_log_probs_from_samples(self, samples: list):
        """Build tensors on this GPU worker and compute per-sample logprobs and entropies.
        Returns a list of dicts (one per sample): {"logprobs": [...], "entropies": [...]} where lists are CPU lists.
        """
        self.actor_module_fsdp.eval()
        device = get_device_id()

        results_per_sample = []
        try:
            sample_tensors = [torch.tensor(s['sample_ids'], dtype=torch.long, device=device) for s in samples]
            pos_tensors = [torch.arange(len(s['sample_ids']), dtype=torch.long, device=device) for s in samples]
            lab_tensors = [
                torch.tensor(
                    s.get('labels', [-100] * len(s.get('sample_ids', []))),
                    dtype=torch.long,
                    device=device,
                )
                for s in samples
            ]
            lens = [t.numel() for t in sample_tensors]

            input_ids = torch.cat(sample_tensors).unsqueeze(0)
            position_ids = torch.cat(pos_tensors).unsqueeze(0)
            labels = torch.cat(lab_tensors).unsqueeze(0)

            outputs = self.actor_module_fsdp(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                labels=labels,
            )
            flat_lp = outputs.loss  # [T]

            flat_ent = getattr(outputs, "entropy", None)
            splits_lp = torch.split(flat_lp, lens)
            splits_ent = torch.split(flat_ent, lens) if flat_ent is not None else [None] * len(lens)
            for lp, et in zip(splits_lp, splits_ent):
                results_per_sample.append({
                    "logprobs": lp.detach().cpu().tolist(),
                    "entropies": (et.detach().cpu().tolist() if et is not None else None),
                })
        except Exception as e:
            logger.error(f"[{self.rank}] compute_log_probs_from_samples error: {e}", exc_info=True)
            raise

        # logger.debug(f"[{self.rank}] LEAVING: compute_log_probs_from_samples")
        return results_per_sample

    def set_trainer_weights_info(self, weights_info: List[tuple]):
        """Stores the weight metadata received from the Trainer."""
        logger.info(f"[{self.rank}] Storing weights info.")
        self.weights_info = weights_info

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """
        Receives weights broadcasted from the TrainerWorker and loads them into
        the FSDP model in a memory-efficient way.
        """
        # logger.info(f"[{self.rank}] ENTERING: sync_weights (as destination)")
        
        if not hasattr(self, 'weights_info') or not self.weights_info:
            raise RuntimeError("weights_info not set on LogProbGPUWorker before syncing weights.")

        # For multi-GPU FSDP, only rank 0 gathers the full state dict to save memory.
        # my_rank_in_fsdp_group = dist.get_rank(self.fsdp_group)
        # is_designated_loader = (my_rank_in_fsdp_group == 0)
        
        received_state_dict = {}
        for hf_key, shape, dtype in self.weights_info:
            # All ranks must participate in broadcast, and all should store the result
            # to construct the full state dict for loading.
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device())
            collective.broadcast(tensor, src_rank=source_rank_in_collective_group, group_name=group_name)
            received_state_dict[hf_key] = tensor
        
        if self.fsdp_world_size > 1:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
            # Rank 0 provides the full dict, others provide an empty one.
            # FSDP handles scattering the weights from rank 0 to all other ranks.
            options = StateDictOptions(full_state_dict=True, cpu_offload=False, broadcast_from_rank0=False)
            set_model_state_dict(
                self.actor_module_fsdp,
                received_state_dict,
                options=options,
            )
        else:
            # For a single GPU worker, a standard load is sufficient.
            self.actor_module_fsdp.load_state_dict(received_state_dict)

        # A barrier is needed to ensure all ranks have loaded weights before proceeding.
        dist.barrier(group=self.fsdp_group)
        # logger.info(f"[{self.rank}] LEAVING: sync_weights (as destination). LogProbWorker successfully synced weights.")

    def load_pretrained_weights_for_testing(self, model_name: str):
        """
        Testing-only method: Load pretrained weights directly from HuggingFace for standalone tests.
        This bypasses the normal orchestrated weight synchronization flow.
        """
        # logger.info(f"[{self.rank}] ENTERING: load_pretrained_weights_for_testing (model={model_name})")
        
        if self.actor_module_fsdp is None:
            raise RuntimeError("Model not initialized. Call init_model first.")
        
        import pickle
        from transformers import AutoModelForCausalLM
        
        # Load pretrained model to get weights
        # logger.info(f"[{self.rank}] Loading pretrained model from {model_name}...")
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Match FSDP model dtype
            # device_map="cpu"  # Load on CPU first
        )
        
        # Use the existing _load_initial_weights method
        state_dict_bytes = pickle.dumps(pretrained_model.state_dict())
        self._load_initial_weights(state_dict_bytes)
        
        # Clean up
        del pretrained_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # logger.info(f"[{self.rank}] LEAVING: load_pretrained_weights_for_testing. Weights loaded successfully.")

# --- vLLM Worker Definition ---

@ray.remote(num_gpus=1)
class RolloutGPUWorker(object):
    """The worker responsible for vLLM-based generation."""
    def __init__(self, config: DictConfig):
        _configure_global_logging()
        self.config = config
        self._is_generator = True
        self.worker_role = "rollout"  # Set role for WandB labeling
        self.vllm_worker_engine = None # This will be the WorkerWrapperBase instance
        self.rank = -1 # Will be set by vLLM init
        self.global_rank = -1  # Global rank in collective group
        self.world_size = -1
        # self.initial_sync_done = False # No longer needed, we will attempt to load weights on every sync.

    def get_global_rank(self) -> int:
        # This worker's rank is local to its own TP group.
        if self.rank == -1:
             self.rank = int(os.environ.get("RANK", -1))
        return self.rank

    def init_vllm_worker(self, all_kwargs: List[Dict[str, Any]]):
        """
        Initializes the low-level vLLM worker engine.
        This method is called via RPC from the ExternalProcessExecutor.
        It is this worker's entrypoint into a distributed environment.
        """
        _configure_global_logging()
        # logger.info(f"[PID:{os.getpid()}] ENTERING: init_vllm_worker")
        
        # This worker's specific rank is determined by the environment variable.
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        
        # Get the specific kwargs for THIS worker from the list.
        kwargs = all_kwargs[self.rank]
        local_rank = kwargs["local_rank"]
        
        # logger.info(f"RolloutWorker on PID {os.getpid()} is initializing its vLLM worker with local_rank: {local_rank} (Group Rank: {self.rank}, World Size: {self.world_size})")
        
        vllm_config = kwargs["vllm_config"]
        # WorkerWrapperBase expects the FULL list of kwargs for all workers.
        self.vllm_worker_engine = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=self.rank)
        self.vllm_worker_engine.init_worker(all_kwargs)
        
        # logger.info(f"[{self.rank}] LEAVING: init_vllm_worker")

    def execute_method(self, method: str, *args, **kwargs):
        """
        Generic method executor called via RPC from ExternalProcessExecutor.
        """
        _configure_global_logging()

        # If 'args' is passed as a keyword argument, unpack it into positional args.
        # This handles how vLLM's collective_rpc packages arguments.
        if 'args' in kwargs:
            new_args = kwargs.pop('args')
            # The 'args' kwarg from vllm is usually a list containing a tuple.
            if isinstance(new_args, list) and len(new_args) > 0 and isinstance(new_args[0], tuple):
                 args = new_args[0] + args
            else:
                 args = tuple(new_args) + args

        if self.vllm_worker_engine is None:
            if method == "init_vllm_worker":
                # The 'all_kwargs' list is passed as the first positional argument.
                result = self.init_vllm_worker(args[0])
                # logger.info(f"[{self.rank}] LEAVING: execute_method (method={method})")
                return result
            raise RuntimeError("vLLM worker engine not initialized.")

        # Special handling for 'initialize_cache' which has a peculiar
        # argument structure passed down from the vLLM engine.
        if method == "initialize_cache":
            # The 'args' tuple is expected to be: (None, (num_gpu_blocks, num_cpu_blocks), None)
            # We must unpack it to pass the correct arguments to the worker's method.
            num_gpu_blocks = args[1][0]
            num_cpu_blocks = args[1][1]
            
            # Note: We call self.vllm_worker_engine.worker.initialize_cache directly
            # to bypass the execute_method wrapper and call the actual implementation.
            result = self.vllm_worker_engine.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)
            # logger.debug(f"[{self.rank}] LEAVING: execute_method (method={method})")
            return result
            
        # For all other methods, delegate to the vllm_worker_engine's execute_method
        # which correctly handles dispatching.
        result = self.vllm_worker_engine.execute_method(method, *args, **kwargs)

        return result
    
    def set_trainer_weights_info(self, weights_info: List[tuple]):
        """Stores the weight metadata received from the Trainer."""
        # logger.info(f"[{self.rank}] Storing weights info for vLLM worker.")
        self.weights_info = weights_info

    async def sync_weights(self, group_name: str, source_rank_in_collective_group: int):
        """Receives weights and loads them into the vLLM engine using ray.util.collective."""
        # logger.info(f"[{self.rank}] ENTERING: sync_weights (vLLM destination)")
        
        if not hasattr(self, 'weights_info') or not self.weights_info:
            raise RuntimeError("weights_info not set on RolloutGPUWorker before syncing weights.")
        
        if not self.vllm_worker_engine:
            logger.error(f"[{self.rank}] vLLM worker engine not initialized. Cannot sync weights.")
            return

        loaded_count = 0
        
        inference_model = self.vllm_worker_engine.model_runner.model

        # All workers participate in all broadcasts. The key is a standard HF key.
        for key, shape, dtype in self.weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device())
            collective.broadcast(tensor, src_rank=source_rank_in_collective_group, group_name=group_name)
            
            # The vLLM model's load_weights method internally handles the key conversion.
            try:
                inference_model.load_weights([(key, tensor)])
                loaded_count += 1
            except Exception as e:
                logger.error(f"[{self.rank}] Error loading weight for key '{key}'. Error: {e}", exc_info=True)
        
        # logger.info(f"[{self.rank}] `load_weights` loop complete. Attempted to load {loaded_count}/{len(self.weights_info)} matched tensors.")
        
        # await collective.barrier(group_name=group_name)
        # logger.info(f"[{self.rank}] LEAVING: sync_weights (vLLM destination).")

    async def init_wandb_shared(self, run_id: str, project: str, entity: str, collective_rank: int):
        """Deprecated: W&B shared init is removed. Manual system metrics are used instead."""
        return