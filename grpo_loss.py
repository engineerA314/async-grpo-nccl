from dataclasses import dataclass
from functools import partial
import os
import time
import torch
torch.set_float32_matmul_precision('high')
from typing import Callable, Optional, Tuple, Union, List
import torch.nn as nn
import torch._dynamo
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
import torch.nn.functional as F

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _fa_cross_entropy_loss
    _FLASH_ATTN_CE_AVAILABLE = True
except Exception:
    _FLASH_ATTN_CE_AVAILABLE = False

DEBUG = False
def debug_print(message):
    return

@dataclass
class GRPOOutput(ModelOutput):
    loss: torch.Tensor = None
    entropy: torch.Tensor = None
    neg_log_ratio: torch.Tensor = None
    loss_clip1: torch.Tensor = None
    loss_unclipped: torch.Tensor = None
    loss_clipped: torch.Tensor = None
    policy_logprobs: torch.Tensor = None

@dataclass
class LogProbsOutput(ModelOutput):
    loss: torch.Tensor = None
    entropy: torch.Tensor = None


def make_grpo_forward(model, temperature: float = 1.0, mode: str = 'training', use_torch_compile: bool = True, loss_chunksize: int | None = None):
    # Compile only lightweight helper(s), not the full loss graph
    global COMPUTE_ENTROPY_FN
    COMPUTE_ENTROPY_FN = entropy_from_logits
    if use_torch_compile:
        try:
            COMPUTE_ENTROPY_FN = torch.compile(entropy_from_logits, mode="reduce-overhead")
        except Exception:
            COMPUTE_ENTROPY_FN = entropy_from_logits
    def _forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        old_logprobs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        clip_low: float = 0.2,
        clip_high: float = 0.4,
        clip_ratio_c: float = 1e32,
        compute_entropy: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else model.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        if mode == 'training':
            # Use memory-efficient per-token logprob computation; avoid compiling heavy graphs
            loss_fn = grpo_loss_and_entropy_ce_from_logsoftmax
            compiled_per_token_fn = None  # intentionally not compiled
            _true_logprobs = None
            
            out = loss_fn(
                lm_head_weights=model.lm_head.weight,
                lm_head_bias=model.lm_head.bias if hasattr(model.lm_head, "bias") else None,
                hidden_states=hidden_states,
                labels=labels,
                temperature=temperature,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_low=clip_low,
                clip_high=clip_high,
                clip_ratio_c=clip_ratio_c,
                loss_chunksize=loss_chunksize,
                per_token_fn=compiled_per_token_fn if compiled_per_token_fn is not None else logprobs_from_logits,
            )
            return GRPOOutput(
                loss=out[0],
                entropy=out[1],
                neg_log_ratio=out[2],
                loss_clip1=out[3],
                loss_unclipped=out[4],
                loss_clipped=out[5],
                policy_logprobs=_true_logprobs,
            )
        elif mode == 'eval':
            # select the eval log-prob function; keep eager for stability
            loss_fn = ce_loss_and_entropy_logsoftmax
            
            out = loss_fn(
                lm_head_weights=model.lm_head.weight,
                lm_head_bias=model.lm_head.bias if hasattr(model.lm_head, "bias") else None,
                hidden_states=hidden_states,
                labels=labels,
                temperature=temperature,
            )
            return LogProbsOutput(
                loss=out[0],
                entropy=out[1],
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    # if use_torch_compile:
    #     model.model = torch.compile(model.model)
    
    model.__original_forward = model.forward
    model.forward = _forward
    return model

def logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    **unused_kwargs,
) -> torch.Tensor:
    """
    Memory-efficient per-token log-prob computation (works with any dtype, does not create N*V log_softmax).
    Prefer using Flash-Attn CE if available; otherwise, use gather - logsumexp method.
    Returns shape: same as labels (BxL or 1xT)
    """
    if _FLASH_ATTN_CE_AVAILABLE:
        return logprobs_from_logits_flash_attn(logits, labels, ignore_index=ignore_index)

    # Safe labels (ignore_index -> 0)
    safe_labels = labels.clone()
    if ignore_index is not None:
        safe_labels[safe_labels == ignore_index] = 0

    # Gather logits at label positions and subtract logsumexp to compute log-prob
    gathered = torch.gather(logits, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits, dim=-1)
    logprobs = gathered - lse

    if ignore_index is not None:
        mask = (labels != ignore_index)
        logprobs = logprobs * mask
    return logprobs

def logprobs_from_logits_flash_attn(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    inplace_backward: bool = True,
    **unused_kwargs,
) -> torch.Tensor:
    """
    Flash-Attn cross-entropy based per-token log-prob computation.
    - logits: [..., V], labels: same shape prefix
    - Returns: log-prob of same shape as labels
    """
    if not _FLASH_ATTN_CE_AVAILABLE:
        raise RuntimeError("Flash-Attn cross_entropy_loss is not available")
    labels = labels.to(logits.device)
    safe_labels = labels.clone()
    if ignore_index is not None:
        safe_labels[safe_labels == ignore_index] = 0
    last_dim = logits.shape[-1]
    flat_logits = logits.reshape(-1, last_dim)
    flat_labels = safe_labels.reshape(-1)
    losses, _ = _fa_cross_entropy_loss(flat_logits, flat_labels, inplace_backward=inplace_backward)
    logprobs = -losses.view_as(safe_labels)
    if ignore_index is not None:
        mask = (labels != ignore_index)
        logprobs = logprobs * mask
    return logprobs

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy for each row of logits: H(p) = logsumexp - sum(p*logits)"""
    with torch.no_grad():
        pd = F.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def fused_log_probs_and_entropy_fn(
    lm_head_weights: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    per_token_fn: Callable = logprobs_from_logits,
    **unused_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic GRPO loss/entropy that uses a per-token logprobs function.
    """
    # Support both 1xT (packed) and BxL (padded) batches
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    V = lm_head_weights.size(0)
    # Align dtypes to avoid mixed-dtype mm under compile/FSDP
    logits = torch.matmul(hidden_states.to(lm_head_weights.dtype), lm_head_weights.t())
    if lm_head_bias is not None:
        logits = logits + lm_head_bias
    logits = logits / temperature
    # Per-token logprob (memory-efficient path). per_token_fn may return (logprobs, logits_flat) or just logprobs
    loss_out = per_token_fn(logits, shifted_labels, vocab_size=V)
    if isinstance(loss_out, tuple):
        loss_bxl = loss_out[0]
        logits_flat = loss_out[1]
    else:
        loss_bxl = loss_out
        B, Lc, V_local = logits.shape
        logits_flat = logits.reshape(-1, V_local)
    # Entropy is computed only for valid (response) token positions (= valid labels)
    B, Lc, V_local = logits.shape
    logits_flat_view = logits.reshape(-1, V_local)
    mask_flat = (shifted_labels.reshape(-1) != -100)
    if mask_flat.any():
        ent_valid = COMPUTE_ENTROPY_FN(logits_flat_view[mask_flat].detach().bfloat16())
        ent_all = torch.zeros(B * Lc, device=logits.device, dtype=torch.float32)
        ent_all[mask_flat] = ent_valid.to(dtype=ent_all.dtype)
    else:
        ent_all = torch.zeros(B * Lc, device=logits.device, dtype=torch.float32)
    return loss_bxl.reshape(-1), ent_all

ce_loss_and_entropy_logsoftmax = partial(
    fused_log_probs_and_entropy_fn, per_token_fn=logprobs_from_logits
)

def grpo_loss_and_entropy_ce_fn(
    lm_head_weights: torch.Tensor,
    lm_head_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    clip_low: float = 0.2,
    clip_high: float = 0.4,
    clip_ratio_c: float = 1e32,
    ce_loss_and_entropy_fn: Callable = ce_loss_and_entropy_logsoftmax,
    per_token_fn: Callable = logprobs_from_logits,
    loss_chunksize: int | None = None,
    **unused_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token negative log probabilities (GRPO loss) and entropies.
    If loss_chunksize is provided, compute along time dimension in chunks to reduce peak memory.
    """
    B, T, _ = hidden_states.shape
    V = lm_head_weights.size(0)
    shifted_labels_full = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()

    if loss_chunksize is not None and loss_chunksize > 0 and loss_chunksize < T:
        device = hidden_states.device
        # store outputs as float32 for downstream stability, but compute logits in bf16
        policy_logprobs_bt = torch.zeros((B, T), device=device, dtype=torch.float32)
        entropy_bt = torch.zeros((B, T), device=device, dtype=torch.float32)

        for start in range(0, T, loss_chunksize):
            end = min(start + loss_chunksize, T)
            hs_chunk = hidden_states[:, start:end, :]
            # Compute logits for chunk: [B, Lc, V]
            logits = torch.matmul(hs_chunk.to(lm_head_weights.dtype), lm_head_weights.t())
            if lm_head_bias is not None:
                logits = logits + lm_head_bias
            logits = logits / temperature
            # Per-token logprobs (memory-efficient) and entropy (response tokens only)
            shifted_labels_chunk = shifted_labels_full[:, start:end]
            loss_out = per_token_fn(logits, shifted_labels_chunk, vocab_size=V)
            if isinstance(loss_out, tuple):
                logprobs_bxl = loss_out[0]
                logits_flat = loss_out[1]
            else:
                logprobs_bxl = loss_out
                Bc, Lc_local, Vc = logits.shape
                logits_flat = logits.reshape(-1, Vc)
            Lc = end - start
            policy_logprobs_bt[:, start:end] = logprobs_bxl.reshape(B, Lc)

            # entropy on valid rows only
            valid_mask = (shifted_labels_chunk.reshape(-1) != -100)
            if valid_mask.any():
                ent_valid = COMPUTE_ENTROPY_FN(logits_flat[valid_mask].detach().bfloat16())
                ent_chunk = torch.zeros(B * Lc, device=device, dtype=torch.float32)
                ent_chunk[valid_mask] = ent_valid.to(dtype=ent_chunk.dtype)
                entropy_bt[:, start:end] = ent_chunk.reshape(B, Lc)

        policy_logprobs = policy_logprobs_bt.reshape(-1)
        entropy = entropy_bt.reshape(-1)
    else:
        # Non-chunked path: use fused fn with provided per_token_fn to avoid N*V full log_softmax materialization
        policy_logprobs, entropy = fused_log_probs_and_entropy_fn(
            lm_head_weights, lm_head_bias, hidden_states, labels, temperature, per_token_fn=per_token_fn
        )

    if old_logprobs is None:
        old_logprobs = policy_logprobs.detach()
    neg_log_ratio = policy_logprobs - old_logprobs
    ratio = torch.exp(neg_log_ratio)

    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * torch.clamp(ratio, lower, upper)
    loss_clip1 = torch.maximum(loss_unclipped, loss_clipped)
    
    # Dual-clip for negative advantages (matching VERL implementation)
    loss_clipped_dual = -advantages * clip_ratio_c
    loss_clip2 = torch.minimum(loss_clip1, loss_clipped_dual)
    # Apply dual clip only when advantages < 0
    is_negative = advantages < 0
    pg_loss_token = torch.where(is_negative, loss_clip2, loss_clip1)

    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    mask_flat = (shifted_labels.reshape(-1) != -100)
    # Mean over masked tokens (VERL token-mean semantics)
    pg_loss_vec = torch.where(mask_flat, pg_loss_token, torch.zeros_like(pg_loss_token))
    denom = mask_flat.sum().clamp(min=1).to(dtype=pg_loss_vec.dtype)
    pg_loss = pg_loss_vec.sum() / denom
    # Return raw log-ratio (detached) so callers can derive proper KL/clipfrac diagnostics
    return pg_loss, entropy, (neg_log_ratio.detach()), loss_clip1.detach(), loss_unclipped.detach(), loss_clipped.detach()

grpo_loss_and_entropy_ce_from_logsoftmax = partial(
    grpo_loss_and_entropy_ce_fn, ce_loss_and_entropy_fn=ce_loss_and_entropy_logsoftmax
)

# @torch.compile
def compute_dual_clip_grpo_loss(
    policy_model,
    minibatch,
    clip_low: float,
    clip_high: float,
    clip_ratio_c: float,
    compute_entropy: bool = True,
    apply_entropy_bonus: bool = True,
    entropy_coeff: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute GRPO loss using the dual-clip PPO algorithm.

    This loss applies two clipping mechanisms:
      - Clipping the probability ratio within [1 - clip_low, 1 + clip_high].
      - For negative advantages, clipping the loss to -advantages * clip_ratio_c.

    Args:
        policy_model: A causal LM policy model to evaluate.
        minibatch: A dict containing batch data keys:
            'batch_ids', 'batch_position_ids', 'labels',
            'reference_output_logprobs', 'advantages', and 'output_indices'.
        clip_low: Lower epsilon for PPO clipping (bounds ratio >= 1 - clip_low).
        clip_high: Upper epsilon for PPO clipping (bounds ratio <= 1 + clip_high).
        clip_ratio_c: Dual clipping constant c for negative advantages.

    Returns:
        loss (Tensor): Scalar policy gradient loss to backpropagate.
        metrics (dict): Dictionary of metric values:
            'loss', 'pg_loss', 'pg_clip', 'pg_clip_lower', 'kl_div', 'entropy'.
    """
    batch_ids = minibatch["batch_ids"]
    batch_position_ids = minibatch["batch_position_ids"]
    output_indices = minibatch["output_indices"]
    old_logprobs = minibatch["reference_output_logprobs"]
    advantages = minibatch["advantages"]

    labels = minibatch["labels"]

    grpo_output = policy_model(
        input_ids=batch_ids,
        attention_mask=minibatch.get("attention_mask", None),
        position_ids=batch_position_ids,
        labels=labels,
        use_cache=False,
        old_logprobs=old_logprobs,
        advantages=advantages,
        clip_low=clip_low,
        clip_high=clip_high,
        clip_ratio_c=clip_ratio_c,
        compute_entropy=compute_entropy,
    )
    
    # Track clipping and loss metrics
    # Aggregate with sums (safe when no valid indices) to avoid NaNs from empty means
    try:
        # Use per-token MEAN for logging to avoid scale explosion when later doing token-weighted averaging
        out_log_ratio = grpo_output.neg_log_ratio[output_indices]
        kl_val = out_log_ratio.abs().mean().item() if out_log_ratio.numel() > 0 else 0.0
    except Exception:
        kl_val = 0.0
    try:
        out_entropy = grpo_output.entropy[output_indices]
        ent_val = out_entropy.mean().item() if out_entropy.numel() > 0 else 0.0
    except Exception:
        ent_val = 0.0
    try:
        old_entropy = minibatch.get("reference_output_entropies", None)
        old_ent_val = (old_entropy[output_indices].mean().item()
                       if old_entropy is not None and old_entropy.numel() > 0 else 0.0)
    except Exception:
        old_ent_val = 0.0

    # Additional diagnostics on output-token positions only
    try:
        log_ratio_out = grpo_output.neg_log_ratio[output_indices].float()
        ratio_out = torch.exp(log_ratio_out)
        approx_kl = (ratio_out - 1.0 - log_ratio_out).mean().item() if log_ratio_out.numel() > 0 else 0.0
        clipfrac_upper = (ratio_out > (1.0 + clip_high)).float().mean().item() if ratio_out.numel() > 0 else 0.0
        clipfrac_lower = (ratio_out < (1.0 - clip_low)).float().mean().item() if ratio_out.numel() > 0 else 0.0
        log_ratio_mean = log_ratio_out.mean().item() if log_ratio_out.numel() > 0 else 0.0
        log_ratio_std = (log_ratio_out.std(unbiased=False).item() if log_ratio_out.numel() > 1 else 0.0)
    except Exception:
        approx_kl = 0.0
        clipfrac_upper = 0.0
        clipfrac_lower = 0.0
        log_ratio_mean = 0.0
        log_ratio_std = 0.0

    # Advantage stats on the very same output positions
    try:
        adv_out = advantages[output_indices].float()
        adv_mean = adv_out.mean().item() if adv_out.numel() > 0 else 0.0
        adv_std = (adv_out.std(unbiased=False).item() if adv_out.numel() > 1 else 0.0)
        adv_pos_frac = (adv_out > 0).float().mean().item() if adv_out.numel() > 0 else 0.0
        adv_zero_frac = (adv_out == 0).float().mean().item() if adv_out.numel() > 0 else 0.0
    except Exception:
        adv_mean = adv_std = adv_pos_frac = adv_zero_frac = 0.0

    # Mask sanity: compare labeled output token count vs output_indices length
    try:
        labeled_nonmasked = (labels != -100).sum().item()
        expected_out = int(output_indices.numel())
    except Exception:
        labeled_nonmasked = 0
        expected_out = 0

    # Loss scales for consistent comparison
    try:
        loss_sum = grpo_output.loss.detach().item()
        loss_per_token = (loss_sum / max(1, expected_out)) if expected_out > 0 else float(loss_sum)
        # Optional: per-sample average if provided
        try:
            num_samples_tensor = minibatch.get("num_samples", None)
            num_samples = int(num_samples_tensor.item()) if num_samples_tensor is not None else 0
        except Exception:
            num_samples = 0
        loss_per_sample = (loss_sum / max(1, num_samples)) if num_samples > 0 else float(loss_sum)
    except Exception:
        loss_sum = grpo_output.loss.detach().item() if hasattr(grpo_output.loss, 'detach') else float(grpo_output.loss)
        loss_per_token = float(loss_sum)
        loss_per_sample = float(loss_sum)

    metrics = {
        "loss": loss_sum,
        "loss_per_token": loss_per_token,
        "loss_per_sample": loss_per_sample,
        "pg_loss": loss_sum,  # Add pg_loss for consistency with VERL logging
        "pg_clip": (grpo_output.loss_clipped > grpo_output.loss_unclipped).detach()[output_indices].sum().item(),
        "kl_div": kl_val,
        "entropy": ent_val,
        "old_policy_entropy": old_ent_val,
        "entropy_coeff": float(entropy_coeff),
        # New diagnostics
        "approx_kl": approx_kl,
        "clipfrac_upper": clipfrac_upper,
        "clipfrac_lower": clipfrac_lower,
        "log_ratio_mean": log_ratio_mean,
        "log_ratio_std": log_ratio_std,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "adv_pos_frac": adv_pos_frac,
        "adv_zero_frac": adv_zero_frac,
    }
    
    # Apply entropy bonus to loss if enabled
    final_loss = grpo_output.loss
    try:
        if compute_entropy and apply_entropy_bonus and float(entropy_coeff) != 0.0:
            # Mean entropy over output token positions
            out_entropy = grpo_output.entropy[output_indices]
            if out_entropy.numel() > 0:
                entropy_loss = out_entropy.mean()
                final_loss = final_loss - float(entropy_coeff) * entropy_loss
    except Exception:
        # Fail-safe: keep original loss if anything goes wrong
        final_loss = grpo_output.loss

    return final_loss, metrics
    
    