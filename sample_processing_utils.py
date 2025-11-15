import random
import numpy as np
import torch
import os
import logging
from numba import njit
from transformers import AutoTokenizer
import torch.distributed as dist
# Configure numba logging
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("GRPO_LOGGING_LEVEL", "INFO"))

def _rank_str():
    """Return a robust rank string for logging.

    Prefer torch.distributed rank when initialized; otherwise fall back to
    environment RANK. LOCAL_RANK is always 0 in our per-actor-per-GPU setup and
    should not be used for identity in logs.
    """
    try:
        import torch.distributed as dist  # local import to avoid hard dep at import time
        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "?"

@njit
def get_output_logits_indices_numba(input_lens, output_lens):
    """
    Compute indices for extracting the output logits from a concatenated logits tensor.
    
    Parameters:
        input_lens (np.ndarray): 1D array of input lengths for each sample.
        output_lens (np.ndarray): 1D array of output lengths for each sample.
        
    For each sample, the logit corresponding to the first output token is at:
        global_index = cumulative_offset + (input_length - 1)
    and there are output_length logits (one per output token).
    
    Returns:
        np.ndarray: 1D array of indices that can be used to select the output logits.
    """
    n_samples = input_lens.shape[0]
    
    # First, compute total number of output indices.
    total_output = 0
    for i in range(n_samples):
        total_output += output_lens[i]
    
    out_indices = np.empty(total_output, dtype=np.int64)
    
    offset = 0    # Cumulative token offset for concatenated samples.
    pos = 0       # Position within out_indices array.
    
    for i in range(n_samples):
        L_in = input_lens[i]
        L_out = output_lens[i]
        # The first output logit's index for this sample is at offset + (L_in - 1)
        start_idx = offset + L_in - 1
        for j in range(L_out):
            out_indices[pos] = start_idx + j
            pos += 1
        # Update offset by the sum of input and output lengths for this sample.
        offset += L_in + L_out
    return out_indices

@njit
def broadcast_values(values, output_lens):
    """
    Broadcast per-sample scalar values to the token level.

    Parameters:
        values : np.ndarray (1D)
            An array of scalar values (e.g., rewards, weights, advantages) for each sample,
            with shape (n_samples,).
        output_lens : np.ndarray (1D of ints)
            An array representing the token counts (output lengths) for each sample,
            with shape (n_samples,).

    Returns:
        np.ndarray (2D)
            An array of broadcasted values, repeated to align with token positions,
            with shape (1, total_tokens), where total_tokens = sum(output_lens).
    """
    # Compute total number of tokens across all samples.
    total_tokens = 0
    num_samples = output_lens.shape[0]
    for i in range(num_samples):
        total_tokens += output_lens[i]
    
    # Initialize the broadcasted values array.
    # broadcasted = np.zeros((1, total_tokens))
    broadcasted = np.zeros(total_tokens)
    
    # Use a running offset to fill in the broadcasted values.
    offset = 0
    for i in range(num_samples):
        length_i = output_lens[i]
        for j in range(length_i):
            broadcasted[offset + j] = values[i]
        offset += length_i

    return broadcasted

def get_output_logits_indices(batched_questions, device):
    input_lens = np.array([s['input_len'] for s in batched_questions])
    output_lens = np.array([s['output_len'] for s in batched_questions])
    output_indices = get_output_logits_indices_numba(input_lens, output_lens)
    output_indices = torch.from_numpy(output_indices).to(device)
    return output_indices, output_lens

def get_input_for_logprobs(batched_questions, output_indices, device):
    batch_ids = torch.cat(
        [torch.tensor(s['sample_ids'], dtype=torch.long) for s in batched_questions]
    ).unsqueeze(0).to(device)
    batch_position_ids = torch.cat(
        [torch.tensor(s['sample_position_ids'], dtype=torch.long) for s in batched_questions]
    ).unsqueeze(0).to(device)
    # Concatenate precomputed per-sample labels
    labels = torch.cat(
        [torch.tensor(s['labels'], dtype=torch.long) for s in batched_questions]
    ).unsqueeze(0).to(device)
    return batch_ids, batch_position_ids, labels

def make_dummy_batch(batched_questions, device):
    """Return a no-op dummy batch for synchronization without affecting metrics"""
    input_ids = [11, 12, 13, 14]
    batch_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    batch_position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
    labels = torch.full((1, len(input_ids)), -100, dtype=torch.long, device=device)
    output_indices = torch.empty((0,), dtype=torch.long, device=device)
    advantages = torch.zeros(len(input_ids), device=device, dtype=torch.float32)
    reference_output_logprobs = torch.zeros(len(input_ids), device=device, dtype=torch.float32)
    output_lens_broadcasted = torch.zeros(len(input_ids), device=device, dtype=torch.float32)
    return {
        "batch_ids": batch_ids,
        "batch_position_ids": batch_position_ids,
        "output_indices": output_indices,
        "advantages": advantages,
        "reference_output_logprobs": reference_output_logprobs,
        "output_lens_broadcasted": output_lens_broadcasted,
        "num_output_tokens_non_masked": torch.tensor(0.0, device=device),
        "num_output_tokens": torch.tensor(0.0, device=device, dtype=torch.float32),
        "num_samples": torch.tensor(0.0, device=device, dtype=torch.float32),
        "max_reward_in_group": torch.tensor(0.0, device=device, dtype=torch.float32),
        "total_modified_reward": torch.tensor(0.0, device=device, dtype=torch.float32),
        "total_non_modified_reward": torch.tensor(0.0, device=device, dtype=torch.float32),
        "num_modified_samples": torch.tensor(0.0, device=device, dtype=torch.float32),
        "delimiter_not_found": torch.tensor(0.0, device=device, dtype=torch.float32),
        "samples": batched_questions,
        "labels": labels,
        "total_reward_rank": torch.tensor(0.0, device=device, dtype=torch.float32),
        "truncated_sample": torch.tensor(0.0, device=device, dtype=torch.float32),
        "advantage_is_zero": torch.tensor(0.0, device=device, dtype=torch.float32),
    }

def post_process_batch(batched_questions, device, constant_length_samples=None):
    # logger.info(f"[{_rank_str()}] post_process_batch: batched_questions shape={len(batched_questions)}")
    
    
    # Handle dummy batch: emit a minimal placeholder batch
    if len(batched_questions) == 1 and batched_questions[0].get('dummy', False):
        return make_dummy_batch(batched_questions, device)
    output_indices, output_lens = get_output_logits_indices(batched_questions, device)
    modified_samples = [q for q in batched_questions if q['modified_reward'] is not None]
    non_modified_samples = [q for q in batched_questions if q['modified_reward'] is None]

    advantages = np.array([s['advantage'] for s in batched_questions])

    sample_lens = np.array([s['input_len'] +s['output_len'] for s in batched_questions])
    advantages = torch.from_numpy(broadcast_values(advantages, sample_lens)).to(device).to(torch.float32)

    if constant_length_samples is None:
        output_lens_broadcasted = torch.from_numpy(broadcast_values(output_lens, sample_lens)).to(device).to(torch.float32)
    else:
        output_lens_broadcasted = torch.ones_like(advantages).to(device).to(torch.float32) * constant_length_samples

    reference_output_logprobs = torch.cat(
        [torch.tensor(s['sample_logprobs']) for s in batched_questions]
    ).to(device).to(torch.float32)
    # Optional: old-policy entropy if provided by logprob worker (robust to partial None)
    try:
        ent_tensors = []
        for s in batched_questions:
            ent = s.get('sample_entropies', None)
            if ent is None:
                # Fallback: zeros with same length as logprobs for this sample
                ent = [0.0] * len(s.get('sample_logprobs') or [])
            ent_tensors.append(torch.tensor(ent))
        reference_output_entropies = torch.cat(ent_tensors).to(device).to(torch.float32) if ent_tensors else torch.zeros_like(reference_output_logprobs)
    except Exception:
        reference_output_entropies = torch.zeros_like(reference_output_logprobs)
    batch_ids, batch_position_ids, labels = get_input_for_logprobs(batched_questions, output_indices, device)
    
    _out = {
        "batch_ids": batch_ids.contiguous(),
        "batch_position_ids": batch_position_ids.contiguous(),
        "output_indices": output_indices.contiguous(),
        "advantages": advantages.contiguous(),
        "reference_output_logprobs": reference_output_logprobs.contiguous(),
        "reference_output_entropies": reference_output_entropies.contiguous(),
        "output_lens_broadcasted": output_lens_broadcasted.contiguous(),
        "num_output_tokens_non_masked": (labels != -100).sum().to(torch.float32),
        "num_output_tokens": torch.tensor(output_lens.sum(), device=device, dtype=torch.float32),
        "num_samples": torch.tensor(len(batched_questions), device=device, dtype=torch.float32),
        "max_reward_in_group": torch.tensor(0.0, device=device, dtype=torch.float32),
        "total_modified_reward": torch.tensor(sum([s['modified_reward'] for s in modified_samples]), device=device, dtype=torch.float32) if len(modified_samples) > 0 else torch.tensor(0.0, device=device, dtype=torch.float32),
        "total_non_modified_reward": torch.tensor(sum([s['reward'] for s in non_modified_samples]), device=device, dtype=torch.float32) if len(non_modified_samples) > 0 else torch.tensor(0.0, device=device, dtype=torch.float32),
        "num_modified_samples": torch.tensor(len(modified_samples), device=device, dtype=torch.float32),
        "delimiter_not_found": torch.tensor(sum([s['delimiter_not_found'] for s in modified_samples]), device=device, dtype=torch.float32) if len(modified_samples) > 0 else torch.tensor(0.0, device=device, dtype=torch.float32),
        "samples": batched_questions,
        "labels": labels,
        "total_reward_rank": torch.tensor(sum([s['reward'] for s in batched_questions]), device=device, dtype=torch.float32),
        "truncated_sample": torch.tensor(sum([s['truncated_sample'] for s in batched_questions]), device=device, dtype=torch.float32),
        "advantage_is_zero": torch.tensor(0.0, device=device, dtype=torch.float32),
    }
    try:
        _out["uids"] = torch.tensor([int(s.get("__uid__", -1)) for s in batched_questions], device=device)
    except Exception:
        pass
    return _out
