"""
Reward function adapters registry.

Defines adapters mapping simple string keys to functions that compute
a {"reward": float, "reward_info": {...}} dict given a sample.
"""

from typing import Dict, Callable, Any
from enum import Enum
import re
import logging

from deepscaler_math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from countdown_reward import format_reward_function, answer_reward_function


def _extract_reference_and_answer(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 'parsed_gt_answer' and 'parsed_attempt' fields into the sample dict
    by splitting sample['sample_text'] on sample['input'].
    """
    original_input = sample['input']
    output = sample['sample_text'].split(original_input)[1]
    # Ground truth answer
    if "\\boxed" in sample.get('answer', ''):
        parsed_gt = extract_answer(sample['answer'])
    else:
        parsed_gt = sample.get('answer')
    # Model attempt
    try:
        parsed_attempt = extract_answer(output)
    except Exception:
        parsed_attempt = ''
    # Annotate sample
    sample['parsed_gt_answer'] = parsed_gt
    sample['parsed_attempt'] = parsed_attempt or ''
    return sample


def mathd_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using mathd (string match) from deepscaler_math_utils.
    Returns a dict with 'reward' (1.0 or 0.0) and 'reward_success'.
    """
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def sympy_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using sympy-based checker from deepscaler_math_utils.
    Returns a dict with 'reward' and 'reward_success'.
    """
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def countdown_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adapter for the Countdown Tasks reward.
    """
    RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
    # Isolate model's generated text by splitting off the prompt
    full_text = sample.get('sample_text', '')
    output = full_text.split(sample.get('input', ''), 1)[1]
    response = "<think>" + output
    # Prepend the RESPONSE_PROMPT to reconstruct the opening <think> tag
    format_r = format_reward_function(response, end_token=sample.get('end_token', ''))
    answer_r = answer_reward_function(response, numbers=sample.get('nums'), target=sample.get('target'))
    reward = format_r * 0.1 + answer_r
    return {"reward": reward, "format_reward": format_r}


# ----------------------------------- CUSTOM IMPLEMENTATION -----------------------------------
def _base_reward_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    A basic reward adapter that checks for the presence of 'reward' in the sample.
    This can be used for datasets that are pre-computed with rewards.
    """
    if "reward" not in sample:
        raise ValueError("Sample does not contain 'reward' key.")
    return {"reward": float(sample["reward"])}


def countdown_reward_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adapter for the Countdown Tasks reward.
    """
    RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
    # Isolate model's generated text by splitting off the prompt
    full_text = sample.get('sample_text', '')
    output = full_text.split(sample.get('input', ''), 1)[1]
    response = "<think>" + output
    # Prepend the RESPONSE_PROMPT to reconstruct the opening <think> tag
    format_r = format_reward_function(response, end_token=sample.get('end_token', ''))
    answer_r = answer_reward_function(response, numbers=sample.get('nums'), target=sample.get('target'))
    reward = format_r * 0.1 + answer_r
    return {"reward": reward, "format_reward": format_r}

# ----------------------------------- END OF CUSTOM IMPLEMENTATION -----------------------------------
class RewardType(str, Enum):
    """
    An enumeration of the available reward adapters.
    This allows for pluggable reward functions in the GRPO pipeline.
    """

    MATHD = "mathd"
    SYMPY = "sympy"
    COUNTDOWN = "countdown"
    BASE = "base" # for datasets that are pre-computed with rewards.
    GSM8K = "gsm8k" # GSM8K math word problems


def _extract_final_answer_from_text(text: str) -> str:
    """Strictly extract value inside <final_answer>...</final_answer>.

    This mirrors the behavior used in the async-grpo and VERL implementations:
    - Only the tag contents are considered
    - Case-insensitive match on the tag name
    - No additional fallbacks (e.g., last number, #### pattern)
    """
    m = re.search(r"<final_answer>\s*([^<]+?)\s*</final_answer>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _normalize_numeric_string(s: str) -> str:
    """Normalize numeric strings by removing commas and trimming spaces.

    Note: We intentionally do NOT strip trailing periods or apply any other
    heuristics so that the behavior is strictly aligned with VERL and
    async-grpo implementations.
    """
    return s.replace(",", "").strip()


def gsm8k_reward_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Strict GSM8K reward: only accept <final_answer>...</final_answer> content.

    Behavior matches async-grpo and VERL final-answer reward semantics.
    """
    full_text = sample.get("sample_text", "")
    prompt = sample.get("input", "")
    # Extract only the model completion (post-prompt) to mirror async-grpo
    completion = full_text[len(prompt):] if prompt and full_text.startswith(prompt) else full_text

    pred = _extract_final_answer_from_text(completion)
    gt = str(sample.get("answer", "")).strip()

    correct = _normalize_numeric_string(pred) == _normalize_numeric_string(gt)
    return {"reward": float(bool(correct))}


REWARD_ADAPTERS: Dict[RewardType, Callable[..., Dict[str, Any]]] = {
    RewardType.MATHD: mathd_adapter,
    RewardType.SYMPY: sympy_adapter,
    RewardType.COUNTDOWN: countdown_adapter,
    RewardType.BASE: _base_reward_adapter, # for datasets that are pre-computed with rewards.
    RewardType.GSM8K: gsm8k_reward_adapter, # GSM8K math word problems
}


def get_reward_adapter(name: RewardType) -> Callable[..., Dict[str, Any]]:
    """
    Look up and return the reward adapter function by RewardType.
    Raises ValueError if not found.
    """
    try:
        return REWARD_ADAPTERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward adapter: {name}") from e 