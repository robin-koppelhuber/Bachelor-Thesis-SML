"""Model loading utilities"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_model(
    model_id: str,
    num_labels: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    torch_dtype: str = "auto",
    ignore_mismatched_sizes: bool = False,
) -> PreTrainedModel:
    """
    Load model from HuggingFace

    Args:
        model_id: HuggingFace model ID
        num_labels: Number of classification labels (optional - if None, uses checkpoint's config)
        cache_dir: Optional cache directory
        device: Optional device to load model on
        torch_dtype: PyTorch dtype ('auto', 'float32', 'float16', 'bfloat16')
        ignore_mismatched_sizes: Whether to ignore size mismatches (for fine-tuned models)

    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_id}")

    # Determine dtype
    if torch_dtype == "auto":
        dtype = None  # Let AutoModel decide
    elif torch_dtype == "float32":
        dtype = torch.float32
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = None

    # Load model - only pass num_labels if explicitly provided
    load_kwargs = {
        "cache_dir": str(cache_dir) if cache_dir else None,
        "torch_dtype": dtype,
        "ignore_mismatched_sizes": ignore_mismatched_sizes,
    }

    # Only add num_labels if provided (don't pass it for fine-tuned models)
    if num_labels is not None:
        load_kwargs["num_labels"] = num_labels

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        **load_kwargs,
    )

    if device is not None:
        model = model.to(device)

    logger.info(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_tokenizer(
    model_id: str,
    cache_dir: Optional[Path] = None,
) -> PreTrainedTokenizer:
    """
    Load tokenizer from HuggingFace

    Args:
        model_id: HuggingFace model ID
        cache_dir: Optional cache directory

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    return tokenizer


def compute_task_vector(
    finetuned_model: PreTrainedModel,
    base_model: PreTrainedModel,
) -> Dict[str, torch.Tensor]:
    """
    Compute task vector (difference between fine-tuned and base model)

    Args:
        finetuned_model: Fine-tuned model
        base_model: Base model

    Returns:
        Dictionary of task vectors (parameter differences)
    """
    logger.info("Computing task vector...")

    task_vector = {}

    finetuned_state = finetuned_model.state_dict()
    base_state = base_model.state_dict()

    for name in finetuned_state.keys():
        if name in base_state:
            task_vector[name] = finetuned_state[name] - base_state[name]

    logger.info(f"  Computed task vector with {len(task_vector)} parameters")
    return task_vector


def apply_task_vector(
    base_model: PreTrainedModel,
    task_vector: Dict[str, torch.Tensor],
    scaling: float = 1.0,
) -> PreTrainedModel:
    """
    Apply task vector to base model

    Args:
        base_model: Base model
        task_vector: Task vector to apply
        scaling: Scaling factor for task vector

    Returns:
        Model with task vector applied
    """
    logger.info(f"Applying task vector (scaling={scaling})...")

    # Get base state
    state_dict = base_model.state_dict()

    # Add scaled task vector
    for name, param in task_vector.items():
        if name in state_dict:
            # Handle shape mismatches (e.g., when task vector is from larger model)
            if state_dict[name].shape == param.shape:
                # Shapes match - apply directly
                state_dict[name] = state_dict[name] + scaling * param
            elif state_dict[name].numel() <= param.numel():
                # Task vector is larger - slice to match base model size
                # This happens when merging models with different num_labels
                flat_param = param.flatten()
                flat_base = state_dict[name].flatten()
                # Take only the first N elements from task vector
                state_dict[name] = (flat_base + scaling * flat_param[:flat_base.numel()]).reshape(state_dict[name].shape)
                logger.debug(f"  Trimmed {name} from {param.shape} to {state_dict[name].shape}")
            else:
                # Base model is larger - this shouldn't happen with our padding approach
                logger.warning(f"  Skipping {name}: base shape {state_dict[name].shape} > task vector shape {param.shape}")

    # Load modified state
    base_model.load_state_dict(state_dict)

    return base_model
