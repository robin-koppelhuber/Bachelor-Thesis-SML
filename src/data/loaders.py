"""Dataset loading utilities"""

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_hf_dataset(
    dataset_path: str,
    subset: Optional[str] = None,
    split: str = "train",
    cache_dir: Optional[Path] = None,
) -> Dataset:
    """
    Load dataset from HuggingFace

    Args:
        dataset_path: HuggingFace dataset path
        subset: Optional dataset subset (e.g., 'mnli' for GLUE)
        split: Dataset split to load
        cache_dir: Optional cache directory

    Returns:
        Loaded dataset
    """
    logger.info(f"Loading dataset: {dataset_path} (subset={subset}, split={split})")

    if subset:
        dataset = load_dataset(
            dataset_path,
            subset,
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    else:
        dataset = load_dataset(
            dataset_path,
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    logger.info(f"  Loaded {len(dataset)} samples")
    return dataset


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    text_column_2: Optional[str] = None,
    label_column: str = "label",
    max_length: int = 512,
    truncation: bool = True,
    padding: str = "max_length",
) -> Dataset:
    """
    Preprocess dataset for classification

    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        text_column: Name of text column
        text_column_2: Optional second text column (for sentence pairs)
        label_column: Name of label column
        max_length: Maximum sequence length
        truncation: Whether to truncate
        padding: Padding strategy

    Returns:
        Preprocessed dataset
    """
    logger.info("Preprocessing dataset...")

    def preprocess_function(examples):
        # Handle single or paired text
        if text_column_2:
            texts = (examples[text_column], examples[text_column_2])
        else:
            texts = examples[text_column]

        # Tokenize
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )

        # Add labels
        tokenized["labels"] = examples[label_column]

        return tokenized

    # Apply preprocessing
    processed = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    logger.info(f"  Preprocessed {len(processed)} samples")
    return processed
