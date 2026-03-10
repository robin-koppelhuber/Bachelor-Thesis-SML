"""Compute reference losses (utopia + nadir) for the Chebyshev training objective.

The utopia loss is the average cross-entropy loss of the task-specific fine-tuned model
on the reference split. The nadir loss is the same for the base model with a randomly
initialized classification head.

Both are evaluated on the first half of a stratified 50/50 split of the test (or
validation) split. The second half is reserved for benchmark evaluation, ensuring that
the Chebyshev training objective is not calibrated on the same data used for final
evaluation.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.data.loaders import load_hf_dataset, preprocess_dataset
from src.models.loaders import load_model, load_tokenizer
from src.utils.cache import get_memory

logger = logging.getLogger(__name__)


def compute_reference_losses(
    cfg: DictConfig,
    task_names: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Compute utopia and nadir losses for the Chebyshev scalarization training objective.

    For each task:
      - utopia_loss: avg cross-entropy of the task-specific fine-tuned model on the
                     reference split (first half of stratified test/validation split)
      - nadir_loss:  avg cross-entropy of the base model (random classification head)
                     on the same reference split

    The reference split is always the FIRST half of a stratified 50/50 split applied
    to the test (or validation, for GLUE tasks) split. The second half is reserved for
    final benchmark evaluation to avoid leakage.

    Args:
        cfg: Hydra configuration
        task_names: List of task names
        device: Device to evaluate on

    Returns:
        Dictionary: {"ag_news": {"utopia": 0.13, "nadir": 1.39}, ...}
    """
    task_names_tuple = tuple(task_names)
    model_id = cfg.model.hf_model_id
    # Convert DictConfig → plain dict so Joblib can hash deterministically across runs
    dataset_configs = {task: OmegaConf.to_container(cfg.datasets[task], resolve=True) for task in task_names}
    reference_num_samples = cfg.benchmark.evaluation.get("reference_num_samples", None)
    split_seed = cfg.get("seed", 42)
    paths = {
        "hf_models_cache_base": str(cfg.paths.hf_models_cache_base) if cfg.paths.hf_models_cache_base else None,
        "hf_models_cache_finetuned": str(cfg.paths.hf_models_cache_finetuned)
        if cfg.paths.hf_models_cache_finetuned
        else None,
        "hf_datasets_cache": str(cfg.paths.hf_datasets_cache) if cfg.paths.hf_datasets_cache else None,
    }
    torch_dtype = cfg.model.loading.torch_dtype
    batch_size = cfg.benchmark.evaluation.batch_size

    return _compute_reference_losses_cached(
        task_names_tuple=task_names_tuple,
        model_id=model_id,
        dataset_configs=dataset_configs,
        reference_num_samples=reference_num_samples,
        split_seed=split_seed,
        paths=paths,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        device_str=str(device),
    )


@get_memory().cache
def _compute_reference_losses_cached(
    task_names_tuple: tuple,
    model_id: str,
    dataset_configs: dict,  # plain dict (converted from DictConfig for stable hashing)
    reference_num_samples: Optional[int],
    split_seed: int,
    paths: dict,
    torch_dtype: str,
    batch_size: int,
    device_str: str,  # str instead of torch.device for stable Joblib hashing
) -> Dict[str, Dict[str, float]]:
    """Cached implementation. Cache invalidates when any argument changes."""
    device = torch.device(device_str)
    # Restore DictConfig for attribute-style access; plain dicts were only needed for hashing
    dataset_configs = {task: OmegaConf.create(cfg_dict) for task, cfg_dict in dataset_configs.items()}
    task_names = list(task_names_tuple)

    logger.info("Cache miss — computing reference losses from scratch")

    tokenizer = load_tokenizer(
        model_id=model_id,
        cache_dir=Path(paths["hf_models_cache_base"]) if paths["hf_models_cache_base"] else None,
    )

    reference_losses = {}

    for task_name in task_names:
        task_cfg = dataset_configs[task_name]
        logger.info(f"\nComputing reference losses for {task_name}...")

        # Load the reference split (first half of stratified test/validation split)
        reference_dataset = _load_reference_split(task_cfg, paths, reference_num_samples, split_seed)

        # Preprocess once for both utopia and nadir
        processed = preprocess_dataset(
            dataset=reference_dataset,
            tokenizer=tokenizer,
            text_column=task_cfg.preprocessing.text_column,
            text_column_2=task_cfg.preprocessing.get("text_column_2", None),
            label_column=task_cfg.preprocessing.label_column,
            label_map=task_cfg.preprocessing.get("label_map", None),
            max_length=task_cfg.preprocessing.max_length,
            truncation=task_cfg.preprocessing.truncation,
            padding=task_cfg.preprocessing.padding,
        )

        dataloader = DataLoader(
            processed,  # ty:ignore[invalid-argument-type]
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        # --- Utopia: fine-tuned model loss ---
        ft_model = load_model(
            model_id=task_cfg.finetuned_checkpoint,
            num_labels=None,  # preserve fine-tuned weights
            cache_dir=Path(paths["hf_models_cache_finetuned"]) if paths["hf_models_cache_finetuned"] else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        ft_model.eval()
        utopia_loss = _compute_average_loss(ft_model, dataloader, device)
        logger.info(f"  Utopia loss ({task_name}): {utopia_loss:.4f}")
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # --- Nadir: base model loss (random classification head) ---
        base_model = load_model(
            model_id=model_id,
            num_labels=task_cfg.num_labels,
            cache_dir=Path(paths["hf_models_cache_base"]) if paths["hf_models_cache_base"] else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        base_model.eval()
        nadir_loss = _compute_average_loss(base_model, dataloader, device)
        logger.info(f"  Nadir loss ({task_name}):  {nadir_loss:.4f}")
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        reference_losses[task_name] = {"utopia": utopia_loss, "nadir": nadir_loss}

        del reference_dataset, processed, dataloader
        gc.collect()

    logger.info(f"\n Reference losses computed: {reference_losses}")
    return reference_losses


def _load_reference_split(task_cfg, paths: dict, num_samples: Optional[int], split_seed: int):
    """
    Load the reference split for utopia/nadir computation.

    Always applies a stratified 50/50 split and returns the FIRST half (the second half
    is reserved for benchmark evaluation). The split is reproducible via split_seed.

    For GLUE tasks (validation == test in config): loads the validation split.
    For other tasks (ag_news, imdb): loads the test split.
    """
    val_split = task_cfg.hf_dataset.split.get("validation", None)
    test_split = task_cfg.hf_dataset.split.test

    # Prefer validation; fall back to test if no separate validation exists
    load_split = val_split if val_split is not None else test_split

    dataset = load_hf_dataset(
        dataset_path=task_cfg.hf_dataset.path,
        subset=task_cfg.hf_dataset.get("subset", None),
        split=load_split,
        cache_dir=Path(paths["hf_datasets_cache"]) if paths["hf_datasets_cache"] else None,
    )

    # Stratified 50/50 split: use first half for reference, second half for evaluation
    label_column = task_cfg.preprocessing.label_column
    split_result = dataset.train_test_split(
        test_size=0.5,
        stratify_by_column=label_column,
        seed=split_seed,
    )
    dataset = split_result["train"]  # first half

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    logger.info(
        f"  Reference split: {len(dataset)} samples "
        f"(first half of '{load_split}', stratify_by='{label_column}', seed={split_seed})"
    )
    return dataset


def _compute_average_loss(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute average cross-entropy loss over a dataloader."""
    total_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def get_evaluation_split(dataset, label_column: str, split_seed: int):
    """
    Return the SECOND half of a stratified 50/50 split for benchmark evaluation.

    Used by the benchmark runner to ensure the evaluation set is disjoint from the
    reference split used to compute the Chebyshev utopia/nadir losses.

    Args:
        dataset: HuggingFace Dataset (already loaded from the test/validation split)
        label_column: Column name to stratify by
        split_seed: Random seed (must match the seed used in _load_reference_split)

    Returns:
        The second half of the stratified split
    """
    split_result = dataset.train_test_split(
        test_size=0.5,
        stratify_by_column=label_column,
        seed=split_seed,
    )
    return split_result["test"]  # second half for evaluation
