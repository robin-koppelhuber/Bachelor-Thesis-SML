"""Compute reference losses (utopia + nadir) for the Chebyshev training objective.

The utopia loss is the average cross-entropy loss of the task-specific fine-tuned model
on the reference split (diagonal of the cross-evaluation matrix):
    z_star_t = R^t(theta^t)

Two nadir methods are supported (controlled by cfg.benchmark.evaluation.nadir_method):

  "cross_eval" (default, recommended):
    The nadir is the worst-case loss on each task among all fine-tuned expert models:
        z_nad_t = max_s R^t(theta^s)
    This is the theoretically correct nadir for the normalised Chebyshev/EPO objective,
    as it bounds the losses of all expert models from above.

  "random_baseline" (legacy):
    The nadir is the loss of the base model with a randomly initialised classification
    head on the same reference split. Kept for reproducibility of old runs.

Both are evaluated on the first half of a stratified 50/50 split of the test (or
validation) split. The second half is reserved for benchmark evaluation, ensuring that
the Chebyshev training objective is not calibrated on the same data used for final
evaluation.
"""

import gc
import logging
import math
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
    Compute utopia and nadir losses for the Chebyshev/EPO training objective.

    For each task:
      - utopia_loss: avg cross-entropy of the task-specific fine-tuned model on the
                     reference split (first half of stratified test/validation split)
      - nadir_loss:  depends on cfg.benchmark.evaluation.nadir_method:
          "cross_eval" (default): max loss on task t across all fine-tuned expert models
          "random_baseline": loss of the base model with random classification head

    The reference split is always the FIRST half of a stratified 50/50 split applied
    to the test (or validation, for GLUE tasks) split. The second half is reserved for
    final benchmark evaluation to avoid leakage.

    Args:
        cfg: Hydra configuration
        task_names: List of task names
        device: Device to evaluate on

    Returns:
        Dictionary: {"cola": {"utopia": 0.38, "nadir": 0.55}, ...}
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
    nadir_method = cfg.benchmark.evaluation.get("nadir_method", "cross_eval")

    return _compute_reference_losses_cached(
        task_names_tuple=task_names_tuple,
        model_id=model_id,
        dataset_configs=dataset_configs,
        reference_num_samples=reference_num_samples,
        split_seed=split_seed,
        paths=paths,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        nadir_method=nadir_method,
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
    nadir_method: str,  # "cross_eval" or "random_baseline"
    device_str: str,  # str instead of torch.device for stable Joblib hashing
) -> Dict[str, Dict[str, float]]:
    """Cached implementation. Cache invalidates when any argument changes."""
    device = torch.device(device_str)
    # Restore DictConfig for attribute-style access; plain dicts were only needed for hashing
    dataset_configs = {task: OmegaConf.create(cfg_dict) for task, cfg_dict in dataset_configs.items()}
    task_names = list(task_names_tuple)

    logger.info(f"Cache miss — computing reference losses from scratch (nadir_method={nadir_method!r})")

    tokenizer = load_tokenizer(
        model_id=model_id,
        cache_dir=Path(paths["hf_models_cache_base"]) if paths["hf_models_cache_base"] else None,
    )

    # --- Step 1: Build per-task reference dataloaders and compute utopia losses ---
    # Utopia = fine-tuned expert on its own task (diagonal of cross-eval matrix)
    dataloaders: Dict[str, DataLoader] = {}
    utopia_losses: Dict[str, float] = {}

    for task_name in task_names:
        task_cfg = dataset_configs[task_name]
        logger.info(f"\nComputing utopia loss for {task_name}...")

        reference_dataset = _load_reference_split(task_cfg, paths, reference_num_samples, split_seed)

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
        dataloaders[task_name] = dataloader

        ft_model = load_model(
            model_id=task_cfg.finetuned_checkpoint,
            num_labels=None,  # preserve fine-tuned weights
            cache_dir=Path(paths["hf_models_cache_finetuned"]) if paths["hf_models_cache_finetuned"] else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        ft_model.eval()
        utopia_losses[task_name] = _compute_average_loss(ft_model, dataloader, device)
        logger.info(f"  Utopia loss ({task_name}): {utopia_losses[task_name]:.4f}")
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        del reference_dataset, processed
        gc.collect()

    # --- Step 2: Compute nadir losses ---
    if nadir_method == "cross_eval":
        nadir_losses = _compute_cross_eval_nadir(
            task_names=task_names,
            dataset_configs=dataset_configs,
            dataloaders=dataloaders,
            utopia_losses=utopia_losses,
            paths=paths,
            torch_dtype=torch_dtype,
            device=device,
        )
    elif nadir_method == "random_baseline":
        nadir_losses = _compute_random_baseline_nadir(
            task_names=task_names,
            dataset_configs=dataset_configs,
            dataloaders=dataloaders,
            model_id=model_id,
            paths=paths,
            torch_dtype=torch_dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unknown nadir_method={nadir_method!r}. Choose 'cross_eval' or 'random_baseline'.")

    # --- Step 3: Assemble and log results ---
    reference_losses = {}
    task_width = max(len(t) for t in task_names)
    logger.info("\n" + "=" * 60)
    logger.info(f"  Reference losses (nadir_method={nadir_method!r})")
    logger.info("=" * 60)
    logger.info(f"  {'Task':<{task_width}}  {'Utopia':>10}  {'Nadir':>10}  {'Scale':>10}")
    logger.info(f"  {'-'*task_width}  {'-'*10}  {'-'*10}  {'-'*10}")
    for task_name in task_names:
        u = utopia_losses[task_name]
        n = nadir_losses[task_name]
        scale = n - u
        if scale <= 0:
            logger.warning(
                f"  ⚠ DEGENERATE: nadir ({n:.4f}) <= utopia ({u:.4f}) for task '{task_name}'. "
                f"The normalization scale is non-positive — check the cross-eval matrix."
            )
        logger.info(f"  {task_name:<{task_width}}  {u:>10.4f}  {n:>10.4f}  {scale:>10.4f}")
        reference_losses[task_name] = {"utopia": u, "nadir": n}
    logger.info("=" * 60)

    return reference_losses


def _compute_cross_eval_nadir(
    task_names: List[str],
    dataset_configs: dict,
    dataloaders: Dict[str, DataLoader],
    utopia_losses: Dict[str, float],
    paths: dict,
    torch_dtype: str,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute nadir via cross-evaluation: for each task t, the nadir is the worst
    (highest) cross-entropy loss on task t across all fine-tuned expert models.

        z_nad_t = max_s R^t(theta^s)

    Each expert theta^s is loaded once and evaluated on all tasks, which reduces
    the number of model loads from T^2 to T (same as for the random baseline).

    Args:
        task_names: Ordered list of task names
        dataset_configs: Plain-dict dataset configs (already converted from DictConfig)
        dataloaders: Pre-built per-task reference dataloaders (keyed by task name)
        utopia_losses: Utopia (diagonal) losses already computed (for logging)
        paths: Path config dict
        torch_dtype: Model loading dtype
        device: Evaluation device

    Returns:
        Dict mapping task_name -> nadir loss (max loss across all experts)
    """
    logger.info("\nComputing cross-eval nadir: evaluating each expert on all tasks...")
    # cross_losses[source_task][eval_task] = loss
    cross_losses: Dict[str, Dict[str, float]] = {}

    for source_task in task_names:
        source_cfg = dataset_configs[source_task]
        logger.info(f"\n  Loading expert for {source_task}...")

        expert_model = load_model(
            model_id=source_cfg.finetuned_checkpoint,
            num_labels=None,  # preserve fine-tuned weights
            cache_dir=Path(paths["hf_models_cache_finetuned"]) if paths["hf_models_cache_finetuned"] else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        expert_model.eval()

        cross_losses[source_task] = {}
        for eval_task in task_names:
            eval_cfg = dataset_configs[eval_task]
            # Check num_labels compatibility
            expert_num_labels = expert_model.config.num_labels
            task_num_labels = eval_cfg["num_labels"]
            if expert_num_labels != task_num_labels:
                logger.warning(
                    f"  ⚠ Skipping expert({source_task}) on task({eval_task}): "
                    f"num_labels mismatch ({expert_num_labels} vs {task_num_labels}). "
                    f"This pair will not contribute to the nadir."
                )
                cross_losses[source_task][eval_task] = float("nan")
                continue

            loss = _compute_average_loss(expert_model, dataloaders[eval_task], device)
            cross_losses[source_task][eval_task] = loss
            logger.info(f"    expert({source_task}) → task({eval_task}): loss={loss:.4f}")

        del expert_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Log the full cross-eval loss matrix
    task_width = max(len(t) for t in task_names)
    logger.info("\n  Cross-eval loss matrix (rows=expert, cols=evaluated task):")
    header = f"  {'expert \\ task':<{task_width}}" + "".join(f"  {t:>10}" for t in task_names)
    logger.info(header)
    for source in task_names:
        row = f"  {source:<{task_width}}"
        for ev in task_names:
            v = cross_losses[source][ev]
            row += f"  {v:>10.4f}" if not (isinstance(v, float) and v != v) else f"  {'skip':>10}"
        logger.info(row)

    # Compute nadir = column-wise max (ignoring NaN)
    nadir_losses: Dict[str, float] = {}
    for eval_task in task_names:
        valid = [
            cross_losses[src][eval_task]
            for src in task_names
            if not (isinstance(cross_losses[src][eval_task], float) and math.isnan(cross_losses[src][eval_task]))
        ]
        if not valid:
            raise RuntimeError(
                f"No valid cross-eval loss for task '{eval_task}' — all expert/task pairs "
                f"were skipped due to num_labels mismatches. Cannot compute nadir."
            )
        nadir_losses[eval_task] = max(valid)

    return nadir_losses


def _compute_random_baseline_nadir(
    task_names: List[str],
    dataset_configs: dict,
    dataloaders: Dict[str, DataLoader],
    model_id: str,
    paths: dict,
    torch_dtype: str,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute nadir via random baseline: loss of the base model with a randomly
    initialised classification head on the reference split (legacy method).
    """
    logger.info("\nComputing random-baseline nadir: evaluating base model with random classification head...")
    nadir_losses: Dict[str, float] = {}

    for task_name in task_names:
        task_cfg = dataset_configs[task_name]
        base_model = load_model(
            model_id=model_id,
            num_labels=task_cfg["num_labels"],
            cache_dir=Path(paths["hf_models_cache_base"]) if paths["hf_models_cache_base"] else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        base_model.eval()
        nadir_losses[task_name] = _compute_average_loss(base_model, dataloaders[task_name], device)
        logger.info(f"  Nadir loss ({task_name}):  {nadir_losses[task_name]:.4f}")
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return nadir_losses


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
