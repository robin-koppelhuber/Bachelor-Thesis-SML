"""Compute reference points (utopia, single-task optima) for multi-objective optimization"""

import logging
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.loaders import load_hf_dataset, preprocess_dataset
from src.evaluation.evaluator import ClassificationEvaluator
from src.models.loaders import load_model, load_tokenizer
from src.utils.cache import get_memory

logger = logging.getLogger(__name__)


def compute_reference_points(
    cfg: DictConfig,
    task_names: List[str],
    metrics: List[str],
    device: torch.device,
    cache_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Compute reference points from single-task fine-tuned models

    This computes the performance of each fine-tuned model on ALL tasks,
    giving us:
    - Single-task optima: Best performance when optimized for one task
    - Utopia point: Ideal performance (max across all tasks)

    Args:
        cfg: Hydra configuration
        task_names: List of task names
        metrics: List of metrics to compute
        device: Device to run on
        cache_dir: Directory to cache results (unused, for API compatibility)

    Returns:
        Dictionary mapping task names to their metrics on all tasks
        Example: {
            "ag_news": {"ag_news_accuracy": 0.95, "imdb_accuracy": 0.70, ...},
            "imdb": {"ag_news_accuracy": 0.65, "imdb_accuracy": 0.92, ...},
            ...
        }
    """
    logger.info("Computing reference points from single-task fine-tuned models...")

    # Create cache key from essential parameters only
    task_names_tuple = tuple(task_names)
    metrics_tuple = tuple(metrics)
    model_id = cfg.model.hf_model_id
    num_samples = cfg.benchmark.evaluation.num_samples

    # Extract only the critical config parameters needed for computation
    # This ensures cache doesn't invalidate on irrelevant config changes
    dataset_configs = {task: cfg.datasets[task] for task in task_names}
    paths = {
        'hf_models_cache_base': str(cfg.paths.hf_models_cache_base) if cfg.paths.hf_models_cache_base else None,
        'hf_models_cache_finetuned': str(cfg.paths.hf_models_cache_finetuned) if cfg.paths.hf_models_cache_finetuned else None,
        'hf_datasets_cache': str(cfg.paths.hf_datasets_cache) if cfg.paths.hf_datasets_cache else None,
    }
    torch_dtype = cfg.model.loading.torch_dtype
    batch_size = cfg.benchmark.evaluation.batch_size
    use_torch_compile = cfg.model.optimization.get("use_torch_compile", True)
    torch_compile_mode_eval = cfg.model.optimization.get("torch_compile_mode_eval", "default")

    # Get cached reference points (Joblib caching via decorator)
    return _compute_reference_points_cached(
        task_names_tuple=task_names_tuple,
        metrics_tuple=metrics_tuple,
        model_id=model_id,
        num_samples=num_samples,
        dataset_configs=dataset_configs,
        paths=paths,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        use_torch_compile=use_torch_compile,
        torch_compile_mode_eval=torch_compile_mode_eval,
        device=device,
    )


@get_memory().cache
def _compute_reference_points_cached(
    task_names_tuple: tuple,
    metrics_tuple: tuple,
    model_id: str,
    num_samples: int,
    dataset_configs: dict,
    paths: dict,
    torch_dtype: str,
    batch_size: int,
    use_torch_compile: bool,
    torch_compile_mode_eval: str,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Cached implementation of reference points computation

    Joblib automatically caches based on all arguments.
    Cache invalidates ONLY when critical parameters change:
    - Task names change
    - Metrics change
    - Model ID changes
    - Number of samples changes
    - Dataset configurations change (preprocessing, checkpoints, etc.)
    - Cache paths change
    - torch_dtype changes
    - batch_size changes
    - Torch compilation settings change
    - Function code changes

    Cache does NOT invalidate when:
    - Logging settings change
    - W&B settings change
    - Random seed changes
    - Other irrelevant config changes
    """
    task_names = list(task_names_tuple)
    metrics = list(metrics_tuple)

    logger.info("Cache miss - computing reference points from scratch")

    # Load tokenizer (shared across all models)
    tokenizer = load_tokenizer(
        model_id=model_id,
        cache_dir=Path(paths['hf_models_cache_base']) if paths['hf_models_cache_base'] else None,
    )

    reference_points = {}

    # For each fine-tuned model (one per task)
    for source_task in task_names:
        logger.info(f"\nEvaluating fine-tuned model for {source_task} on all tasks...")
        dataset_cfg = dataset_configs[source_task]

        # Load the fine-tuned model for this task
        finetuned_model = load_model(
            model_id=dataset_cfg.finetuned_checkpoint,
            num_labels=None,  # Don't override for fine-tuned models
            cache_dir=Path(paths['hf_models_cache_finetuned'])
            if paths['hf_models_cache_finetuned']
            else None,
            device=device,
            torch_dtype=torch_dtype,
        )
        finetuned_model.eval()

        task_metrics = {}

        # Evaluate on all tasks
        for eval_task in task_names:
            logger.info(f"  Evaluating on {eval_task}...")
            eval_dataset_cfg = dataset_configs[eval_task]

            # Load test dataset
            test_dataset = load_hf_dataset(
                dataset_path=eval_dataset_cfg.hf_dataset.path,
                subset=eval_dataset_cfg.hf_dataset.get("subset", None),
                split=eval_dataset_cfg.hf_dataset.split.test,
                cache_dir=Path(paths['hf_datasets_cache']) if paths['hf_datasets_cache'] else None,
            )

            test_dataset_processed = preprocess_dataset(
                dataset=test_dataset,
                tokenizer=tokenizer,
                text_column=eval_dataset_cfg.preprocessing.text_column,
                text_column_2=eval_dataset_cfg.preprocessing.get("text_column_2", None),
                label_column=eval_dataset_cfg.preprocessing.label_column,
                label_map=eval_dataset_cfg.preprocessing.get("label_map", None),
                max_length=eval_dataset_cfg.preprocessing.max_length,
                truncation=eval_dataset_cfg.preprocessing.truncation,
                padding=eval_dataset_cfg.preprocessing.padding,
            )

            # Limit samples if specified
            if num_samples:
                test_dataset_processed = test_dataset_processed.select(
                    range(min(num_samples, len(test_dataset_processed)))
                )

            # Create dataloader
            test_dataloader = DataLoader(
                test_dataset_processed,
                batch_size=batch_size,
                shuffle=False,
            )

            # Evaluate
            evaluator = ClassificationEvaluator(
                model=finetuned_model,
                tokenizer=tokenizer,
                device=str(device),
                batch_size=batch_size,
                use_torch_compile=use_torch_compile,
                torch_compile_mode=torch_compile_mode_eval,
            )

            result = evaluator.evaluate(
                dataloader=test_dataloader,
                task_name=eval_task,
                metrics=metrics,
            )

            # Store metrics with task prefix
            for metric_name, metric_value in result.metrics.items():
                task_metrics[f"{eval_task}_{metric_name}"] = metric_value

            logger.info(f"    {eval_task}: {result.metrics}")

            # Clean up
            del test_dataset, test_dataset_processed, test_dataloader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        reference_points[source_task] = task_metrics

        # Clean up model
        del finetuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info("✓ Computed reference points (Joblib will cache automatically)")
    return reference_points


def get_utopia_point(
    reference_points: Dict[str, Dict[str, float]],
    task_names: List[str],
    metric_name: str,
) -> Dict[str, float]:
    """
    Extract utopia point from reference points

    The utopia point is the ideal (but usually unachievable) point that combines
    the best performance on each task.

    Args:
        reference_points: Output from compute_reference_points()
        task_names: List of task names
        metric_name: Metric to extract (e.g., "accuracy", "f1_macro")

    Returns:
        Dictionary mapping task names to their best possible performance
        Example: {"ag_news": 0.95, "imdb": 0.92, "mnli": 0.88, "mrpc": 0.91}
    """
    utopia = {}

    for task in task_names:
        # Find the maximum performance on this task across all fine-tuned models
        key = f"{task}_{metric_name}"
        max_performance = max(
            ref_point[key] for ref_point in reference_points.values()
        )
        utopia[task] = max_performance

    return utopia


def get_single_task_optima(
    reference_points: Dict[str, Dict[str, float]],
    task_names: List[str],
    metric_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Extract single-task optimal points from reference points

    For each task, returns the performance when a model is optimized only for that task.

    Args:
        reference_points: Output from compute_reference_points()
        task_names: List of task names
        metric_name: Metric to extract

    Returns:
        Dictionary mapping task names to their performance on all tasks
        Example: {
            "ag_news": {"ag_news": 0.95, "imdb": 0.70, "mnli": 0.65, "mrpc": 0.68},
            "imdb": {"ag_news": 0.65, "imdb": 0.92, "mnli": 0.62, "mrpc": 0.70},
            ...
        }
    """
    single_task_optima = {}

    for source_task in task_names:
        optima = {}
        for eval_task in task_names:
            key = f"{eval_task}_{metric_name}"
            optima[eval_task] = reference_points[source_task][key]
        single_task_optima[source_task] = optima

    return single_task_optima
