"""Evaluation script for saved training-based models

This script evaluates pre-trained models saved by training-based methods
(e.g., Chebyshev, EPO) without re-training them.

Usage:
    python -m src.benchmarks.poc.evaluate_saved model_path=path/to/model.safetensors
    python -m src.benchmarks.poc.evaluate_saved  # Evaluates all models in cache_dir
"""

import hashlib
import json
from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.loaders import load_hf_dataset, preprocess_dataset
from src.evaluation.evaluator import ClassificationEvaluator, EvaluationResult
from src.models.loaders import apply_task_vector, compute_task_vector, load_model, load_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def flatten_task_vector(task_vector_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten task vector dictionary to 1D tensor"""
    return torch.cat([v.flatten() for v in task_vector_dict.values()])


def unflatten_task_vector(
    flat_vector: torch.Tensor, task_vector_template: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Unflatten 1D tensor back to task vector dictionary"""
    unflattened = {}
    offset = 0
    for name, template in task_vector_template.items():
        numel = template.numel()
        shape = template.shape
        unflattened[name] = flat_vector[offset : offset + numel].reshape(shape)
        offset += numel
    return unflattened


def load_saved_model_state(
    model_path: Path,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Load saved model state dict from safetensors or torch format

    Args:
        model_path: Path to saved model
        device: Device to load onto

    Returns:
        State dict
    """
    logger.info(f"Loading saved model from {model_path}...")

    # Try loading with safetensors first
    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(model_path), device=str(device))
        logger.info("  ✓ Loaded model (safetensors format)")
        return state_dict
    except (ImportError, Exception) as e:
        logger.debug(f"  Safetensors loading failed: {e}, trying torch.load")

    # Fallback to torch.load
    state_dict = torch.load(model_path, map_location=device)
    logger.info("  ✓ Loaded model (torch format)")
    return state_dict


def evaluate_saved_model(
    cfg: DictConfig,
    model_path: Path,
    device: torch.device,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate a single saved model

    Args:
        cfg: Hydra configuration
        model_path: Path to saved model
        device: Device to run on

    Returns:
        Dictionary of evaluation results per task
    """
    logger.info("=" * 80)
    logger.info(f"Evaluating saved model: {model_path.name}")
    logger.info("=" * 80)

    # Load tokenizer
    tokenizer = load_tokenizer(
        model_id=cfg.model.hf_model_id,
        cache_dir=Path(cfg.paths.hf_models_cache_base) if cfg.paths.hf_models_cache_base else None,
    )

    # Create base model factory
    def create_base_model(num_labels: int):
        return load_model(
            model_id=cfg.model.hf_model_id,
            num_labels=num_labels,
            cache_dir=Path(cfg.paths.hf_models_cache_base)
            if cfg.paths.hf_models_cache_base
            else None,
            device=device,
            torch_dtype=cfg.model.loading.torch_dtype,
        )

    # Load dataset configs
    dataset_configs = {}
    for task_name in cfg.benchmark.tasks:
        dataset_configs[task_name] = cfg.datasets[task_name]

    # Load base model and compute task vector from saved model
    logger.info("\nComputing task vector from saved model...")
    max_labels = max(ds.num_labels for ds in dataset_configs.values())
    base_model = create_base_model(max_labels)
    base_state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # Load trained model state
    trained_state_dict = load_saved_model_state(model_path, device)

    # Compute task vector
    task_vector_dict = {
        name: param.cpu() - base_state_dict[name]
        for name, param in trained_state_dict.items()
        if name in base_state_dict
    }

    # Create task vector template for unflattening
    task_vector_template = task_vector_dict

    # Flatten for consistency (even though we'll unflatten immediately)
    merged_flat = flatten_task_vector(task_vector_dict)
    merged_task_vector = unflatten_task_vector(merged_flat, task_vector_template)

    logger.info(f"  Task vector size: {merged_flat.shape[0]:,} parameters")

    # Clean up
    del base_model, trained_state_dict, task_vector_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate on each task
    task_results = {}

    for task_name in cfg.benchmark.tasks:
        logger.info(f"\nEvaluating on task: {task_name}")

        dataset_cfg = dataset_configs[task_name]

        # Create base model with correct num_labels for this task
        merged_model = create_base_model(dataset_cfg.num_labels)

        # Apply merged task vector
        merged_model = apply_task_vector(merged_model, merged_task_vector, scaling=1.0)
        merged_model.eval()

        # Load and preprocess test dataset
        test_dataset = load_hf_dataset(
            dataset_path=dataset_cfg.hf_dataset.path,
            subset=dataset_cfg.hf_dataset.get("subset", None),
            split=dataset_cfg.hf_dataset.split.test,
            cache_dir=Path(cfg.paths.hf_datasets_cache)
            if cfg.paths.hf_datasets_cache
            else None,
        )

        test_dataset_processed = preprocess_dataset(
            dataset=test_dataset,
            tokenizer=tokenizer,
            text_column=dataset_cfg.preprocessing.text_column,
            text_column_2=dataset_cfg.preprocessing.get("text_column_2", None),
            label_column=dataset_cfg.preprocessing.label_column,
            max_length=dataset_cfg.preprocessing.max_length,
            truncation=dataset_cfg.preprocessing.truncation,
            padding=dataset_cfg.preprocessing.padding,
        )

        # Limit samples if specified
        if cfg.benchmark.evaluation.num_samples:
            test_dataset_processed = test_dataset_processed.select(
                range(min(cfg.benchmark.evaluation.num_samples, len(test_dataset_processed)))
            )

        # Create dataloader
        test_dataloader = DataLoader(
            test_dataset_processed,
            batch_size=cfg.benchmark.evaluation.batch_size,
            shuffle=False,
        )

        # Evaluate
        evaluator = ClassificationEvaluator(
            model=merged_model,
            tokenizer=tokenizer,
            device=str(device),
            batch_size=cfg.benchmark.evaluation.batch_size,
            use_torch_compile=cfg.model.optimization.get("use_torch_compile", True),
            torch_compile_mode=cfg.model.optimization.get("torch_compile_mode_eval", "default"),
        )

        result = evaluator.evaluate(
            dataloader=test_dataloader,
            task_name=task_name,
            metrics=cfg.benchmark.evaluation.metrics,
        )

        task_results[task_name] = result

        # Clean up for next task
        del merged_model, test_dataset, test_dataset_processed, test_dataloader, evaluator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return task_results


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point"""

    # Setup device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    logger.info(f"Using device: {device}")

    # Check if specific model path provided
    if hasattr(cfg, 'model_path') and cfg.model_path:
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return

        # Evaluate single model
        results = evaluate_saved_model(cfg, model_path, device)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Results")
        logger.info("=" * 80)
        for task_name, result in results.items():
            logger.info(f"\n{task_name}:")
            for metric, value in result.metrics.items():
                logger.info(f"  {metric}: {value:.4f}")

    else:
        # Evaluate all models in cache directory
        cache_dir = Path(cfg.benchmark.training.cache_dir) if hasattr(cfg.benchmark, 'training') else None

        if not cache_dir or not cache_dir.exists():
            logger.error(f"Cache directory not found or not configured: {cache_dir}")
            logger.info("Please specify a model path using: model_path=path/to/model.safetensors")
            return

        # Find all saved models
        model_files = list(cache_dir.glob("*.safetensors")) + list(cache_dir.glob("*.pt")) + list(cache_dir.glob("*.pth"))

        if not model_files:
            logger.error(f"No saved models found in: {cache_dir}")
            return

        logger.info(f"Found {len(model_files)} saved models in {cache_dir}")

        # Evaluate each model
        all_results = {}
        for model_path in model_files:
            try:
                results = evaluate_saved_model(cfg, model_path, device)
                all_results[model_path.name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_path.name}: {e}")
                continue

        # Print summary for all models
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Summary (All Models)")
        logger.info("=" * 80)

        for model_name, results in all_results.items():
            logger.info(f"\n{model_name}:")
            for task_name, result in results.items():
                logger.info(f"  {task_name}:")
                for metric, value in result.metrics.items():
                    logger.info(f"    {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
