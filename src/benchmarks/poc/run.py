"""Proof of Concept benchmark runner

This module implements the POC benchmark for model merging with roberta-base.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.loaders import load_hf_dataset, preprocess_dataset
from src.evaluation.evaluator import ClassificationEvaluator, EvaluationResult
from src.methods.registry import MethodRegistry
from src.models.loaders import apply_task_vector, compute_task_vector, load_model, load_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def flatten_task_vector(task_vector_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten task vector dictionary to 1D tensor

    Args:
        task_vector_dict: Dictionary of parameter name -> tensor

    Returns:
        Flattened 1D tensor
    """
    return torch.cat([v.flatten() for v in task_vector_dict.values()])


def unflatten_task_vector(
    flat_vector: torch.Tensor, task_vector_template: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Unflatten 1D tensor back to task vector dictionary

    Args:
        flat_vector: Flattened 1D tensor
        task_vector_template: Template dictionary with parameter shapes

    Returns:
        Dictionary of parameter name -> tensor
    """
    unflattened = {}
    offset = 0

    for name, template in task_vector_template.items():
        numel = template.numel()
        shape = template.shape
        unflattened[name] = flat_vector[offset : offset + numel].reshape(shape)
        offset += numel

    return unflattened


def run_poc_benchmark(cfg: DictConfig, device: torch.device) -> Dict:
    """
    Run proof of concept benchmark

    Args:
        cfg: Hydra configuration
        device: Device to run on

    Returns:
        Dictionary of results
    """
    logger.info("=" * 80)
    logger.info("Starting POC Benchmark")
    logger.info("=" * 80)
    logger.info(f"Tasks: {cfg.benchmark.tasks}")
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Device: {device}")

    # Step 1: Load base model and tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading base model")
    logger.info("=" * 80)

    tokenizer = load_tokenizer(
        model_id=cfg.model.hf_model_id,
        cache_dir=Path(cfg.paths.hf_models_cache_base) if cfg.paths.hf_models_cache_base else None,
    )

    # We'll need to reload base model multiple times, so create a factory function
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

    # Step 2: Load fine-tuned models and compute task vectors
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Computing task vectors")
    logger.info("=" * 80)

    task_vectors_dict = {}
    dataset_configs = {}

    for task_name in cfg.benchmark.tasks:
        logger.info(f"\nProcessing task: {task_name}")

        # Get dataset config
        dataset_cfg = cfg.datasets[task_name]
        dataset_configs[task_name] = dataset_cfg

        # Load base model for this task
        base_model = create_base_model(num_labels=dataset_cfg.num_labels)

        # Load fine-tuned model
        logger.info(f"  Loading fine-tuned model: {dataset_cfg.finetuned_checkpoint}")
        finetuned_model = load_model(
            model_id=dataset_cfg.finetuned_checkpoint,
            num_labels=dataset_cfg.num_labels,
            cache_dir=Path(cfg.paths.hf_models_cache_finetuned)
            if cfg.paths.hf_models_cache_finetuned
            else None,
            device=device,
            torch_dtype=cfg.model.loading.torch_dtype,
        )

        # Compute task vector
        task_vector = compute_task_vector(finetuned_model, base_model)

        # Store flattened version for merging
        task_vectors_dict[task_name] = flatten_task_vector(task_vector)

        logger.info(f"  Task vector shape: {task_vectors_dict[task_name].shape}")

        # Clean up
        del base_model, finetuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Get a template for unflattening later (use first task's structure)
    first_task = cfg.benchmark.tasks[0]
    base_model_template = create_base_model(num_labels=dataset_configs[first_task].num_labels)
    finetuned_model_template = load_model(
        model_id=dataset_configs[first_task].finetuned_checkpoint,
        num_labels=dataset_configs[first_task].num_labels,
        cache_dir=Path(cfg.paths.hf_models_cache_finetuned) if cfg.paths.hf_models_cache_finetuned else None,
        device=device,
        torch_dtype=cfg.model.loading.torch_dtype,
    )
    task_vector_template = compute_task_vector(finetuned_model_template, base_model_template)
    del base_model_template, finetuned_model_template
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 3: Initialize merging method
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Initializing merging method")
    logger.info("=" * 80)

    method = MethodRegistry.create(cfg.method.name, **cfg.method.params)
    logger.info(f"Method: {method}")

    # Step 4: Merge task vectors for each preference vector and evaluate
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Merging and evaluation")
    logger.info("=" * 80)

    all_results = []

    for pref_idx, preference_vector in enumerate(cfg.benchmark.preference_vectors):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Preference vector {pref_idx + 1}: {preference_vector}")
        logger.info(f"{'=' * 80}")

        preference_array = np.array(preference_vector, dtype=np.float32)

        # Merge task vectors
        logger.info("Merging task vectors...")
        merged_flat = method.merge(
            task_vectors=task_vectors_dict, preference_vector=preference_array
        )

        # Unflatten merged vector
        merged_task_vector = unflatten_task_vector(merged_flat, task_vector_template)

        # For each task, evaluate the merged model
        task_results = {}

        for task_name in cfg.benchmark.tasks:
            logger.info(f"\nEvaluating on task: {task_name}")

            dataset_cfg = dataset_configs[task_name]

            # Create base model
            merged_model = create_base_model(num_labels=dataset_cfg.num_labels)

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
            )

            result = evaluator.evaluate(
                dataloader=test_dataloader,
                task_name=task_name,
                metrics=cfg.benchmark.evaluation.metrics,
            )

            task_results[task_name] = result

            # Clean up
            del merged_model, test_dataset, test_dataset_processed, test_dataloader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Store results for this preference vector
        all_results.append(
            {
                "preference_vector": preference_vector,
                "task_results": task_results,
            }
        )

        # Log summary
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Results for preference vector {preference_vector}:")
        for task_name, result in task_results.items():
            logger.info(f"  {task_name}: {result.metrics}")

    # Step 5: Summarize results
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Complete!")
    logger.info("=" * 80)

    results = {
        "benchmark_name": cfg.benchmark.name,
        "method": cfg.method.name,
        "tasks": cfg.benchmark.tasks,
        "all_results": all_results,
        "status": "completed",
    }

    return results
