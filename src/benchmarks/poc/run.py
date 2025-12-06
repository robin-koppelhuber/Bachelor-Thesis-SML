"""Proof of Concept benchmark runner

This module implements the POC benchmark for model merging with roberta-base.
"""

from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.loaders import load_hf_dataset, preprocess_dataset
from src.evaluation.evaluator import ClassificationEvaluator, EvaluationResult
from src.methods.registry import MethodRegistry
from src.models.loaders import apply_task_vector, compute_task_vector, load_model, load_tokenizer
from src.utils.cache import setup_cache, get_memory
from src.utils.config_validator import validate_poc_benchmark_config, validate_training_config
from src.utils.logger import get_logger, setup_logging

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


def pad_task_vectors_to_match(
    task_vectors_dict: Dict[str, torch.Tensor],
    target_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Pad task vectors to match target size by appending zeros

    This preserves fine-tuned classification head weights while allowing
    models with different num_labels to be merged. Smaller task vectors
    get zero-padded, meaning they contribute nothing to the extra label weights.

    Args:
        task_vectors_dict: Dictionary mapping task names to flattened task vectors
        target_size: Target size for all task vectors

    Returns:
        Dictionary with all task vectors padded to target_size
    """
    padded_vectors = {}

    for task_name, task_vector in task_vectors_dict.items():
        current_size = task_vector.shape[0]

        if current_size < target_size:
            # Pad with zeros to reach target size
            padding_size = target_size - current_size
            padded_vector = torch.cat([
                task_vector,
                torch.zeros(padding_size, dtype=task_vector.dtype, device=task_vector.device)
            ])
            padded_vectors[task_name] = padded_vector
            logger.debug(f"  Padded {task_name}: {current_size} -> {target_size} (+{padding_size} zeros)")
        elif current_size > target_size:
            raise ValueError(
                f"Task vector for {task_name} is larger ({current_size}) than target size ({target_size})"
            )
        else:
            # Already correct size
            padded_vectors[task_name] = task_vector

    return padded_vectors


def unpad_task_vector(
    padded_vector: torch.Tensor,
    original_size: int,
) -> torch.Tensor:
    """
    Remove padding from a task vector

    Args:
        padded_vector: Padded task vector
        original_size: Original size before padding

    Returns:
        Unpadded task vector
    """
    return padded_vector[:original_size]


def load_cached_trained_model_as_task_vector(
    cache_path: Path,
    create_base_model_fn,
    max_num_labels: int,
    method,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a cached trained model and return it as a flattened task vector

    Args:
        cache_path: Path to cached model checkpoint
        create_base_model_fn: Function to create base model with given num_labels
        max_num_labels: Maximum number of labels across tasks
        method: Training method instance with _load_trained_model method
        device: Device to load model on

    Returns:
        Flattened task vector (trained_params - base_params)
    """
    # Create base model and get its state
    base_model = create_base_model_fn(max_num_labels)
    base_state_dict = {k: v.cpu() for k, v in base_model.state_dict().items()}

    # Create trained model and load from cache
    trained_model = create_base_model_fn(max_num_labels)
    method._load_trained_model(trained_model, str(cache_path), device=device)

    # Compute task vector (difference from base)
    trained_state_dict = trained_model.state_dict()
    task_vector_dict = {
        name: param.cpu() - base_state_dict[name]
        for name, param in trained_state_dict.items()
        if name in base_state_dict
    }

    return flatten_task_vector(task_vector_dict)


def perform_evaluation(
    merged_task_vector: Dict[str, torch.Tensor],
    task_name: str,
    dataset_cfg: DictConfig,
    tokenizer,
    create_base_model_fn,
    device: torch.device,
    cfg: DictConfig,
) -> EvaluationResult:
    """
    Perform evaluation on a single task

    This function is wrapped by Joblib cache for automatic result caching.

    Args:
        merged_task_vector: Merged task vector to evaluate
        task_name: Name of the task
        dataset_cfg: Dataset configuration
        tokenizer: Tokenizer
        create_base_model_fn: Function to create base model
        device: Device to run on
        cfg: Full configuration

    Returns:
        Evaluation result
    """
    # Create base model
    merged_model = create_base_model_fn(num_labels=dataset_cfg.num_labels)

    # Apply merged task vector
    merged_model = apply_task_vector(merged_model, merged_task_vector, scaling=1.0)
    merged_model.eval()

    # Load and preprocess test dataset
    test_dataset = load_hf_dataset(
        dataset_path=dataset_cfg.hf_dataset.path,
        subset=dataset_cfg.hf_dataset.get("subset", None),
        split=dataset_cfg.hf_dataset.split.test,
        cache_dir=Path(cfg.paths.hf_datasets_cache) if cfg.paths.hf_datasets_cache else None,
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

    # Clean up
    del merged_model, test_dataset, test_dataset_processed, test_dataloader
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


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

    # Validate configuration before execution
    logger.info("Validating configuration...")
    validate_poc_benchmark_config(cfg)
    validate_training_config(cfg)

    # Initialize Joblib cache for evaluation results
    cache_enabled = cfg.benchmark.get('cache_enabled', True)
    if cache_enabled:
        eval_cache_dir = Path(cfg.paths.evaluation_cache)
        setup_cache(eval_cache_dir, verbose=0)
        logger.info(f"Evaluation cache enabled: {eval_cache_dir}")

        # Clear cache if requested
        if cfg.benchmark.get('clear_cache', False):
            from src.utils.cache import clear_cache
            clear_cache()
            logger.info("Cleared evaluation cache")
    else:
        logger.warning("Evaluation caching disabled")

    logger.info(f"Execution mode: {cfg.benchmark.mode}")
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

    # Step 2: Initialize merging method (check type first to optimize)
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Initializing merging method")
    logger.info("=" * 80)

    method = MethodRegistry.create(cfg.method.name, **cfg.method.params)
    logger.info(f"Method: {method}")

    # Import to check method type
    from src.methods.base import BaseTrainingMethod

    # Check if method requires task vectors (task-arithmetic methods)
    requires_task_vectors = not isinstance(method, BaseTrainingMethod)

    # Step 3: Prepare dataset configs and optionally compute task vectors
    logger.info("\n" + "=" * 80)
    if requires_task_vectors:
        logger.info("Step 3: Computing task vectors for task-arithmetic method")
    else:
        logger.info("Step 3: Preparing dataset configs (skipping task vector computation)")
    logger.info("=" * 80)

    task_vectors_dict = {}
    task_vector_template = None
    dataset_configs = {}

    # Collect dataset configs (always needed)
    for task_name in cfg.benchmark.tasks:
        dataset_cfg = cfg.datasets[task_name]
        dataset_configs[task_name] = dataset_cfg

    # Only compute task vectors if needed by the method
    if requires_task_vectors:
        # First pass: compute task vectors with their natural sizes
        task_vector_sizes = {}  # Track original sizes for each task

        for task_name in cfg.benchmark.tasks:
            logger.info(f"\nProcessing task: {task_name}")
            dataset_cfg = dataset_configs[task_name]

            # Load base model for this task (with task-specific num_labels)
            base_model = create_base_model(num_labels=dataset_cfg.num_labels)

            # Load fine-tuned model (don't pass num_labels to preserve fine-tuned weights)
            logger.info(f"  Loading fine-tuned model: {dataset_cfg.finetuned_checkpoint}")
            finetuned_model = load_model(
                model_id=dataset_cfg.finetuned_checkpoint,
                num_labels=None,  # Don't pass num_labels for fine-tuned models
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
            task_vector_sizes[task_name] = task_vectors_dict[task_name].shape[0]

            logger.info(f"  Task vector shape: {task_vectors_dict[task_name].shape}")

            # Clean up
            del base_model, finetuned_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Pad all task vectors to the maximum size
        max_task_vector_size = max(task_vector_sizes.values())
        logger.info(f"\nPadding task vectors to maximum size: {max_task_vector_size}")
        # Keep tensors on their original device (don't force CPU)
        task_vectors_dict = pad_task_vectors_to_match(task_vectors_dict, max_task_vector_size)

        # Store original sizes for later unpadding
        task_vector_original_sizes = task_vector_sizes

        # Get a template for unflattening later (use task with max labels for template)
        # Find task with maximum labels
        max_labels_task = max(dataset_configs.items(), key=lambda x: x[1].num_labels)[0]
        logger.info(f"Using {max_labels_task} as template (has maximum num_labels)")

        base_model_template = create_base_model(num_labels=dataset_configs[max_labels_task].num_labels)
        finetuned_model_template = load_model(
            model_id=dataset_configs[max_labels_task].finetuned_checkpoint,
            num_labels=None,  # Don't pass num_labels for fine-tuned models
            cache_dir=Path(cfg.paths.hf_models_cache_finetuned) if cfg.paths.hf_models_cache_finetuned else None,
            device=device,
            torch_dtype=cfg.model.loading.torch_dtype,
        )
        task_vector_template = compute_task_vector(finetuned_model_template, base_model_template)
        del base_model_template, finetuned_model_template
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.info("  ✓ Skipping task vector computation for training-based method")
        logger.info(f"  Collected {len(dataset_configs)} dataset configurations")

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

        # Get merged task vector (via merging or training)
        from src.methods.base import BaseTrainingMethod

        is_training_based = isinstance(method, BaseTrainingMethod)

        if is_training_based:
            # Training-based method: train new model and get task vector
            import hashlib
            import json

            # Generate unique identifier for this training configuration
            # Convert OmegaConf objects to plain Python types for JSON serialization
            params = OmegaConf.to_container(cfg.method.params, resolve=True)

            # Use INCLUSION (whitelist) approach: only include behavior-critical parameters
            training_critical_params = {
                # Core training hyperparameters (affect optimization)
                'learning_rate',                # Optimizer learning rate
                'batch_size',                   # Batch size per GPU
                'weight_decay',                 # L2 regularization strength
                'gradient_accumulation_steps',  # Effective batch size = batch_size * this
                'max_grad_norm',                # Gradient clipping threshold
                # Method-specific algorithm parameters
                'use_augmented',                # Chebyshev: augmented scalarization
                'epsilon',                      # Chebyshev: augmentation coefficient
                'normalize_preferences',        # Normalize preference weights
                'utopia_point',                 # Chebyshev: manual utopia point
                'nadir_point',                  # Chebyshev: manual nadir point
                # Data-affecting parameters (change what data is seen)
                'use_fp16',                     # Mixed precision (numerical precision)
                'max_samples_per_task',         # Data subsetting (limits training data)
                # TIES merging parameters
                'k',                            # Top-k fraction to keep
                'lambda_merge',                 # Merge scaling coefficient
                'scaling_method',               # Parameter scaling strategy
                'use_preferences',              # Use preference weighting
                'sign_consensus_method',        # Sign resolution strategy
            }
            # Only include parameters that exist in config and are in the critical set
            training_params = {k: v for k, v in params.items() if k in training_critical_params}

            config_str = json.dumps({
                "method": cfg.method.name,
                "preference": preference_array.tolist(),
                "params": training_params,  # Only training-related params
                "tasks": sorted(OmegaConf.to_container(cfg.benchmark.tasks, resolve=True)),
            }, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

            # Construct cache path for trained model
            cache_dir = Path(cfg.paths.trained_models_cache)
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_filename = f"{cfg.method.name}_{config_hash}.safetensors"
            model_cache_path = cache_dir / model_filename

            # Determine execution mode
            mode = cfg.benchmark.get('mode', 'train_eval')
            force_retrain = cfg.benchmark.get('force_retrain', False)
            model_exists = model_cache_path.exists()

            # Mode logic:
            # - eval_only: Require cached model (fail if missing)
            # - train_eval: Use cache if exists and not force_retrain, else train
            # - train_only: Skip if cache exists and not force_retrain, else train
            should_train = False
            should_load_cache = False

            if mode == 'eval_only':
                if not model_exists:
                    raise FileNotFoundError(
                        f"eval_only mode requires cached model: {model_cache_path}\n"
                        f"Train first with mode=train_eval or mode=train_only"
                    )
                should_load_cache = True
            elif mode == 'train_only':
                if model_exists and not force_retrain:
                    logger.info(f"Skipping training (cached model exists): {model_cache_path}")
                    should_load_cache = True
                else:
                    should_train = True
            else:  # train_eval (default)
                if model_exists and not force_retrain:
                    should_load_cache = True
                else:
                    should_train = True

            if should_load_cache:
                logger.info(f"Loading cached trained model: {model_cache_path}")
                max_labels = max(ds.num_labels for ds in dataset_configs.values())
                merged_flat = load_cached_trained_model_as_task_vector(
                    model_cache_path, create_base_model, max_labels, method, device
                )
                logger.info("  ✓ Loaded from cache")
            elif should_train:
                logger.info(f"Training multi-task model (mode={mode})...")

                # Save path for trained model (always save when training)
                save_path = str(model_cache_path)
                logger.info(f"  Will save to: {save_path}")

                merged_flat = method.train(
                    base_model=cfg.model.hf_model_id,
                    dataset_configs=dataset_configs,
                    preference_vector=preference_array,
                    model_cache_dir=str(Path(cfg.paths.hf_models_cache_base)) if cfg.paths.hf_models_cache_base else None,
                    finetuned_model_cache_dir=str(Path(cfg.paths.hf_models_cache_finetuned)) if cfg.paths.hf_models_cache_finetuned else None,
                    dataset_cache_dir=str(Path(cfg.paths.hf_datasets_cache)) if cfg.paths.hf_datasets_cache else None,
                    save_path=save_path,
                    epoch_checkpoint_dir=str(Path(cfg.paths.checkpoints_dir)),
                    model_identifier=config_hash,
                )
            else:
                # Should not reach here, but handle gracefully
                raise RuntimeError(f"Invalid execution state: mode={mode}, should_train={should_train}, should_load_cache={should_load_cache}")
        else:
            # Parameter merging method: merge pre-computed task vectors
            logger.info("Merging task vectors...")
            merged_flat = method.merge(
                task_vectors=task_vectors_dict,
                preference_vector=preference_array
            )

        # Unflatten merged vector (create template if not already available)
        if task_vector_template is None:
            # Create template on-demand for training-based methods
            first_task = cfg.benchmark.tasks[0]
            base_model_template = create_base_model(num_labels=dataset_configs[first_task].num_labels)
            finetuned_model_template = load_model(
                model_id=dataset_configs[first_task].finetuned_checkpoint,
                num_labels=None,  # Don't pass num_labels for fine-tuned models
                cache_dir=Path(cfg.paths.hf_models_cache_finetuned) if cfg.paths.hf_models_cache_finetuned else None,
                device=device,
                torch_dtype=cfg.model.loading.torch_dtype,
            )
            task_vector_template = compute_task_vector(finetuned_model_template, base_model_template)
            del base_model_template, finetuned_model_template
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("  Created task vector template for evaluation")

        merged_task_vector = unflatten_task_vector(merged_flat, task_vector_template)

        # For each task, evaluate the merged model (with Joblib caching)
        task_results = {}

        for task_name in cfg.benchmark.tasks:
            logger.info(f"\nEvaluating on task: {task_name}")

            dataset_cfg = dataset_configs[task_name]

            if cache_enabled:
                # Use Joblib-cached evaluation
                memory = get_memory()

                # For training-based methods, include model identifier in cache key
                model_id = config_hash if is_training_based else None

                # Create cached version of evaluation function
                # Joblib automatically hashes merged_task_vector and all arguments
                @memory.cache
                def _cached_perform_evaluation(
                    task_vec_hash,  # Hash of task vector for cache key
                    method_name,
                    pref_vec_tuple,
                    task,
                    model_id_param,
                ):
                    """Cached wrapper - Joblib handles all caching automatically"""
                    logger.info(f"  Cache miss - performing evaluation for {task}")
                    return perform_evaluation(
                        merged_task_vector=merged_task_vector,
                        task_name=task,
                        dataset_cfg=dataset_cfg,
                        tokenizer=tokenizer,
                        create_base_model_fn=create_base_model,
                        device=device,
                        cfg=cfg,
                    )

                # Create deterministic hash of task vector for cache key
                import hashlib
                task_vec_bytes = str(sorted(merged_task_vector.items())).encode()
                task_vec_hash = hashlib.md5(task_vec_bytes).hexdigest()[:12]

                result = _cached_perform_evaluation(
                    task_vec_hash=task_vec_hash,
                    method_name=cfg.method.name,
                    pref_vec_tuple=tuple(preference_vector),
                    task=task_name,
                    model_id_param=model_id,
                )

                logger.info(f"  ✓ {result}")
            else:
                # No caching - perform evaluation directly
                result = perform_evaluation(
                    merged_task_vector=merged_task_vector,
                    task_name=task_name,
                    dataset_cfg=dataset_cfg,
                    tokenizer=tokenizer,
                    create_base_model_fn=create_base_model,
                    device=device,
                    cfg=cfg,
                )
                logger.info(f"  {result}")

            task_results[task_name] = result

            # Log individual task results to W&B
            try:
                import wandb
                if wandb.run:
                    # Create preference vector string for grouping
                    pref_str = "_".join([f"{p:.2f}" for p in preference_vector])
                    log_dict = {}
                    for metric_name, metric_value in result.metrics.items():
                        log_dict[f"eval/{task_name}/{metric_name}"] = metric_value
                    # Also log with preference prefix for easier comparison
                    for metric_name, metric_value in result.metrics.items():
                        log_dict[f"eval_pref_{pref_str}/{task_name}/{metric_name}"] = metric_value
                    # Use pref_idx as step so different preference vectors show as different points
                    wandb.log(log_dict, step=pref_idx)
            except (ImportError, AttributeError):
                pass

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

        # Log aggregate metrics to W&B for this preference vector
        try:
            import wandb
            if wandb.run:
                pref_str = "_".join([f"{p:.2f}" for p in preference_vector])
                # Calculate average metrics across tasks
                all_metrics = {}
                for task_name, result in task_results.items():
                    for metric_name, metric_value in result.metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)

                # Log averages
                avg_log_dict = {"preference_vector_str": pref_str}
                for metric_name, values in all_metrics.items():
                    avg_value = sum(values) / len(values)
                    avg_log_dict[f"eval_avg/{metric_name}"] = avg_value
                    avg_log_dict[f"eval_pref_{pref_str}/avg_{metric_name}"] = avg_value

                # Use pref_idx as step so different preference vectors show as different points
                wandb.log(avg_log_dict, step=pref_idx)
        except (ImportError, AttributeError):
            pass

    # Step 5: Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)

    figures = {}
    try:
        from src.visualization.generator import generate_all_visualizations

        # Determine output directory for plots (under current run directory)
        output_dir = Path(cfg.paths.visualizations_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving visualizations to: {output_dir}")

        # Generate all visualizations
        primary_metric = cfg.benchmark.evaluation.metrics[0] if cfg.benchmark.evaluation.metrics else "f1_macro"
        figures = generate_all_visualizations(
            all_results=all_results,
            task_names=cfg.benchmark.tasks,
            metric_name=primary_metric,
            output_dir=output_dir,
            method_name=cfg.method.name,
        )

        logger.info(f"✓ Generated {len(figures)} visualizations")

    except Exception as e:
        import traceback
        logger.error(f"Failed to generate visualizations: {e}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.warning("Continuing without visualizations. Check the error above for details.")

    # Step 6: Summarize results
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Complete!")
    logger.info("=" * 80)

    results = {
        "benchmark_name": cfg.benchmark.name,
        "method": cfg.method.name,
        "tasks": cfg.benchmark.tasks,
        "all_results": all_results,
        "figures": figures,
        "status": "completed",
    }

    return results

