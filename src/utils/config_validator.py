"""Configuration validation utilities

Validates Hydra configurations before benchmark execution to:
- Fail fast with clear error messages
- Prevent wasted compute time
- Document configuration assumptions
"""

import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails"""
    pass


def validate_preference_vector(pref_vector: List[float], num_tasks: int, index: int = 0) -> None:
    """
    Validate a single preference vector

    Args:
        pref_vector: Preference weights for each task
        num_tasks: Expected number of tasks
        index: Index for error messages

    Raises:
        ConfigValidationError: If validation fails
    """
    if len(pref_vector) != num_tasks:
        raise ConfigValidationError(
            f"Preference vector {index} has {len(pref_vector)} values but {num_tasks} tasks are configured.\n"
            f"  Preference vector: {pref_vector}\n"
            f"  Tasks: {num_tasks}"
        )

    pref_sum = sum(pref_vector)
    if abs(pref_sum - 1.0) > 1e-6:
        raise ConfigValidationError(
            f"Preference vector {index} must sum to 1.0, got {pref_sum:.6f}\n"
            f"  Preference vector: {pref_vector}"
        )

    if any(p < 0 for p in pref_vector):
        raise ConfigValidationError(
            f"Preference vector {index} contains negative values: {pref_vector}\n"
            "  All preferences must be non-negative."
        )


def validate_poc_benchmark_config(cfg: DictConfig) -> None:
    """
    Validate POC benchmark configuration

    Args:
        cfg: Hydra configuration

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []

    # Validate tasks
    if not cfg.benchmark.tasks:
        errors.append("No tasks configured in benchmark.tasks")

    num_tasks = len(cfg.benchmark.tasks)

    # Validate preference vectors
    if not cfg.benchmark.preference_vectors:
        errors.append("No preference vectors configured")
    else:
        for i, pref_vector in enumerate(cfg.benchmark.preference_vectors):
            try:
                validate_preference_vector(pref_vector, num_tasks, i)
            except ConfigValidationError as e:
                errors.append(str(e))

    # Validate execution mode
    valid_modes = ['train_eval', 'train_only', 'eval_only']
    mode = cfg.benchmark.get('mode', 'train_eval')
    if mode not in valid_modes:
        errors.append(
            f"Invalid benchmark.mode: {mode}. Must be one of {valid_modes}"
        )

    # Validate cache directories if caching enabled
    if cfg.benchmark.get('cache_enabled', True):
        try:
            eval_cache_dir = Path(cfg.paths.evaluation_cache)
            eval_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create evaluation cache directory: {e}")

        try:
            trained_cache_dir = Path(cfg.paths.trained_models_cache)
            trained_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create trained models cache directory: {e}")

    # Validate dataset configs exist for all tasks
    for task_name in cfg.benchmark.tasks:
        if task_name not in cfg.datasets:
            errors.append(f"No dataset configuration found for task: {task_name}")

    # Raise all errors together
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)

    logger.info("✓ Configuration validation passed")


def validate_training_config(cfg: DictConfig) -> None:
    """
    Validate training-specific configuration

    Args:
        cfg: Hydra configuration

    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []

    # Validate training parameters
    if hasattr(cfg.method, 'params'):
        params = cfg.method.params

        # Check common training params
        if 'learning_rate' in params and params.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {params.learning_rate}")

        if 'batch_size' in params and params.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {params.batch_size}")

        if 'num_epochs' in params and params.num_epochs < 1:
            errors.append(f"num_epochs must be >= 1, got {params.num_epochs}")

        # Validate Chebyshev-specific params
        if cfg.method.name == 'chebyshev':
            if 'epsilon' in params and params.epsilon < 0:
                errors.append(f"Chebyshev epsilon must be >= 0, got {params.epsilon}")

        # Validate TIES-specific params
        if cfg.method.name == 'ties':
            if 'k' in params and not (0 < params.k <= 1):
                errors.append(f"TIES k must be in (0, 1], got {params.k}")

    if errors:
        error_msg = "Training configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)
