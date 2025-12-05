"""Evaluation result caching utilities"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _to_native_python(obj):
    """Convert OmegaConf objects to native Python types"""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (DictConfig, dict)):
        return {k: _to_native_python(v) for k, v in obj.items()}
    elif isinstance(obj, (ListConfig, list)):
        return [_to_native_python(item) for item in obj]
    else:
        return obj


def _compute_cache_key(
    method: str,
    preference_vector: List,
    task_name: str,
    metric_names: List,
    model_identifier: Optional[str] = None,
) -> str:
    """
    Compute cache key for evaluation results

    Args:
        method: Method name
        preference_vector: Preference vector
        task_name: Task name
        metric_names: List of metrics
        model_identifier: Optional unique identifier for the model (e.g., training config hash)
                         This is crucial for training-based methods to avoid cache collisions

    Returns:
        MD5 hash as cache key
    """
    # Convert to native Python types (handles OmegaConf objects)
    preference_vector = _to_native_python(preference_vector)
    metric_names = _to_native_python(metric_names)

    # Create deterministic hash from parameters
    key_data = {
        "method": method,
        "preference": preference_vector,
        "task": task_name,
        "metrics": sorted(metric_names),
    }

    # Include model identifier if provided (important for training-based methods)
    if model_identifier:
        key_data["model_id"] = model_identifier

    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_evaluation(
    cache_dir: Path,
    method: str,
    preference_vector: list,
    task_name: str,
    metric_names: list,
    model_identifier: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Get cached evaluation result if available

    Args:
        cache_dir: Cache directory
        method: Merging method name
        preference_vector: Preference vector
        task_name: Task name
        metric_names: List of metric names
        model_identifier: Optional unique identifier for the model

    Returns:
        Cached metrics dict or None if not found
    """
    cache_key = _compute_cache_key(method, preference_vector, task_name, metric_names, model_identifier)
    cache_file = cache_dir / f"eval_{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cached = json.load(f)

        logger.debug(f"Cache hit for {task_name} with {method} [{preference_vector}]")
        return cached["metrics"]

    except Exception as e:
        logger.warning(f"Failed to load cache from {cache_file}: {e}")
        return None


def save_evaluation_to_cache(
    cache_dir: Path,
    method: str,
    preference_vector: list,
    task_name: str,
    metric_names: list,
    metrics: Dict[str, float],
    model_identifier: Optional[str] = None,
) -> None:
    """
    Save evaluation result to cache

    Args:
        cache_dir: Cache directory
        method: Merging method name
        preference_vector: Preference vector
        task_name: Task name
        metric_names: List of metric names
        metrics: Computed metrics
        model_identifier: Optional unique identifier for the model
    """
    cache_key = _compute_cache_key(method, preference_vector, task_name, metric_names, model_identifier)
    cache_file = cache_dir / f"eval_{cache_key}.json"

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        cache_data = {
            "method": method,
            "preference_vector": preference_vector,
            "task": task_name,
            "metric_names": metric_names,
            "metrics": metrics,
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Cached evaluation for {task_name}")

    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_file}: {e}")


def clear_evaluation_cache(cache_dir: Path, method: Optional[str] = None) -> int:
    """
    Clear evaluation cache

    Args:
        cache_dir: Cache directory
        method: Optional method name to clear only that method's cache

    Returns:
        Number of files deleted
    """
    if not cache_dir.exists():
        return 0

    count = 0
    pattern = f"eval_*.json" if method is None else f"eval_{method}_*.json"

    for cache_file in cache_dir.glob(pattern):
        try:
            cache_file.unlink()
            count += 1
        except Exception as e:
            logger.warning(f"Failed to delete {cache_file}: {e}")

    if count > 0:
        logger.info(f"Cleared {count} cached evaluation(s)")

    return count
