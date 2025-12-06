"""Evaluation result caching using Joblib Memory

Joblib provides robust, automatic caching with:
- Automatic hash computation for function arguments
- Cache invalidation when function code changes
- Compression and memory mapping
- Handles numpy arrays, dicts, and complex objects
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from joblib import Memory

logger = logging.getLogger(__name__)

# Global memory instance (initialized by setup_cache)
_memory: Optional[Memory] = None


def setup_cache(cache_dir: Optional[Path], verbose: int = 0) -> Memory:
    """
    Initialize joblib Memory for caching

    Args:
        cache_dir: Directory for cache storage (None = no caching)
        verbose: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Memory instance
    """
    global _memory

    if cache_dir is None:
        logger.info("Caching disabled (no cache directory specified)")
        _memory = Memory(location=None, verbose=verbose)
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache enabled: {cache_dir}")
        _memory = Memory(location=str(cache_dir), verbose=verbose)

    return _memory


def get_memory() -> Memory:
    """
    Get global Memory instance

    Returns:
        Memory instance (raises if not initialized)
    """
    if _memory is None:
        raise RuntimeError("Cache not initialized. Call setup_cache() first.")
    return _memory


def clear_cache(cache_dir: Optional[Path] = None) -> None:
    """
    Clear all cached results

    Args:
        cache_dir: Cache directory to clear (uses global memory if None)
    """
    if cache_dir is not None:
        memory = Memory(location=str(cache_dir), verbose=0)
        memory.clear()
        logger.info(f"Cleared cache: {cache_dir}")
    elif _memory is not None:
        _memory.clear()
        logger.info("Cleared global cache")
    else:
        logger.warning("No cache to clear")


# Cached evaluation function
def cached_evaluate_model(
    evaluate_fn,
    model_state: Dict[str, Any],
    task_name: str,
    metric_names: tuple,
    method_name: str,
    preference_vector: tuple,
    model_identifier: Optional[str] = None,
):
    """
    Cached wrapper for model evaluation

    This function is decorated with @memory.cache at runtime.
    Joblib automatically hashes all arguments and caches results.

    Args:
        evaluate_fn: Function that performs actual evaluation
        model_state: Model state dict (hashable)
        task_name: Task name
        metric_names: Tuple of metric names (tuple for hashability)
        method_name: Merging method name
        preference_vector: Preference vector (tuple for hashability)
        model_identifier: Optional identifier for training-based methods

    Returns:
        Evaluation metrics dict
    """
    # Joblib handles caching automatically based on argument hashes
    return evaluate_fn(model_state, task_name, metric_names)
