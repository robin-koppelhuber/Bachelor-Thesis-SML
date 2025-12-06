"""Evaluation result caching using Joblib Memory

Joblib provides robust, automatic caching with:
- Automatic hash computation for function arguments
- Cache invalidation when function code changes
- Compression and memory mapping
- Handles numpy arrays, dicts, and complex objects
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from joblib import Memory

logger = logging.getLogger(__name__)

# Global memory instance
_memory: Optional[Memory] = None


def setup_cache(cache_dir: Path, verbose: int = 0) -> Memory:
    """
    Initialize Joblib Memory for caching evaluation results

    Args:
        cache_dir: Directory for cache storage
        verbose: Verbosity level (0=silent, 1=info, 10=debug)

    Returns:
        Memory instance
    """
    global _memory

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _memory = Memory(location=str(cache_dir), verbose=verbose)
    logger.info(f"Joblib cache initialized: {cache_dir}")

    return _memory


def get_memory() -> Memory:
    """
    Get global Memory instance

    Returns:
        Memory instance

    Raises:
        RuntimeError: If cache not initialized
    """
    if _memory is None:
        raise RuntimeError(
            "Cache not initialized. Call setup_cache() first in main.py or run.py"
        )
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


def cached_evaluation(
    method_name: str,
    preference_vector: tuple,  # tuple for hashability
    task_name: str,
    model_identifier: Optional[str] = None,
):
    """
    Decorator factory for cached evaluation

    Joblib automatically hashes all arguments and caches results.
    When the function code changes, cache is automatically invalidated.

    Args:
        method_name: Merging method name
        preference_vector: Preference vector (as tuple for hashing)
        task_name: Task name
        model_identifier: Optional identifier for training-based methods

    Returns:
        Cached function wrapper
    """
    memory = get_memory()

    # Create a unique cache key by combining all parameters
    # Joblib will hash this along with the function code
    def decorator(func):
        # Tag the cached function with metadata for debugging
        cached_func = memory.cache(func)
        cached_func.__cache_metadata__ = {
            'method': method_name,
            'preference': preference_vector,
            'task': task_name,
            'model_id': model_identifier,
        }
        return cached_func

    return decorator


def evaluate_model_cached(
    evaluate_fn,
    method_name: str,
    preference_vector: tuple,
    task_name: str,
    model_identifier: Optional[str],
    **eval_kwargs
) -> Dict[str, float]:
    """
    Cached wrapper for model evaluation

    This function is automatically cached by Joblib based on all arguments.

    Args:
        evaluate_fn: Function that performs actual evaluation
        method_name: Merging method name
        preference_vector: Preference vector (tuple for hashability)
        task_name: Task name
        model_identifier: Optional identifier for training-based methods
        **eval_kwargs: Additional arguments passed to evaluate_fn

    Returns:
        Evaluation metrics dict
    """
    memory = get_memory()

    # Create a cached version of the evaluation function
    # Joblib hashes ALL arguments including nested dicts, arrays, etc.
    @memory.cache
    def _cached_eval(method, pref, task, model_id, **kwargs):
        logger.debug(f"Cache miss - evaluating {task} with {method}")
        return evaluate_fn(**kwargs)

    result = _cached_eval(
        method_name,
        preference_vector,
        task_name,
        model_identifier,
        **eval_kwargs
    )

    return result
