"""Evaluation metrics computation"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        predictions: Predicted labels
        labels: True labels
        metrics: List of metric names to compute

    Returns:
        Dictionary mapping metric names to values
    """
    results = {}

    for metric in metrics:
        if metric == "accuracy":
            results["accuracy"] = float(accuracy_score(labels, predictions))

        elif metric == "f1_macro":
            results["f1_macro"] = float(
                f1_score(labels, predictions, average="macro", zero_division=0)
            )

        elif metric == "f1_weighted":
            results["f1_weighted"] = float(
                f1_score(labels, predictions, average="weighted", zero_division=0)
            )

        elif metric == "f1_micro":
            results["f1_micro"] = float(
                f1_score(labels, predictions, average="micro", zero_division=0)
            )

        elif metric == "precision_macro":
            results["precision_macro"] = float(
                precision_score(labels, predictions, average="macro", zero_division=0)
            )

        elif metric == "recall_macro":
            results["recall_macro"] = float(
                recall_score(labels, predictions, average="macro", zero_division=0)
            )

        elif metric == "mcc":
            results["mcc"] = float(matthews_corrcoef(labels, predictions))

        elif metric == "cohen_kappa":
            results["cohen_kappa"] = float(cohen_kappa_score(labels, predictions))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


def compute_per_class_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class F1, precision, and recall

    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Optional list of class names (defaults to numeric labels)

    Returns:
        Dictionary mapping class names to metric dicts
    """
    # Get unique classes
    unique_classes = sorted(set(labels) | set(predictions))

    if class_names is None:
        class_names = [f"class_{i}" for i in unique_classes]
    elif len(class_names) != len(unique_classes):
        raise ValueError(
            f"Number of class names ({len(class_names)}) doesn't match "
            f"number of classes ({len(unique_classes)})"
        )

    # Compute per-class metrics
    f1_scores = f1_score(labels, predictions, average=None, zero_division=0, labels=unique_classes)
    precision_scores = precision_score(
        labels, predictions, average=None, zero_division=0, labels=unique_classes
    )
    recall_scores = recall_score(
        labels, predictions, average=None, zero_division=0, labels=unique_classes
    )

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "f1": float(f1_scores[i]),
            "precision": float(precision_scores[i]),
            "recall": float(recall_scores[i]),
        }

    return per_class_metrics


def compute_confusion_matrix_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix

    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: Optional list of class names

    Returns:
        Tuple of (confusion_matrix, class_names)
    """
    # Get unique classes
    unique_classes = sorted(set(labels) | set(predictions))

    if class_names is None:
        class_names = [f"class_{i}" for i in unique_classes]

    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=unique_classes)

    return cm, class_names


def compute_weighted_score(
    task_metrics: Dict[str, Dict[str, float]],
    preference_vector: np.ndarray,
    metric_name: str = "accuracy",
) -> float:
    """
    Compute weighted multi-task score based on preference vector

    Args:
        task_metrics: Dictionary mapping task names to metric dicts
        preference_vector: Preference weights for each task
        metric_name: Metric to weight

    Returns:
        Weighted score
    """
    task_names = sorted(task_metrics.keys())
    scores = np.array([task_metrics[name][metric_name] for name in task_names])
    weighted_score = np.dot(preference_vector, scores)
    return float(weighted_score)


def compute_task_interference(
    results_across_preferences: List[Dict[str, Dict[str, float]]],
    metric_name: str = "f1_macro",
) -> Dict[Tuple[str, str], float]:
    """
    Compute task interference as pairwise correlation between task performances
    across different preference vectors.

    Negative correlation indicates interference: when Task A improves, Task B degrades.

    Args:
        results_across_preferences: List of result dicts (one per preference vector)
                                   Each dict maps task_name -> metrics
        metric_name: Metric to use for interference computation

    Returns:
        Dictionary mapping (task1, task2) pairs to correlation coefficients
        Values near -1 indicate strong interference
    """
    if len(results_across_preferences) < 2:
        return {}

    # Get task names from first result
    task_names = sorted(results_across_preferences[0].keys())

    # Build matrix: rows = preference vectors, cols = tasks
    performance_matrix = []
    for result in results_across_preferences:
        row = [result[task][metric_name] for task in task_names]
        performance_matrix.append(row)

    performance_matrix = np.array(performance_matrix)

    # Compute pairwise correlations
    interference = {}
    for i, task1 in enumerate(task_names):
        for j, task2 in enumerate(task_names):
            if i < j:  # Only compute upper triangle
                corr = np.corrcoef(performance_matrix[:, i], performance_matrix[:, j])[0, 1]
                interference[(task1, task2)] = float(corr)

    return interference


def compute_preference_alignment(
    requested_preference: np.ndarray,
    achieved_scores: Dict[str, float],
    metric_name: str = "f1_macro",
) -> Dict[str, float]:
    """
    Measure how well achieved performance aligns with requested preference vector

    Args:
        requested_preference: Requested preference weights (should sum to 1)
        achieved_scores: Dictionary mapping task names to achieved scores
        metric_name: Name of the metric (for documentation)

    Returns:
        Dictionary with alignment metrics:
        - cosine_similarity: Cosine similarity between requested and normalized achieved
        - mse: Mean squared error between requested and normalized achieved
        - max_deviation: Maximum absolute deviation from requested preference
    """
    task_names = sorted(achieved_scores.keys())

    # Get achieved scores in same order as tasks
    achieved_vec = np.array([achieved_scores[task] for task in task_names])

    # Normalize achieved scores to sum to 1 (like preference vector)
    achieved_normalized = achieved_vec / achieved_vec.sum()

    # Compute alignment metrics
    # Cosine similarity (1.0 = perfect alignment, -1.0 = opposite)
    cosine_sim = np.dot(requested_preference, achieved_normalized) / (
        np.linalg.norm(requested_preference) * np.linalg.norm(achieved_normalized)
    )

    # Mean squared error
    mse = np.mean((requested_preference - achieved_normalized) ** 2)

    # Maximum deviation
    max_dev = np.max(np.abs(requested_preference - achieved_normalized))

    return {
        "cosine_similarity": float(cosine_sim),
        "mse": float(mse),
        "max_deviation": float(max_dev),
    }


def compute_hypervolume_indicator(
    pareto_points: np.ndarray,
    reference_point: np.ndarray,
) -> float:
    """
    Compute hypervolume indicator for Pareto frontier

    The hypervolume indicator measures the volume of objective space dominated
    by the Pareto frontier. Higher values indicate better Pareto coverage.

    NOTE: This is a placeholder/template implementation for 2D case.
    For production use, consider using pygmo library for efficient computation.

    Args:
        pareto_points: Array of shape (N, M) where N is number of points
                      and M is number of objectives
        reference_point: Reference point for hypervolume computation
                        (typically worst acceptable performance)

    Returns:
        Hypervolume value (higher is better)

    Example:
        >>> # For 2D case (2 tasks)
        >>> pareto_points = np.array([[0.8, 0.6], [0.7, 0.7], [0.6, 0.8]])
        >>> reference = np.array([0.0, 0.0])
        >>> hv = compute_hypervolume_indicator(pareto_points, reference)

    TODO: For production, use:
        ```python
        import pygmo as pg
        hv = pg.hypervolume(pareto_points)
        return hv.compute(reference_point)
        ```
    """
    if len(pareto_points) == 0:
        return 0.0

    n_objectives = pareto_points.shape[1]

    if n_objectives == 2:
        # Simple 2D implementation
        # Sort points by first objective (descending)
        sorted_points = pareto_points[np.argsort(-pareto_points[:, 0])]

        hypervolume = 0.0
        prev_x = reference_point[0]

        for point in sorted_points:
            width = point[0] - prev_x
            height = point[1] - reference_point[1]
            if width > 0 and height > 0:
                hypervolume += width * height
            prev_x = point[0]

        return float(hypervolume)
    else:
        # For higher dimensions, use placeholder
        # In production, use pygmo or similar library
        raise NotImplementedError(
            f"Hypervolume computation for {n_objectives}D not implemented. "
            "Please use pygmo library: pip install pygmo"
        )
