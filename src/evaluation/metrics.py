"""Evaluation metrics computation"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


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
