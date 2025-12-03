"""Pareto frontier visualization"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_pareto_frontier_2d(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D Pareto frontier (maximization)

    Args:
        points: Array of shape (N, 2) with objective values

    Returns:
        Tuple of (pareto_points, pareto_indices)
    """
    # Sort by first objective (descending)
    sorted_indices = np.argsort(points[:, 0])[::-1]
    pareto_indices = []

    current_max_y = -np.inf
    for idx in sorted_indices:
        if points[idx, 1] > current_max_y:
            pareto_indices.append(idx)
            current_max_y = points[idx, 1]

    pareto_indices = np.array(pareto_indices)
    pareto_points = points[pareto_indices]

    # Sort pareto points by first objective for plotting
    sort_order = np.argsort(pareto_points[:, 0])
    pareto_points = pareto_points[sort_order]
    pareto_indices = pareto_indices[sort_order]

    return pareto_points, pareto_indices


def plot_pareto_frontier_2d(
    results: Dict[str, Tuple[float, float]],
    task_names: Tuple[str, str],
    method_groups: Optional[Dict[str, List[str]]] = None,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D Pareto frontier

    Args:
        results: Dictionary mapping config names to (task1_score, task2_score)
        task_names: Names of the two tasks
        method_groups: Optional grouping of configs by method
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    # Convert to array
    points = np.array(list(results.values()))
    config_names = list(results.keys())

    # Compute Pareto frontier
    pareto_points, pareto_indices = compute_pareto_frontier_2d(points)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all points
    if method_groups:
        # Color by method
        for method, configs in method_groups.items():
            method_points = np.array([results[cfg] for cfg in configs if cfg in results])
            if len(method_points) > 0:
                ax.scatter(
                    method_points[:, 0],
                    method_points[:, 1],
                    label=method,
                    s=50,
                    alpha=0.6,
                )
    else:
        ax.scatter(points[:, 0], points[:, 1], s=50, alpha=0.6, label="Configurations")

    # Plot Pareto frontier
    ax.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "r--",
        linewidth=2,
        label="Pareto Frontier",
    )
    ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        s=100,
        c="red",
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=10,
    )

    # Labels
    ax.set_xlabel(f"{task_names[0]} Performance")
    ax.set_ylabel(f"{task_names[1]} Performance")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Pareto Frontier: {task_names[0]} vs {task_names[1]}")

    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        from src.visualization.plots import save_plot

        save_plot(fig, save_path)

    return fig
