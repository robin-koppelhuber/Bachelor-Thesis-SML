"""Core plotting utilities"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 6)


def save_plot(
    fig: plt.Figure,
    save_path: Path,
    formats: List[str] = ["png", "pdf"],
) -> None:
    """
    Save plot in multiple formats

    Args:
        fig: Matplotlib figure
        save_path: Path without extension
        formats: List of formats to save
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = save_path.with_suffix(f".{fmt}")
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot: {output_path}")


def plot_task_performance(
    results: Dict[str, Dict[str, float]],
    metric_name: str = "accuracy",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-task performance as bar chart

    Args:
        results: Dictionary mapping task names to metric dicts
        metric_name: Metric to plot
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    tasks = list(results.keys())
    scores = [results[task][metric_name] for task in tasks]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(tasks, scores, color=sns.color_palette("husl", len(tasks)))
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_xlabel("Task")
    ax.set_ylim(0, 1.0)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Per-Task {metric_name.replace('_', ' ').title()}")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_method_comparison(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    task_names: List[str],
    metric_name: str = "accuracy",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare multiple methods across tasks

    Args:
        method_results: Nested dict: method -> task -> metrics
        task_names: List of task names
        metric_name: Metric to plot
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    methods = list(method_results.keys())
    num_tasks = len(task_names)
    num_methods = len(methods)

    x = np.arange(num_tasks)
    width = 0.8 / num_methods

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        scores = [method_results[method][task][metric_name] for task in task_names]
        offset = width * (i - num_methods / 2 + 0.5)
        ax.bar(x + offset, scores, width, label=method)

    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_xlabel("Task")
    ax.set_title(f"Method Comparison: {metric_name.replace('_', ' ').title()}")
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig
