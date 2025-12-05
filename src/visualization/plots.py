"""Core plotting utilities"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def plot_radar_chart(
    task_results: Dict[str, float],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    color: str = "blue",
    alpha: float = 0.25,
) -> plt.Figure:
    """
    Plot radar chart for multi-task performance

    Args:
        task_results: Dictionary mapping task names to scores
        metric_name: Name of the metric being plotted
        save_path: Optional path to save figure
        title: Optional title
        color: Fill color for radar chart
        alpha: Transparency for fill

    Returns:
        Matplotlib figure
    """
    tasks = list(task_results.keys())
    scores = list(task_results.values())
    num_tasks = len(tasks)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # Complete the circle
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    # Plot data
    ax.plot(angles, scores_plot, "o-", linewidth=2, color=color, label=metric_name)
    ax.fill(angles, scores_plot, alpha=alpha, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)

    if title:
        ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Multi-Task Performance: {metric_name}", pad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_multi_radar_chart(
    results_dict: Dict[str, Dict[str, float]],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot multiple radar charts for comparing different configurations

    Args:
        results_dict: Dictionary mapping config names to task results
        metric_name: Name of the metric being plotted
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    if not results_dict:
        raise ValueError("results_dict cannot be empty")

    # Get task names from first config
    tasks = list(next(iter(results_dict.values())).keys())
    num_tasks = len(tasks)

    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Plot each configuration
    colors = sns.color_palette("husl", len(results_dict))
    for i, (config_name, task_results) in enumerate(results_dict.items()):
        scores = [task_results[task] for task in tasks]
        scores_plot = scores + [scores[0]]  # Complete the circle

        ax.plot(angles, scores_plot, "o-", linewidth=2, color=colors[i], label=config_name)
        ax.fill(angles, scores_plot, alpha=0.15, color=colors[i])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    if title:
        ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Multi-Configuration Comparison: {metric_name}", pad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_performance_heatmap(
    performance_matrix: np.ndarray,
    task_names: List[str],
    preference_labels: List[str],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap of performance across tasks and preference vectors

    Args:
        performance_matrix: Array of shape (num_preferences, num_tasks)
        task_names: List of task names
        preference_labels: List of preference vector labels
        metric_name: Name of the metric being plotted
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    sns.heatmap(
        performance_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=task_names,
        yticklabels=preference_labels,
        cbar_kws={"label": metric_name},
        ax=ax,
    )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Preference Vector", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Performance Heatmap: {metric_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    task_name: str,
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix heatmap

    Args:
        cm: Confusion matrix array
        class_names: List of class names
        task_name: Name of the task
        save_path: Optional path to save figure
        normalize: Whether to normalize by row (true labels)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = cm_normalized
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix: {task_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_task_interference_matrix(
    interference_matrix: np.ndarray,
    task_names: List[str],
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot task interference correlation matrix

    Args:
        interference_matrix: Correlation matrix between tasks
        task_names: List of task names
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with diverging colormap
    # Red = negative correlation (interference), Blue = positive correlation (synergy)
    sns.heatmap(
        interference_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=task_names,
        yticklabels=task_names,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title("Task Interference Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_preference_alignment(
    requested: np.ndarray,
    achieved: np.ndarray,
    task_names: List[str],
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot requested vs achieved preference alignment

    Args:
        requested: Requested preference vector
        achieved: Achieved (normalized) performance vector
        task_names: List of task names
        save_path: Optional path to save figure
        title: Optional title

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(task_names))
    width = 0.35

    # Bar chart comparison
    ax1.bar(x - width / 2, requested, width, label="Requested", alpha=0.8, color="steelblue")
    ax1.bar(x + width / 2, achieved, width, label="Achieved (normalized)", alpha=0.8, color="coral")
    ax1.set_xlabel("Task", fontsize=12)
    ax1.set_ylabel("Weight", fontsize=12)
    ax1.set_title("Preference Alignment: Bar Chart", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Scatter plot with diagonal
    ax2.scatter(requested, achieved, s=100, alpha=0.7, color="steelblue")
    for i, task in enumerate(task_names):
        ax2.annotate(task, (requested[i], achieved[i]), xytext=(5, 5), textcoords="offset points")

    # Add perfect alignment diagonal
    max_val = max(requested.max(), achieved.max())
    ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Alignment")

    ax2.set_xlabel("Requested Preference", fontsize=12)
    ax2.set_ylabel("Achieved Performance (normalized)", fontsize=12)
    ax2.set_title("Preference Alignment: Scatter", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_method_comparison_dashboard(
    method_results: Dict[str, Dict[str, Dict[str, float]]],
    task_names: List[str],
    metrics: List[str] = ["f1_macro", "accuracy"],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create comprehensive dashboard comparing multiple methods

    Args:
        method_results: Nested dict: method -> task -> metrics
        task_names: List of task names
        metrics: List of metrics to plot
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    methods = list(method_results.keys())
    num_tasks = len(task_names)
    num_methods = len(methods)

    x = np.arange(num_tasks)
    width = 0.8 / num_methods

    colors = sns.color_palette("husl", num_methods)

    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx]

        for method_idx, method in enumerate(methods):
            scores = [method_results[method][task][metric_name] for task in task_names]
            offset = width * (method_idx - num_methods / 2 + 0.5)
            ax.bar(x + offset, scores, width, label=method, color=colors[method_idx], alpha=0.8)

        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Task", fontsize=12)
        ax.set_title(f"{metric_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Method Comparison Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig
