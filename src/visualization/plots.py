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


def format_task_name(name: str) -> str:
    """
    Format task name for better display

    Examples:
        ag_news -> AG News
        imdb -> IMDB
        mnli -> MNLI
        mrpc -> MRPC

    Args:
        name: Task name to format

    Returns:
        Formatted task name
    """
    parts = name.replace('_', ' ').split()
    return ' '.join([part.upper() if len(part) <= 4 else part.title() for part in parts])


def format_metric_name(name: str) -> str:
    """
    Format metric name for better display

    Examples:
        f1_macro -> F1 Macro
        accuracy -> Accuracy

    Args:
        name: Metric name to format

    Returns:
        Formatted metric name
    """
    return name.replace('_', ' ').title()


def save_plot(
    fig: plt.Figure,
    save_path: Path,
    formats: List[str] = ["png"],
) -> None:
    """
    Save plot in multiple formats

    Args:
        fig: Matplotlib figure
        save_path: Path without extension
        formats: List of formats to save (default: PNG only)
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

    # Format task names for display
    formatted_tasks = [format_task_name(task) for task in tasks]
    formatted_metric = format_metric_name(metric_name)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(formatted_tasks)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.grid(True)

    if title:
        ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"Multi-Task Performance: {formatted_metric}", pad=20, fontsize=12, fontweight="bold")

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

    # Format task names for display
    formatted_task_names = [format_task_name(name) for name in task_names]
    formatted_metric = format_metric_name(metric_name)

    # Create heatmap (RdYlGn is appropriate since higher is better for all metrics)
    sns.heatmap(
        performance_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        xticklabels=formatted_task_names,
        yticklabels=preference_labels,
        cbar_kws={"label": formatted_metric},
        ax=ax,
    )

    ax.set_xlabel("Task", fontsize=12, fontweight='bold')
    ax.set_ylabel("Preference Vector", fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Performance Heatmap: {formatted_metric}", fontsize=14, fontweight="bold")

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

    # Format task names for display
    formatted_task_names = [format_task_name(name) for name in task_names]

    # Create heatmap with diverging colormap
    # Red = positive correlation (synergy), Blue = negative correlation (interference)
    # Note: This is appropriate since we want high correlation to be good (red)
    sns.heatmap(
        interference_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=formatted_task_names,
        yticklabels=formatted_task_names,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_xlabel("Task", fontsize=12, fontweight='bold')
    ax.set_ylabel("Task", fontsize=12, fontweight='bold')

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

    # Format task names for display
    formatted_task_names = [format_task_name(name) for name in task_names]

    x = np.arange(len(task_names))
    width = 0.35

    # Bar chart comparison
    ax1.bar(x - width / 2, requested, width, label="Requested Preference", alpha=0.8, color="steelblue")
    ax1.bar(x + width / 2, achieved, width, label="Achieved (÷ sum)", alpha=0.8, color="coral")
    ax1.set_xlabel("Task", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Relative Weight", fontsize=12, fontweight='bold')
    ax1.set_title("Preference vs. Achieved Performance", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(formatted_task_names, rotation=45, ha="right")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Add explanation text
    ax1.text(0.02, 0.98, "Achieved normalized by sum of all task scores",
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Scatter plot with diagonal
    ax2.scatter(requested, achieved, s=100, alpha=0.7, color="steelblue")
    for i, task in enumerate(formatted_task_names):
        ax2.annotate(task, (requested[i], achieved[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Add perfect alignment diagonal
    max_val = max(requested.max(), achieved.max())
    ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Alignment")

    ax2.set_xlabel("Requested Preference Weight", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Achieved Score (÷ sum)", fontsize=12, fontweight='bold')
    ax2.set_title("Preference Alignment Scatter", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_parallel_coordinates(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> plt.Figure:
    """
    Plot parallel coordinates visualization for comparing preference vectors

    This visualization shows all tasks on parallel vertical axes, with each
    preference vector as a line connecting its performance across tasks.

    Args:
        all_results: List of result dictionaries with preference_vector and task_results
        task_names: List of task names
        metric_name: Metric to visualize
        save_path: Optional path to save figure
        reference_points: Optional reference points from single-task fine-tuned models

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Format task names and metric
    formatted_task_names = [format_task_name(name) for name in task_names]
    formatted_metric = format_metric_name(metric_name)

    num_tasks = len(task_names)
    x_positions = np.arange(num_tasks)

    # Color code by preference vector type
    def categorize_preference(pref_vec):
        """Categorize preference vector as equal, extreme, or balanced"""
        max_val = max(pref_vec)
        if len(set(pref_vec)) == 1:
            return "equal", "#2ca02c"  # Green
        elif max_val >= 0.7:
            return "extreme", "#d62728"  # Red
        else:
            return "balanced", "#1f77b4"  # Blue

    # Plot each preference vector
    for result in all_results:
        pref_vec = result["preference_vector"]
        task_results = result["task_results"]

        # Extract scores
        scores = [task_results[task].metrics[metric_name] for task in task_names]

        # Categorize and get color
        category, color = categorize_preference(pref_vec)

        # Create label
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"

        # Plot line
        ax.plot(x_positions, scores, 'o-', linewidth=2, markersize=8,
                alpha=0.6, color=color, label=pref_label)

    # Plot reference points (single-task fine-tuned models) if provided
    if reference_points:
        for task_name in task_names:
            if task_name in reference_points:
                ref_scores = [reference_points[task_name].get(f"{t}_{metric_name}", 0)
                             for t in task_names]
                ax.plot(x_positions, ref_scores, 's--', linewidth=2, markersize=10,
                        alpha=0.8, color='gold', markeredgecolor='black', markeredgewidth=1.5)

    # Styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(formatted_task_names, fontsize=12, fontweight='bold')
    ax.set_ylabel(formatted_metric, fontsize=13, fontweight='bold')
    ax.set_xlabel("Task", fontsize=13, fontweight='bold')
    ax.set_title(f"Parallel Coordinates: Multi-Task Performance Comparison\n{formatted_metric}",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.grid(axis='x', alpha=0.2)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.6, label='Equal Preference'),
        Patch(facecolor='#d62728', alpha=0.6, label='Extreme Preference (≥0.7)'),
        Patch(facecolor='#1f77b4', alpha=0.6, label='Balanced Preference'),
    ]
    if reference_points:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='s', color='gold', markeredgecolor='black',
                   markeredgewidth=1.5, markersize=10, linestyle='--',
                   label='Fine-tuned Single-Task')
        )

    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_distance_to_utopia(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> plt.Figure:
    """
    Plot distance to utopia point for each preference vector

    Shows which preference vectors achieve solutions closest to the ideal utopia point.

    Args:
        all_results: List of result dictionaries with preference_vector and task_results
        task_names: List of task names
        metric_name: Metric to use
        save_path: Optional path to save figure
        reference_points: Optional reference points to compute utopia point

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Format metric
    formatted_metric = format_metric_name(metric_name)

    # Compute utopia point (max performance on each task)
    if reference_points:
        # Use reference points to get utopia
        utopia_scores = []
        for task in task_names:
            max_score = 0
            for ref_task, ref_metrics in reference_points.items():
                score = ref_metrics.get(f"{task}_{metric_name}", 0)
                max_score = max(max_score, score)
            utopia_scores.append(max_score)
        utopia_point = np.array(utopia_scores)
    else:
        # Use current results to compute utopia
        utopia_point = np.array([
            max(result["task_results"][task].metrics[metric_name] for result in all_results)
            for task in task_names
        ])

    # Calculate distances and extract info
    distances = []
    pref_labels = []
    colors = []

    def categorize_preference(pref_vec):
        """Categorize preference vector"""
        max_val = max(pref_vec)
        if len(set(pref_vec)) == 1:
            return "#2ca02c"  # Green - equal
        elif max_val >= 0.7:
            return "#d62728"  # Red - extreme
        else:
            return "#1f77b4"  # Blue - balanced

    for result in all_results:
        pref_vec = result["preference_vector"]
        task_results = result["task_results"]

        # Extract scores
        scores = np.array([task_results[task].metrics[metric_name] for task in task_names])

        # Calculate Euclidean distance to utopia
        distance = np.linalg.norm(scores - utopia_point)
        distances.append(distance)

        # Create label
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
        pref_labels.append(pref_label)

        # Get color
        colors.append(categorize_preference(pref_vec))

    # Sort by distance (ascending)
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    pref_labels = np.array(pref_labels)[sorted_indices]
    colors = np.array(colors)[sorted_indices]

    # Create bar chart
    y_positions = np.arange(len(distances))
    bars = ax.barh(y_positions, distances, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for i, (bar, dist) in enumerate(zip(bars, distances)):
        ax.text(dist + 0.005, bar.get_y() + bar.get_height()/2,
                f'{dist:.4f}', va='center', fontsize=9, fontweight='bold')

    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pref_labels, fontsize=10)
    ax.set_xlabel(f"Euclidean Distance to Utopia Point", fontsize=12, fontweight='bold')
    ax.set_ylabel("Preference Vector", fontsize=12, fontweight='bold')
    ax.set_title(f"Distance to Utopia Point Analysis\n{formatted_metric} (Lower is Better)",
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.7, edgecolor='black', label='Equal Preference'),
        Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Extreme Preference (≥0.7)'),
        Patch(facecolor='#1f77b4', alpha=0.7, edgecolor='black', label='Balanced Preference'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    # Add annotation about utopia point
    utopia_str = f"Utopia: [{', '.join([f'{x:.3f}' for x in utopia_point])}]"
    ax.text(0.02, 0.98, utopia_str, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        save_plot(fig, save_path)

    return fig


def plot_performance_recovery(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str = "f1_macro",
    save_path: Optional[Path] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> plt.Figure:
    """
    Plot performance recovery analysis comparing achieved vs expected performance

    Tests the hypothesis: Multi-task model should achieve at least
    (preference_weight × single_task_performance) for each task.

    Shows both absolute values and recovery percentages.

    Args:
        all_results: List of result dictionaries with preference_vector and task_results
        task_names: List of task names
        metric_name: Metric to analyze
        save_path: Optional path to save figure
        reference_points: Single-task fine-tuned model performance (required)

    Returns:
        Matplotlib figure
    """
    if not reference_points:
        raise ValueError("reference_points required for performance recovery analysis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Format names
    formatted_task_names = [format_task_name(name) for name in task_names]
    formatted_metric = format_metric_name(metric_name)

    # Get single-task performance for each task (best performance on that task)
    single_task_perf = {}
    for task in task_names:
        max_score = 0
        for ref_task, ref_metrics in reference_points.items():
            score = ref_metrics.get(f"{task}_{metric_name}", 0)
            if score > max_score:
                max_score = score
        single_task_perf[task] = max_score

    # Prepare data for plotting
    num_tasks = len(task_names)
    num_prefs = len(all_results)

    # Colors by preference type
    def categorize_preference(pref_vec):
        max_val = max(pref_vec)
        if len(set(pref_vec)) == 1:
            return "#2ca02c"  # Green - equal
        elif max_val >= 0.7:
            return "#d62728"  # Red - extreme
        else:
            return "#1f77b4"  # Blue - balanced

    # Calculate expected and achieved for each preference vector and task
    for pref_idx, result in enumerate(all_results):
        pref_vec = result["preference_vector"]
        task_results = result["task_results"]
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
        color = categorize_preference(pref_vec)

        x_positions = np.arange(num_tasks) + pref_idx * 0.12

        expected_vals = []
        achieved_vals = []
        recovery_rates = []

        for i, task in enumerate(task_names):
            # Expected: preference_weight × single_task_performance
            expected = pref_vec[i] * single_task_perf[task]
            expected_vals.append(expected)

            # Achieved: actual multi-task performance
            achieved = task_results[task].metrics[metric_name]
            achieved_vals.append(achieved)

            # Recovery rate: achieved / expected (in percentage)
            if expected > 0:
                recovery_rate = (achieved / expected) * 100
            else:
                recovery_rate = 0
            recovery_rates.append(recovery_rate)

        # Plot 1: Absolute values (expected vs achieved)
        width = 0.1
        ax1.bar(x_positions - width/2, expected_vals, width,
                alpha=0.6, color=color, edgecolor='black', linewidth=0.5)
        ax1.bar(x_positions + width/2, achieved_vals, width,
                alpha=0.9, color=color, edgecolor='black', linewidth=1,
                label=pref_label if pref_idx < 7 else None)  # Limit legend entries

        # Plot 2: Recovery percentage
        bars = ax2.bar(x_positions, recovery_rates, width * 2,
                      alpha=0.8, color=color, edgecolor='black', linewidth=1,
                      label=pref_label if pref_idx < 7 else None)

        # Color bars in plot 2 based on performance
        for bar, rate in zip(bars, recovery_rates):
            if rate >= 100:
                bar.set_facecolor('#2ecc71')  # Green for exceeding expectation
                bar.set_alpha(0.8)
            elif rate >= 80:
                bar.set_facecolor('#f39c12')  # Orange for close
                bar.set_alpha(0.7)
            else:
                bar.set_facecolor('#e74c3c')  # Red for underperforming
                bar.set_alpha(0.7)

    # Styling for Plot 1 (Absolute Values)
    ax1.set_xticks(np.arange(num_tasks) + (num_prefs - 1) * 0.12 / 2)
    ax1.set_xticklabels(formatted_task_names, fontsize=11, fontweight='bold')
    ax1.set_ylabel(f'{formatted_metric} (Absolute)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax1.set_title(f'Performance Recovery: Expected vs. Achieved\n{formatted_metric}',
                  fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)

    # Add legend with custom patches
    from matplotlib.patches import Patch
    legend_elements1 = [
        Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Expected (weight × single-task)'),
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', linewidth=2, label='Achieved (multi-task)'),
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', fontsize=9, framealpha=0.9)

    # Add explanation text
    ax1.text(0.02, 0.98,
             'Expected = preference_weight × single_task_performance',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Styling for Plot 2 (Recovery Percentage)
    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='100% Recovery')
    ax2.axhline(y=80, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax2.set_xticks(np.arange(num_tasks) + (num_prefs - 1) * 0.12 / 2)
    ax2.set_xticklabels(formatted_task_names, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Recovery Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax2.set_title(f'Performance Recovery Rate\n{formatted_metric} (Higher is Better)',
                  fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend for recovery rates
    legend_elements2 = [
        Patch(facecolor='#2ecc71', alpha=0.8, edgecolor='black', label='≥100%: Exceeds expectation'),
        Patch(facecolor='#f39c12', alpha=0.7, edgecolor='black', label='80-100%: Close to expectation'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', label='<80%: Underperforming'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=9, framealpha=0.9)

    # Add explanation text
    ax2.text(0.02, 0.02,
             'Recovery % = (achieved / expected) × 100',
             transform=ax2.transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

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
