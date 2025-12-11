"""Pareto frontier visualization"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    single_task_optima: Optional[Dict[str, Tuple[float, float]]] = None,
) -> plt.Figure:
    """
    Plot 2D Pareto frontier with reference points and hypervolume visualization

    Args:
        results: Dictionary mapping config names to (task1_score, task2_score)
        task_names: Names of the two tasks
        method_groups: Optional grouping of configs by method
        save_path: Optional path to save figure
        title: Optional title
        single_task_optima: Optional dict with single-task optimal points from fine-tuned models
                           e.g., {"task1_optimal": (0.95, 0.60), "task2_optimal": (0.65, 0.92)}

    Returns:
        Matplotlib figure
    """
    # Convert to array
    points = np.array(list(results.values()))
    config_names = list(results.keys())

    # Compute Pareto frontier
    pareto_points, pareto_indices = compute_pareto_frontier_2d(points)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Compute reference points
    # If single-task optima are provided, use them to compute utopia point
    # Otherwise fall back to max from current results
    if single_task_optima and len(single_task_optima) > 0:
        # Extract all points including single-task optima
        all_points = list(results.values()) + list(single_task_optima.values())
        all_points_array = np.array(all_points)
        utopia_point = np.array([np.max(all_points_array[:, 0]), np.max(all_points_array[:, 1])])
    else:
        utopia_point = np.array([np.max(points[:, 0]), np.max(points[:, 1])])

    nadir_point = np.array([np.min(points[:, 0]), np.min(points[:, 1])])

    # # COMMENTED OUT: Visualize hypervolume (area dominated by Pareto front)
    # # Draw filled region from nadir point to Pareto front
    # if len(pareto_points) > 0:
    #     # Create polygon for hypervolume visualization
    #     hypervolume_x = [nadir_point[0]]
    #     hypervolume_y = [nadir_point[1]]

    #     for point in pareto_points:
    #         hypervolume_x.append(point[0])
    #         hypervolume_y.append(nadir_point[1])
    #         hypervolume_x.append(point[0])
    #         hypervolume_y.append(point[1])

    #     # Close the polygon
    #     hypervolume_x.append(pareto_points[-1][0])
    #     hypervolume_y.append(nadir_point[1])
    #     hypervolume_x.append(nadir_point[0])
    #     hypervolume_y.append(nadir_point[1])

    #     ax.fill(hypervolume_x, hypervolume_y, alpha=0.15, color='green',
    #             label='Hypervolume', zorder=1)

    # Plot all points
    non_pareto_mask = np.ones(len(points), dtype=bool)
    non_pareto_mask[pareto_indices] = False

    if np.any(non_pareto_mask):
        ax.scatter(
            points[non_pareto_mask, 0],
            points[non_pareto_mask, 1],
            s=60,
            alpha=0.4,
            color='gray',
            label="Dominated Solutions",
            zorder=3,
        )

    # Plot Pareto frontier line
    ax.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "r--",
        linewidth=2.5,
        label="Pareto Frontier",
        zorder=5,
    )

    # Plot Pareto optimal points (we'll label them selectively below)
    pareto_scatter = ax.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        s=150,
        c="red",
        marker="*",
        edgecolors="darkred",
        linewidths=1.5,
        label="Pareto Optimal",
        zorder=6,
    )

    # Store pareto point info for labeling
    pareto_config_names = [config_names[idx] for idx in pareto_indices]

    # Plot utopia point (ideal but usually unachievable)
    ax.scatter(
        utopia_point[0],
        utopia_point[1],
        s=200,
        c="gold",
        marker="D",
        edgecolors="black",
        linewidths=2,
        label="Utopia Point",
        zorder=7,
    )

    # # COMMENTED OUT: Plot nadir point (worst on Pareto front)
    # ax.scatter(
    #     nadir_point[0],
    #     nadir_point[1],
    #     s=100,
    #     c="purple",
    #     marker="s",
    #     edgecolors="black",
    #     linewidths=1.5,
    #     label="Nadir Point",
    #     zorder=7,
    # )

    # Plot single-task optimal points if provided
    if single_task_optima:
        colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Distinct colors
        markers_list = ["^", "v", "<", ">"]  # Different triangle orientations

        for i, (opt_name, opt_point) in enumerate(single_task_optima.items()):
            marker = markers_list[i % len(markers_list)]
            color = colors_list[i % len(colors_list)]

            # Create cleaner label: "Fine-tuned: AG News" instead of "Ag News Optimal"
            task_name = opt_name.replace('_optimal', '').replace('_', ' ')
            # Capitalize properly: ag_news -> AG News, imdb -> IMDB, mnli -> MNLI, mrpc -> MRPC
            task_name_parts = task_name.split()
            formatted_name = ' '.join([part.upper() if len(part) <= 4 else part.title() for part in task_name_parts])
            label = f"Fine-tuned: {formatted_name}"

            ax.scatter(
                opt_point[0],
                opt_point[1],
                s=150,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=1.5,
                label=label,
                zorder=7,
            )

            # Draw line from single-task optimum to utopia point
            ax.plot(
                [opt_point[0], utopia_point[0]],
                [opt_point[1], utopia_point[1]],
                "--",
                alpha=0.3,
                color=color,
                linewidth=1,
                zorder=2,
            )

    # Find and annotate solution closest to utopia point
    distances_to_utopia = np.linalg.norm(pareto_points - utopia_point, axis=1)
    closest_idx = np.argmin(distances_to_utopia)
    closest_point = pareto_points[closest_idx]

    # Smart labeling: Label only key points to avoid clutter
    # 1. Closest to utopia (always)
    # 2. Extreme points (best on each axis)
    # 3. Optionally: corner points of Pareto front

    labeled_points = set()

    # Label closest to utopia
    closest_config = pareto_config_names[closest_idx]
    ax.annotate(
        f"Closest to Utopia\n{closest_config}\n({closest_point[0]:.3f}, {closest_point[1]:.3f})",
        xy=closest_point,
        xytext=(closest_point[0] + 0.05, closest_point[1] - 0.08),
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", lw=1.5),
        zorder=8,
    )
    labeled_points.add(closest_idx)

    # Label extreme points (best on each axis) if different from closest
    best_x_idx = np.argmax(pareto_points[:, 0])
    best_y_idx = np.argmax(pareto_points[:, 1])

    if best_x_idx not in labeled_points and len(pareto_points) > 1:
        point = pareto_points[best_x_idx]
        config = pareto_config_names[best_x_idx]
        ax.annotate(
            f"Best {format_task_name(task_names[0])}\n{config}",
            xy=point,
            xytext=(point[0] - 0.05, point[1] + 0.05),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.6),
            arrowprops=dict(arrowstyle="->", lw=1),
            zorder=8,
        )
        labeled_points.add(best_x_idx)

    if best_y_idx not in labeled_points and len(pareto_points) > 1:
        point = pareto_points[best_y_idx]
        config = pareto_config_names[best_y_idx]
        ax.annotate(
            f"Best {format_task_name(task_names[1])}\n{config}",
            xy=point,
            xytext=(point[0] + 0.05, point[1] - 0.05),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.6),
            arrowprops=dict(arrowstyle="->", lw=1),
            zorder=8,
        )
        labeled_points.add(best_y_idx)

    # Add a note about unlabeled points
    if len(pareto_points) > len(labeled_points):
        unlabeled_count = len(pareto_points) - len(labeled_points)
        ax.text(
            0.98, 0.02, f"+ {unlabeled_count} other Pareto-optimal solution(s)",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='right',
            style='italic',
            color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray')
        )

    # Labels and styling
    # Format task names properly (ag_news -> AG News, imdb -> IMDB, etc.)
    def format_task_name(name: str) -> str:
        parts = name.replace('_', ' ').split()
        return ' '.join([part.upper() if len(part) <= 4 else part.title() for part in parts])

    formatted_task1 = format_task_name(task_names[0])
    formatted_task2 = format_task_name(task_names[1])

    ax.set_xlabel(f"{formatted_task1} Performance", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"{formatted_task2} Performance", fontsize=12, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Pareto Frontier Analysis: {formatted_task1} vs {formatted_task2}",
                    fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(max(0, nadir_point[0] - 0.05), min(1.0, utopia_point[0] + 0.05))
    ax.set_ylim(max(0, nadir_point[1] - 0.05), min(1.0, utopia_point[1] + 0.05))

    # Add text box with statistics
    hypervolume = _compute_hypervolume_2d(pareto_points, nadir_point)
    n_pareto = len(pareto_points)
    stats_text = f"Pareto Solutions: {n_pareto}\nHypervolume: {hypervolume:.3f}"
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    if save_path:
        from src.visualization.plots import save_plot
        save_plot(fig, save_path)

    return fig


def _compute_hypervolume_2d(pareto_points: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute 2D hypervolume indicator

    Args:
        pareto_points: Pareto optimal points (N, 2)
        reference_point: Reference point (worst case)

    Returns:
        Hypervolume value
    """
    if len(pareto_points) == 0:
        return 0.0

    # Sort by first objective (descending for maximization)
    sorted_points = pareto_points[np.argsort(-pareto_points[:, 0])]

    hypervolume = 0.0
    prev_x = reference_point[0]

    for point in sorted_points:
        width = point[0] - prev_x
        height = point[1] - reference_point[1]
        if width > 0 and height > 0:
            hypervolume += width * height
        prev_x = point[0]

    return hypervolume
