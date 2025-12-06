"""Visualization generator for benchmark results"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import (
    compute_confusion_matrix_metrics,
    compute_preference_alignment,
    compute_task_interference,
)
from src.visualization.pareto import compute_pareto_frontier_2d, plot_pareto_frontier_2d
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_method_comparison_dashboard,
    plot_multi_radar_chart,
    plot_performance_heatmap,
    plot_preference_alignment,
    plot_radar_chart,
    plot_task_interference_matrix,
)

logger = logging.getLogger(__name__)


def generate_all_visualizations(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str = "f1_macro",
    output_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate all visualizations for benchmark results

    Args:
        all_results: List of result dictionaries, one per preference vector
                    Each dict has: preference_vector, task_results
        task_names: List of task names
        metric_name: Primary metric to use for visualizations
        output_dir: Optional directory to save plots
        method_name: Optional method name for titles
        reference_points: Optional reference points from single-task fine-tuned models

    Returns:
        Dictionary mapping visualization names to matplotlib figures
    """
    logger.info("Generating comprehensive visualizations...")
    figures = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance heatmap (all preference vectors × all tasks)
    try:
        fig = generate_performance_heatmap(
            all_results,
            task_names,
            metric_name,
            output_dir / "performance_heatmap" if output_dir else None,
            method_name,
        )
        figures["performance_heatmap"] = fig
    except Exception as e:
        logger.error(f"Failed to generate performance heatmap: {e}")

    # 2. Radar charts for each preference vector
    try:
        radar_figs = generate_radar_charts(
            all_results,
            task_names,
            metric_name,
            output_dir / "radar_charts" if output_dir else None,
            max_charts=5,  # Limit to avoid too many plots
        )
        figures.update(radar_figs)
    except Exception as e:
        logger.error(f"Failed to generate radar charts: {e}")

    # 3. Task interference matrix
    try:
        fig = generate_task_interference_viz(
            all_results,
            task_names,
            metric_name,
            output_dir / "task_interference" if output_dir else None,
        )
        if fig:
            figures["task_interference_matrix"] = fig
    except Exception as e:
        logger.error(f"Failed to generate task interference visualization: {e}")

    # 4. Pareto frontiers for all task pairs
    try:
        pareto_figs = generate_pareto_frontiers(
            all_results,
            task_names,
            metric_name,
            output_dir / "pareto_frontiers" if output_dir else None,
            method_name,
            reference_points,
        )
        figures.update(pareto_figs)
    except Exception as e:
        logger.error(f"Failed to generate Pareto frontiers: {e}")

    # 5. Preference alignment plots for selected preference vectors
    try:
        alignment_figs = generate_preference_alignment_plots(
            all_results,
            task_names,
            metric_name,
            output_dir / "preference_alignment" if output_dir else None,
            max_plots=3,  # Show only most interesting cases
        )
        figures.update(alignment_figs)
    except Exception as e:
        logger.error(f"Failed to generate preference alignment plots: {e}")

    logger.info(f"Generated {len(figures)} visualizations successfully")
    return figures


def generate_performance_heatmap(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_path: Optional[Path],
    method_name: Optional[str] = None,
) -> plt.Figure:
    """Generate performance heatmap across preference vectors and tasks"""
    # Build performance matrix
    performance_matrix = []
    preference_labels = []

    for result in all_results:
        pref_vec = result["preference_vector"]
        task_results = result["task_results"]

        # Create label for preference vector
        pref_label = f"[{', '.join([f'{x:.2f}' for x in pref_vec])}]"
        preference_labels.append(pref_label)

        # Extract performance for each task
        row = [task_results[task].metrics[metric_name] for task in task_names]
        performance_matrix.append(row)

    performance_matrix = np.array(performance_matrix)

    # Create plot
    title = f"Performance Heatmap: {method_name}" if method_name else None
    fig = plot_performance_heatmap(
        performance_matrix,
        task_names,
        preference_labels,
        metric_name,
        save_path,
        title,
    )

    return fig


def generate_radar_charts(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    max_charts: int = 5,
) -> Dict[str, plt.Figure]:
    """Generate radar charts for selected preference vectors"""
    figures = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Select interesting preference vectors to visualize
    # Include: equal preference, extreme preferences, and a few others
    selected_results = select_representative_preferences(all_results, max_charts)

    for i, result in enumerate(selected_results):
        pref_vec = result["preference_vector"]
        task_results = result["task_results"]

        # Extract scores for each task
        task_scores = {task: task_results[task].metrics[metric_name] for task in task_names}

        # Create label
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"

        # Generate plot
        save_path = save_dir / f"radar_chart_{i}" if save_dir else None
        fig = plot_radar_chart(
            task_scores,
            metric_name,
            save_path,
            title=f"Preference: {pref_label}",
        )

        figures[f"radar_chart_{i}"] = fig

    return figures


def generate_task_interference_viz(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_path: Optional[Path],
) -> Optional[plt.Figure]:
    """Generate task interference correlation matrix"""
    if len(all_results) < 2:
        logger.warning("Need at least 2 preference vectors to compute task interference")
        return None

    # Extract task results across preference vectors
    results_across_prefs = []
    for result in all_results:
        task_metrics = {}
        for task in task_names:
            task_metrics[task] = result["task_results"][task].metrics
        results_across_prefs.append(task_metrics)

    # Compute interference
    interference_dict = compute_task_interference(results_across_prefs, metric_name)

    # Build correlation matrix
    num_tasks = len(task_names)
    interference_matrix = np.eye(num_tasks)  # Diagonal is 1.0 (perfect correlation with self)

    for (task1, task2), corr in interference_dict.items():
        i = task_names.index(task1)
        j = task_names.index(task2)
        interference_matrix[i, j] = corr
        interference_matrix[j, i] = corr  # Symmetric

    # Generate plot
    fig = plot_task_interference_matrix(
        interference_matrix,
        task_names,
        save_path,
    )

    return fig


def generate_pareto_frontiers(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, plt.Figure]:
    """Generate Pareto frontier plots for all task pairs"""
    figures = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    num_tasks = len(task_names)

    # Generate for each task pair
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            task1, task2 = task_names[i], task_names[j]

            # Extract scores for this task pair
            results_dict = {}

            for idx, result in enumerate(all_results):
                pref_vec = result["preference_vector"]
                pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"

                score1 = result["task_results"][task1].metrics[metric_name]
                score2 = result["task_results"][task2].metrics[metric_name]

                results_dict[pref_label] = (score1, score2)

            # Get single-task optimal points from reference points (fine-tuned models)
            single_task_optima = None
            if reference_points:
                from src.benchmarks.reference_points import get_single_task_optima

                single_task_optima_full = get_single_task_optima(reference_points, task_names, metric_name)

                # Extract only the scores for this task pair
                single_task_optima = {}
                for source_task, task_scores in single_task_optima_full.items():
                    # Only include if this source task is one of the two tasks being plotted
                    if source_task == task1 or source_task == task2:
                        score1 = task_scores[task1]
                        score2 = task_scores[task2]
                        single_task_optima[f"{source_task}_optimal"] = (score1, score2)

            # Generate Pareto frontier
            save_path = save_dir / f"pareto_{task1}_vs_{task2}" if save_dir else None
            title = f"Pareto Frontier Analysis: {task1} vs {task2}"
            if method_name:
                title += f" ({method_name})"

            fig = plot_pareto_frontier_2d(
                results_dict,
                (task1, task2),
                save_path=save_path,
                title=title,
                single_task_optima=single_task_optima if single_task_optima else None,
            )

            figures[f"pareto_{task1}_vs_{task2}"] = fig

    return figures


def generate_preference_alignment_plots(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    max_plots: int = 3,
) -> Dict[str, plt.Figure]:
    """Generate preference alignment plots for selected preference vectors"""
    figures = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Select interesting preference vectors
    selected_results = select_representative_preferences(all_results, max_plots)

    for i, result in enumerate(selected_results):
        pref_vec = np.array(result["preference_vector"])
        task_results = result["task_results"]

        # Extract achieved scores
        achieved_scores = {task: task_results[task].metrics[metric_name] for task in task_names}

        # Compute alignment metrics
        alignment_metrics = compute_preference_alignment(pref_vec, achieved_scores, metric_name)

        # Get achieved vector in same order as task_names
        achieved_vec = np.array([achieved_scores[task] for task in task_names])

        # Generate plot
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
        title = f"Preference Alignment: {pref_label}\n"
        title += f"Cosine Similarity: {alignment_metrics['cosine_similarity']:.3f}"

        save_path = save_dir / f"alignment_{i}" if save_dir else None
        fig = plot_preference_alignment(
            pref_vec,
            achieved_vec / achieved_vec.sum(),  # Normalize
            task_names,
            save_path,
            title,
        )

        figures[f"preference_alignment_{i}"] = fig

    return figures


def select_representative_preferences(
    all_results: List[Dict],
    max_count: int,
) -> List[Dict]:
    """
    Select representative preference vectors for visualization

    Prioritizes:
    1. Equal preference (if exists)
    2. Extreme preferences (single task focused)
    3. Random selection of others

    Args:
        all_results: List of all result dictionaries
        max_count: Maximum number to select

    Returns:
        List of selected result dictionaries
    """
    if len(all_results) <= max_count:
        return all_results

    selected = []

    # Try to find equal preference
    for result in all_results:
        pref_vec = result["preference_vector"]
        if len(set(pref_vec)) == 1:  # All elements are equal
            selected.append(result)
            break

    # Try to find extreme preferences (one task gets >= 0.7)
    for result in all_results:
        if len(selected) >= max_count:
            break

        pref_vec = result["preference_vector"]
        if max(pref_vec) >= 0.7:
            if result not in selected:
                selected.append(result)

    # Fill remaining slots with random selection
    remaining = [r for r in all_results if r not in selected]
    import random

    random.shuffle(remaining)
    while len(selected) < max_count and remaining:
        selected.append(remaining.pop(0))

    return selected
