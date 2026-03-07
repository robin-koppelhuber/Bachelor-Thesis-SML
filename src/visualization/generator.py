"""Visualization generator for benchmark results"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import ListConfig

from src.evaluation.metrics import (
    compute_preference_alignment,
    compute_task_interference,
)
from src.visualization.pareto import plot_normalized_pareto_frontier_2d, plot_pareto_frontier_2d
from src.visualization.plots import (
    format_metric_name,
    format_task_name,
    plot_distance_to_utopia,
    plot_normalized_skill_retention,
    plot_parallel_coordinates,
    plot_performance_heatmap,
    plot_performance_recovery,
    plot_preference_alignment,
    plot_radar_chart,
    plot_task_interference_matrix,
)

logger = logging.getLogger(__name__)

# All supported single-metric plot types
AVAILABLE_PLOT_TYPES: FrozenSet[str] = frozenset(
    {
        "heatmap",
        "parallel_coordinates",
        "distance_to_utopia",
        "performance_recovery",
        "normalized_skill_retention",
        "normalized_pareto_frontiers",
        "radar_charts",
        "task_interference",
        "preference_alignment",
        "pareto_frontiers",
    }
)


def resolve_plot_types(plots_spec: Any) -> FrozenSet[str]:
    """
    Resolve a plot specification to a frozenset of plot type names.

    Args:
        plots_spec: One of:
            - "all"                     → all available plot types
            - ["all"]                   → all available plot types
            - ["heatmap", "pareto_frontiers", ...]  → specific types
            - []                        → no plots

    Returns:
        FrozenSet of plot type names to generate
    """
    if plots_spec == "all":
        return AVAILABLE_PLOT_TYPES
    if isinstance(plots_spec, (list, ListConfig)):
        if "all" in plots_spec:
            return AVAILABLE_PLOT_TYPES
        return frozenset(plots_spec)
    return frozenset()


def generate_all_visualizations(
    all_results: List[Dict],
    task_names: List[str],
    metrics_config: List[Any],
    output_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
    cross_metric_plots_config: Optional[List[Any]] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate all visualizations for benchmark results.

    Args:
        all_results: List of result dicts, one per preference vector.
                     Each dict: {preference_vector, task_results}
        task_names: List of task names
        metrics_config: List of metric configs from the benchmark config.
                        Each item has .name (str) and .plots (str | list).
        output_dir: Optional directory to save plots
        method_name: Optional method name for titles
        reference_points: Optional reference points from single-task fine-tuned models
        cross_metric_plots_config: Optional list of cross-metric plot specs (extensible stub)

    Returns:
        Dictionary mapping visualization names to matplotlib figures
    """
    logger.info("Generating comprehensive visualizations...")
    figures: Dict[str, plt.Figure] = {}

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating visualizations for {len(metrics_config)} metrics")

    for metric_cfg in metrics_config:
        metric_name: str = metric_cfg.name
        plot_types = resolve_plot_types(metric_cfg.get("plots", []))

        if not plot_types:
            logger.debug(f"No plots requested for metric: {metric_name}")
            continue

        logger.info(f"Generating plots for metric '{metric_name}': {sorted(plot_types)}")

        if "heatmap" in plot_types:
            try:
                save_path = output_dir / f"heatmap_{metric_name}" if output_dir else None
                fig = _generate_performance_heatmap(all_results, task_names, metric_name, save_path, method_name)
                figures[f"heatmap_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate heatmap for {metric_name}: {e}")

        if "parallel_coordinates" in plot_types:
            try:
                save_path = output_dir / f"parallel_coordinates_{metric_name}" if output_dir else None
                fig = plot_parallel_coordinates(all_results, task_names, metric_name, save_path, reference_points)
                figures[f"parallel_coordinates_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate parallel_coordinates for {metric_name}: {e}")

        if "distance_to_utopia" in plot_types:
            try:
                save_path = output_dir / f"distance_to_utopia_{metric_name}" if output_dir else None
                fig = plot_distance_to_utopia(all_results, task_names, metric_name, save_path, reference_points)
                figures[f"distance_to_utopia_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate distance_to_utopia for {metric_name}: {e}")

        if "performance_recovery" in plot_types:
            try:
                save_path = output_dir / f"performance_recovery_{metric_name}" if output_dir else None
                fig = plot_performance_recovery(all_results, task_names, metric_name, save_path, reference_points)
                figures[f"performance_recovery_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate performance_recovery for {metric_name}: {e}")

        if "radar_charts" in plot_types:
            try:
                save_dir = output_dir / f"radar_charts_{metric_name}" if output_dir else None
                radar_figs = _generate_radar_charts(all_results, task_names, metric_name, save_dir, max_charts=5)
                figures.update(radar_figs)
            except Exception as e:
                logger.error(f"Failed to generate radar_charts for {metric_name}: {e}")

        if "task_interference" in plot_types:
            try:
                save_path = output_dir / f"task_interference_{metric_name}" if output_dir else None
                fig = _generate_task_interference_viz(all_results, task_names, metric_name, save_path)
                if fig:
                    figures[f"task_interference_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate task_interference for {metric_name}: {e}")

        if "preference_alignment" in plot_types:
            try:
                save_dir = output_dir / f"preference_alignment_{metric_name}" if output_dir else None
                alignment_figs = _generate_preference_alignment_plots(
                    all_results, task_names, metric_name, save_dir, max_plots=3
                )
                figures.update(alignment_figs)
            except Exception as e:
                logger.error(f"Failed to generate preference_alignment for {metric_name}: {e}")

        if "pareto_frontiers" in plot_types:
            try:
                save_dir = output_dir / f"pareto_frontiers_{metric_name}" if output_dir else None
                pareto_figs = _generate_pareto_frontiers(
                    all_results, task_names, metric_name, save_dir, method_name, reference_points
                )
                figures.update(pareto_figs)
            except Exception as e:
                logger.error(f"Failed to generate pareto_frontiers for {metric_name}: {e}")

        if "normalized_skill_retention" in plot_types:
            try:
                save_path = output_dir / f"normalized_skill_retention_{metric_name}" if output_dir else None
                fig = plot_normalized_skill_retention(all_results, task_names, metric_name, save_path, reference_points)
                figures[f"normalized_skill_retention_{metric_name}"] = fig
            except Exception as e:
                logger.error(f"Failed to generate normalized_skill_retention for {metric_name}: {e}")

        if "normalized_pareto_frontiers" in plot_types:
            try:
                save_dir = output_dir / f"normalized_pareto_frontiers_{metric_name}" if output_dir else None
                norm_pareto_figs = _generate_normalized_pareto_frontiers(
                    all_results, task_names, metric_name, save_dir, method_name, reference_points
                )
                figures.update(norm_pareto_figs)
            except Exception as e:
                logger.error(f"Failed to generate normalized_pareto_frontiers for {metric_name}: {e}")

    # Cross-metric plots
    cross_figures = _generate_cross_metric_visualizations(
        all_results,
        task_names,
        metrics_config,
        cross_metric_plots_config,
        output_dir,
        reference_points,
    )
    figures.update(cross_figures)

    logger.info(f"Generated {len(figures)} visualizations successfully")
    return figures


def _generate_cross_metric_visualizations(
    all_results: List[Dict],
    task_names: List[str],
    metrics_config: List[Any],
    cross_metric_plots_config: Optional[List[Any]],
    output_dir: Optional[Path],
    reference_points: Optional[Dict],
) -> Dict[str, plt.Figure]:
    """
    Generate cross-metric plots (plots involving multiple metrics simultaneously).

    Currently a stub: no built-in plot types. Future types can be added by implementing
    a handler here and adding the type name to the config.
    """
    figures: Dict[str, plt.Figure] = {}
    if not cross_metric_plots_config:
        return figures

    for plot_cfg in cross_metric_plots_config:
        plot_type = plot_cfg.get("type", "")
        logger.warning(f"Unknown or unimplemented cross-metric plot type: '{plot_type}'. Skipping.")

    return figures


def _generate_performance_heatmap(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_path: Optional[Path],
    method_name: Optional[str] = None,
) -> plt.Figure:
    """Extract data and call the dumb heatmap plotting function."""
    performance_matrix = []
    preference_labels = []

    for result in all_results:
        pref_vec = result["preference_vector"]
        pref_label = f"[{', '.join([f'{x:.2f}' for x in pref_vec])}]"
        preference_labels.append(pref_label)
        row = [result["task_results"][task].metrics[metric_name] for task in task_names]
        performance_matrix.append(row)

    performance_matrix_np = np.array(performance_matrix)
    title = f"Performance Heatmap: {method_name}" if method_name else None

    return plot_performance_heatmap(
        performance_matrix_np,
        task_names,
        preference_labels,
        metric_name,
        save_path,
        title,
    )


def _generate_radar_charts(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    max_charts: int = 5,
) -> Dict[str, plt.Figure]:
    """Extract data and generate radar charts for representative preference vectors."""
    figures: Dict[str, plt.Figure] = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    selected_results = _select_representative_preferences(all_results, max_charts)

    for i, result in enumerate(selected_results):
        pref_vec = result["preference_vector"]
        task_scores = {task: result["task_results"][task].metrics[metric_name] for task in task_names}
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
        save_path = save_dir / f"radar_chart_{metric_name}_{i}" if save_dir else None
        fig = plot_radar_chart(
            task_scores,
            metric_name,
            save_path,
            title=f"Preference: {pref_label}",
        )
        figures[f"radar_chart_{metric_name}_{i}"] = fig

    return figures


def _generate_task_interference_viz(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_path: Optional[Path],
) -> Optional[plt.Figure]:
    """Extract data, compute interference metric, and generate correlation matrix plot."""
    if len(all_results) < 2:
        logger.warning("Need at least 2 preference vectors to compute task interference")
        return None

    results_across_prefs = [
        {task: result["task_results"][task].metrics for task in task_names} for result in all_results
    ]

    interference_dict = compute_task_interference(results_across_prefs, metric_name)

    num_tasks = len(task_names)
    interference_matrix = np.eye(num_tasks)
    for (task1, task2), corr in interference_dict.items():
        i = task_names.index(task1)
        j = task_names.index(task2)
        interference_matrix[i, j] = corr
        interference_matrix[j, i] = corr

    return plot_task_interference_matrix(interference_matrix, task_names, save_path)


def _generate_pareto_frontiers(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, plt.Figure]:
    """Extract data and generate Pareto frontier plots for all task pairs."""
    figures: Dict[str, plt.Figure] = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    num_tasks = len(task_names)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            task1, task2 = task_names[i], task_names[j]

            results_dict = {}
            for result in all_results:
                pref_vec = result["preference_vector"]
                pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
                score1 = result["task_results"][task1].metrics[metric_name]
                score2 = result["task_results"][task2].metrics[metric_name]
                results_dict[pref_label] = (score1, score2)

            single_task_optima = None
            if reference_points:
                from src.benchmarks.reference_points import get_single_task_optima

                optima_full = get_single_task_optima(reference_points, task_names, metric_name)
                single_task_optima = {
                    f"{src}_optimal": (scores[task1], scores[task2])
                    for src, scores in optima_full.items()
                    if src in (task1, task2)
                }

            save_path = save_dir / f"pareto_{task1}_vs_{task2}" if save_dir else None
            title = f"Pareto Frontier: {task1} vs {task2}"
            if method_name:
                title += f" ({method_name})"

            fig = plot_pareto_frontier_2d(
                results_dict,
                (task1, task2),
                save_path=save_path,
                title=title,
                single_task_optima=single_task_optima or None,
            )
            figures[f"pareto_{task1}_vs_{task2}_{metric_name}"] = fig

    return figures


def _generate_normalized_pareto_frontiers(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, plt.Figure]:
    """Generate normalized Pareto frontier plots (kappa-ratio axes) for all task pairs.

    Requires cohen_kappa to be present in both task_results and reference_points.
    Skips silently if cohen_kappa is unavailable.
    """
    figures: Dict[str, plt.Figure] = {}

    if not reference_points:
        logger.warning("reference_points required for normalized_pareto_frontiers; skipping.")
        return figures

    # Check cohen_kappa availability
    sample_metrics = all_results[0]["task_results"][task_names[0]].metrics
    if "cohen_kappa" not in sample_metrics:
        logger.warning(
            "cohen_kappa not found in task results; skipping normalized_pareto_frontiers. "
            "Add 'cohen_kappa' to the benchmark metrics config."
        )
        return figures

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    from src.benchmarks.reference_points import get_diagonal_optima, get_single_task_optima

    diagonal_kappas = get_diagonal_optima(reference_points, task_names, "cohen_kappa")

    num_tasks = len(task_names)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            task1, task2 = task_names[i], task_names[j]

            # Collect kappa scores for each preference vector
            results_dict = {}
            for result in all_results:
                pref_vec = result["preference_vector"]
                pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
                kappa1 = result["task_results"][task1].metrics["cohen_kappa"]
                kappa2 = result["task_results"][task2].metrics["cohen_kappa"]
                results_dict[pref_label] = (kappa1, kappa2)

            # Single-task optima kappas (off-diagonal: ft model for one task, evaluated on both)
            optima_full = get_single_task_optima(reference_points, task_names, "cohen_kappa")
            single_task_optima = {
                f"{src}_optimal": (scores[task1], scores[task2])
                for src, scores in optima_full.items()
                if src in (task1, task2)
            }

            save_path = save_dir / f"norm_pareto_{task1}_vs_{task2}" if save_dir else None
            title = f"Normalized Pareto Frontier: {task1} vs {task2}"
            if method_name:
                title += f" ({method_name})"

            fig = plot_normalized_pareto_frontier_2d(
                results=results_dict,
                task_names=(task1, task2),
                diagonal_kappas=diagonal_kappas,
                save_path=save_path,
                title=title,
                single_task_optima=single_task_optima or None,
            )
            figures[f"norm_pareto_{task1}_vs_{task2}_{metric_name}"] = fig

    return figures


def _generate_preference_alignment_plots(
    all_results: List[Dict],
    task_names: List[str],
    metric_name: str,
    save_dir: Optional[Path],
    max_plots: int = 3,
) -> Dict[str, plt.Figure]:
    """Extract data and generate preference alignment plots for representative vectors."""
    figures: Dict[str, plt.Figure] = {}

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    selected_results = _select_representative_preferences(all_results, max_plots)

    for i, result in enumerate(selected_results):
        pref_vec = np.array(result["preference_vector"])
        task_results = result["task_results"]
        achieved_scores = {task: task_results[task].metrics[metric_name] for task in task_names}
        alignment_metrics = compute_preference_alignment(pref_vec, achieved_scores, metric_name)

        achieved_vec = np.array([achieved_scores[task] for task in task_names])
        pref_label = f"[{', '.join([f'{x:.1f}' for x in pref_vec])}]"
        title = f"Preference Alignment: {pref_label}\n"
        title += f"Cosine Similarity: {alignment_metrics['cosine_similarity']:.3f}"

        save_path = save_dir / f"alignment_{metric_name}_{i}" if save_dir else None
        fig = plot_preference_alignment(
            pref_vec,
            achieved_vec / achieved_vec.sum(),
            task_names,
            save_path,
            title,
        )
        figures[f"preference_alignment_{metric_name}_{i}"] = fig

    return figures


def _select_representative_preferences(
    all_results: List[Dict],
    max_count: int,
) -> List[Dict]:
    """
    Select representative preference vectors for visualization. May result in some preference vectors not being visualized

    Priority: equal preference → extreme preferences (max ≥ 0.7) → random fill.
    """
    if len(all_results) <= max_count:
        return all_results

    selected = []

    for result in all_results:
        if len(set(result["preference_vector"])) == 1:
            selected.append(result)
            break

    for result in all_results:
        if len(selected) >= max_count:
            break
        if max(result["preference_vector"]) >= 0.7 and result not in selected:
            selected.append(result)

    remaining = [r for r in all_results if r not in selected]
    import random

    random.shuffle(remaining)
    while len(selected) < max_count and remaining:
        selected.append(remaining.pop(0))

    return selected


def export_results_table(
    all_results: List[Dict],
    task_names: List[str],
    metrics: List[str],
    output_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    reference_points: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """
    Export comprehensive results table to JSON and CSV.

    Creates a table with all metrics for all preference vectors and tasks,
    including reference points from single-task fine-tuned models.

    Args:
        all_results: List of result dictionaries
        task_names: List of task names
        metrics: List of metric name strings
        output_dir: Optional directory to save exports
        method_name: Optional method name for labeling
        reference_points: Optional reference points from single-task models

    Returns:
        Dictionary containing the comprehensive results table
    """
    logger.info("Exporting comprehensive results table...")

    def to_python(obj):
        """Convert OmegaConf objects to regular Python types"""
        if hasattr(obj, "__iter__") and not isinstance(obj, str):
            if hasattr(obj, "items"):
                return {k: to_python(v) for k, v in obj.items()}
            else:
                return [to_python(item) for item in obj]
        return obj

    task_names = to_python(task_names)
    metrics = to_python(metrics)

    rows = []

    for result in all_results:
        pref_vec = to_python(result["preference_vector"])
        pref_label = f"[{', '.join([f'{x:.2f}' for x in pref_vec])}]"

        for task in task_names:
            task_result = result["task_results"][task]
            row = {
                "preference_vector": pref_label,
                "preference_raw": list(pref_vec),
                "task": str(task),
                "task_formatted": format_task_name(task),
                "source": str(method_name) if method_name else "merged",
            }
            for metric in metrics:
                if metric in task_result.metrics:
                    metric_value = task_result.metrics[metric]
                    row[metric] = float(metric_value) if metric_value is not None else None
                    row[f"{metric}_formatted"] = format_metric_name(metric)
            rows.append(row)

    if reference_points:
        for source_task, ref_metrics in reference_points.items():
            for task in task_names:
                row = {
                    "preference_vector": f"Fine-tuned: {format_task_name(source_task)}",
                    "preference_raw": None,
                    "task": str(task),
                    "task_formatted": format_task_name(task),
                    "source": f"finetuned_{source_task}",
                }
                for metric in metrics:
                    metric_key = f"{task}_{metric}"
                    if metric_key in ref_metrics:
                        metric_value = ref_metrics[metric_key]
                        row[metric] = float(metric_value) if metric_value is not None else None
                        row[f"{metric}_formatted"] = format_metric_name(metric)
                    else:
                        row[metric] = None
                rows.append(row)

    df = pd.DataFrame(rows)

    summary_stats = {}
    method_name_str = str(method_name) if method_name else "merged"
    merged_results = df[
        df["source"].str.startswith("merged") | df["source"].str.startswith("train") | (df["source"] == method_name_str)
    ]
    if not merged_results.empty:
        for task in task_names:
            task_data = merged_results[merged_results["task"] == task]
            summary_stats[f"{task}_avg"] = {
                metric: float(task_data[metric].mean()) if metric in task_data.columns else None for metric in metrics
            }

    best_per_task = {}
    for task in task_names:
        task_data = merged_results[merged_results["task"] == task]
        best_per_task[task] = {}
        for metric in metrics:
            if metric in task_data.columns and not task_data[metric].isna().all():
                best_idx = task_data[metric].idxmax()
                best_row = task_data.loc[best_idx]
                best_per_task[task][metric] = {
                    "preference_vector": best_row["preference_vector"],
                    "value": float(best_row[metric]),
                }

    utopia_distances = {}
    for result in all_results:
        pref_vec = to_python(result["preference_vector"])
        pref_label = f"[{', '.join([f'{x:.2f}' for x in pref_vec])}]"

        for metric in metrics:
            if reference_points:
                utopia_scores = []
                for task in task_names:
                    max_score = max(ref_metrics.get(f"{task}_{metric}", 0) for ref_metrics in reference_points.values())
                    utopia_scores.append(max_score)
                utopia_point = np.array(utopia_scores)
            else:
                utopia_point = np.array(
                    [max(r["task_results"][task].metrics[metric] for r in all_results) for task in task_names]
                )

            scores = np.array([result["task_results"][task].metrics[metric] for task in task_names])
            distance = float(np.linalg.norm(scores - utopia_point))

            if pref_label not in utopia_distances:
                utopia_distances[pref_label] = {}
            utopia_distances[pref_label][metric] = {
                "distance": distance,
                "utopia_point": utopia_point.tolist(),
            }

    export_data = {
        "method": str(method_name) if method_name else None,
        "tasks": list(task_names),
        "metrics": list(metrics),
        "results": rows,
        "summary_statistics": summary_stats,
        "best_per_task": best_per_task,
        "utopia_distances": utopia_distances,
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "comprehensive_results.json"
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=2)
        logger.info(f"Saved JSON results to: {json_path}")

        csv_path = output_dir / "comprehensive_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to: {csv_path}")

        if summary_stats:
            summary_df = pd.DataFrame(summary_stats).T
            summary_csv_path = output_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_csv_path)
            logger.info(f"Saved summary statistics to: {summary_csv_path}")

        if utopia_distances:
            dist_rows = []
            for pref_label, metric_dists in utopia_distances.items():
                for metric, dist_data in metric_dists.items():
                    dist_rows.append(
                        {
                            "preference_vector": pref_label,
                            "metric": metric,
                            "distance": dist_data["distance"],
                            "utopia_point": str(dist_data["utopia_point"]),
                        }
                    )
            dist_df = pd.DataFrame(dist_rows)
            dist_csv_path = output_dir / "utopia_distances.csv"
            dist_df.to_csv(dist_csv_path, index=False)
            logger.info(f"Saved utopia distances to: {dist_csv_path}")

    logger.info("Results export completed successfully")
    return export_data
