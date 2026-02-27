"""Unit tests for src/visualization/plots.py (dumb plotting functions)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.visualization.plots import (
    format_metric_name,
    format_task_name,
    plot_distance_to_utopia,
    plot_parallel_coordinates,
    plot_performance_heatmap,
    plot_performance_recovery,
    plot_radar_chart,
    plot_task_interference_matrix,
)

output_dir = Path(__file__).resolve().parent / "test_output"

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ag_news", "AG NEWS"),  # both "ag" (2) and "news" (4) have len ≤ 4 → fully uppercased
        ("imdb", "IMDB"),
        ("mnli", "MNLI"),
        ("mrpc", "MRPC"),
        ("cola", "COLA"),
        ("qnli", "QNLI"),
        ("sst2", "SST2"),
    ],
)
def test_format_task_name(name, expected):
    assert format_task_name(name) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 Macro"),
        ("f1_weighted", "F1 Weighted"),
        ("precision_macro", "Precision Macro"),
        ("mcc", "Mcc"),
        ("cohen_kappa", "Cohen Kappa"),
    ],
)
def test_format_metric_name(name, expected):
    assert format_metric_name(name) == expected


# ---------------------------------------------------------------------------
# plot_performance_heatmap
# ---------------------------------------------------------------------------


def test_plot_performance_heatmap_returns_figure(performance_matrix, task_names):
    pref_labels = ["[0.25, 0.25, 0.25, 0.25]", "[0.70, 0.10, 0.10, 0.10]"]
    fig = plot_performance_heatmap(
        performance_matrix, task_names, pref_labels, "accuracy", save_path=output_dir / "heatmap_test"
    )
    assert isinstance(fig, plt.Figure)


def test_plot_performance_heatmap_with_title(performance_matrix, task_names):
    pref_labels = ["equal", "focused"]
    fig = plot_performance_heatmap(
        performance_matrix,
        task_names,
        pref_labels,
        "f1_macro",
        save_path=output_dir / "heatmap_title_test",
        title="Test Title",
    )
    assert isinstance(fig, plt.Figure)
    assert "Test Title" in fig.axes[0].get_title()


# ---------------------------------------------------------------------------
# plot_radar_chart
# ---------------------------------------------------------------------------


def test_plot_radar_chart_returns_figure(task_names):
    task_scores = {"ag_news": 0.69, "imdb": 0.52, "mnli": 0.51, "mrpc": 0.63}
    fig = plot_radar_chart(task_scores, "accuracy", output_dir / "radar_charts_test")
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_parallel_coordinates
# ---------------------------------------------------------------------------


def test_plot_parallel_coordinates_returns_figure(all_results, task_names):
    fig = plot_parallel_coordinates(all_results, task_names, "accuracy", save_path=output_dir / "parallel_coords_test")
    assert isinstance(fig, plt.Figure)


def test_plot_parallel_coordinates_with_reference_points(all_results, task_names, reference_points):
    fig = plot_parallel_coordinates(
        all_results,
        task_names,
        "accuracy",
        save_path=output_dir / "parallel_coords_ref_test",
        reference_points=reference_points,
    )
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_distance_to_utopia
# ---------------------------------------------------------------------------


def test_plot_distance_to_utopia_no_reference_points(all_results, task_names):
    fig = plot_distance_to_utopia(all_results, task_names, "accuracy", save_path=output_dir / "utopia_test")
    assert isinstance(fig, plt.Figure)


def test_plot_distance_to_utopia_with_reference_points(all_results, task_names, reference_points):
    fig = plot_distance_to_utopia(
        all_results, task_names, "accuracy", save_path=output_dir / "utopia_ref_test", reference_points=reference_points
    )
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_performance_recovery
# ---------------------------------------------------------------------------


def test_plot_performance_recovery_requires_reference_points(all_results, task_names):
    with pytest.raises(ValueError):
        plot_performance_recovery(all_results, task_names, "accuracy", reference_points=None)


def test_plot_performance_recovery_returns_figure(all_results, task_names, reference_points):
    fig = plot_performance_recovery(
        all_results,
        task_names,
        "accuracy",
        save_path=output_dir / "perf_recovery_test",
        reference_points=reference_points,
    )
    assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_task_interference_matrix
# ---------------------------------------------------------------------------


def test_plot_task_interference_matrix_returns_figure(task_names):
    matrix = np.array(
        [
            [1.0, 0.8, -0.2, 0.5],
            [0.8, 1.0, 0.3, 0.6],
            [-0.2, 0.3, 1.0, -0.1],
            [0.5, 0.6, -0.1, 1.0],
        ]
    )
    fig = plot_task_interference_matrix(matrix, task_names, save_path=output_dir / "interference_matrix_test")
    assert isinstance(fig, plt.Figure)
