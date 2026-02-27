"""Unit tests for src/visualization/pareto.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.pareto import compute_pareto_frontier_2d, plot_pareto_frontier_2d

output_dir = Path(__file__).resolve().parent / "test_output"


# ---------------------------------------------------------------------------
# compute_pareto_frontier_2d
# ---------------------------------------------------------------------------


def test_pareto_all_dominated():
    """Only one point dominates all others."""
    points = np.array(
        [
            [0.9, 0.9],  # dominates everything
            [0.5, 0.5],
            [0.3, 0.7],
        ]
    )
    pareto_pts, pareto_idx = compute_pareto_frontier_2d(points)
    assert len(pareto_pts) == 1
    np.testing.assert_array_equal(pareto_pts[0], [0.9, 0.9])


def test_pareto_all_optimal():
    """All points are Pareto optimal (no dominance)."""
    points = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
        ]
    )
    pareto_pts, pareto_idx = compute_pareto_frontier_2d(points)
    assert len(pareto_pts) == 3


def test_pareto_mixed():
    """Typical mixed case: some dominated, some not."""
    points = np.array(
        [
            [0.8, 0.4],  # Pareto optimal
            [0.6, 0.6],  # Pareto optimal
            [0.4, 0.8],  # Pareto optimal
            [0.5, 0.3],  # dominated by [0.8, 0.4]
            [0.3, 0.5],  # dominated by [0.6, 0.6]
        ]
    )
    pareto_pts, pareto_idx = compute_pareto_frontier_2d(points)
    assert len(pareto_pts) == 3
    # Pareto points should be sorted by first objective
    assert pareto_pts[0, 0] <= pareto_pts[-1, 0]


def test_pareto_single_point():
    """Single point is always Pareto optimal."""
    points = np.array([[0.75, 0.60]])
    pareto_pts, pareto_idx = compute_pareto_frontier_2d(points)
    assert len(pareto_pts) == 1


# ---------------------------------------------------------------------------
# plot_pareto_frontier_2d
# ---------------------------------------------------------------------------


def test_plot_pareto_frontier_2d_returns_figure():
    results = {
        "[0.25, 0.25]": (0.69, 0.52),
        "[0.70, 0.10]": (0.83, 0.48),
        "[0.10, 0.70]": (0.55, 0.70),
    }
    fig = plot_pareto_frontier_2d(results, ("ag_news", "imdb"), save_path=output_dir / "pareto_2d_test")
    assert isinstance(fig, plt.Figure)


def test_plot_pareto_frontier_2d_with_optima():
    results = {
        "[0.25, 0.25]": (0.69, 0.52),
        "[0.70, 0.10]": (0.83, 0.48),
        "[0.10, 0.70]": (0.55, 0.70),
    }
    single_task_optima = {
        "ag_news_optimal": (0.947, 0.953),
        "imdb_optimal": (0.600, 0.953),
    }
    fig = plot_pareto_frontier_2d(
        results,
        ("ag_news", "imdb"),
        save_path=output_dir / "pareto_2d_optima_test",
        single_task_optima=single_task_optima,
        title="Test Pareto",
    )
    assert isinstance(fig, plt.Figure)
