"""Unit tests for generator.py utilities (resolve_plot_types, etc.)."""

import pytest

from src.visualization.generator import AVAILABLE_PLOT_TYPES, resolve_plot_types


def test_resolve_plot_types_all_string():
    result = resolve_plot_types("all")
    assert result == AVAILABLE_PLOT_TYPES


def test_resolve_plot_types_list_with_all():
    result = resolve_plot_types(["all"])
    assert result == AVAILABLE_PLOT_TYPES


def test_resolve_plot_types_specific_list():
    result = resolve_plot_types(["heatmap", "pareto_frontiers"])
    assert result == frozenset({"heatmap", "pareto_frontiers"})


def test_resolve_plot_types_empty_list():
    result = resolve_plot_types([])
    assert result == frozenset()


def test_resolve_plot_types_none():
    result = resolve_plot_types(None)
    assert result == frozenset()


def test_available_plot_types_contains_expected():
    expected = {
        "heatmap",
        "parallel_coordinates",
        "distance_to_utopia",
        "performance_recovery",
        "radar_charts",
        "task_interference",
        "preference_alignment",
        "pareto_frontiers",
    }
    assert AVAILABLE_PLOT_TYPES == expected
