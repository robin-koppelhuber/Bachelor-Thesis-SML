"""Shared fixtures for visualization unit tests.

Real metric values are sourced from tests/summary_statistics.csv and
tests/utopia_distances.csv produced by export_results_table().
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless backend — must be set before any pyplot import


TASK_NAMES = ["ag_news", "imdb", "mnli", "mrpc"]


class FakeEvalResult:
    """Minimal stand-in for EvaluationResult used in plotting code."""

    def __init__(self, metrics: dict):
        self.metrics = metrics


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def task_names():
    return list(TASK_NAMES)


@pytest.fixture
def all_results():
    """Two preference vectors with realistic metric values (from CSV data)."""
    return [
        {
            "preference_vector": [0.25, 0.25, 0.25, 0.25],
            "task_results": {
                "ag_news": FakeEvalResult({"accuracy": 0.694, "f1_macro": 0.683}),
                "imdb": FakeEvalResult({"accuracy": 0.516, "f1_macro": 0.399}),
                "mnli": FakeEvalResult({"accuracy": 0.505, "f1_macro": 0.470}),
                "mrpc": FakeEvalResult({"accuracy": 0.628, "f1_macro": 0.542}),
            },
        },
        {
            "preference_vector": [0.7, 0.1, 0.1, 0.1],
            "task_results": {
                "ag_news": FakeEvalResult({"accuracy": 0.830, "f1_macro": 0.820}),
                "imdb": FakeEvalResult({"accuracy": 0.480, "f1_macro": 0.370}),
                "mnli": FakeEvalResult({"accuracy": 0.475, "f1_macro": 0.440}),
                "mrpc": FakeEvalResult({"accuracy": 0.600, "f1_macro": 0.510}),
            },
        },
    ]


@pytest.fixture
def reference_points():
    """Single-task fine-tuned model optima (utopia_point column from CSV data)."""
    return {
        "ag_news": {
            "ag_news_accuracy": 0.947,
            "ag_news_f1_macro": 0.947,
            "imdb_accuracy": 0.953,
            "imdb_f1_macro": 0.953,
            "mnli_accuracy": 0.881,
            "mnli_f1_macro": 0.880,
            "mrpc_accuracy": 0.912,
            "mrpc_f1_macro": 0.897,
        },
        "imdb": {
            "ag_news_accuracy": 0.600,
            "ag_news_f1_macro": 0.590,
            "imdb_accuracy": 0.953,
            "imdb_f1_macro": 0.953,
            "mnli_accuracy": 0.500,
            "mnli_f1_macro": 0.490,
            "mrpc_accuracy": 0.680,
            "mrpc_f1_macro": 0.660,
        },
        "mnli": {
            "ag_news_accuracy": 0.610,
            "ag_news_f1_macro": 0.600,
            "imdb_accuracy": 0.510,
            "imdb_f1_macro": 0.500,
            "mnli_accuracy": 0.881,
            "mnli_f1_macro": 0.880,
            "mrpc_accuracy": 0.700,
            "mrpc_f1_macro": 0.680,
        },
        "mrpc": {
            "ag_news_accuracy": 0.620,
            "ag_news_f1_macro": 0.610,
            "imdb_accuracy": 0.520,
            "imdb_f1_macro": 0.510,
            "mnli_accuracy": 0.490,
            "mnli_f1_macro": 0.480,
            "mrpc_accuracy": 0.912,
            "mrpc_f1_macro": 0.897,
        },
    }


@pytest.fixture
def performance_matrix():
    """(2 pref_vectors × 4 tasks) float array for heatmap tests."""
    return np.array(
        [
            [0.694, 0.516, 0.505, 0.628],
            [0.830, 0.480, 0.475, 0.600],
        ]
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    matplotlib.pyplot.close("all")
