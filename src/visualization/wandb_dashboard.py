"""W&B Custom Dashboard Configuration

This module provides pre-configured W&B dashboard layouts for multi-task
model merging benchmarks.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def create_benchmark_dashboard(
    project_name: str,
    entity: Optional[str] = None,
) -> Dict:
    """
    Create a comprehensive W&B dashboard for benchmark results

    This dashboard includes:
    - Performance metrics across tasks and preference vectors
    - Pareto frontier analysis
    - Task interference metrics
    - Preference alignment tracking
    - Method comparison charts

    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)

    Returns:
        Dashboard configuration dictionary

    Usage:
        ```python
        import wandb
        from src.visualization.wandb_dashboard import create_benchmark_dashboard

        # Initialize W&B
        wandb.init(project="my_project")

        # Get dashboard config
        dashboard_config = create_benchmark_dashboard("my_project")

        # Note: Dashboard must be created via W&B UI or API
        # This function provides the recommended layout configuration
        ```
    """
    dashboard_config = {
        "project": project_name,
        "entity": entity,
        "sections": [
            # Section 1: Overview
            {
                "name": "Overview",
                "panels": [
                    {
                        "title": "Benchmark Summary",
                        "type": "run-comparer",
                        "config": {
                            "metrics": [
                                "eval_avg/f1_macro",
                                "eval_avg/accuracy",
                                "preference_alignment/cosine_similarity",
                            ]
                        },
                    },
                    {
                        "title": "Results Summary Table",
                        "type": "table",
                        "config": {"table_name": "results_summary"},
                    },
                ],
            },
            # Section 2: Per-Task Performance
            {
                "name": "Task Performance",
                "panels": [
                    {
                        "title": "AG News Performance",
                        "type": "line",
                        "config": {
                            "metrics": [
                                "eval/ag_news/f1_macro",
                                "eval/ag_news/accuracy",
                            ]
                        },
                    },
                    {
                        "title": "IMDB Performance",
                        "type": "line",
                        "config": {
                            "metrics": [
                                "eval/imdb/f1_macro",
                                "eval/imdb/accuracy",
                            ]
                        },
                    },
                    {
                        "title": "MNLI Performance",
                        "type": "line",
                        "config": {
                            "metrics": [
                                "eval/mnli/f1_macro",
                                "eval/mnli/accuracy",
                            ]
                        },
                    },
                    {
                        "title": "MRPC Performance",
                        "type": "line",
                        "config": {
                            "metrics": [
                                "eval/mrpc/f1_macro",
                                "eval/mrpc/accuracy",
                            ]
                        },
                    },
                ],
            },
            # Section 3: Multi-Task Analysis
            {
                "name": "Multi-Task Analysis",
                "panels": [
                    {
                        "title": "Average F1 Macro Across Tasks",
                        "type": "line",
                        "config": {"metrics": ["eval_avg/f1_macro"]},
                    },
                    {
                        "title": "Task Interference",
                        "type": "table",
                        "config": {"table_name": "task_interference"},
                    },
                    {
                        "title": "Preference Alignment Metrics",
                        "type": "line",
                        "config": {
                            "metrics": [
                                "preference_alignment/cosine_similarity",
                                "preference_alignment/mse",
                                "preference_alignment/max_deviation",
                            ]
                        },
                    },
                ],
            },
            # Section 4: Visualizations
            {
                "name": "Visualizations",
                "panels": [
                    {
                        "title": "Performance Heatmap",
                        "type": "image",
                        "config": {"media_keys": ["viz/performance_heatmap"]},
                    },
                    {
                        "title": "Task Interference Matrix",
                        "type": "image",
                        "config": {"media_keys": ["viz/task_interference_matrix"]},
                    },
                    {
                        "title": "Radar Charts",
                        "type": "image",
                        "config": {
                            "media_keys": [
                                "viz/radar_chart_0",
                                "viz/radar_chart_1",
                                "viz/radar_chart_2",
                            ]
                        },
                    },
                ],
            },
            # Section 5: Pareto Analysis
            {
                "name": "Pareto Frontiers",
                "panels": [
                    {
                        "title": "AG News vs IMDB",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_ag_news_vs_imdb"]},
                    },
                    {
                        "title": "AG News vs MNLI",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_ag_news_vs_mnli"]},
                    },
                    {
                        "title": "AG News vs MRPC",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_ag_news_vs_mrpc"]},
                    },
                    {
                        "title": "IMDB vs MNLI",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_imdb_vs_mnli"]},
                    },
                    {
                        "title": "IMDB vs MRPC",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_imdb_vs_mrpc"]},
                    },
                    {
                        "title": "MNLI vs MRPC",
                        "type": "image",
                        "config": {"media_keys": ["viz/pareto_mnli_vs_mrpc"]},
                    },
                ],
            },
            # Section 6: Method Comparison
            {
                "name": "Method Comparison",
                "panels": [
                    {
                        "title": "F1 Macro by Method",
                        "type": "parallel-coordinates",
                        "config": {
                            "metrics": [
                                "eval/ag_news/f1_macro",
                                "eval/imdb/f1_macro",
                                "eval/mnli/f1_macro",
                                "eval/mrpc/f1_macro",
                            ]
                        },
                    },
                    {
                        "title": "Accuracy by Method",
                        "type": "parallel-coordinates",
                        "config": {
                            "metrics": [
                                "eval/ag_news/accuracy",
                                "eval/imdb/accuracy",
                                "eval/mnli/accuracy",
                                "eval/mrpc/accuracy",
                            ]
                        },
                    },
                ],
            },
        ],
    }

    return dashboard_config


def log_dashboard_url(
    project_name: str,
    entity: Optional[str] = None,
    dashboard_name: str = "Benchmark Dashboard",
) -> str:
    """
    Generate W&B dashboard URL

    Args:
        project_name: W&B project name
        entity: W&B entity (username or team)
        dashboard_name: Name of the dashboard

    Returns:
        URL to the dashboard
    """
    if entity:
        url = f"https://wandb.ai/{entity}/{project_name}/reports/{dashboard_name}"
    else:
        url = f"https://wandb.ai/projects/{project_name}/reports/{dashboard_name}"

    return url


def create_custom_charts() -> List[Dict]:
    """
    Create custom chart configurations for W&B

    Returns:
        List of chart configurations

    Usage:
        ```python
        import wandb

        charts = create_custom_charts()
        for chart in charts:
            wandb.log({chart["name"]: wandb.plot.line_series(...)})
        ```
    """
    charts = [
        {
            "name": "task_performance_comparison",
            "title": "Task Performance Comparison",
            "type": "bar",
            "keys": [
                "eval/ag_news/f1_macro",
                "eval/imdb/f1_macro",
                "eval/mnli/f1_macro",
                "eval/mrpc/f1_macro",
            ],
            "labels": ["AG News", "IMDB", "MNLI", "MRPC"],
        },
        {
            "name": "preference_vector_heatmap",
            "title": "Performance Across Preference Vectors",
            "type": "heatmap",
            "description": "Shows performance for each task across different preference vectors",
        },
        {
            "name": "pareto_frontier_comparison",
            "title": "Pareto Frontier Analysis",
            "type": "scatter",
            "description": "Compare Pareto frontiers for different methods",
        },
    ]

    return charts


def setup_wandb_dashboard(
    run,
    task_names: List[str],
    preference_vectors: List[List[float]],
    method_name: str,
) -> None:
    """
    Setup W&B dashboard with custom panels and charts

    This function configures the W&B run with custom visualizations
    that will appear in the workspace UI.

    Args:
        run: Active W&B run object
        task_names: List of task names
        preference_vectors: List of preference vectors
        method_name: Name of the merging method

    Usage:
        ```python
        import wandb
        from src.visualization.wandb_dashboard import setup_wandb_dashboard

        run = wandb.init(project="my_project")
        setup_wandb_dashboard(
            run,
            task_names=["ag_news", "imdb", "mnli", "mrpc"],
            preference_vectors=[[0.25, 0.25, 0.25, 0.25], ...],
            method_name="ties"
        )
        ```
    """
    try:
        # Define custom summary metrics
        run.define_metric("eval_avg/f1_macro", summary="max")
        run.define_metric("eval_avg/accuracy", summary="max")
        run.define_metric("preference_alignment/cosine_similarity", summary="mean")
        run.define_metric("preference_alignment/mse", summary="min")

        # Define per-task metrics
        for task in task_names:
            run.define_metric(f"eval/{task}/f1_macro", summary="max")
            run.define_metric(f"eval/{task}/accuracy", summary="max")

        # Log dashboard configuration to run config
        run.config.update(
            {
                "dashboard": {
                    "task_names": task_names,
                    "num_preference_vectors": len(preference_vectors),
                    "method": method_name,
                }
            }
        )

        logger.info("✓ W&B dashboard configured successfully")

    except Exception as e:
        logger.error(f"Failed to setup W&B dashboard: {e}")


def create_report_template(
    project_name: str,
    run_ids: List[str],
    title: str = "Model Merging Benchmark Report",
) -> str:
    """
    Create a W&B report template with benchmark results

    Args:
        project_name: W&B project name
        run_ids: List of W&B run IDs to include
        title: Report title

    Returns:
        Report template as markdown string

    Usage:
        Save this to a .md file and upload to W&B Reports:
        https://docs.wandb.ai/guides/reports
    """
    report_md = f"""# {title}

## Executive Summary

This report presents the results of multi-task model merging experiments
across AG News, IMDB, MNLI, and MRPC datasets.

## Methodology

We evaluate multiple model merging approaches:
- TIES (Task Interference and Enrichment Search)
- Chebyshev Fine-tuning
- EPO (Exact Pareto Optimization)
- Simple Averaging

Each method is tested with multiple preference vectors to understand
trade-offs between tasks.

## Results

### Performance Overview

${{runset:{project_name}:{','.join(run_ids)}}}

### Task Performance

#### AG News
${{line:{{"lineKey":"eval/ag_news/f1_macro","title":"AG News F1 Macro"}}}}

#### IMDB
${{line:{{"lineKey":"eval/imdb/f1_macro","title":"IMDB F1 Macro"}}}}

#### MNLI
${{line:{{"lineKey":"eval/mnli/f1_macro","title":"MNLI F1 Macro"}}}}

#### MRPC
${{line:{{"lineKey":"eval/mrpc/f1_macro","title":"MRPC F1 Macro"}}}}

### Multi-Task Analysis

#### Average Performance
${{line:{{"lineKey":"eval_avg/f1_macro","title":"Average F1 Macro"}}}}

#### Task Interference
${{table:{{"tableKey":"task_interference"}}}}

### Visualizations

#### Performance Heatmap
${{image:{{"imageKey":"viz/performance_heatmap"}}}}

#### Task Interference Matrix
${{image:{{"imageKey":"viz/task_interference_matrix"}}}}

### Pareto Frontiers

#### AG News vs IMDB
${{image:{{"imageKey":"viz/pareto_ag_news_vs_imdb"}}}}

## Conclusions

[Your conclusions here]

## References

- TIES: https://arxiv.org/abs/2306.01708
- Chebyshev Scalarization: Multi-objective optimization technique
- EPO: Exact Pareto Optimization via MGDA

---

*Report generated for W&B project: {project_name}*
"""

    return report_md
