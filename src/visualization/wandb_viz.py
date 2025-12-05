"""Weights & Biases visualization utilities"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def log_figures_to_wandb(
    figures: Dict[str, plt.Figure],
    step: Optional[int] = None,
) -> None:
    """
    Log matplotlib figures to W&B

    Args:
        figures: Dictionary mapping figure names to matplotlib figures
        step: Optional step number for logging

    Requires:
        wandb must be imported and initialized before calling this function
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping figure logging")
            return

        log_dict = {}
        for name, fig in figures.items():
            log_dict[f"viz/{name}"] = wandb.Image(fig)
            plt.close(fig)  # Close figure to free memory

        wandb.log(log_dict, step=step)
        logger.info(f"Logged {len(figures)} figures to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping figure logging")
    except Exception as e:
        logger.error(f"Failed to log figures to W&B: {e}")


def log_confusion_matrix_to_wandb(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    task_name: str,
    step: Optional[int] = None,
) -> None:
    """
    Log confusion matrix to W&B

    Args:
        predictions: Predicted labels
        labels: True labels
        class_names: List of class names
        task_name: Name of the task
        step: Optional step number for logging
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping confusion matrix logging")
            return

        wandb.log(
            {
                f"confusion_matrix/{task_name}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels.tolist(),
                    preds=predictions.tolist(),
                    class_names=class_names,
                )
            },
            step=step,
        )
        logger.info(f"Logged confusion matrix for {task_name} to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping confusion matrix logging")
    except Exception as e:
        logger.error(f"Failed to log confusion matrix to W&B: {e}")


def create_visualization_artifact(
    plot_files: List[Path],
    artifact_name: str,
    artifact_type: str = "plots",
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create and log W&B artifact containing visualization files

    Args:
        plot_files: List of paths to plot files
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (default: "plots")
        description: Optional description
        metadata: Optional metadata dictionary
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping artifact creation")
            return

        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata,
        )

        for plot_file in plot_files:
            if plot_file.exists():
                artifact.add_file(str(plot_file))
            else:
                logger.warning(f"Plot file not found: {plot_file}")

        wandb.log_artifact(artifact)
        logger.info(f"Created and logged artifact '{artifact_name}' with {len(plot_files)} files")

    except ImportError:
        logger.warning("wandb not installed, skipping artifact creation")
    except Exception as e:
        logger.error(f"Failed to create visualization artifact: {e}")


def log_pareto_frontier_to_wandb(
    results: Dict[str, tuple],
    task_names: tuple,
    pareto_indices: np.ndarray,
    step: Optional[int] = None,
) -> None:
    """
    Log Pareto frontier analysis to W&B as a table

    Args:
        results: Dictionary mapping config names to (score1, score2) tuples
        task_names: Names of the two tasks
        pareto_indices: Indices of Pareto-optimal points
        step: Optional step number for logging
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping Pareto frontier logging")
            return

        # Prepare table data
        table_data = []
        config_names = list(results.keys())

        for i, (config_name, scores) in enumerate(results.items()):
            is_pareto = i in pareto_indices
            table_data.append([
                config_name,
                float(scores[0]),
                float(scores[1]),
                is_pareto,
            ])

        # Create W&B table
        table = wandb.Table(
            columns=["configuration", task_names[0], task_names[1], "is_pareto_optimal"],
            data=table_data,
        )

        wandb.log({f"pareto_analysis/{task_names[0]}_vs_{task_names[1]}": table}, step=step)
        logger.info(f"Logged Pareto frontier for {task_names[0]} vs {task_names[1]} to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping Pareto frontier logging")
    except Exception as e:
        logger.error(f"Failed to log Pareto frontier to W&B: {e}")


def log_metrics_summary(
    summary_metrics: Dict[str, Any],
) -> None:
    """
    Log summary metrics to W&B summary

    Args:
        summary_metrics: Dictionary of metrics to log to summary
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping summary metrics logging")
            return

        wandb.summary.update(summary_metrics)
        logger.info(f"Logged {len(summary_metrics)} summary metrics to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping summary metrics logging")
    except Exception as e:
        logger.error(f"Failed to log summary metrics to W&B: {e}")


def log_task_interference_to_wandb(
    interference_dict: Dict[tuple, float],
    step: Optional[int] = None,
) -> None:
    """
    Log task interference metrics to W&B

    Args:
        interference_dict: Dictionary mapping (task1, task2) to correlation coefficient
        step: Optional step number for logging
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping interference logging")
            return

        # Prepare table data
        table_data = []
        for (task1, task2), correlation in interference_dict.items():
            table_data.append([task1, task2, float(correlation)])

        # Create W&B table
        table = wandb.Table(
            columns=["task_1", "task_2", "correlation"],
            data=table_data,
        )

        wandb.log({"task_interference": table}, step=step)

        # Also log as individual metrics for easy tracking
        for (task1, task2), correlation in interference_dict.items():
            wandb.log({f"interference/{task1}_vs_{task2}": correlation}, step=step)

        logger.info(f"Logged task interference metrics to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping interference logging")
    except Exception as e:
        logger.error(f"Failed to log task interference to W&B: {e}")


def log_preference_alignment_to_wandb(
    preference_vector: np.ndarray,
    alignment_metrics: Dict[str, float],
    task_names: List[str],
    step: Optional[int] = None,
) -> None:
    """
    Log preference alignment metrics to W&B

    Args:
        preference_vector: Requested preference vector
        alignment_metrics: Dictionary with alignment metrics (cosine_similarity, mse, max_deviation)
        task_names: List of task names
        step: Optional step number for logging
    """
    try:
        import wandb

        if not wandb.run:
            logger.warning("W&B run not initialized, skipping alignment logging")
            return

        # Log alignment metrics
        log_dict = {
            "preference_alignment/cosine_similarity": alignment_metrics["cosine_similarity"],
            "preference_alignment/mse": alignment_metrics["mse"],
            "preference_alignment/max_deviation": alignment_metrics["max_deviation"],
        }

        wandb.log(log_dict, step=step)
        logger.info(f"Logged preference alignment metrics to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping alignment logging")
    except Exception as e:
        logger.error(f"Failed to log preference alignment to W&B: {e}")
