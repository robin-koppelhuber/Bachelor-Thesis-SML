"""Regenerate plots and results tables from saved raw predictions.

Loads raw_predictions.npz + the Hydra config snapshot from a previous run and
re-runs the full visualization pipeline without loading any models.

Usage:
    # Regenerate with the original config:
    uv run python scripts/replot.py <run_dir>

    # Override benchmark config (e.g. to add new metrics):
    uv run python scripts/replot.py <run_dir> \\
        --benchmark-config configs/benchmark/glue-2-label.yaml

    # Write to a custom output directory:
    uv run python scripts/replot.py <run_dir> --output-dir /tmp/new_plots

Where <run_dir> is the Hydra output directory of the original run, e.g.:
    outputs/glue_2_label/2025-01-15_12-34-56/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_config(run_dir: Path, benchmark_config: Optional[Path]):
    """Load the Hydra config snapshot, optionally merging an updated benchmark config."""
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    if benchmark_config is not None:
        if not benchmark_config.exists():
            raise FileNotFoundError(f"Benchmark config not found: {benchmark_config}")
        override = OmegaConf.load(benchmark_config)
        # Merge only the benchmark section (keeps method, paths, etc. from original run)
        if "benchmark" in override:
            cfg.benchmark = OmegaConf.merge(cfg.benchmark, override.benchmark)
            logger.info(f"Merged benchmark config from: {benchmark_config}")

    return cfg


def _load_npz(run_dir: Path):
    """
    Try to load raw_predictions.npz.  Returns the NpzFile, or None if absent.

    Looks in <run_dir>/visualizations/ (the default viz output dir) and
    also directly in <run_dir>/ as a fallback.
    """
    candidates = [
        run_dir / "visualizations" / "raw_predictions.npz",
        run_dir / "raw_predictions.npz",
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Loading raw predictions from: {path}")
            return np.load(path, allow_pickle=True)
    return None


def _reconstruct_all_results(npz, cfg, metric_names: List[str]) -> List[Dict]:
    """
    Rebuild the all_results list (one entry per preference vector) from npz data.

    Each entry has the shape expected by generate_all_visualizations:
        {
            "preference_vector": [float, ...],
            "task_results":      {task_name: EvaluationResult},
        }
    """
    from src.evaluation.evaluator import EvaluationResult
    from src.evaluation.metrics import compute_classification_metrics

    task_names = list(npz["task_names"])
    pref_vecs = npz["preference_vectors"]  # shape (n_prefs, n_tasks)

    all_results = []
    for pref_idx, pref_vec in enumerate(pref_vecs):
        task_results = {}
        for task in task_names:
            preds_key = f"pref_{pref_idx}_{task}_predictions"
            labels_key = f"pref_{pref_idx}_{task}_labels"

            if preds_key not in npz or labels_key not in npz:
                logger.warning(f"Missing predictions for pref {pref_idx}, task {task}; skipping entry")
                break

            predictions = npz[preds_key].astype(np.int32)
            labels = npz[labels_key].astype(np.int32)

            # Recompute metrics (allows adding new metric names via updated benchmark config)
            metrics = compute_classification_metrics(predictions, labels, metric_names)

            task_results[task] = EvaluationResult(
                task_name=task,
                metrics=metrics,
                num_samples=len(labels),
                predictions=predictions,
                labels=labels,
            )
        else:
            # Only append if all tasks completed successfully
            all_results.append(
                {
                    "preference_vector": pref_vec.tolist(),
                    "task_results": task_results,
                }
            )

    logger.info(f"Reconstructed {len(all_results)} preference-vector entries")
    return all_results


def _reconstruct_all_results_from_json(json_path: Path, task_names: List[str], metric_names: List[str]) -> List[Dict]:
    """
    Fallback: reconstruct all_results from comprehensive_results.json metric floats.

    Predictions/labels are not available, so EvaluationResult objects will have
    predictions=None and labels=None.  New metrics cannot be computed.
    """
    from src.evaluation.evaluator import EvaluationResult

    logger.warning("raw_predictions.npz not found — falling back to comprehensive_results.json (metric floats only).")
    logger.warning("New metrics cannot be computed without raw predictions.")

    with open(json_path) as f:
        data = json.load(f)

    # Group rows by preference_vector label, keeping only merged/method rows
    from collections import OrderedDict

    pref_buckets: OrderedDict = OrderedDict()
    for row in data["results"]:
        source = row.get("source", "")
        if source.startswith("finetuned_"):
            continue
        pref_label = row["preference_vector"]
        if pref_label not in pref_buckets:
            pref_buckets[pref_label] = {"preference_raw": row.get("preference_raw"), "task_rows": {}}
        pref_buckets[pref_label]["task_rows"][row["task"]] = row

    all_results = []
    for pref_label, bucket in pref_buckets.items():
        pref_vec = bucket["preference_raw"]
        if pref_vec is None:
            continue
        task_results = {}
        for task in task_names:
            row = bucket["task_rows"].get(task, {})
            metrics = {m: row[m] for m in metric_names if m in row and row[m] is not None}
            task_results[task] = EvaluationResult(
                task_name=task,
                metrics=metrics,
                num_samples=0,
                predictions=None,
                labels=None,
            )
        all_results.append({"preference_vector": pref_vec, "task_results": task_results})

    logger.info(f"Reconstructed {len(all_results)} preference-vector entries from JSON")
    return all_results


def _reconstruct_reference_points(json_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Rebuild reference_points from rows in comprehensive_results.json whose
    source starts with 'finetuned_'.

    Returns dict: {source_task: {f"{eval_task}_{metric}": float}}
    """
    with open(json_path) as f:
        data = json.load(f)

    metrics = data.get("metrics", [])
    reference_points: Dict[str, Dict[str, float]] = {}

    for row in data["results"]:
        source = row.get("source", "")
        if not source.startswith("finetuned_"):
            continue
        source_task = source[len("finetuned_"):]
        task = row["task"]
        if source_task not in reference_points:
            reference_points[source_task] = {}
        for metric in metrics:
            if metric in row and row[metric] is not None:
                reference_points[source_task][f"{task}_{metric}"] = float(row[metric])

    logger.info(f"Reconstructed reference points for {len(reference_points)} source tasks")
    return reference_points


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate visualizations from a previous benchmark run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dir", type=Path, help="Hydra output directory of the original run")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional benchmark YAML to merge (e.g. to add new metrics or plot types)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output directory for regenerated plots (default: <run_dir>/visualizations_replot_<timestamp>)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        logger.error(f"run_dir does not exist or is not a directory: {run_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = _load_config(run_dir, args.benchmark_config)
    task_names = list(cfg.benchmark.tasks)
    metric_names = [m.name for m in cfg.benchmark.evaluation.metrics]
    logger.info(f"Tasks:   {task_names}")
    logger.info(f"Metrics: {metric_names}")

    # ------------------------------------------------------------------
    # 2. Load raw predictions (or fall back to JSON)
    # ------------------------------------------------------------------
    json_path = run_dir / "visualizations" / "comprehensive_results.json"
    if not json_path.exists():
        # Try directly in run_dir for older runs
        json_path = run_dir / "comprehensive_results.json"

    npz = _load_npz(run_dir)
    if npz is not None:
        all_results = _reconstruct_all_results(npz, cfg, metric_names)
    elif json_path.exists():
        all_results = _reconstruct_all_results_from_json(json_path, task_names, metric_names)
    else:
        logger.error("Neither raw_predictions.npz nor comprehensive_results.json found in the run directory.")
        sys.exit(1)

    if not all_results:
        logger.error("No results to visualize — check the run directory.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Reconstruct reference points from the JSON
    # ------------------------------------------------------------------
    reference_points = {}
    if json_path.exists():
        reference_points = _reconstruct_reference_points(json_path)
    else:
        logger.warning("comprehensive_results.json not found — reference points will be omitted")

    # ------------------------------------------------------------------
    # 4. Set output directory
    # ------------------------------------------------------------------
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = run_dir / f"visualizations_replot_{ts}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ------------------------------------------------------------------
    # 5. Run visualization pipeline (mirrors run.py step 5)
    # ------------------------------------------------------------------
    from src.visualization.generator import export_results_table, generate_all_visualizations

    figures = generate_all_visualizations(
        all_results=all_results,
        task_names=task_names,
        metrics_config=list(cfg.benchmark.evaluation.metrics),
        output_dir=output_dir,
        method_name=cfg.method.name,
        reference_points=reference_points or None,
        cross_metric_plots_config=list(cfg.benchmark.evaluation.get("cross_metric_plots", [])),
    )
    logger.info(f"Generated {len(figures)} visualizations")

    export_results_table(
        all_results=all_results,
        task_names=task_names,
        metrics=metric_names,
        output_dir=output_dir,
        method_name=cfg.method.name,
        reference_points=reference_points or None,
    )

    logger.info(f"Done. Results written to: {output_dir}")


if __name__ == "__main__":
    main()
