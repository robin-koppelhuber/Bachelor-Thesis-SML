"""Main entry point for the benchmarking framework"""

import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from src.utils.device import get_device
from src.utils.logger import setup_logging
from src.utils.seeding import set_seed
from src.utils.wandb_utils import finish_wandb, init_wandb

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point

    Args:
        cfg: Hydra configuration
    """
    # Setup logging (Hydra has already changed cwd to output directory)
    # Log file will be created in Hydra's output directory (e.g., outputs/2024-01-15/10-30-45/)
    log_file = Path.cwd() / f"{cfg.benchmark.name}.log"
    setup_logging(
        log_level=cfg.logging.level,
        log_file=log_file if cfg.logging.log_to_file else None,
        console_format=cfg.logging.console_format,
    )

    logger.info("=" * 80)
    logger.info(f"Starting benchmark: {cfg.benchmark.name}")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    set_seed(cfg.seed)
    logger.info(f"Set random seed: {cfg.seed}")

    # Get device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")

    # Initialize W&B
    wandb_run = None
    if cfg.logging.log_to_wandb:
        # Set wandb dir to output folder (Hydra's cwd)
        wandb_dir = Path.cwd()

        wandb_run = init_wandb(
            config=cfg,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.benchmark.name}_{cfg.method.name}",
            tags=[cfg.benchmark.name, cfg.method.name] + (cfg.wandb.tags or []),
            group=cfg.wandb.group,
            notes=cfg.wandb.notes,
            mode=cfg.wandb.mode,
            dir=wandb_dir,
        )
        if wandb_run:
            logger.info(f"Initialized W&B run: {wandb_run.name}")

            # Setup custom W&B dashboard
            try:
                from src.visualization.wandb_dashboard import setup_wandb_dashboard

                setup_wandb_dashboard(
                    run=wandb_run,
                    task_names=cfg.benchmark.tasks,
                    preference_vectors=cfg.benchmark.preference_vectors,
                    method_name=cfg.method.name,
                )
            except Exception as e:
                logger.warning(f"Failed to setup W&B dashboard: {e}")

    # Print config
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Run benchmark based on config
    try:
        if cfg.benchmark.name == "proof_of_concept":
            from src.benchmarks.poc.run import run_poc_benchmark

            results = run_poc_benchmark(cfg, device)
            logger.info("\n" + "=" * 80)
            logger.info("Benchmark Results:")
            logger.info("=" * 80)
            logger.info(f"Status: {results['status']}")
            logger.info(f"Method: {results['method']}")
            logger.info(f"Tasks: {results['tasks']}")

            # Log summary table to W&B if enabled
            if wandb_run:
                import wandb

                # Create summary table for all preference vectors and tasks
                table_data = []
                for result_entry in results.get("all_results", []):
                    pref_vec = result_entry["preference_vector"]
                    pref_str = str(pref_vec)
                    task_results = result_entry["task_results"]

                    for task_name, task_result in task_results.items():
                        row = {
                            "preference_vector": pref_str,
                            "task": task_name,
                        }
                        row.update(task_result.metrics)
                        table_data.append(row)

                # Create W&B table
                if table_data:
                    columns = list(table_data[0].keys())
                    table = wandb.Table(columns=columns, data=[list(row.values()) for row in table_data])
                    wandb.log({"results_summary": table})

                    # Log final summary metrics
                    wandb.summary["num_preference_vectors"] = len(results["all_results"])
                    wandb.summary["num_tasks"] = len(results["tasks"])
                    wandb.summary["method"] = results["method"]
                    wandb.summary["status"] = results["status"]

                # Log visualizations to W&B if available
                if "figures" in results and results["figures"]:
                    logger.info("Logging visualizations to W&B...")
                    try:
                        from src.visualization.wandb_viz import (
                            create_visualization_artifact,
                            log_figures_to_wandb,
                        )

                        # Log figures as images
                        log_figures_to_wandb(results["figures"])

                        # Create artifact with all saved plot files
                        viz_dir = Path.cwd() / "visualizations"
                        if viz_dir.exists():
                            plot_files = list(viz_dir.glob("**/*.png")) + list(viz_dir.glob("**/*.pdf"))
                            if plot_files:
                                create_visualization_artifact(
                                    plot_files=plot_files,
                                    artifact_name=f"visualizations_{cfg.method.name}_{cfg.benchmark.name}",
                                    artifact_type="plots",
                                    description=f"Visualizations for {cfg.method.name} on {cfg.benchmark.name}",
                                    metadata={
                                        "method": cfg.method.name,
                                        "benchmark": cfg.benchmark.name,
                                        "num_tasks": len(results["tasks"]),
                                        "num_preference_vectors": len(results["all_results"]),
                                    },
                                )
                                logger.info(f"✓ Created W&B artifact with {len(plot_files)} plot files")

                    except Exception as e:
                        logger.error(f"Failed to log visualizations to W&B: {e}")

                # Log dashboard URL and create report template
                try:
                    from src.visualization.wandb_dashboard import (
                        create_report_template,
                        log_dashboard_url,
                    )

                    # Log dashboard URL
                    dashboard_url = log_dashboard_url(
                        project_name=cfg.wandb.project,
                        entity=cfg.wandb.entity,
                        dashboard_name="Benchmark-Dashboard",
                    )
                    logger.info(f"\n{'=' * 80}")
                    logger.info("W&B Dashboard")
                    logger.info(f"{'=' * 80}")
                    logger.info(f"View your results at: {wandb_run.get_url()}")
                    logger.info(f"Suggested dashboard URL: {dashboard_url}")

                    # Create and save report template
                    report_md = create_report_template(
                        project_name=cfg.wandb.project,
                        run_ids=[wandb_run.id],
                        title=f"{cfg.method.name} Model Merging Benchmark",
                    )

                    report_path = Path.cwd() / "wandb_report_template.md"
                    report_path.write_text(report_md)
                    logger.info(f"\n✓ Report template saved to: {report_path}")
                    logger.info("  Upload this to W&B Reports: https://docs.wandb.ai/guides/reports")

                except Exception as e:
                    logger.warning(f"Failed to create report template: {e}")

        else:
            raise ValueError(f"Unknown benchmark: {cfg.benchmark.name}")

    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        raise

    # Cleanup
    if wandb_run:
        finish_wandb()
        logger.info("Finished W&B run")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
