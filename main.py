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
    # Setup logging
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
        wandb_run = init_wandb(
            config=cfg,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.benchmark.name}_{cfg.method.name}",
            tags=[cfg.benchmark.name, cfg.method.name] + (cfg.wandb.tags or []),
            group=cfg.wandb.group,
            notes=cfg.wandb.notes,
            mode=cfg.wandb.mode,
        )
        if wandb_run:
            logger.info(f"Initialized W&B run: {wandb_run.name}")

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

            # Log to W&B if enabled
            if wandb_run:
                import wandb

                for result_entry in results.get("all_results", []):
                    pref_vec = result_entry["preference_vector"]
                    task_results = result_entry["task_results"]

                    # Create wandb log dict
                    log_dict = {"preference_vector": pref_vec}
                    for task_name, task_result in task_results.items():
                        for metric_name, metric_value in task_result.metrics.items():
                            log_dict[f"{task_name}/{metric_name}"] = metric_value

                    wandb.log(log_dict)

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
