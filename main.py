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

    # TODO: Implement benchmark runner
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark runner not implemented yet!")
    logger.info("=" * 80)

    # Cleanup
    if wandb_run:
        finish_wandb()
        logger.info("Finished W&B run")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
