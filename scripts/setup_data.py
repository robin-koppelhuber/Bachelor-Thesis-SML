"""Script to download and cache datasets"""

import logging
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig

from src.data.loaders import load_hf_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Download and cache all POC datasets"""
    # Load Hydra config
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="config")

    # Use configured HuggingFace cache directory
    cache_dir = Path(cfg.paths.hf_datasets_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading datasets for benchmark...")
    logger.info(f"Benchmark: {cfg.benchmark.name}")
    logger.info(f"Cache directory: {cache_dir}")

    # Download each dataset from benchmark config
    for task_name in cfg.benchmark.tasks:
        logger.info(f"\nDownloading {task_name}...")

        # Get dataset config
        dataset_cfg = cfg.datasets[task_name]

        # Determine splits to download
        split_mapping = dataset_cfg.hf_dataset.split
        splits = list(split_mapping.values())

        for split in splits:
            try:
                dataset = load_hf_dataset(
                    dataset_path=dataset_cfg.hf_dataset.path,
                    subset=dataset_cfg.hf_dataset.get("subset"),
                    split=split,
                    cache_dir=cache_dir,
                )
                logger.info(f"  ✓ {split}: {len(dataset)} samples")
            except Exception as e:
                logger.error(f"  ✗ Failed to download {split}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Dataset download complete!")
    logger.info(f"Datasets cached in: {cache_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
