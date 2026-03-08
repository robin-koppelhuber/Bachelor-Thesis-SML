"""Script to download and cache datasets"""

import argparse
import logging
from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig

from src.data.loaders import load_hf_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_dataset(dataset_cfg, cache_dir: Path) -> None:
    """Download all splits for a single dataset config."""
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


def main():
    parser = argparse.ArgumentParser(description="Download and cache datasets")
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Download all datasets declared in config (covers all benchmarks). "
        "Default: only the datasets used by the selected benchmark.",
    )
    args = parser.parse_args()

    # Load Hydra config
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="config")

    # Use configured HuggingFace cache directory
    cache_dir = Path(cfg.paths.hf_datasets_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading datasets...")
    logger.info(f"Cache directory: {cache_dir}")

    if args.all_benchmarks:
        # Download every dataset declared in config (covers all benchmarks)
        logger.info("Mode: all benchmarks — downloading all declared datasets")
        tasks_to_download = list(cfg.datasets.keys())
    else:
        # Only datasets required by the currently selected benchmark
        logger.info(f"Mode: benchmark '{cfg.benchmark.name}' only")
        tasks_to_download = list(cfg.benchmark.tasks)

    logger.info(f"Datasets to download: {tasks_to_download}\n")

    for task_name in tasks_to_download:
        logger.info(f"Downloading {task_name}...")
        dataset_cfg = cfg.datasets[task_name]
        download_dataset(dataset_cfg, cache_dir)

    logger.info("\n" + "=" * 80)
    logger.info("Dataset download complete!")
    logger.info(f"Datasets cached in: {cache_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
