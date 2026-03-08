"""Script to download and cache models"""

import argparse
import logging
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from src.models.loaders import load_model, load_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download and cache models")
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Download all fine-tuned models declared in config (covers all benchmarks). "
        "Default: only models used by the selected benchmark.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. cluster=euler",
    )
    args = parser.parse_args()

    # Load Hydra config
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="config", overrides=args.overrides)

    # Use configured HuggingFace cache directories
    cache_dir_base = Path(cfg.paths.hf_models_cache_base)
    cache_dir_finetuned = Path(cfg.paths.hf_models_cache_finetuned)

    cache_dir_base.mkdir(parents=True, exist_ok=True)
    cache_dir_finetuned.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading models...")
    logger.info(f"Base model cache:      {cache_dir_base}")
    logger.info(f"Fine-tuned model cache: {cache_dir_finetuned}")

    if args.all_benchmarks:
        logger.info("Mode: all benchmarks — downloading all declared fine-tuned models")
        tasks_to_download = list(cfg.datasets.keys())
    else:
        logger.info(f"Mode: benchmark '{cfg.benchmark.name}' only")
        tasks_to_download = list(cfg.benchmark.tasks)

    logger.info(f"Tasks to download: {tasks_to_download}\n")

    # Download base model
    base_model_id = cfg.model.hf_model_id
    logger.info(f"Downloading base model: {base_model_id}")
    try:
        load_tokenizer(base_model_id, cache_dir=cache_dir_base)
        load_model(
            base_model_id,
            num_labels=2,  # Placeholder, will be overridden at runtime
            cache_dir=cache_dir_base,
            device=torch.device("cpu"),
        )
        logger.info("  ✓ Base model downloaded")
    except Exception as e:
        logger.error(f"  ✗ Failed to download base model: {e}")

    # Download fine-tuned models
    for task_name in tasks_to_download:
        dataset_cfg = cfg.datasets[task_name]
        checkpoint = dataset_cfg.finetuned_checkpoint

        logger.info(f"\nDownloading {task_name}: {checkpoint}")
        try:
            load_model(
                checkpoint,
                num_labels=dataset_cfg.num_labels,
                cache_dir=cache_dir_finetuned,
                device=torch.device("cpu"),
            )
            logger.info(f"  ✓ {task_name} model downloaded")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {task_name}: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("Model download complete!")
    logger.info(f"Base models cached in:       {cache_dir_base}")
    logger.info(f"Fine-tuned models cached in: {cache_dir_finetuned}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
