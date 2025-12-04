"""Script to download and cache models"""

import logging
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig

from src.models.loaders import load_model, load_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Download and cache all POC models"""
    # Load Hydra config
    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="config")

    # Use configured HuggingFace cache directory for models
    cache_dir_base = Path(cfg.paths.hf_models_cache_base)
    cache_dir_finetuned = Path(cfg.paths.hf_models_cache_finetuned)

    cache_dir_base.mkdir(parents=True, exist_ok=True)
    cache_dir_finetuned.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading models for benchmark...")
    logger.info(f"Benchmark: {cfg.benchmark.name}")
    logger.info(f"Cache directory: {cache_dir_base}")

    # Download base model
    base_model_id = cfg.model.hf_model_id
    logger.info(f"\nDownloading base model: {base_model_id}")
    try:
        tokenizer = load_tokenizer(base_model_id, cache_dir=cache_dir_base)
        model = load_model(
            base_model_id,
            num_labels=2,  # Placeholder, will be overridden
            cache_dir=cache_dir_base,
            device=torch.device("cpu"),
        )
        logger.info("  ✓ Base model downloaded")
    except Exception as e:
        logger.error(f"  ✗ Failed to download base model: {e}")

    # Download fine-tuned models from dataset configs
    for task_name in cfg.benchmark.tasks:
        dataset_cfg = cfg.datasets[task_name]
        checkpoint = dataset_cfg.finetuned_checkpoint

        logger.info(f"\nDownloading {task_name}: {checkpoint}")
        try:
            model = load_model(
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
    logger.info(f"Models cached in: {cache_dir_base}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
