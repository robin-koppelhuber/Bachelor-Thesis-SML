"""Weights & Biases integration utilities"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from omegaconf import DictConfig, OmegaConf

# Load environment variables from .env file
# This is safe to call multiple times - it won't override existing env vars
load_dotenv()

WandbMode = Literal["online", "offline", "disabled", "shared"]


def init_wandb(
    config: DictConfig,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    notes: Optional[str] = None,
    mode: Optional[WandbMode] = None,
    dir: Optional[Path] = None,
) -> Optional[Any]:
    """
    Initialize Weights & Biases run

    Args:
        config: Full experiment configuration
        project: W&B project name
        entity: W&B entity (username or team)
        name: Run name
        tags: Run tags
        group: Run group
        notes: Run notes
        mode: W&B mode (online, offline, disabled)
        dir: W&B directory

    Returns:
        W&B run object or None if disabled

    Note:
        WANDB_API_KEY must be set in environment variables (via .env file)
    """
    # Check if W&B is available
    if not WANDB_AVAILABLE:
        return None

    # Check if W&B is enabled
    wandb_config = config.get("wandb", {})
    enabled = wandb_config.get("enabled", True)

    if not enabled or mode == "disabled":
        return None

    # Use environment variables as fallback for missing values
    # Note: load_dotenv() at module level loads .env into os.environ
    project = project or os.getenv("WANDB_PROJECT")
    entity = entity or os.getenv("WANDB_ENTITY")
    if mode is None:
        env_mode = os.getenv("WANDB_MODE", "online")
        if env_mode in ("online", "offline", "disabled", "shared"):
            mode = env_mode  # type: ignore
        else:
            mode = "online"

    # Convert OmegaConf to dict for W&B
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)

    # Initialize W&B
    # WANDB_API_KEY is automatically read from environment by wandb.init()
    # It was loaded from .env by load_dotenv() at module import time
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=tags,
        group=group,
        notes=notes,
        config=config_dict,
        mode=mode,
        dir=str(dir) if dir else None,
    )

    return run


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    """
    Log metrics to W&B

    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return

    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

    wandb.log(metrics, step=step)


def log_artifact(
    artifact_path: Path,
    artifact_name: str,
    artifact_type: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log artifact to W&B

    Args:
        artifact_path: Path to artifact
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (model, dataset, etc.)
        metadata: Optional metadata dictionary
    """
    if not WANDB_AVAILABLE or not wandb.run:
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )

    if artifact_path.is_dir():
        artifact.add_dir(str(artifact_path))
    else:
        artifact.add_file(str(artifact_path))

    wandb.log_artifact(artifact)


def finish_wandb() -> None:
    """Finish W&B run"""
    if wandb.run:
        wandb.finish()


# test wandb
if __name__ == "__main__":
    import random

    import hydra

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        run = None
        if cfg.logging.log_to_wandb:
            run = init_wandb(
                config=cfg,
                name=f"{cfg.benchmark.name}_{cfg.method.name}",
                tags=[cfg.benchmark.name, cfg.method.name],
            )

        epochs = 10
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2**-epoch - random.random() / epoch - offset
            loss = 2**-epoch + random.random() / epoch + offset

            # Log metrics to wandb.
            run.log({"acc": acc, "loss": loss})

        # Finish the run and upload any remaining data.
        run.finish()

    main()
