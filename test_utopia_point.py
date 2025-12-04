"""Test script to verify utopia point computation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from omegaconf import DictConfig

from src.methods import ChebyshevFineTuning
from src.utils.device import get_device


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Test utopia point computation"""

    print("=" * 80)
    print("UTOPIA POINT COMPUTATION TEST")
    print("=" * 80)

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create Chebyshev method
    method = ChebyshevFineTuning(
        num_epochs=1,
        batch_size=8,
        utopia_point=None,  # Force computation
    )

    # Get task configurations
    task_names = sorted(cfg.datasets.keys())
    dataset_configs = cfg.datasets

    print(f"\nTasks: {task_names}")
    print("\nComputing utopia point...")

    # Compute utopia point
    import numpy as np
    preference_vector = np.array([0.25, 0.25, 0.25, 0.25])

    utopia_point = method._compute_utopia_point(
        task_names=task_names,
        preference_vector=preference_vector,
        dataset_configs=dataset_configs,
    )

    print("\n" + "=" * 80)
    print("UTOPIA POINT COMPUTED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nUtopia point tensor: {utopia_point}")
    print(f"Shape: {utopia_point.shape}")
    print(f"Values: {utopia_point.tolist()}")

    # Interpret results
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    for i, task_name in enumerate(task_names):
        utopia_val = utopia_point[i].item()
        loss_val = -utopia_val
        print(f"\n{task_name}:")
        print(f"  Utopia point (negative loss): {utopia_val:.4f}")
        print(f"  Best achievable loss: {loss_val:.4f}")
        print(f"  Approximate accuracy: {(1.0 - min(loss_val, 1.0)) * 100:.1f}%")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
