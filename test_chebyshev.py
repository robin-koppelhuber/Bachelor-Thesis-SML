"""Quick test script for Chebyshev fine-tuning implementation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.benchmarks.poc.run import run_poc_benchmark
from src.utils.device import get_device


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Run minimal Chebyshev test"""

    # Override config for quick testing
    cfg.method.name = "chebyshev"
    cfg.method.params.num_epochs = 1  # Just 1 epoch for testing
    cfg.method.params.batch_size = 8  # Smaller batch size
    cfg.benchmark.preference_vectors = [[0.25, 0.25, 0.25, 0.25]]  # Single preference
    cfg.benchmark.evaluation.num_samples = 50  # Limit evaluation samples
    cfg.wandb.mode = "disabled"  # Disable W&B for testing

    print("=" * 80)
    print("CHEBYSHEV FINE-TUNING TEST")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Method: {cfg.method.name}")
    print(f"  Epochs: {cfg.method.params.num_epochs}")
    print(f"  Batch size: {cfg.method.params.batch_size}")
    print(f"  Preference vectors: {cfg.benchmark.preference_vectors}")
    print(f"  Evaluation samples: {cfg.benchmark.evaluation.num_samples}")
    print("=" * 80)

    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Run benchmark
    try:
        results = run_poc_benchmark(cfg, device)

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nStatus: {results['status']}")
        print(f"Method: {results['method']}")
        print(f"\nResults:")
        for result_entry in results['all_results']:
            pref = result_entry['preference_vector']
            print(f"\n  Preference: {pref}")
            for task_name, eval_result in result_entry['task_results'].items():
                print(f"    {task_name}: {eval_result.metrics}")

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
