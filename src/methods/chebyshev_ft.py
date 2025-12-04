"""Chebyshev Scalarization Fine-Tuning

Fine-tuning from a base model using Chebyshev scalarization to optimize
for a given preference vector over multiple tasks.

The Chebyshev scalarization minimizes:
    max_i { w_i * (f_i^* - f_i(x)) }

Where:
- w_i is the preference weight for task i
- f_i^* is the utopia point (best performance on task i)
- f_i(x) is the current performance on task i

For numerical stability, we typically use the augmented Chebyshev:
    max_i { w_i * (f_i^* - f_i(x)) } + epsilon * sum_i { w_i * (f_i^* - f_i(x)) }
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from src.methods.base import BaseTrainingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("chebyshev")
class ChebyshevFineTuning(BaseTrainingMethod):
    """
    Fine-tuning with Chebyshev scalarization

    This is a training-based method that fine-tunes from the base model
    using Chebyshev scalarization to optimize for a preference vector.
    """

    def __init__(
        self,
        use_augmented: bool = True,
        epsilon: float = 0.001,
        utopia_point: Optional[np.ndarray] = None,
        nadir_point: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Initialize Chebyshev fine-tuning

        Args:
            use_augmented: Use augmented Chebyshev scalarization
            epsilon: Augmentation weight for smoothness
            utopia_point: Best performance per task (computed if None)
            nadir_point: Worst performance per task (computed if None)
            **kwargs: Base training parameters (learning_rate, num_epochs, etc.)
        """
        super().__init__(**kwargs)
        self.use_augmented = use_augmented
        self.epsilon = epsilon
        self.utopia_point = utopia_point
        self.nadir_point = nadir_point

    def _compute_multi_task_loss(
        self,
        task_losses: torch.Tensor,
        preference_vector: torch.Tensor,
        utopia_point: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute Chebyshev scalarization loss

        Args:
            task_losses: Losses for each task (n_tasks,)
            preference_vector: Preference weights (n_tasks,)
            utopia_point: Best performance per task (n_tasks,)

        Returns:
            Scalar loss value
        """
        # Weighted deviations from utopia point
        # Using negative loss as performance (higher is better)
        weighted_deviations = preference_vector * (utopia_point - (-task_losses))

        # Chebyshev: minimize maximum weighted deviation
        chebyshev_loss = -weighted_deviations.max()

        # Augmented Chebyshev: add sum term for smoothness
        if self.use_augmented:
            chebyshev_loss += self.epsilon * weighted_deviations.sum()

        return chebyshev_loss

    def _get_training_kwargs(
        self, task_names: list, preference_vector: np.ndarray, dataset_configs: Dict, cache_dir: Optional[str] = None
    ) -> Dict:
        """
        Get Chebyshev-specific training parameters

        Computes utopia point for use in training loop

        Args:
            task_names: List of task names
            preference_vector: Preference weights
            dataset_configs: Dataset configurations
            cache_dir: Optional cache directory for loading fine-tuned models

        Returns:
            Dictionary with 'utopia_point' for training
        """
        utopia_point = self._compute_utopia_point(task_names, preference_vector, dataset_configs, cache_dir=cache_dir)
        return {"utopia_point": utopia_point}

    def _compute_utopia_point(
        self,
        task_names: list,
        preference_vector: np.ndarray,
        dataset_configs: Dict,
        cache_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute utopia point (best achievable performance per task)

        Loads pre-trained single-task models and evaluates them on validation
        data to get actual best performance per task.

        We use negative loss as the performance metric (higher is better)

        Args:
            task_names: List of task names
            preference_vector: Preference weights
            dataset_configs: Dataset configurations
            cache_dir: Optional cache directory for loading fine-tuned models

        Returns:
            Utopia point tensor (n_tasks,)
        """
        if self.utopia_point is not None:
            # Use provided utopia point from config
            logger.info("Using utopia point from config")
            return torch.tensor(self.utopia_point, dtype=torch.float32)

        logger.info("\nComputing utopia point from fine-tuned models...")
        if cache_dir:
            logger.info(f"  Using model cache: {cache_dir}")

        # Compute actual performance from fine-tuned models
        import gc

        from torch.utils.data import DataLoader
        from transformers import default_data_collator

        from src.data.loaders import load_hf_dataset, preprocess_dataset
        from src.models.loaders import load_model, load_tokenizer
        from src.utils.device import get_device

        device = get_device()
        utopia_losses = []

        for task_name in task_names:
            task_cfg = dataset_configs[task_name]
            finetuned_checkpoint = task_cfg.finetuned_checkpoint

            logger.info(f"  Evaluating {task_name} ({finetuned_checkpoint})...")

            # Load fine-tuned model
            tokenizer = load_tokenizer(finetuned_checkpoint, cache_dir=cache_dir)
            model = load_model(
                model_id=finetuned_checkpoint,
                num_labels=task_cfg.num_labels,
                device=device,
                cache_dir=cache_dir,
            )
            model.eval()

            # Load validation data (small sample for speed)
            dataset = load_hf_dataset(
                dataset_path=task_cfg.hf_dataset.path,
                subset=task_cfg.hf_dataset.get("subset", None),
                split=task_cfg.hf_dataset.split.get("validation", task_cfg.hf_dataset.split.test),
            )

            # Limit to 200 samples for speed
            if len(dataset) > 200:
                dataset = dataset.select(range(200))

            # Preprocess
            processed = preprocess_dataset(
                dataset=dataset,
                tokenizer=tokenizer,
                text_column=task_cfg.preprocessing.text_column,
                text_column_2=task_cfg.preprocessing.get("text_column_2", None),
                label_column=task_cfg.preprocessing.label_column,
                max_length=task_cfg.preprocessing.max_length,
                truncation=task_cfg.preprocessing.truncation,
                padding=task_cfg.preprocessing.padding,
            )

            # Create dataloader
            dataloader = DataLoader(
                processed,
                batch_size=32,
                shuffle=False,
                collate_fn=default_data_collator,
            )

            # Evaluate
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    num_batches += 1

            avg_loss = total_loss / num_batches
            utopia_losses.append(-avg_loss)  # Negative loss (higher is better)

            logger.info(f"    Average loss: {avg_loss:.4f} -> Utopia: {-avg_loss:.4f}")

            # Clear GPU memory for next model
            del model
            del tokenizer
            del dataloader
            del processed
            del dataset
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        utopia_tensor = torch.tensor(utopia_losses, dtype=torch.float32)

        logger.info(f"\nUtopia point (negative loss): {utopia_tensor.tolist()}")

        return utopia_tensor
