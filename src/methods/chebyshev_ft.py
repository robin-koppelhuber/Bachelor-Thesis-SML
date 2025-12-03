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

from src.methods.base import BaseMergingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("chebyshev")
class ChebyshevFineTuning(BaseMergingMethod):
    """
    Fine-tuning with Chebyshev scalarization

    This is a training-based method that fine-tunes from the base model
    using Chebyshev scalarization to optimize for a preference vector.
    """

    def __init__(
        self,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        use_augmented: bool = True,
        epsilon: float = 0.001,
        normalize_preferences: bool = True,
        utopia_point: Optional[np.ndarray] = None,
        nadir_point: Optional[np.ndarray] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        """
        Initialize Chebyshev fine-tuning

        Args:
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            use_augmented: Use augmented Chebyshev scalarization
            epsilon: Augmentation weight for smoothness
            normalize_preferences: Normalize preference vector
            utopia_point: Best performance per task (computed if None)
            nadir_point: Worst performance per task (computed if None)
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional parameters
        """
        super().__init__(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            use_augmented=use_augmented,
            epsilon=epsilon,
            normalize_preferences=normalize_preferences,
            utopia_point=utopia_point,
            nadir_point=nadir_point,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            **kwargs,
        )
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_augmented = use_augmented
        self.epsilon = epsilon
        self.normalize_preferences = normalize_preferences
        self.utopia_point = utopia_point
        self.nadir_point = nadir_point
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

    def merge(
        self,
        task_vectors: Dict[str, torch.Tensor],
        preference_vector: np.ndarray,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        This method requires training, not simple merging of task vectors

        For the POC, we'll need to implement the actual training loop.
        This is a placeholder that raises NotImplementedError.

        Args:
            task_vectors: Not used for this method
            preference_vector: Preference weights for each task
            base_model: Base model to fine-tune from

        Returns:
            Trained model parameters (as task vector from base)

        Raises:
            NotImplementedError: This method requires implementation of training loop
        """
        # TODO: Implement Chebyshev fine-tuning training loop
        # This requires:
        # 1. Loading training data for all tasks
        # 2. Setting up optimizer and scheduler
        # 3. Training loop with Chebyshev scalarization loss
        # 4. Computing task vector from trained model

        logger.error("Chebyshev fine-tuning not implemented yet")
        raise NotImplementedError(
            "Chebyshev fine-tuning requires training loop implementation. "
            "This method cannot be used with simple task vector merging."
        )

    def compute_chebyshev_loss(
        self,
        task_losses: torch.Tensor,
        preference_vector: torch.Tensor,
        utopia_point: torch.Tensor,
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
        weighted_deviations = preference_vector * (utopia_point - task_losses)

        # Chebyshev: minimize maximum weighted deviation
        chebyshev_loss = -weighted_deviations.max()

        # Augmented Chebyshev: add sum term for smoothness
        if self.use_augmented:
            chebyshev_loss += self.epsilon * weighted_deviations.sum()

        return chebyshev_loss
