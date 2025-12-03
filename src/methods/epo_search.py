"""Exact Pareto Optimization (EPO) Search

Paper: "Multi-Task Learning as Multi-Objective Optimization" (Liu et al., 2021)
https://arxiv.org/abs/2108.00597

EPO uses preference-aware Multiple Gradient Descent Algorithm (MGDA) to find
Pareto optimal solutions. At each training step, it solves a quadratic program
to find the optimal weighting of task gradients that minimizes the overall loss
while respecting the preference vector.

The key idea is to find the optimal task weights that:
1. Keep all task gradients moving in a descent direction
2. Respect the preference vector (bias toward preferred tasks)
3. Find Pareto optimal solutions (no task can improve without hurting another)
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from src.methods.base import BaseMergingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("epo")
class EPOSearch(BaseMergingMethod):
    """
    Exact Pareto Optimization (EPO) Search

    This is a training-based method that uses preference-aware MGDA to find
    Pareto optimal solutions for multi-task learning.
    """

    def __init__(
        self,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 16,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        solver: str = "quadprog",
        n_tasks: int = 4,
        use_preferences: bool = True,
        normalize_preferences: bool = True,
        normalization_type: str = "loss",
        update_weights_every: int = 1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        """
        Initialize EPO search

        Args:
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            solver: QP solver to use ('quadprog', 'cvxpy', 'scipy')
            n_tasks: Number of tasks
            use_preferences: Use preference vector for weighting
            normalize_preferences: Normalize preference vector
            normalization_type: Gradient normalization ('loss', 'loss+', 'none')
            update_weights_every: Update task weights every N steps
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
            solver=solver,
            n_tasks=n_tasks,
            use_preferences=use_preferences,
            normalize_preferences=normalize_preferences,
            normalization_type=normalization_type,
            update_weights_every=update_weights_every,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            **kwargs,
        )
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.solver = solver
        self.n_tasks = n_tasks
        self.use_preferences = use_preferences
        self.normalize_preferences = normalize_preferences
        self.normalization_type = normalization_type
        self.update_weights_every = update_weights_every
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

        For the POC, we'll need to implement the actual training loop with EPO.
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
        # TODO: Implement EPO training loop
        # This requires:
        # 1. Loading training data for all tasks
        # 2. Setting up optimizer and scheduler
        # 3. Training loop with EPO gradient weighting
        # 4. Solving QP at each step to find optimal task weights
        # 5. Computing task vector from trained model

        logger.error("EPO search not implemented yet")
        raise NotImplementedError(
            "EPO search requires training loop implementation. "
            "This method cannot be used with simple task vector merging."
        )

    def solve_epo_qp(
        self,
        task_gradients: torch.Tensor,
        preference_vector: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve EPO quadratic program to find optimal task weights

        The QP minimizes:
            || sum_i w_i * g_i ||^2
        subject to:
            w_i >= 0
            sum_i w_i = 1
            w approximates preference_vector

        Args:
            task_gradients: Gradients for each task (n_tasks, n_params)
            preference_vector: Preference weights (n_tasks,)

        Returns:
            Optimal task weights (n_tasks,)
        """
        # TODO: Implement QP solver
        # Use specified solver (quadprog, cvxpy, or scipy)

        logger.warning("EPO QP solver not implemented yet - using preference vector")
        return preference_vector
