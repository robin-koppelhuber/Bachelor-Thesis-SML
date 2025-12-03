"""TIES: Resolving Interference When Merging Models

Paper: https://arxiv.org/abs/2306.01708

TIES merging consists of three steps:
1. Trim: Remove small parameter changes (keep top-k by magnitude)
2. Elect Sign: Resolve sign conflicts by taking majority vote
3. Disjoint Merge: Average parameters with the same sign, weighted by preference
"""

from typing import Dict, Optional
import torch
import numpy as np
import logging

from src.methods.base import BaseMergingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("ties")
class TIESMerging(BaseMergingMethod):
    """
    TIES: Trim, Elect Sign, and Merge

    Implementation of TIES merging with preference-aware weighting.
    """

    def __init__(
        self,
        k: float = 0.2,
        lambda_merge: float = 1.0,
        scaling_method: str = "magnitude",
        use_preferences: bool = True,
        normalize_preferences: bool = True,
        sign_consensus_method: str = "majority",
        **kwargs,
    ):
        """
        Initialize TIES merging

        Args:
            k: Top-k fraction to keep (trim step)
            lambda_merge: Merge coefficient (scaling factor)
            scaling_method: How to scale merged vector ('magnitude', 'sign', 'none')
            use_preferences: Whether to use preference vector for weighting
            normalize_preferences: Whether to normalize preference vector
            sign_consensus_method: Method for sign election ('majority', 'weighted')
            **kwargs: Additional parameters
        """
        super().__init__(
            k=k,
            lambda_merge=lambda_merge,
            scaling_method=scaling_method,
            use_preferences=use_preferences,
            normalize_preferences=normalize_preferences,
            sign_consensus_method=sign_consensus_method,
            **kwargs,
        )
        self.k = k
        self.lambda_merge = lambda_merge
        self.scaling_method = scaling_method
        self.use_preferences = use_preferences
        self.normalize_preferences = normalize_preferences
        self.sign_consensus_method = sign_consensus_method

    def merge(
        self,
        task_vectors: Dict[str, torch.Tensor],
        preference_vector: np.ndarray,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Merge task vectors using TIES method

        Args:
            task_vectors: Dictionary mapping task names to task vectors
            preference_vector: Preference weights for each task
            base_model: Not used for this method

        Returns:
            Merged parameter vector
        """
        # Normalize preferences if requested
        if self.normalize_preferences:
            preference_vector = preference_vector / preference_vector.sum()

        # Get task names in consistent order
        task_names = sorted(task_vectors.keys())
        n_tasks = len(task_names)

        logger.debug(f"Merging {n_tasks} tasks with TIES (k={self.k})")

        # Stack task vectors
        task_vector_list = [task_vectors[name] for name in task_names]
        stacked_vectors = torch.stack(task_vector_list, dim=0)  # (n_tasks, n_params)

        # Step 1: Trim - Keep only top-k parameters by magnitude
        trimmed_vectors = self._trim(stacked_vectors)

        # Step 2: Elect Sign - Resolve sign conflicts
        sign_vector = self._elect_sign(trimmed_vectors, preference_vector)

        # Step 3: Disjoint Merge - Average parameters with same sign
        merged_vector = self._disjoint_merge(
            trimmed_vectors, sign_vector, preference_vector
        )

        # Apply merge coefficient
        merged_vector = self.lambda_merge * merged_vector

        return merged_vector

    def _trim(self, task_vectors: torch.Tensor) -> torch.Tensor:
        """
        Trim: Keep only top-k parameters by magnitude per task

        Args:
            task_vectors: Stacked task vectors (n_tasks, n_params)

        Returns:
            Trimmed task vectors
        """
        # TODO: Implement trimming
        # For each task vector, keep only top-k% parameters by magnitude
        # Set other parameters to zero

        # Placeholder: Return as-is for now
        logger.warning("TIES trimming not implemented yet - using all parameters")
        return task_vectors

    def _elect_sign(
        self, task_vectors: torch.Tensor, preference_vector: np.ndarray
    ) -> torch.Tensor:
        """
        Elect Sign: Determine consensus sign for each parameter

        Args:
            task_vectors: Trimmed task vectors (n_tasks, n_params)
            preference_vector: Preference weights

        Returns:
            Sign vector (-1, 0, or 1 for each parameter)
        """
        # TODO: Implement sign election
        # For each parameter, determine the majority sign (positive or negative)
        # Can use preference_vector to weight votes

        # Placeholder: Use simple sign of mean for now
        logger.warning(
            "TIES sign election not implemented yet - using sign of mean"
        )
        mean_vector = task_vectors.mean(dim=0)
        return torch.sign(mean_vector)

    def _disjoint_merge(
        self,
        task_vectors: torch.Tensor,
        sign_vector: torch.Tensor,
        preference_vector: np.ndarray,
    ) -> torch.Tensor:
        """
        Disjoint Merge: Average parameters that agree with consensus sign

        Args:
            task_vectors: Trimmed task vectors (n_tasks, n_params)
            sign_vector: Consensus sign for each parameter
            preference_vector: Preference weights

        Returns:
            Merged vector
        """
        # TODO: Implement disjoint merge
        # For each parameter, average values from tasks that agree with consensus sign
        # Weight by preference_vector if use_preferences is True

        # Placeholder: Simple weighted average for now
        logger.warning(
            "TIES disjoint merge not implemented yet - using weighted average"
        )
        preference_tensor = torch.tensor(
            preference_vector, dtype=task_vectors.dtype, device=task_vectors.device
        )
        preference_tensor = preference_tensor.view(-1, 1)  # (n_tasks, 1)

        if self.use_preferences:
            merged_vector = (task_vectors * preference_tensor).sum(dim=0)
        else:
            merged_vector = task_vectors.mean(dim=0)

        return merged_vector
