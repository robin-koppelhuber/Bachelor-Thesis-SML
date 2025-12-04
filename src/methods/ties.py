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
            Trimmed task vectors with bottom (1-k) parameters set to zero
        """
        if self.k >= 1.0:
            # Keep all parameters if k >= 1.0
            logger.debug("k >= 1.0, keeping all parameters")
            return task_vectors

        n_tasks, n_params = task_vectors.shape
        k_params = int(n_params * self.k)

        if k_params == 0:
            logger.warning(f"k={self.k} results in 0 parameters, keeping at least 1")
            k_params = 1

        logger.debug(f"Trimming to top-{k_params} params ({self.k*100:.1f}%) per task")

        # For each task vector, keep only top-k by absolute magnitude
        trimmed_vectors = torch.zeros_like(task_vectors)

        for i in range(n_tasks):
            # Get absolute values
            abs_values = torch.abs(task_vectors[i])

            # Find top-k indices
            _, topk_indices = torch.topk(abs_values, k=k_params, largest=True)

            # Keep only top-k values (preserve original signs)
            trimmed_vectors[i, topk_indices] = task_vectors[i, topk_indices]

        return trimmed_vectors

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
        if self.sign_consensus_method == "majority":
            # Simple majority: sign of sum across tasks
            sign_vector = torch.sign(task_vectors.sum(dim=0))
            logger.debug("Using majority sign election")

        elif self.sign_consensus_method == "weighted":
            # Weighted by preference vector
            preference_tensor = torch.tensor(
                preference_vector,
                dtype=task_vectors.dtype,
                device=task_vectors.device,
            ).view(-1, 1)

            # Weighted sum
            weighted_sum = (task_vectors * preference_tensor).sum(dim=0)
            sign_vector = torch.sign(weighted_sum)
            logger.debug("Using preference-weighted sign election")

        else:
            raise ValueError(
                f"Unknown sign_consensus_method: {self.sign_consensus_method}"
            )

        # Resolve zeros (ties) to majority sign
        # Count positive vs negative occurrences
        num_positive = (task_vectors > 0).sum(dim=0).float()
        num_negative = (task_vectors < 0).sum(dim=0).float()

        # Where sign is zero, use majority
        zero_mask = sign_vector == 0
        sign_vector[zero_mask] = torch.where(
            num_positive[zero_mask] >= num_negative[zero_mask],
            torch.ones_like(sign_vector[zero_mask]),
            -torch.ones_like(sign_vector[zero_mask]),
        )

        return sign_vector

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
        # Create mask for values that agree with elected sign
        # For positive sign: keep positive values
        # For negative sign: keep negative values
        # Shape: (n_tasks, n_params)
        sign_agreement_mask = torch.where(
            sign_vector.unsqueeze(0) > 0,
            task_vectors > 0,  # If elected sign is positive, keep positive values
            task_vectors < 0,  # If elected sign is negative, keep negative values
        )

        # Apply mask to get only agreeing values
        selected_values = task_vectors * sign_agreement_mask

        if self.use_preferences:
            # Weighted average by preference
            preference_tensor = torch.tensor(
                preference_vector,
                dtype=task_vectors.dtype,
                device=task_vectors.device,
            ).view(-1, 1)

            # Weight the selected values
            weighted_values = selected_values * preference_tensor

            # Sum weighted values
            numerator = weighted_values.sum(dim=0)

            # Sum of weights for non-zero values
            # Only count weights where the value was actually selected (non-zero)
            weight_mask = sign_agreement_mask.float() * preference_tensor
            denominator = weight_mask.sum(dim=0)

            # Avoid division by zero
            denominator = torch.clamp(denominator, min=1e-8)

            merged_vector = numerator / denominator

            logger.debug("Using preference-weighted disjoint merge")

        else:
            # Simple mean of agreeing values
            non_zero_counts = sign_agreement_mask.sum(dim=0).float()
            numerator = selected_values.sum(dim=0)

            # Avoid division by zero
            denominator = torch.clamp(non_zero_counts, min=1)

            merged_vector = numerator / denominator

            logger.debug("Using unweighted disjoint merge")

        return merged_vector
