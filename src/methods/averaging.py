"""Simple averaging baseline for model merging"""

from typing import Dict, Optional

import numpy as np
import torch

from src.methods.base import BaseMergingMethod
from src.methods.registry import MethodRegistry


@MethodRegistry.register("averaging")
class AveragingMerging(BaseMergingMethod):
    """
    Simple weighted averaging of task vectors

    This is a baseline method that performs weighted averaging of task vectors
    according to the preference vector.
    """

    def __init__(
        self,
        normalize_preferences: bool = True,
        **kwargs,
    ):
        """
        Initialize averaging method

        Args:
            normalize_preferences: Whether to normalize preference vector to sum to 1
            **kwargs: Additional parameters
        """
        super().__init__(normalize_preferences=normalize_preferences, **kwargs)
        self.normalize_preferences = normalize_preferences

    def merge(
        self,
        task_vectors: Dict[str, torch.Tensor],
        preference_vector: np.ndarray,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Merge task vectors via weighted averaging

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

        # Weighted sum of task vectors
        merged_vector = None
        for i, task_name in enumerate(task_names):
            weight = preference_vector[i]
            task_vector = task_vectors[task_name]

            if merged_vector is None:
                merged_vector = weight * task_vector
            else:
                merged_vector += weight * task_vector

        return merged_vector
