"""Base class for model merging methods"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import numpy as np


# ABC / abstract base class forces subclasses to implement all abstract methods
class BaseMergingMethod(ABC):
    """Base class for model merging methods"""

    def __init__(self, **kwargs):
        """
        Initialize merging method

        Args:
            **kwargs: Method-specific parameters
        """
        self.params = kwargs

    @abstractmethod
    def merge(
        self,
        task_vectors: Dict[str, torch.Tensor],
        preference_vector: np.ndarray,
        base_model: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """
        Merge task vectors according to preference vector

        Args:
            task_vectors: Dictionary mapping task names to task vectors
            preference_vector: Preference weights for each task
            base_model: Optional base model for methods that need it

        Returns:
            Merged parameter vector
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"
