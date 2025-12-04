"""Model merging methods

This module imports all merging methods to ensure they are registered.
"""

# Import base classes
from src.methods.base import BaseMergingMethod, BaseTrainingMethod  # noqa: F401

# Import all methods to trigger registration
from src.methods.averaging import AveragingMerging  # noqa: F401
from src.methods.ties import TIESMerging  # noqa: F401
from src.methods.chebyshev_ft import ChebyshevFineTuning  # noqa: F401

# Registry will be imported from base module
from src.methods.registry import MethodRegistry  # noqa: F401

__all__ = [
    "MethodRegistry",
    "BaseMergingMethod",
    "BaseTrainingMethod",
    "AveragingMerging",
    "TIESMerging",
    "ChebyshevFineTuning",
]
