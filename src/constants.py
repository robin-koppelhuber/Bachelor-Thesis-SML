"""Project-wide constants

This module contains only true constants that never vary and don't belong in configs.
For configurable values (models, datasets, paths), use Hydra configs instead.
"""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent

# Standardized metric names (for consistency across codebase)
# These match scikit-learn's metric naming conventions
METRIC_ACCURACY = "accuracy"
METRIC_F1_MACRO = "f1_macro"
METRIC_F1_WEIGHTED = "f1_weighted"
METRIC_F1_MICRO = "f1_micro"
METRIC_PRECISION_MACRO = "precision_macro"
METRIC_RECALL_MACRO = "recall_macro"
