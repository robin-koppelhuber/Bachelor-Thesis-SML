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

        logger.info(f"Merging {n_tasks} tasks with TIES (k={self.k})")

        # Stack task vectors
        task_vector_list = [task_vectors[name] for name in task_names]
        logger.info(f"  Stacking {n_tasks} task vectors...")
        stacked_vectors = torch.stack(task_vector_list, dim=0)  # (n_tasks, n_params)
        logger.info(f"  Stacked shape: {stacked_vectors.shape}, device: {stacked_vectors.device}, dtype: {stacked_vectors.dtype}")

        # Step 1: Trim - Keep only top-k parameters by magnitude
        logger.info(f"  Step 1: Trimming (k={self.k})...")
        trimmed_vectors = self._trim(stacked_vectors)
        logger.info(f"  ✓ Trimming complete")

        # Free original stacked vectors to save memory
        del stacked_vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Step 2: Elect Sign - Resolve sign conflicts
        logger.info(f"  Step 2: Electing signs...")
        sign_vector = self._elect_sign(trimmed_vectors, preference_vector)
        logger.info(f"  ✓ Sign election complete")

        # Step 3: Disjoint Merge - Average parameters with same sign
        logger.info(f"  Step 3: Disjoint merge...")
        merged_vector = self._disjoint_merge(
            trimmed_vectors, sign_vector, preference_vector
        )
        logger.info(f"  ✓ Disjoint merge complete")

        # Free intermediate tensors
        del trimmed_vectors, sign_vector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        logger.info(f"    Trimming to top-{k_params:,} params ({self.k*100:.1f}%) per task from {n_params:,} total")
        logger.info(f"    Device: {task_vectors.device}, Memory: {task_vectors.element_size() * task_vectors.nelement() / 1024**3:.2f} GB")

        # For each task vector, keep only top-k by absolute magnitude
        logger.info(f"    Allocating output tensor ({task_vectors.element_size() * task_vectors.nelement() / 1024**3:.2f} GB)...")
        trimmed_vectors = torch.zeros_like(task_vectors)
        logger.info(f"    ✓ Allocation complete")

        for i in range(n_tasks):
            logger.info(f"    Processing task {i+1}/{n_tasks}...")

            # Memory-efficient approach: use sampling-based threshold on large CPU tensors
            # topk and quantile on large CPU tensors can cause memory issues or segfaults
            if task_vectors.device.type == 'cpu' and n_params > 10_000_000:
                logger.info(f"      Using sampling-based threshold (memory-efficient for large CPU tensors)")
                # Get absolute values
                abs_values = torch.abs(task_vectors[i])

                # Sample a subset to estimate threshold (much faster than quantile on full tensor)
                # Using 5M samples gives better accuracy while still avoiding segfaults
                sample_size = min(5_000_000, n_params)
                sample_indices = torch.randperm(n_params)[:sample_size]
                sampled_abs = abs_values[sample_indices]

                # Find threshold on sample that would keep approximately top-k
                threshold = torch.quantile(sampled_abs, 1.0 - self.k)

                # Apply threshold to full tensor
                mask = abs_values >= threshold
                trimmed_vectors[i] = torch.where(mask, task_vectors[i], torch.zeros_like(task_vectors[i]))

                actual_k = mask.sum().item()
                logger.info(f"      ✓ Kept {actual_k:,} parameters (target: {k_params:,}, threshold: {threshold:.6f})")
            else:
                # Standard topk approach (fast on GPU or small tensors)
                abs_values = torch.abs(task_vectors[i])
                logger.info(f"      Finding top-{k_params:,} indices...")
                _, topk_indices = torch.topk(abs_values, k=k_params, largest=True)
                logger.info(f"      ✓ Found top-k indices")

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
        logger.info(f"      Computing sign vector...")

        if self.sign_consensus_method == "majority":
            # Simple majority: sign of sum across tasks (memory-efficient)
            sign_vector = torch.sign(task_vectors.sum(dim=0))
            logger.debug("Using majority sign election")

        elif self.sign_consensus_method == "weighted":
            # Weighted by preference vector
            preference_tensor = torch.tensor(
                preference_vector,
                dtype=task_vectors.dtype,
                device=task_vectors.device,
            ).view(-1, 1)

            # Weighted sum (accumulate to save memory)
            weighted_sum = (task_vectors * preference_tensor).sum(dim=0)
            sign_vector = torch.sign(weighted_sum)
            del weighted_sum  # Free memory immediately
            logger.debug("Using preference-weighted sign election")

        else:
            raise ValueError(
                f"Unknown sign_consensus_method: {self.sign_consensus_method}"
            )

        # Resolve zeros (ties) to majority sign
        # Only process where sign is zero to save memory
        zero_mask = sign_vector == 0
        num_zeros = zero_mask.sum().item()

        if num_zeros > 0:
            logger.info(f"      Resolving {num_zeros:,} tied parameters...")
            # Count positive vs negative for zero positions only (memory-efficient)
            # Process each task vector to avoid creating large boolean matrices
            num_positive_zeros = torch.zeros(num_zeros, dtype=torch.float32, device=task_vectors.device)
            num_negative_zeros = torch.zeros(num_zeros, dtype=torch.float32, device=task_vectors.device)

            zero_indices = torch.where(zero_mask)[0]
            for i in range(task_vectors.shape[0]):
                task_zero_vals = task_vectors[i, zero_indices]
                num_positive_zeros += (task_zero_vals > 0).float()
                num_negative_zeros += (task_zero_vals < 0).float()

            # Assign majority sign
            sign_vector[zero_indices] = torch.where(
                num_positive_zeros >= num_negative_zeros,
                torch.ones_like(num_positive_zeros),
                -torch.ones_like(num_positive_zeros),
            )

            del num_positive_zeros, num_negative_zeros, zero_indices

        del zero_mask
        logger.info(f"      ✓ Sign vector computed")
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
        logger.info(f"      Computing disjoint merge...")

        # Initialize result tensors
        numerator = torch.zeros(task_vectors.shape[1], dtype=task_vectors.dtype, device=task_vectors.device)
        denominator = torch.zeros(task_vectors.shape[1], dtype=task_vectors.dtype, device=task_vectors.device)

        # Prepare preference weights if needed
        if self.use_preferences:
            preference_tensor = torch.tensor(
                preference_vector,
                dtype=task_vectors.dtype,
                device=task_vectors.device,
            )

        # Process each task vector individually to save memory
        # This avoids creating large (n_tasks, n_params) boolean masks
        for i in range(task_vectors.shape[0]):
            task_vec = task_vectors[i]

            # Create agreement mask for this task only (saves memory)
            # For positive elected sign: keep positive task values
            # For negative elected sign: keep negative task values
            pos_sign = sign_vector > 0
            neg_sign = sign_vector < 0
            task_pos = task_vec > 0
            task_neg = task_vec < 0

            agreement_mask = (pos_sign & task_pos) | (neg_sign & task_neg)

            # Apply mask
            selected_values = torch.where(agreement_mask, task_vec, torch.zeros_like(task_vec))

            if self.use_preferences:
                # Weight by preference
                weight = preference_tensor[i]
                numerator += selected_values * weight
                denominator += agreement_mask.float() * weight
            else:
                numerator += selected_values
                denominator += agreement_mask.float()

        # Avoid division by zero
        if self.use_preferences:
            denominator = torch.clamp(denominator, min=1e-8)
            logger.debug("Using preference-weighted disjoint merge")
        else:
            denominator = torch.clamp(denominator, min=1.0)
            logger.debug("Using unweighted disjoint merge")

        merged_vector = numerator / denominator

        logger.info(f"      ✓ Disjoint merge computed")
        return merged_vector
