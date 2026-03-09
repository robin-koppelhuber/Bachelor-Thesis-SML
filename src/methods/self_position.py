"""Self Position Search — Convex Hull Pareto Optimizer

Optimizes mixing coefficients α over single-task fine-tuned checkpoints so that
the merged model θ(α) = base_θ + Σ_k α_k · τ_k minimizes the preference-weighted
multi-task loss:

    min_α  Σ_k r_k · L_k(θ(α))
    s.t.   α = softmax(β),  β ∈ R^T  (unconstrained reparametrization)

Only T mixing logits β are trainable — model parameters are NOT trained; they are
deterministically reconstructed at each gradient step from the fixed task vectors.

Gradient computation (chain rule):
    ∇_α_j (Σ_k r_k L_k) = ⟨∇_θ (Σ_k r_k L_k), τ_j⟩

This requires one combined backward pass (not T separate passes) + T CPU dot
products with the task vectors — cheaper than EPO.

Adam is used on β (unconstrained logits); softmax ensures α ∈ Δ_T at all times
without explicit simplex projection.

Optional TIES preprocessing (all α-independent, precomputed once):
  1. Trim: per-task top-k% by magnitude
  2. Elect global sign: γ_p = sign(Σ_k τ̂_k_p)  [unweighted — no α dependency]
  3. Disjoint mask: τ̃_k_p = 0 where sign(τ̂_k_p) ≠ γ_p

Reference: "Improving General Text Embedding Model: Tackling Task Conflict and
           Data Imbalance through Model Merging" (arXiv:2410.15035).
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from src.methods.base import BaseTrainingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


@MethodRegistry.register("self_position")
class SelfPositionSearch(BaseTrainingMethod):
    """
    Self Position Search: preference-weighted convex hull optimizer.

    Stays within the convex hull of fine-tuned checkpoints (H1 optimized).
    Optimizes mixing logits β via Adam; reconstructs θ(α) from fixed task vectors.

    Hypothesis class: H1 (optimized) — uses training data, stays in convex hull.
    Compare with:
      - TIES: H1 zero-shot (no optimization, no training data)
      - Chebyshev/EPO: H2 unconstrained (full fine-tuning, moves beyond convex hull)
    """

    def __init__(
        self,
        lr_alpha: float = 5e-3,
        ties_preprocessing: bool = False,
        ties_k: float = 0.2,
        invert_preference: bool = False,
        **kwargs,
    ):
        """
        Args:
            lr_alpha: Adam learning rate for mixing logits β. Default 5e-3 (paper).
            ties_preprocessing: If True, apply TIES steps 1–3 (trim + global sign
                election + disjoint mask) to task vectors before optimization.
                All steps are α-independent and can be precomputed once.
            ties_k: Top-k fraction to keep in TIES trimming. Default 0.2.
            invert_preference: If True, transform r → normalise(1/r) before use.
                For ablations when the opposite preference convention is needed.
            **kwargs: Forwarded to BaseTrainingMethod (num_epochs, batch_size, etc.).
        """
        super().__init__(**kwargs)
        self.lr_alpha = lr_alpha
        self.ties_preprocessing = ties_preprocessing
        self.ties_k = ties_k
        self.invert_preference = invert_preference

        # Instance state — initialized per training run in _get_training_kwargs
        self._beta: Optional[torch.Tensor] = None
        self._alpha_optimizer: Optional[torch.optim.Adam] = None
        self._task_vectors_cpu: Optional[Dict[str, torch.Tensor]] = None
        self._base_params_cpu: Optional[torch.Tensor] = None
        self._task_names_ordered: Optional[List[str]] = None

        # Captured in overridden train() for use in _load_task_vectors
        self._base_model_id: Optional[str] = None
        self._model_cache_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Override train() to capture base_model and model_cache_dir
    # ------------------------------------------------------------------

    def train(self, base_model: str, *args, model_cache_dir=None, **kwargs):
        """Capture base model ID and cache dir, then delegate to BaseTrainingMethod."""
        self._base_model_id = base_model
        self._model_cache_dir = model_cache_dir
        return super().train(base_model, *args, model_cache_dir=model_cache_dir, **kwargs)

    # ------------------------------------------------------------------
    # BaseTrainingMethod interface
    # ------------------------------------------------------------------

    def _compute_multi_task_loss(
        self,
        task_losses: torch.Tensor,
        preference_vector: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fallback to weighted-sum scalarization.

        Not called during normal Self Position training (which overrides
        _train_epoch). Defined to satisfy the abstract method contract.
        """
        return (preference_vector * task_losses).sum()

    def _get_training_kwargs(
        self,
        task_names: list,
        preference_vector,
        dataset_configs: Dict,
        cache_dir: Optional[str] = None,
        reference_losses: Optional[Dict] = None,
    ) -> Dict:
        """
        Initialize mixing logits β and Adam optimizer for this training run.

        Called once per preference vector by BaseTrainingMethod.train().
        Resets all per-run state so multiple preference vectors work correctly.
        """
        T = len(task_names)
        self._task_names_ordered = list(task_names)

        # Reset task vectors — loaded lazily on first _train_epoch call
        self._task_vectors_cpu = None
        self._base_params_cpu = None

        # β initialized to zeros → α = softmax(0) = [1/T, ..., 1/T]
        self._beta = torch.zeros(T, requires_grad=True)
        self._alpha_optimizer = torch.optim.Adam([self._beta], lr=self.lr_alpha)

        logger.info(
            f"Self Position init: T={T}, lr_alpha={self.lr_alpha}, TIES preprocessing={self.ties_preprocessing}"
        )
        logger.info(f"  Initial α = [{1 / T:.4f}] * {T}  (from β=0, softmax)")

        return {"finetuned_model_cache_dir": cache_dir}

    # ------------------------------------------------------------------
    # Task vector loading
    # ------------------------------------------------------------------

    def _load_task_vectors(
        self,
        model: torch.nn.Module,
        task_names: List[str],
        dataset_configs: Dict,
        finetuned_model_cache_dir: Optional[str],
    ) -> None:
        """
        Load and store task vectors τ_k = ft_k_params − base_k_params.

        Task vectors are stored as flat float32 CPU tensors in the same
        ordering as model.parameters() so that dot products with flat gradients
        are well-defined.

        Handles classifier head size mismatches (different num_labels per task):
        the extra rows (from max_labels − num_labels_k) are zero-padded.

        If self.ties_preprocessing is True, applies TIES steps 1–3 after loading.

        Called once on the first epoch of the first preference vector.
        """
        from pathlib import Path

        from src.models.loaders import compute_task_vector, load_model

        logger.info("Loading task vectors for Self Position...")

        # Save base model params as flat float32 CPU tensor (max-labels model)
        self._base_params_cpu = torch.cat([p.detach().cpu().float().flatten() for p in model.parameters()])
        base_size = self._base_params_cpu.shape[0]
        logger.info(f"  Base model: {base_size:,} parameters")

        self._task_vectors_cpu = {}

        for task_name in task_names:
            dataset_cfg = dataset_configs[task_name]

            # Load task-specific base (same num_labels as ft so shapes match for subtraction)
            task_base = load_model(
                model_id=self._base_model_id,
                num_labels=dataset_cfg.num_labels,
                cache_dir=Path(self._model_cache_dir) if self._model_cache_dir else None,
                device=torch.device("cpu"),
                zero_classifier=True,
            )
            ft_model = load_model(
                model_id=dataset_cfg.finetuned_checkpoint,
                num_labels=None,
                cache_dir=Path(finetuned_model_cache_dir) if finetuned_model_cache_dir else None,
                device=torch.device("cpu"),
            )

            # Reuse existing utility: returns {param_name: ft_param - base_param}
            tv_dict = compute_task_vector(ft_model, task_base)
            del task_base, ft_model

            # Build flat τ_k in model.named_parameters() order — must match the
            # ordering used by gradient collection in _train_epoch.
            # Classifier head size may differ (task num_labels < max_labels):
            # those extra rows are zero-padded so τ_k has the same size as base_params.
            tau_parts = []
            for name, base_param in model.named_parameters():
                expected_size = base_param.numel()
                if name in tv_dict:
                    diff = tv_dict[name].cpu().float().flatten()
                    actual_size = diff.shape[0]
                    if actual_size == expected_size:
                        tau_parts.append(diff)
                    elif actual_size < expected_size:
                        padded = torch.zeros(expected_size)
                        padded[:actual_size] = diff
                        tau_parts.append(padded)
                    else:
                        logger.warning(
                            f"  {task_name}/{name}: diff size {actual_size} > base size {expected_size} — truncating"
                        )
                        tau_parts.append(diff[:expected_size])
                else:
                    tau_parts.append(torch.zeros(expected_size))

            tau_k = torch.cat(tau_parts)

            logger.info(
                f"  {task_name}: τ_k size={tau_k.shape[0]:,}, "
                f"norm={tau_k.norm():.4f}, "
                f"nnz={(tau_k != 0).sum().item():,}"
            )
            self._task_vectors_cpu[task_name] = tau_k

        if self.ties_preprocessing:
            self._apply_ties_preprocessing(task_names)

    def _apply_ties_preprocessing(self, task_names: List[str]) -> None:
        """
        Apply TIES steps 1–3 to task vectors (all α-independent).

        Step 1 — Trim: keep top-k% params by magnitude per task, zero the rest.
        Step 2 — Elect global sign: γ_p = sign(Σ_k τ̂_k_p)  [unweighted sum].
        Step 3 — Disjoint mask: set τ̃_k_p = 0 where sign(τ̂_k_p) ≠ γ_p.

        Self Position then optimizes α over the masked task vectors {τ̃_k}.
        """
        logger.info(f"Applying TIES preprocessing (k={self.ties_k})...")
        T = len(task_names)
        stacked = torch.stack([self._task_vectors_cpu[name] for name in task_names], dim=0)  # (T, N)
        n_params = stacked.shape[1]
        k_params = max(1, int(n_params * self.ties_k))

        # Step 1: Trim (per-task)
        trimmed = torch.zeros_like(stacked)
        for i in range(T):
            abs_vals = stacked[i].abs()
            if n_params > 10_000_000:
                # Sampling-based threshold for large tensors (avoids OOM)
                sample_size = min(5_000_000, n_params)
                sample_idx = torch.randperm(n_params)[:sample_size]
                threshold = torch.quantile(abs_vals[sample_idx], 1.0 - self.ties_k)
                mask = abs_vals >= threshold
            else:
                _, topk_idx = torch.topk(abs_vals, k=k_params)
                mask = torch.zeros(n_params, dtype=torch.bool)
                mask[topk_idx] = True
            trimmed[i] = torch.where(mask, stacked[i], torch.zeros_like(stacked[i]))
            nnz = mask.sum().item()
            logger.info(f"  Trim {task_names[i]}: kept {nnz:,}/{n_params:,} params")

        # Step 2: Elect global sign (unweighted — α-independent)
        global_sign = torch.sign(trimmed.sum(dim=0))  # (N,)

        # Step 3: Disjoint mask
        for i, task_name in enumerate(task_names):
            conflict_mask = (torch.sign(trimmed[i]) != global_sign) & (trimmed[i] != 0)
            trimmed[i][conflict_mask] = 0.0
            self._task_vectors_cpu[task_name] = trimmed[i]
            nnz_after = (trimmed[i] != 0).sum().item()
            logger.info(f"  Mask {task_name}: {nnz_after:,} params after disjoint mask")

        logger.info("  TIES preprocessing complete")

    # ------------------------------------------------------------------
    # Model reconstruction
    # ------------------------------------------------------------------

    def _set_model_to_merged(self, model: torch.nn.Module, alpha_values: "np.ndarray") -> None:
        """
        In-place: set model params to base_θ + Σ_k α_k · τ_k.

        All computation is on CPU (task vectors are CPU float32); result is
        cast to each parameter's original dtype and copied to its device.
        """
        merged_tv = sum(
            float(alpha_values[i]) * self._task_vectors_cpu[name] for i, name in enumerate(self._task_names_ordered)
        )
        merged_params = self._base_params_cpu + merged_tv

        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                numel = p.numel()
                p.data.copy_(merged_params[idx : idx + numel].reshape(p.shape).to(dtype=p.dtype, device=p.device))
                idx += numel

    # ------------------------------------------------------------------
    # Core: Self Position training epoch
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: torch.nn.Module,
        task_dataloaders: Dict,
        task_names: List[str],
        dataset_configs: Dict,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        preference_vector,
        epoch: int,
        scaler: Optional[Any] = None,
        wandb_prefix: Optional[str] = None,
        finetuned_model_cache_dir: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Run one Self Position training epoch.

        For each step:
          1. Compute α = softmax(β)  [with autograd graph from β]
          2. Set model params to base + Σ α_k τ_k  [detached α — no grad through params]
          3. T forward passes → per-task losses; accumulate combined_loss = Σ r_k L_k
          4. combined_loss.backward() → ∇_θ(Σ r_k L_k) in model.parameters().grad
          5. g = flat_gradient(model) on CPU float32
          6. grad_α_j = ⟨g, τ_j⟩  for j = 1..T  (T CPU dot products)
          7. alpha.backward(grad_α) → β.grad via autograd through softmax
          8. alpha_optimizer.step() → update β
        """
        import numpy as np

        # First call: load task vectors (model is at base checkpoint here)
        if self._task_vectors_cpu is None:
            self._load_task_vectors(model, task_names, dataset_configs, finetuned_model_cache_dir)

        model.train()
        device = next(model.parameters()).device
        T = len(task_names)

        # Preference vector (optionally inverted)
        r = preference_vector.copy().astype(np.float64)
        if self.invert_preference:
            r = 1.0 / np.maximum(r, 1e-8)
        r = np.maximum(r, 1e-8)
        r = r / r.sum()

        # Autocast context for memory-efficient forward passes
        if self.use_fp16 and torch.cuda.is_available():
            from torch.amp import autocast as _autocast

            autocast_ctx = _autocast("cuda")
        else:
            autocast_ctx = nullcontext()

        # Task iterators
        task_iterators = {name: iter(dl) for name, dl in task_dataloaders.items()}
        # Use max across tasks: shorter datasets cycle, longer datasets run fully.
        try:
            steps_per_epoch = max(len(dl) for dl in task_dataloaders.values())
        except TypeError:
            steps_per_epoch = (self.max_samples_per_task or 10_000) // self.batch_size

        logger.info(
            f"\nEpoch {epoch + 1}/{self.num_epochs}  —  "
            f"Self Position steps: {steps_per_epoch}  —  "
            f"α = {F.softmax(self._beta.detach(), dim=0).numpy().round(4).tolist()}"
        )

        log_interval = max(10, steps_per_epoch // 4)  # ~4 log points per epoch regardless of length
        total_loss = 0.0
        num_steps = 0

        for step in range(steps_per_epoch):
            # ----------------------------------------------------------------
            # 1. Fetch one batch per task
            # ----------------------------------------------------------------
            task_batches: Dict[str, Dict] = {}
            all_available = True
            for task_name in task_names:
                try:
                    batch = next(task_iterators[task_name])
                except StopIteration:
                    # Restart iterator (works for both streaming and non-streaming)
                    task_iterators[task_name] = iter(task_dataloaders[task_name])
                    try:
                        batch = next(task_iterators[task_name])
                    except StopIteration:
                        all_available = False
                        break
                task_batches[task_name] = batch

            if not all_available:
                logger.info(f"  Early stop at step {step + 1} (data exhausted)")
                break

            # ----------------------------------------------------------------
            # 2. Compute α = softmax(β) and reconstruct merged model
            # ----------------------------------------------------------------
            alpha = F.softmax(self._beta, dim=0)  # non-leaf; autograd graph to β
            alpha_np = alpha.detach().numpy()

            self._set_model_to_merged(model, alpha_np)
            optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------------------
            # 3. Forward passes: accumulate combined_loss = Σ r_k L_k
            # ----------------------------------------------------------------
            combined_loss = None
            task_losses_scalar: List[float] = []

            for k, task_name in enumerate(task_names):
                batch = task_batches[task_name]
                labels = batch.pop("labels").to(device)
                task_cfg = dataset_configs[task_name]

                with autocast_ctx:
                    outputs = model(**{kk: vv.to(device) for kk, vv in batch.items()})
                    logits = outputs.logits[:, : task_cfg.num_labels]
                    loss_k = F.cross_entropy(logits, labels)

                batch["labels"] = labels  # restore for potential reuse

                task_losses_scalar.append(loss_k.item())
                rk = float(r[k])
                combined_loss = rk * loss_k if combined_loss is None else combined_loss + rk * loss_k

            # ----------------------------------------------------------------
            # 4. Single backward pass → ∇_θ(Σ r_k L_k)
            # Self Position intentionally bypasses GradScaler: we need the raw
            # float32 gradient values for the dot products with task vectors.
            # autocast is used only for forward memory savings.
            # ----------------------------------------------------------------
            combined_loss.backward()

            # ----------------------------------------------------------------
            # 5. Collect flat gradient (CPU float32, in model.parameters() order)
            # ----------------------------------------------------------------
            g = torch.cat([p.grad.detach().cpu().float().flatten() for p in model.parameters() if p.grad is not None])

            # ----------------------------------------------------------------
            # 6. Compute ∇_α_j = ⟨g, τ_j⟩ for j = 1..T (T CPU dot products)
            # ----------------------------------------------------------------
            grad_alpha = torch.tensor([g.dot(self._task_vectors_cpu[name]).item() for name in self._task_names_ordered])

            # ----------------------------------------------------------------
            # 7. Propagate ∇_α through softmax → ∇_β; Adam step on β
            # alpha.backward(grad_alpha) computes β.grad = J_softmax^T · grad_alpha
            # ----------------------------------------------------------------
            self._alpha_optimizer.zero_grad()
            alpha.backward(gradient=grad_alpha)
            self._alpha_optimizer.step()

            # ----------------------------------------------------------------
            # 8. Compatibility: step scheduler and keep scaler state in sync
            # ----------------------------------------------------------------
            scheduler.step()
            if scaler is not None:
                scaler.update()

            step_loss = float(combined_loss.item())
            total_loss += step_loss
            num_steps += 1

            # Periodic logging (~4 log points per epoch regardless of epoch length)
            if (step + 1) % log_interval == 0 or step == steps_per_epoch - 1:
                alpha_log = F.softmax(self._beta.detach(), dim=0).numpy().round(4).tolist()
                logger.info(
                    f"  Step {step + 1}/{steps_per_epoch} — "
                    f"Loss: {step_loss:.4f} — "
                    f"Task losses: {[f'{l:.4f}' for l in task_losses_scalar]} — "
                    f"α: {alpha_log}"
                )

                try:
                    import wandb

                    if wandb.run:
                        prefix = wandb_prefix or "train"
                        global_step = epoch * steps_per_epoch + step + 1
                        log_dict = {f"{prefix}/step_loss": step_loss}
                        for tname, tl, ta in zip(task_names, task_losses_scalar, alpha_np):
                            log_dict[f"{prefix}/task_loss/{tname}"] = tl
                            log_dict[f"{prefix}/alpha/{tname}"] = float(ta)
                        wandb.log(log_dict, step=global_step)
                except (ImportError, AttributeError, Exception):
                    pass

        # Set model to final merged position (for checkpoint saving by base class)
        alpha_final = F.softmax(self._beta.detach(), dim=0).numpy()
        self._set_model_to_merged(model, alpha_final)

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        return avg_loss
