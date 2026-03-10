"""Exact Pareto Optimal (EPO) Search Fine-Tuning

Fine-tuning from a base model using the EPO algorithm to find preference-exact
Pareto optimal solutions.

Papers:
  Primary:  Mahapatra & Rajan, "Multi-task learning with user preferences:
            Gradient descent with controlled ascent in Pareto optimization"
            ICML 2020. https://arxiv.org/abs/2010.06313
  Extended: Mahapatra & Rajan, "Exact Pareto Optimal Search for Multi-Task
            Learning and Multi-Criteria Decision-Making"
            arXiv 2021 (v2 2023). https://arxiv.org/abs/2108.00597

Algorithm overview
------------------
At each training step, EPO:
1. Computes per-task gradients via T separate backward passes.
2. Builds the gradient Gram matrix G ∈ R^{T×T}, where G[i,j] = ⟨∇L_i, ∇L_j⟩.
3. Solves a small LP (T+1 variables) to find mixing weights α that maximise
   the minimum preference-normalised descent across all tasks:

       maximise   t
       subject to (G α)_i / r_i ≥ t   ∀ i
                  α ≥ 0,  Σ α_i = 1

   When t ≤ 0 the LP cannot find a joint descent direction (gradient conflicts
   or Pareto-optimal reached) → fall back to min-norm MGDA QP:

       minimise   α^T G α
       subject to α ≥ 0,  Σ α_i = 1

4. Applies the combined gradient Σ α_i ∇L_i to the model.

Memory optimisation
-------------------
With T backward passes per step, naively storing all gradient vectors on GPU
would require T × |θ| extra memory (≈ 2 GB for RoBERTa-base, T=4).
The `cpu_gradient_offload=True` flag (default) immediately moves each
per-task gradient to CPU after the backward pass and zeros the GPU gradient,
keeping peak overhead to one backward pass at a time.

Preference vector convention
-----------------------------
The convention used here — high r_i = more important task i = lower loss for
task i at convergence — is the SAME as Chebyshev.  This follows from the LP
objective: maximising (G α)_i / r_i means task i with larger r_i receives
proportionally more gradient descent.

This differs from the Mahapatra & Rajan paper's "preference ray" condition
(L(θ*) ∝ r, which has the opposite semantic).  We believe the LP formulation
above better matches the intended use of r in a benchmark comparison against
Chebyshev.  The `invert_preference` parameter (default False) is provided for
ablations if the alternative convention is needed.

Normalization
-------------
When reference_losses are provided (utopia + nadir per task), losses are
converted to normalised excess risk before being passed to the EPO LP:

    e_i = (l_i − utopia_i) / (nadir_i − utopia_i)

This maps [utopia, nadir] → [0, 1] per task, making tasks with different loss
scales comparable in the LP — identical rationale as in Chebyshev.
"""

import logging
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.methods.base import BaseTrainingMethod
from src.methods.registry import MethodRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EPO LP / QP solver
# ---------------------------------------------------------------------------


class EPO_LP:
    """
    Solves the EPO LP and MGDA QP to find optimal gradient mixing weights.

    The LP (not at Pareto-optimal):
        maximise   t
        subject to (G α)_i / r_i ≥ t   ∀ i
                   α ≥ 0,  Σ α_i = 1

    When t ≤ 0 (gradient conflicts / Pareto-optimal), falls back to the
    min-norm MGDA QP:
        minimise   α^T G α
        subject to α ≥ 0,  Σ α_i = 1

    Both problems have T variables (plus one slack for the LP) and are solved
    with scipy — runtime is < 1 ms for T ≤ 20 tasks.
    """

    def __init__(self, n_tasks: int, pref_eps: float = 1e-3):
        self.n_tasks = n_tasks
        self.pref_eps = pref_eps

    def get_alpha(
        self,
        losses: np.ndarray,
        preference: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        """
        Compute gradient mixing weights for one EPO step.

        Args:
            losses:     Current (possibly normalised) loss values, shape (T,).
            preference: Preference vector, shape (T,), positive, sums to 1.
            G:          Gradient Gram matrix, shape (T, T).
                        G[i,j] = ⟨∇L_i, ∇L_j⟩ (possibly with normalised grads).

        Returns:
            alpha: Mixing weights, shape (T,), non-negative, sums to 1.
        """
        # Try EPO LP first (Pareto improvement direction)
        alpha, t_opt = self._solve_epo_lp(preference, G)

        if t_opt > self.pref_eps:
            # LP found a valid preference-aligned descent direction
            return alpha

        # t ≤ 0: no joint descent matching the preference → fall back to MGDA
        mu = losses / (losses.sum() + 1e-12)
        phi = (mu / np.maximum(preference, 1e-12)).max() - (mu / np.maximum(preference, 1e-12)).min()
        if phi < self.pref_eps:
            logger.debug("EPO: on preference ray — using MGDA (min-norm)")
        else:
            logger.debug(f"EPO: LP infeasible (t={t_opt:.4f}) — falling back to MGDA")

        return self._solve_mgda(G)

    # ------------------------------------------------------------------
    # Private solvers
    # ------------------------------------------------------------------

    def _solve_epo_lp(self, preference: np.ndarray, G: np.ndarray):
        """
        Maximise t s.t. (G α)_i / r_i ≥ t, α ≥ 0, Σα = 1.

        Returns (alpha, t_optimal).
        """
        from scipy.optimize import linprog

        T = self.n_tasks
        r = np.maximum(preference, 1e-12)

        # Variables: [α_0, ..., α_{T-1}, t]
        # Minimise -t  (i.e. maximise t)
        c = np.zeros(T + 1)
        c[-1] = -1.0

        # Inequality: -(G α)_i / r_i + t ≤ 0  →  A_ub @ x ≤ b_ub
        # (G α)_i = Σ_j G[i,j] α_j  →  coefficient of α_j in row i = G[i,j]
        A_ub = np.zeros((T, T + 1))
        A_ub[:, :T] = -G / r[:, None]  # -G[i,j] / r_i
        A_ub[:, T] = 1.0  # +1 for t
        b_ub = np.zeros(T)

        # Equality: Σα = 1  (t is unconstrained)
        A_eq = np.zeros((1, T + 1))
        A_eq[0, :T] = 1.0
        b_eq = np.array([1.0])

        # Bounds: 0 ≤ α_i ≤ 1, t unbounded
        bounds = [(0.0, 1.0)] * T + [(None, None)]

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            return np.ones(T) / T, -1.0  # fallback to MGDA

        alpha = np.maximum(result.x[:T], 0.0)
        s = alpha.sum()
        alpha = alpha / s if s > 1e-12 else np.ones(T) / T
        t_opt = float(-result.fun)  # we minimised -t, so t = -obj
        return alpha, t_opt

    def _solve_mgda(self, G: np.ndarray) -> np.ndarray:
        """
        Min-norm QP: minimise α^T G α  s.t. α ≥ 0, Σα = 1.

        Uses Frank-Wolfe iterations (fast for small T) followed by one SLSQP
        polish step for accuracy.
        """
        T = self.n_tasks

        # Frank-Wolfe initialisation: start at uniform
        alpha = np.ones(T) / T

        for _ in range(200):
            grad = 2.0 * G @ alpha  # gradient of α^T G α
            i_min = int(np.argmin(grad))  # FW direction: e_{i_min}
            s = np.zeros(T)
            s[i_min] = 1.0
            d = s - alpha
            # Line search: minimise f(alpha + γ d) = f + γ d^T ∇f + γ² d^T G d
            dGd = float(d @ G @ d)
            if dGd < 1e-12:
                break
            gamma = -float(d @ grad) / (2.0 * dGd)
            gamma = max(0.0, min(1.0, gamma))
            alpha = alpha + gamma * d
            if gamma < 1e-8:
                break

        alpha = np.maximum(alpha, 0.0)
        s = alpha.sum()
        if s < 1e-12:
            return np.ones(T) / T
        return alpha / s


# ---------------------------------------------------------------------------
# EPO Fine-Tuning method
# ---------------------------------------------------------------------------


@MethodRegistry.register("epo")
class EPOFineTuning(BaseTrainingMethod):
    """
    Multi-task fine-tuning with Exact Pareto Optimal (EPO) Search.

    Overrides _train_epoch() to implement the EPO gradient manipulation loop.
    At each step, T separate backward passes are performed to obtain per-task
    gradients; mixing weights are then found via the EPO LP (or MGDA fallback).

    Preference convention: high r_i = more important task i = lower loss for
    task i at convergence (same as Chebyshev).
    """

    def __init__(
        self,
        normalize_gradients: bool = True,
        cpu_gradient_offload: bool = True,
        invert_preference: bool = False,
        pref_eps: float = 1e-3,
        utopia_point: Optional[np.ndarray] = None,
        nadir_point: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Args:
            normalize_gradients:  Normalise each per-task gradient by its L2
                                  norm before computing the Gram matrix.  Makes
                                  the LP scale-invariant with respect to task
                                  gradient magnitudes.  Default True.
            cpu_gradient_offload: Move each per-task gradient to CPU immediately
                                  after the backward pass to cap peak GPU memory
                                  at one backward pass worth of activations.
                                  Default True.
            invert_preference:    If True, transform r → normalise(1/r) before
                                  passing to the EPO LP.  Use for ablation if
                                  the alternative semantic convention is needed.
                                  Default False.
            pref_eps:             Tolerance for detecting that the EPO LP found
                                  a valid descent (t > pref_eps) vs. falling
                                  back to MGDA.  Default 1e-3.
            utopia_point:         Optional pre-set utopia point (overridden by
                                  reference_losses from the benchmark runner).
            nadir_point:          Optional pre-set nadir point (same).
            **kwargs:             Forwarded to BaseTrainingMethod (learning_rate,
                                  num_epochs, batch_size, etc.).
        """
        super().__init__(**kwargs)
        self.normalize_gradients = normalize_gradients
        self.cpu_gradient_offload = cpu_gradient_offload
        self.invert_preference = invert_preference
        self.pref_eps = pref_eps
        self.utopia_point = utopia_point
        self.nadir_point = nadir_point

    # ------------------------------------------------------------------
    # BaseTrainingMethod interface
    # ------------------------------------------------------------------

    def _compute_multi_task_loss(
        self,
        task_losses: torch.Tensor,
        preference_vector: torch.Tensor,
        utopia_point: Optional[torch.Tensor] = None,
        nadir_point: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Not called during normal EPO training (which overrides _train_epoch).
        Used by _compute_validation_loss in base.py to evaluate the same
        normalized objective EPO trains against.

        When utopia/nadir are available (standard path), computes
        preference-weighted normalized excess risk:
            Σ_k w_k * (L_k - u_k) / (n_k - u_k)
        Falls back to raw preference-weighted sum if not provided.
        """
        if utopia_point is not None and nadir_point is not None:
            scale = (nadir_point - utopia_point).clamp(min=1e-8)
            deviations = (task_losses - utopia_point) / scale
            return (preference_vector * deviations).sum()
        return (preference_vector * task_losses).sum()

    def _get_training_kwargs(
        self,
        task_names: list,
        preference_vector: np.ndarray,
        dataset_configs: Dict,
        cache_dir: Optional[str] = None,
        reference_losses: Optional[Dict] = None,
    ) -> Dict:
        """
        Build utopia_point and nadir_point tensors from reference_losses.

        When reference_losses is provided (standard benchmark path), builds
        tensors directly from precomputed values — avoids redundant model
        loading and uses the correct disjoint reference split.

        Falls back to pre-set self.utopia_point / self.nadir_point if available,
        or to unnormalised losses (legacy).
        """
        if reference_losses is not None:
            utopia = torch.tensor([reference_losses[t]["utopia"] for t in task_names], dtype=torch.float32)
            nadir = torch.tensor([reference_losses[t]["nadir"] for t in task_names], dtype=torch.float32)
            logger.info(f"EPO utopia losses (from reference): {utopia.tolist()}")
            logger.info(f"EPO nadir  losses (from reference): {nadir.tolist()}")
            return {"utopia_point": utopia, "nadir_point": nadir}

        if self.utopia_point is not None and self.nadir_point is not None:
            logger.info("EPO: using utopia/nadir from config")
            return {
                "utopia_point": torch.tensor(self.utopia_point, dtype=torch.float32),
                "nadir_point": torch.tensor(self.nadir_point, dtype=torch.float32),
            }

        logger.warning(
            "EPO: reference_losses not provided and no utopia/nadir in config. "
            "Running without loss normalisation (EPO LP will use raw losses). "
            "Pass reference_losses from the benchmark runner for the correct formulation."
        )
        return {"utopia_point": None, "nadir_point": None}

    # ------------------------------------------------------------------
    # Core: EPO training epoch (overrides BaseTrainingMethod._train_epoch)
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: torch.nn.Module,
        task_dataloaders: Dict,
        task_names: List[str],
        dataset_configs: Dict,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        preference_vector: np.ndarray,
        epoch: int,
        scaler: Optional[Any] = None,
        wandb_prefix: Optional[str] = None,
        utopia_point: Optional[torch.Tensor] = None,
        nadir_point: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> float:
        """
        Run one EPO training epoch.

        For each step:
          1. Draw one batch per task.
          2. For each task: forward → backward → move gradient to CPU → zero GPU grad.
          3. Build gradient Gram matrix on CPU (numpy).
          4. Convert losses to normalised excess risk (if utopia/nadir provided).
          5. Solve EPO LP (or MGDA fallback) for mixing weights α.
          6. Compute combined gradient Σ α_i g_i on CPU → assign to model.
          7. Clip, step, schedule.
        """
        model.train()
        total_loss = 0.0
        num_steps = 0
        T = len(task_names)

        # Initialise EPO LP solver
        epo_lp = EPO_LP(n_tasks=T, pref_eps=self.pref_eps)

        # Pre-compute utopia / nadir as numpy for fast per-step use
        utopia_np = utopia_point.numpy() if utopia_point is not None else None
        nadir_np = nadir_point.numpy() if nadir_point is not None else None

        # Compute preference vector (with optional inversion)
        r = preference_vector.copy().astype(np.float64)
        if self.invert_preference:
            r = 1.0 / np.maximum(r, 1e-12)
        r = np.maximum(r, 1e-12)
        r = r / r.sum()

        # Create iterators for each task
        task_iterators = {name: iter(dl) for name, dl in task_dataloaders.items()}

        # Number of steps = length of shortest dataloader
        # Use max across tasks: shorter datasets cycle, longer datasets run fully.
        try:
            steps_per_epoch = max(len(dl) for dl in task_dataloaders.values())
        except TypeError:
            steps_per_epoch = (self.max_samples_per_task or 10000) // self.batch_size

        logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}  —  EPO training steps: {steps_per_epoch}")

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
                task_batches[task_name] = {k: v.to(model.device) for k, v in batch.items()}

            if not all_available:
                logger.info(f"  Early stop at step {step + 1} (data exhausted)")
                break

            # ----------------------------------------------------------------
            # 2. Per-task forward + backward → gradients on CPU
            # ----------------------------------------------------------------
            task_losses_scalar: List[float] = []
            task_grads: List[torch.Tensor] = []  # flat float32; device = CPU or GPU per flag

            for task_name in task_names:
                batch = task_batches[task_name]
                task_cfg = dataset_configs[task_name]

                optimizer.zero_grad(set_to_none=True)

                labels = batch.pop("labels")

                # Use autocast for fp16 forward-pass memory savings if enabled.
                # GradScaler is not used (see backward comment below).
                if self.use_fp16 and torch.cuda.is_available():
                    from torch.amp import autocast

                    ctx = autocast("cuda")
                else:
                    ctx = nullcontext()

                with ctx:
                    outputs = model(**batch)
                    logits = outputs.logits[:, : task_cfg.num_labels]
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)

                batch["labels"] = labels  # restore

                # EPO uses standard (unscaled) backward passes.
                # GradScaler is intentionally not used here: EPO requires per-task
                # gradient values for the Gram matrix, and we immediately convert to
                # float32.  Using autocast for the forward pass still gives fp16
                # activation memory savings without needing loss scaling.
                loss.backward()

                task_losses_scalar.append(loss.item())

                # Collect gradient as a flat float32 vector.
                # cpu_gradient_offload=True: move to CPU immediately to cap GPU memory
                #   (keeps peak GPU overhead to one backward pass at a time).
                # cpu_gradient_offload=False: keep on GPU; Gram + combined grad computed
                #   on GPU with cuBLAS — eliminates ~2 GB PCIe transfer per step.
                if self.cpu_gradient_offload:
                    grad_flat = torch.cat(
                        [p.grad.detach().cpu().float().flatten() for p in model.parameters() if p.grad is not None]
                    )
                else:
                    grad_flat = torch.cat(
                        [p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None]
                    )
                task_grads.append(grad_flat)
                optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------------------
            # 3. Build Gram matrix + compute combined gradient
            # ----------------------------------------------------------------
            losses_np = np.array(task_losses_scalar, dtype=np.float64)

            if self.cpu_gradient_offload:
                # ── CPU / numpy path (original; low GPU memory) ──────────────
                # Stay in float32: 4×125M params×8B(float64) ≈ 4 GB
                grads_np = np.stack([g.numpy() for g in task_grads])  # (T, n_params), float32
                del task_grads

                if self.normalize_gradients:
                    norms = np.linalg.norm(grads_np, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-8)
                    grads_for_gram = (grads_np / norms).astype(np.float32)
                else:
                    grads_for_gram = grads_np

                G = (grads_for_gram @ grads_for_gram.T).astype(np.float64)  # (T, T)
                del grads_for_gram

                # ── losses → EPO LP ──────────────────────────────────────────
                if utopia_np is not None and nadir_np is not None:
                    scale = np.maximum(nadir_np - utopia_np, 1e-8)
                    epo_losses = (losses_np - utopia_np) / scale
                else:
                    epo_losses = losses_np

                alpha = epo_lp.get_alpha(epo_losses, r, G)  # (T,)

                combined_grad_cpu = torch.from_numpy((alpha.astype(np.float32) @ grads_np).copy())
                del grads_np

                idx = 0
                for p in model.parameters():
                    numel = p.numel()
                    if p.requires_grad:
                        p.grad = combined_grad_cpu[idx : idx + numel].reshape(p.shape).to(p.device)
                    idx += numel

            else:
                # ── GPU / cuBLAS path (fast; needs ~2 GB extra VRAM) ─────────
                grads_gpu = torch.stack(task_grads)  # (T, n_params), float32, GPU
                del task_grads

                if self.normalize_gradients:
                    norms = grads_gpu.norm(dim=1, keepdim=True).clamp_(min=1e-8)
                    grads_for_gram = grads_gpu / norms
                else:
                    grads_for_gram = grads_gpu

                # Only (T, T) = 4×4 floats transferred CPU↔GPU
                G = (grads_for_gram @ grads_for_gram.T).cpu().double().numpy()

                # ── losses → EPO LP ──────────────────────────────────────────
                if utopia_np is not None and nadir_np is not None:
                    scale = np.maximum(nadir_np - utopia_np, 1e-8)
                    epo_losses = (losses_np - utopia_np) / scale
                else:
                    epo_losses = losses_np

                alpha = epo_lp.get_alpha(epo_losses, r, G)  # (T,)

                alpha_gpu = torch.tensor(alpha, dtype=torch.float32, device=grads_gpu.device)
                combined_grad_gpu = alpha_gpu @ grads_gpu  # (n_params,) on GPU
                del grads_gpu

                idx = 0
                for p in model.parameters():
                    numel = p.numel()
                    if p.requires_grad:
                        p.grad = combined_grad_gpu[idx : idx + numel].reshape(p.shape)
                    idx += numel

            # ----------------------------------------------------------------
            # 7. Gradient clipping, optimiser step, scheduler step
            # EPO bypasses GradScaler (see per-task backward comment above).
            # ----------------------------------------------------------------
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Track scalar loss for logging (α-weighted combination)
            step_loss = float(np.dot(alpha, task_losses_scalar))
            total_loss += step_loss
            num_steps += 1

            # Periodic logging
            if (step + 1) % 100 == 0 or step == steps_per_epoch - 1:
                logger.info(
                    f"  Step {step + 1}/{steps_per_epoch} — "
                    f"Loss: {step_loss:.4f} — "
                    f"Task losses: {[f'{l:.4f}' for l in task_losses_scalar]} — "
                    f"alpha: {[f'{a:.3f}' for a in alpha]}"
                )

                try:
                    import wandb

                    if wandb.run:
                        prefix = wandb_prefix or "train"
                        local_step = epoch * steps_per_epoch + step + 1
                        log_dict = {f"{prefix}/step_loss": step_loss, "local_step": local_step}
                        for tname, tl, ta in zip(task_names, task_losses_scalar, alpha):
                            log_dict[f"{prefix}/task_loss/{tname}"] = tl
                            log_dict[f"{prefix}/alpha/{tname}"] = ta
                        wandb.log(log_dict)
                except (ImportError, AttributeError, Exception):
                    pass

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
        return avg_loss
