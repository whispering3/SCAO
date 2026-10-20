"""
SparsePreconditioner
====================
Per-layer Kronecker-factored sparse preconditioner for SCAO.

Maintains low-rank approximations of the left (L) and right (R) curvature
factor EMAs:
    L_t ≈ U_l diag(S_l) U_l^T   (m × m approximated at rank k)
    R_t ≈ U_r diag(S_r) U_r^T   (n × n approximated at rank k)

The preconditioned gradient is:
    G_precond = U_l diag(S_l^{-1/4}) (U_l^T G U_r) diag(S_r^{-1/4}) U_r^T

For 1-D parameters (biases, LayerNorm), falls back to Adam-style diagonal.
For parameters where max(m, n) > max_precond_dim, uses sparse block-diagonal
preconditioning: the gradient matrix is partitioned into contiguous blocks
along the larger dimension, each of size ≤ max_precond_dim, and an independent
Kronecker preconditioner is applied to each block.  This preserves curvature
information across the full gradient while keeping the per-block eigendecomp
cost bounded at O(max_precond_dim³).

Mixed-precision note
--------------------
All curvature statistics (L_ema, R_ema) and eigenfactors (U_l, S_l, U_r, S_r)
are stored in **float32** regardless of the parameter dtype.  This is required
because ``torch.linalg.eigh`` does not support bfloat16 or float16.
Gradients are cast to float32 before statistics updates, and preconditioned
gradients are cast back to the original parameter dtype before returning.

torch.compile note
------------------
``update_curvature()`` and ``_update_eigenfactors()`` are decorated with
``@torch.compiler.disable`` because they contain Python control flow
(try/except, adaptive_rank loop) that cannot be traced.  They are called
infrequently (every T_precond steps) so graph-breaks there are harmless.

``precondition()`` and ``natural_grad_norm()`` contain only tensor ops and are
compile-friendly when invoked from a ``torch.compile``-d training step.

Changelog
---------
v3 patch:
  - R5 support: ``k`` is now a property with a setter that safely truncates
    U_l, S_l, U_r, S_r when SCAO._maybe_adjust_rank() assigns a new rank
    externally between eigenfactor updates.  Without this, an external
    ``precond.k = new_k`` would leave the eigenfactor shapes stale, causing
    a shape mismatch in ``precondition()``.
  - Internal storage uses ``_k`` to avoid recursion in ``__init__``.
  - Block-diagonal mode propagates external rank changes to sub-blocks
    and recomputes the average k.
"""

from __future__ import annotations

import math
import warnings
from typing import cast

import torch
from torch import Tensor

from .cuda import (
    fused_kronecker_precond as _cuda_fused_precond,
)
from .cuda import (
    int8_ema_update as _cuda_int8_ema_update,
)
from .utils import (
    adaptive_rank,
    dequantize_sym_int8,
    from_2d,
    quantize_sym_int8,
    to_2d,
)

# float32 is the minimum precision required for stable eigendecomposition.
# bfloat16/float16 are NOT supported by torch.linalg.eigh on CPU.
_PRECOND_DTYPE = torch.float32


class SparsePreconditioner:
    """
    Low-rank Kronecker preconditioner for a single parameter tensor.

    State tensors (all on the same device as the parameter):
        L_ema     : (m, m) - full EMA of left curvature factor (temp, during update)
        R_ema     : (n, n) - full EMA of right curvature factor (temp, during update)
        U_l       : (m, k) - left eigenvectors
        S_l       : (k,)   - left eigenvalues (descending)
        U_r       : (n, k) - right eigenvectors
        S_r       : (k,)   - right eigenvalues (descending)
        k         : current rank (int, property — assignment triggers safe resize)

    v3 note on ``k``
    ----------------
    ``k`` is now a property.  Assigning ``precond.k = new_k`` (as done by
    SCAO._maybe_adjust_rank for R5) will:
      1. Clamp new_k to [k_min, k_max].
      2. Truncate U_l/S_l/U_r/S_r to the new rank (no-op if rank grows —
         the next eigenfactor update will expand them correctly).
      3. Update the internal ``_k`` counter.
    This prevents shape mismatches in ``precondition()`` when the external
    rank assignment happens between eigenfactor updates.
    """

    def __init__(
        self,
        param: Tensor,
        epsilon_sparse: float = 0.05,
        k_min: int = 8,
        k_max: int = 128,
        rho: float = 0.999,
        max_precond_dim: int = 4096,
        use_newton_schulz: bool = False,
        use_int8_ema: bool = False,
        factor_mode: str = "full",
        curvature_mode: str = "dense",
        curvature_rank: int | None = None,
    ) -> None:
        shape = param.shape
        device = param.device
        param_dtype = param.dtype  # original dtype — used to cast outputs back

        # Reshape to 2D to determine (m, n)
        p2d, _ = to_2d(param)
        m, n = p2d.shape

        self.original_shape = shape
        self.m = m
        self.n = n
        self.epsilon_sparse = epsilon_sparse
        if factor_mode not in {"full", "wide"}:
            raise ValueError("factor_mode must be 'full' or 'wide'")
        if curvature_mode != "dense" or curvature_rank is not None:
            raise ValueError(
                "Only the stable dense curvature path is supported; "
                "use curvature_mode='dense' and curvature_rank=None"
            )
        self.factor_mode = factor_mode
        self.curvature_mode = curvature_mode
        # ``wide`` is intended for rectangular LoRA adapters. For a square
        # matrix there is no rank-r side to omit, so retain the full factor.
        self._tracks_left = factor_mode == "full" or m >= n
        self._tracks_right = factor_mode == "full" or n >= m

        # Clamp k_min/k_max to matrix dimensions to avoid invalid ranks in small layers
        d_min = min(m, n)
        self.k_min = min(k_min, d_min)
        self.k_max = min(k_max, d_min // 2) if d_min > 2 * self.k_min else self.k_min
        # A low-rank sketch keeps a small covariance subspace. Keeping twice
        # the active-factor budget preserves room for new gradient directions
        # without reintroducing a dense d x d EMA.
        self.curvature_rank = curvature_rank or max(2 * self.k_max, self.k_min)

        self.rho = rho
        # use_int8_ema: store L_ema/R_ema as int8 tensors with float32 scale factors.
        # Reduces EMA memory from (m²+n²)*4 bytes to (m²+n²)*1 + 8 bytes — 4× reduction.
        # Only applied to the Kronecker path (block-diagonal delegates to sub-preconditioners).
        self.use_int8_ema = use_int8_ema
        self.device = device
        # param_dtype: dtype of the parameter (may be bf16/fp16)
        self.param_dtype = param_dtype

        # Decide whether to use Kronecker preconditioner, block-diagonal, or diagonal fallback.
        # - Kronecker: standard case, max(m,n) ≤ max_precond_dim
        # - Block-diagonal: large layers, max(m,n) > max_precond_dim, splits the larger
        #   dimension into blocks of size ≤ max_precond_dim and applies an independent
        #   Kronecker preconditioner per block.  Matches the paper's Algorithm 1 claim.
        # - Diagonal: only for 1-D params (biases, LayerNorm scales)
        self.use_block_diagonal = (
            param.ndim >= 2
            and m > 1
            and n > 1
            and max(m, n) > max_precond_dim
        )
        self.use_kronecker = (
            param.ndim >= 2
            and m > 1
            and n > 1
            and max(m, n) <= max_precond_dim
        )

        if use_newton_schulz:
            warnings.warn(
                "use_newton_schulz is not active in the low-rank adaptive "
                "preconditioner path; falling back to eigendecomposition.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.use_newton_schulz = False

        if self.use_block_diagonal:
            # Block-diagonal preconditioning:
            # Split the gradient along the larger dimension into B ≤ max_precond_dim
            # blocks.  Each block gets its own independent Kronecker preconditioner.
            # This matches the paper's Algorithm 1 for large-matrix layers.
            large_dim = m if m >= n else n
            small_dim = n if m >= n else m
            self._block_dim = 0 if m >= n else 1  # 0 = split rows, 1 = split cols
            block_size = max_precond_dim
            n_full = large_dim // block_size
            remainder = large_dim % block_size
            self._block_sizes: list[int] = [block_size] * n_full
            if remainder > 0:
                self._block_sizes.append(remainder)
            self._blocks: list[SparsePreconditioner] = []
            for bs in self._block_sizes:
                fake_shape = (bs, small_dim) if self._block_dim == 0 else (small_dim, bs)
                fake_param = param.new_empty(fake_shape)
                blk = SparsePreconditioner(
                    param=fake_param,
                    epsilon_sparse=epsilon_sparse,
                    k_min=k_min,
                    k_max=k_max,
                    rho=rho,
                    max_precond_dim=max_precond_dim,
                    use_newton_schulz=use_newton_schulz,
                    use_int8_ema=use_int8_ema,
                    factor_mode=factor_mode,
                    curvature_mode=curvature_mode,
                    curvature_rank=curvature_rank,
                )
                self._blocks.append(blk)
            # _k tracks the average rank across blocks.
            # Use _k directly to bypass the property setter (no eigenfactors here).
            self._k: int = self._blocks[0]._k if self._blocks else k_min

        elif self.use_kronecker:
            k_init = min(k_min, min(m, n))
            # Use _k directly in __init__ to avoid triggering the property setter
            # before U_l/S_l/U_r/S_r are initialised.
            self._k = k_init

            # Curvature EMAs and eigenfactors are ALWAYS stored in float32.
            # This is required because linalg.eigh is not implemented for
            # bfloat16 or float16 on CPU (and is numerically unstable at
            # reduced precision on GPU).
            #
            # Fix 1 — initialize L/R as scaled identity, NOT zeros.
            # A zero init means the EMA accumulates for T_precond steps with
            # L ≈ 0, so when the preconditioner first fires the matrix is
            # rank-1 and ill-conditioned, causing a hard divergence at step
            # T_precond.  Initializing with eps * I gives a well-conditioned
            # starting point: eigenvalues begin near 1e-4, S^{-1/4} ≈ 10,
            # which is safely handled by the adaptive-eps regularization.
            _eye_eps = 1e-4
            if curvature_mode == "low_rank":
                # Low-rank covariance sketch B where B @ B.T approximates the
                # EMA. It is intentionally float32: the sketch is already far
                # smaller than the dense int8 representation and is updated by
                # a numerically sensitive thin SVD.
                if self._tracks_left:
                    self.L_sketch = torch.zeros(
                        m, min(m, self.curvature_rank), device=device, dtype=_PRECOND_DTYPE
                    )
                if self._tracks_right:
                    self.R_sketch = torch.zeros(
                        n, min(n, self.curvature_rank), device=device, dtype=_PRECOND_DTYPE
                    )
            else:
                L_init = torch.eye(m, device=device, dtype=_PRECOND_DTYPE) * _eye_eps
                R_init = torch.eye(n, device=device, dtype=_PRECOND_DTYPE) * _eye_eps

            if curvature_mode == "dense" and use_int8_ema:
                # Store EMA as int8 + float scale for 4× memory reduction.
                # (m²+n²) bytes int8 vs (m²+n²)*4 bytes fp32.
                if self._tracks_left:
                    self.L_ema_scale: float | Tensor
                    self.L_ema_q, self.L_ema_scale = quantize_sym_int8(L_init)
                if self._tracks_right:
                    self.R_ema_scale: float | Tensor
                    self.R_ema_q, self.R_ema_scale = quantize_sym_int8(R_init)
                # float32 shadow used only during eigendecomposition — not stored between steps
            elif curvature_mode == "dense":
                if self._tracks_left:
                    self.L_ema = L_init
                if self._tracks_right:
                    self.R_ema = R_init

            self.U_l = torch.eye(m, k_init, device=device, dtype=_PRECOND_DTYPE)
            self.S_l = torch.ones(k_init, device=device, dtype=_PRECOND_DTYPE)
            self.U_r = torch.eye(n, k_init, device=device, dtype=_PRECOND_DTYPE)
            self.S_r = torch.ones(k_init, device=device, dtype=_PRECOND_DTYPE)
            self._refresh_inverse_spectrum_cache()
        else:
            # Diagonal fallback: maintain per-element variance estimate in float32
            self.diag_ema = torch.zeros(shape, device=device, dtype=_PRECOND_DTYPE)
            self._diag_bias_factor: float = 1e-8  # updated in update_curvature
            self._k = 1

        # Step counter for this preconditioner (updated externally)
        self.precond_step: int = 0

    def _refresh_inverse_spectrum_cache(self, eps: float = 1e-8) -> None:
        """Refresh derived inverse spectrum tensors after eigenfactor changes."""
        if not self.use_kronecker:
            return
        self.S_l_inv4 = self.S_l.clamp(min=eps).pow(-0.25)
        self.S_r_inv4 = self.S_r.clamp(min=eps).pow(-0.25)
        self.S_l_inv2 = self.S_l.clamp(min=eps).pow(-0.5)
        self.S_r_inv2 = self.S_r.clamp(min=eps).pow(-0.5)
        # In wide-only mode this avoids forming ``S_inv4 - 1`` for every
        # adapter and every optimizer step.  These tensors are refreshed only
        # when the eigensystem changes.
        self.S_l_correction = self.S_l_inv4 - 1.0
        self.S_r_correction = self.S_r_inv4 - 1.0

    # ------------------------------------------------------------------
    # R5: k property — safe external rank assignment
    # ------------------------------------------------------------------

    @property
    def k(self) -> int:
        """Current preconditioner rank."""
        return self._k

    @k.setter
    def k(self, new_k: int) -> None:
        """
        Assign a new rank, clamping to [k_min, k_max] and truncating
        eigenfactor tensors if the rank decreases.

        Called by SCAO._maybe_adjust_rank() (R5).  Safe to call between
        eigenfactor updates: truncation keeps U/S consistent so that
        ``precondition()`` never sees a shape mismatch.

        If new_k > current k, eigenfactors are NOT expanded here — they
        will be expanded to the correct size on the next
        ``_update_eigenfactors()`` call, which recomputes U/S from scratch.
        The interim mismatch (k says N but tensors have M < N columns) is
        handled by clamping new_k to the current tensor width until the
        next eigenfactor update.
        """
        new_k = int(min(self.k_max, max(self.k_min, new_k)))

        if self.use_block_diagonal:
            # Propagate to sub-blocks and recompute average rank.
            for blk in self._blocks:
                blk.k = new_k
            self._k = sum(b._k for b in self._blocks) // max(len(self._blocks), 1)
            return

        if self.use_kronecker:
            if new_k < self._k:
                # Truncate eigenfactors to the smaller rank immediately so
                # precondition() stays consistent until the next eigh call.
                self.U_l = self.U_l[:, :new_k].contiguous()
                self.S_l = self.S_l[:new_k].contiguous()
                self.U_r = self.U_r[:, :new_k].contiguous()
                self.S_r = self.S_r[:new_k].contiguous()
                self._refresh_inverse_spectrum_cache()
            elif new_k > self._k:
                # Rank increase: clamp to the current tensor width.
                # _update_eigenfactors will expand to new_k on the next update.
                actual_cols = self.U_l.shape[1]
                new_k = min(new_k, actual_cols) if new_k > actual_cols else new_k
                # If tensors are already wider (shouldn't happen normally), just accept.

        self._k = new_k

    # ------------------------------------------------------------------
    # Curvature accumulation (called every T_precond optimizer steps)
    # ------------------------------------------------------------------

    @torch.no_grad()
    @torch.compiler.disable
    def update_curvature(self, grad: Tensor) -> None:
        """
        Update the EMA of curvature factors given the current gradient.
        Then recompute the low-rank eigenfactors and select the new rank.

        This is the *expensive* operation — call every T_precond steps only.
        Decorated with @torch.compiler.disable because it contains Python
        control flow (try/except, rank-selection loop) that cannot be traced.

        Args:
            grad: raw gradient of the parameter (same shape as param).
                  Any dtype is accepted — will be cast to float32 internally.
        """
        self.precond_step += 1

        # Cast gradient to float32 for numerically stable statistics.
        # This is the key fix for bfloat16/float16 parameter support.
        g2d_f32, _ = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: delegate each slice to its own sub-preconditioner.
        if self.use_block_diagonal:
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks, strict=True):
                if self._block_dim == 0:
                    g_blk = g2d_f32[offset:offset + bs, :]
                else:
                    g_blk = g2d_f32[:, offset:offset + bs]
                blk.update_curvature(g_blk)
                offset += bs
            # Recompute average rank across blocks after their eigenfactor updates.
            self._k = sum(b._k for b in self._blocks) // max(len(self._blocks), 1)
            return

        if not self.use_kronecker:
            sq = g2d_f32.squeeze(0) if g2d_f32.shape[0] == 1 else g2d_f32
            self.diag_ema.mul_(self.rho).add_((1.0 - self.rho) * sq.reshape_as(self.diag_ema).pow(2))
            # Bias-correct the diagonal EMA so early estimates are not ~0.
            # Same principle as Adam's bias correction for exp_avg_sq.
            bias_factor = 1.0 - self.rho ** self.precond_step
            self._diag_bias_factor = max(bias_factor, 1e-8)
            return

        alpha = 1.0 - self.rho
        low_rank_left: tuple[Tensor, Tensor] | None = None
        low_rank_right: tuple[Tensor, Tensor] | None = None
        if self.curvature_mode == "low_rank":
            if self._tracks_left:
                self.L_sketch, low_rank_left = self._update_low_rank_sketch(
                    self.L_sketch, g2d_f32, alpha, left=True
                )
            if self._tracks_right:
                self.R_sketch, low_rank_right = self._update_low_rank_sketch(
                    self.R_sketch, g2d_f32, alpha, left=False
                )
        elif self.use_int8_ema:
            # Route through the CUDA/Triton helper when it is available. The
            # fallback is numerically equivalent to the explicit PyTorch path.
            if self._tracks_left:
                l_update = alpha * (g2d_f32 @ g2d_f32.T)
                self.L_ema_q, self.L_ema_scale = _cuda_int8_ema_update(
                    self.L_ema_q, self.L_ema_scale, l_update, self.rho
                )
            if self._tracks_right:
                r_update = alpha * (g2d_f32.T @ g2d_f32)
                self.R_ema_q, self.R_ema_scale = _cuda_int8_ema_update(
                    self.R_ema_q, self.R_ema_scale, r_update, self.rho
                )
        else:
            if self._tracks_left:
                self.L_ema.mul_(self.rho).add_(alpha * (g2d_f32 @ g2d_f32.T))
            if self._tracks_right:
                self.R_ema.mul_(self.rho).add_(alpha * (g2d_f32.T @ g2d_f32))

        # EMA bias correction: the EMA starts at 0 so early estimates are
        # (1-rho^t) * true_value instead of true_value. Divide by (1-rho^t)
        # to debias — same trick as Adam's bias correction for exp_avg_sq.
        # Without this, S_l eigenvalues are ~0 for the first ~1/(1-rho) steps,
        # making S_l^{-1/4} blow up and the preconditioned gradient useless.
        bias_factor = 1.0 - self.rho ** self.precond_step
        self._update_eigenfactors(bias_factor, low_rank_left, low_rank_right)

    def _update_low_rank_sketch(
        self,
        sketch: Tensor,
        gradient: Tensor,
        alpha: float,
        *,
        left: bool,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Update a decayed covariance sketch and return its eigensystem.

        If ``B @ B.T`` approximates the prior covariance, concatenating
        ``sqrt(rho) * B`` with ``sqrt(alpha) * G`` represents the next EMA
        before truncation. A thin SVD keeps only ``curvature_rank`` columns.
        This turns dense covariance storage and eigendecomposition into a
        bounded low-rank update suitable for rectangular LoRA matrices.
        """
        sample = gradient if left else gradient.T
        candidate = torch.cat((sketch * math.sqrt(self.rho), sample * math.sqrt(alpha)), dim=1)
        vectors, singular_values, _ = torch.linalg.svd(candidate, full_matrices=False)
        width = sketch.shape[1]
        vectors = vectors[:, :width].contiguous()
        singular_values = singular_values[:width].contiguous()
        return vectors * singular_values.unsqueeze(0), (singular_values.square(), vectors)

    @torch.no_grad()
    @torch.compiler.disable
    def _update_eigenfactors(
        self,
        bias_factor: float = 1.0,
        low_rank_left: tuple[Tensor, Tensor] | None = None,
        low_rank_right: tuple[Tensor, Tensor] | None = None,
    ) -> None:
        """
        Eigendecompose L_ema and R_ema, select adaptive rank, store U/S factors.

        Args:
            bias_factor: EMA bias correction factor = (1 - rho^precond_step).
                         Dividing L_ema and R_ema by this factor debiases the
                         EMA so that eigenvalues reflect true curvature magnitude
                         from the first update onward.  Default 1.0 (no correction).

        All computations are in float32 (_PRECOND_DTYPE).
        Decorated with @torch.compiler.disable (try/except + Python loops).

        v3 note: ``_k`` is assigned directly here (bypassing the property setter)
        because U/S tensors are being replaced in full — no truncation needed.
        """
        eps = 1e-8
        def eig_factor(ema: Tensor, dim: int) -> tuple[Tensor, Tensor]:
            debiased = ema / max(bias_factor, eps)
            regularizer = max(eps, 1e-4 * debiased.trace().item() / dim)
            matrix = debiased.clone()
            matrix.diagonal().add_(regularizer)
            try:
                spectrum, vectors = torch.linalg.eigh(matrix)
            except torch.linalg.LinAlgError:
                matrix.diagonal().add_(eps)
                spectrum, vectors = torch.linalg.eigh(matrix)
            return spectrum.flip(0).clamp(min=0.0), vectors.flip(1)

        def sketch_factor(
            factor: tuple[Tensor, Tensor], dim: int
        ) -> tuple[Tensor, Tensor]:
            spectrum, vectors = factor
            spectrum = spectrum / max(bias_factor, eps)
            regularizer = max(eps, 1e-4 * spectrum.sum().item() / dim)
            return spectrum.add(regularizer), vectors

        # Preserve CPU RNG state so preconditioner updates do not perturb
        # the data-loading shuffle order — important for fair comparisons.
        rng_state = torch.get_rng_state()
        if self.curvature_mode == "low_rank":
            left_factor = (
                sketch_factor(low_rank_left, self.m) if low_rank_left is not None else None
            )
            right_factor = (
                sketch_factor(low_rank_right, self.n) if low_rank_right is not None else None
            )
        elif self.use_int8_ema:
            l_ema = (
                dequantize_sym_int8(self.L_ema_q, self.L_ema_scale)
                if self._tracks_left else None
            )
            r_ema = (
                dequantize_sym_int8(self.R_ema_q, self.R_ema_scale)
                if self._tracks_right else None
            )
        else:
            l_ema = self.L_ema if self._tracks_left else None
            r_ema = self.R_ema if self._tracks_right else None

        if self.curvature_mode != "low_rank":
            left_factor = eig_factor(l_ema, self.m) if l_ema is not None else None
            right_factor = eig_factor(r_ema, self.n) if r_ema is not None else None
        torch.set_rng_state(rng_state)

        # In wide-only mode, select rank from the only tracked factor.
        k_l = (
            adaptive_rank(left_factor[0], self.epsilon_sparse, self.k_min, self.k_max)
            if left_factor is not None else self.k_max
        )
        k_r = (
            adaptive_rank(right_factor[0], self.epsilon_sparse, self.k_min, self.k_max)
            if right_factor is not None else self.k_max
        )
        k_new = min(k_l, k_r)
        # Apply momentum on rank change: avoid oscillating between ranks
        k_new = max(k_new, self._k - 1)   # allow rank to drop by at most 1 per update
        k_new = min(k_new, self._k + 4)   # allow rank to grow by at most 4 per update
        k_new = int(min(self.k_max, max(self.k_min, k_new)))

        # Assign _k directly: U/S tensors are being replaced in full below,
        # so the property setter's truncation logic is not needed here.
        self._k = k_new

        # Store tracked factors and identity factors for the untracked side.
        # Keeping the latter avoids special shapes in state/checkpoint logic.
        if left_factor is not None:
            self.S_l, self.U_l = left_factor[0][:k_new].contiguous(), left_factor[1][:, :k_new].contiguous()
        else:
            self.U_l = torch.eye(self.m, k_new, device=self.device, dtype=_PRECOND_DTYPE)
            self.S_l = torch.ones(k_new, device=self.device, dtype=_PRECOND_DTYPE)
        if right_factor is not None:
            self.S_r, self.U_r = right_factor[0][:k_new].contiguous(), right_factor[1][:, :k_new].contiguous()
        else:
            self.U_r = torch.eye(self.n, k_new, device=self.device, dtype=_PRECOND_DTYPE)
            self.S_r = torch.ones(k_new, device=self.device, dtype=_PRECOND_DTYPE)
        self._refresh_inverse_spectrum_cache()

    # ------------------------------------------------------------------
    # Preconditioning  (called every optimizer step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precondition(self, grad: Tensor) -> Tensor:
        """
        Apply the sparse preconditioner to the gradient.

        Uses identity + low-rank Kronecker correction:

            G_proj    = U_l^T G U_r                          # (k, k) projection
            G_scaled  = diag(S_l^{-1/4}) G_proj diag(S_r^{-1/4})  # curvature scaling
            G_precond = G + U_l (G_scaled - G_proj) U_r^T    # identity + correction

        This is equivalent to the preconditioner:
            P = I + U_l (diag(S_l^{-1/4}) - I) U_l^T ⊗ U_r (diag(S_r^{-1/4}) - I) U_r^T

        which acts as identity on the complement and applies curvature scaling within
        the low-rank subspace.  Crucially, no gradient signal is discarded — the
        previous formulation (pure projection) silently zeroed the complement,
        discarding ~(1 - k/m)(1 - k/n) fraction of the gradient at small rank.

        For the diagonal fallback:
            g_precond = g / (diag_ema^{1/4} + eps)

        Inputs of any float dtype (fp32, bf16, fp16) are accepted.
        Internal computations run in float32; the result is cast back to
        the original gradient dtype before returning.

        Args:
            grad: gradient tensor (same shape as param)

        Returns:
            Preconditioned gradient (same shape and dtype as grad)
        """
        orig_dtype = grad.dtype
        g2d, orig_shape = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: precondition each slice independently, then reassemble.
        if self.use_block_diagonal:
            result = torch.empty_like(g2d)
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks, strict=True):
                if self._block_dim == 0:
                    g_blk = g2d[offset:offset + bs, :]
                    result[offset:offset + bs, :] = blk.precondition(g_blk).to(_PRECOND_DTYPE)
                else:
                    g_blk = g2d[:, offset:offset + bs]
                    result[:, offset:offset + bs] = blk.precondition(g_blk).to(_PRECOND_DTYPE)
                offset += bs
            return from_2d(result, orig_shape).to(orig_dtype)

        if not self.use_kronecker:
            bias = getattr(self, "_diag_bias_factor", 1.0)
            denom = (self.diag_ema / bias).reshape_as(g2d).pow(0.25).add_(1e-8)
            result = g2d / denom
            return from_2d(result, orig_shape).to(orig_dtype)

        if self._tracks_left and self._tracks_right:
            G_precond = _cuda_fused_precond(
                self.U_l, self.S_l_inv4, self.U_r, self.S_r_inv4, g2d
            )
        elif self._tracks_left:
            # One-sided LoRA preconditioning:
            #   G + U @ ((S^-1/4 - I) * (U.T @ G))
            # ``addmm`` dispatches the final operation to cuBLAS and avoids
            # materialising a separate correction tensor before addition.
            projected = self.U_l.T @ g2d
            projected.mul_(self.S_l_correction.unsqueeze(1))
            G_precond = torch.addmm(g2d, self.U_l, projected)
        else:
            projected = g2d @ self.U_r
            projected.mul_(self.S_r_correction.unsqueeze(0))
            G_precond = torch.addmm(g2d, projected, self.U_r.T)

        return from_2d(G_precond, orig_shape).to(orig_dtype)

    # ------------------------------------------------------------------
    # Curvature-aware gradient norm  (for natural gradient clipping)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def natural_grad_norm(self, grad: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Compute the approximate natural gradient norm:

            ||g||_F^2 ≈ g^T (U_l S_l^{-1/2} U_l^T ⊗ U_r S_r^{-1/2} U_r^T) g

        Computation is in float32; returns a float32 scalar.

        Returns:
            Scalar tensor with the natural gradient norm.
        """
        g2d, _ = to_2d(grad.to(_PRECOND_DTYPE))

        # Block-diagonal: sum squared norms over blocks, then sqrt.
        if self.use_block_diagonal:
            sq_norms: list[Tensor] = []
            offset = 0
            for bs, blk in zip(self._block_sizes, self._blocks, strict=True):
                if self._block_dim == 0:
                    g_blk = g2d[offset:offset + bs, :]
                else:
                    g_blk = g2d[:, offset:offset + bs]
                sq_norms.append(blk.natural_grad_norm(g_blk, eps=eps).pow(2))
                offset += bs
            return torch.stack(sq_norms).sum().sqrt()

        if not self.use_kronecker:
            bias = getattr(self, "_diag_bias_factor", 1.0)
            denom = (self.diag_ema / bias).reshape_as(g2d).pow(0.5).add_(eps)
            return (g2d.pow(2) / denom).sum().sqrt()

        if self._tracks_left and self._tracks_right:
            G_proj = (self.U_l.T @ g2d) @ self.U_r
            G_scaled = self.S_l_inv2.unsqueeze(1) * G_proj * self.S_r_inv2.unsqueeze(0)
        elif self._tracks_left:
            G_proj = self.U_l.T @ g2d
            G_scaled = self.S_l_inv2.unsqueeze(1) * G_proj
        else:
            G_proj = g2d @ self.U_r
            G_scaled = G_proj * self.S_r_inv2.unsqueeze(0)

        return cast(Tensor, G_scaled.norm(p="fro"))

    # ------------------------------------------------------------------
    # State dict helpers (for checkpointing)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        d: dict = {
            "precond_step": self.precond_step,
            "k": self._k,  # use _k directly to avoid setter side-effects
            "use_kronecker": self.use_kronecker,
            "use_block_diagonal": self.use_block_diagonal,
            "factor_mode": self.factor_mode,
            "curvature_mode": self.curvature_mode,
            "curvature_rank": self.curvature_rank,
        }
        if self.use_block_diagonal:
            d["blocks"] = [blk.state_dict() for blk in self._blocks]
        elif self.use_kronecker:
            if self.curvature_mode == "low_rank":
                if self._tracks_left:
                    d["L_sketch"] = self.L_sketch
                if self._tracks_right:
                    d["R_sketch"] = self.R_sketch
            elif self.use_int8_ema:
                if self._tracks_left:
                    d.update({"L_ema_q": self.L_ema_q, "L_ema_scale": self.L_ema_scale})
                if self._tracks_right:
                    d.update({"R_ema_q": self.R_ema_q, "R_ema_scale": self.R_ema_scale})
            else:
                if self._tracks_left:
                    d["L_ema"] = self.L_ema
                if self._tracks_right:
                    d["R_ema"] = self.R_ema
            d.update({
                "U_l": self.U_l,
                "S_l": self.S_l,
                "U_r": self.U_r,
                "S_r": self.S_r,
            })
        else:
            d["diag_ema"] = self.diag_ema
        return d

    def memory_bytes(self) -> int:
        """Return total bytes occupied by all internal state tensors."""
        total = 0
        if self.use_block_diagonal:
            return sum(blk.memory_bytes() for blk in self._blocks)
        if self.use_kronecker:
            if self.curvature_mode == "low_rank":
                if self._tracks_left:
                    total += self.L_sketch.numel() * self.L_sketch.element_size()
                if self._tracks_right:
                    total += self.R_sketch.numel() * self.R_sketch.element_size()
            elif self.use_int8_ema:
                # int8 EMA: 1 byte/element (int8) + 8 bytes scale per factor (Python float64)
                if self._tracks_left:
                    total += self.L_ema_q.numel() + 8
                if self._tracks_right:
                    total += self.R_ema_q.numel() + 8
            else:
                if self._tracks_left:
                    total += self.L_ema.numel() * self.L_ema.element_size()
                if self._tracks_right:
                    total += self.R_ema.numel() * self.R_ema.element_size()
            for t in (self.U_l, self.S_l, self.U_r, self.S_r):
                total += t.numel() * t.element_size()
            for t in (
                self.S_l_inv4,
                self.S_r_inv4,
                self.S_l_inv2,
                self.S_r_inv2,
                self.S_l_correction,
                self.S_r_correction,
            ):
                total += t.numel() * t.element_size()
        else:
            total += self.diag_ema.numel() * self.diag_ema.element_size()
        return total

    def load_state_dict(self, state: dict) -> None:
        state_mode = state.get("factor_mode", "full")
        if state_mode != self.factor_mode:
            raise ValueError(
                f"Cannot load factor_mode={state_mode!r} into factor_mode={self.factor_mode!r}"
            )
        state_curvature_mode = state.get("curvature_mode", "dense")
        if state_curvature_mode != self.curvature_mode:
            raise ValueError(
                "Cannot load curvature_mode="
                f"{state_curvature_mode!r} into curvature_mode={self.curvature_mode!r}"
            )
        self.precond_step = state["precond_step"]
        # Assign _k directly to avoid the setter's truncation logic during load;
        # eigenfactors are restored to exactly the saved shape below.
        self._k = state["k"]
        if self.use_block_diagonal:
            for blk, blk_state in zip(self._blocks, state["blocks"], strict=True):
                blk.load_state_dict(blk_state)
        elif self.use_kronecker:
            if self.curvature_mode == "low_rank":
                if self._tracks_left:
                    self.L_sketch = state["L_sketch"].to(
                        device=self.device, dtype=_PRECOND_DTYPE
                    )
                if self._tracks_right:
                    self.R_sketch = state["R_sketch"].to(
                        device=self.device, dtype=_PRECOND_DTYPE
                    )
            elif self.use_int8_ema:
                if self._tracks_left:
                    self.L_ema_q = state["L_ema_q"].to(device=self.device, dtype=torch.int8)
                    self.L_ema_scale = float(state["L_ema_scale"])
                if self._tracks_right:
                    self.R_ema_q = state["R_ema_q"].to(device=self.device, dtype=torch.int8)
                    self.R_ema_scale = float(state["R_ema_scale"])
            else:
                if self._tracks_left:
                    self.L_ema.copy_(state["L_ema"])
                if self._tracks_right:
                    self.R_ema.copy_(state["R_ema"])
            # Eigenfactors are always stored in _PRECOND_DTYPE (float32)
            self.U_l = state["U_l"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.S_l = state["S_l"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.U_r = state["U_r"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self.S_r = state["S_r"].to(device=self.device, dtype=_PRECOND_DTYPE)
            self._refresh_inverse_spectrum_cache()
        else:
            self.diag_ema.copy_(state["diag_ema"])


def _broadcast_precond(
    precond: SparsePreconditioner,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> None:
    """
    Broadcast all preconditioner state tensors from rank 0 to all ranks.

    Handles all three preconditioner modes (Kronecker, block-diagonal, diagonal)
    and both EMA storage formats (float32 and int8).  Also syncs the step counter
    and adaptive rank ``k`` so that subsequent updates remain numerically identical
    across all ranks.

    Args:
        precond: the SparsePreconditioner instance to synchronise.
        process_group: optional process group (default: the global default group).

    Notes:
        This function is called by ``SCAO.sync_preconditioner()``.  It is not
        intended to be called directly unless you manage the distributed state
        yourself.
    """
    import torch.distributed as dist

    # Sync step counter from rank 0.
    step_t = torch.tensor([precond.precond_step], dtype=torch.int64, device=precond.device)
    dist.broadcast(step_t, src=0, group=process_group)
    precond.precond_step = int(step_t.item())

    if precond.use_block_diagonal:
        for blk in precond._blocks:
            _broadcast_precond(blk, process_group)
        # Recompute average rank after sub-blocks are synced.
        precond._k = sum(b._k for b in precond._blocks) // max(len(precond._blocks), 1)
        return

    if precond.use_kronecker:
        # Sync the adaptive rank k; non-rank-0 processes must resize tensors if
        # the checkpoint was saved at a different rank than their current state.
        k_t = torch.tensor([precond._k], dtype=torch.int64, device=precond.device)
        dist.broadcast(k_t, src=0, group=process_group)
        k_new = int(k_t.item())

        if k_new != precond._k:
            # Use _k directly: we're about to replace the tensors in full,
            # so the setter's truncation logic is not appropriate here.
            precond._k = k_new
            precond.U_l = torch.empty(precond.m, k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.S_l = torch.empty(k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.U_r = torch.empty(precond.n, k_new, dtype=_PRECOND_DTYPE, device=precond.device)
            precond.S_r = torch.empty(k_new, dtype=_PRECOND_DTYPE, device=precond.device)

        # Broadcast EMA accumulators.
        if precond.curvature_mode == "low_rank":
            if precond._tracks_left:
                dist.broadcast(precond.L_sketch, src=0, group=process_group)
            if precond._tracks_right:
                dist.broadcast(precond.R_sketch, src=0, group=process_group)
        elif precond.use_int8_ema:
            if precond._tracks_left:
                dist.broadcast(precond.L_ema_q, src=0, group=process_group)
            if precond._tracks_right:
                dist.broadcast(precond.R_ema_q, src=0, group=process_group)
            # Keep scale factors on-device after synchronization so subsequent
            # fallback EMA updates avoid a host round trip.
            scale_attrs = []
            if precond._tracks_left:
                scale_attrs.append("L_ema_scale")
            if precond._tracks_right:
                scale_attrs.append("R_ema_scale")
            for attr in scale_attrs:
                value = getattr(precond, attr)
                t = (
                    value.reshape(1)
                    if isinstance(value, Tensor)
                    else torch.tensor([value], device=precond.device)
                )
                dist.broadcast(t, src=0, group=process_group)
                setattr(precond, attr, t.squeeze(0))
        else:
            if precond._tracks_left:
                dist.broadcast(precond.L_ema, src=0, group=process_group)
            if precond._tracks_right:
                dist.broadcast(precond.R_ema, src=0, group=process_group)

        # Broadcast eigenfactors (in-place: tensors already have the right shape).
        dist.broadcast(precond.U_l, src=0, group=process_group)
        dist.broadcast(precond.S_l, src=0, group=process_group)
        dist.broadcast(precond.U_r, src=0, group=process_group)
        dist.broadcast(precond.S_r, src=0, group=process_group)
        precond._refresh_inverse_spectrum_cache()

    else:
        # Diagonal fallback
        dist.broadcast(precond.diag_ema, src=0, group=process_group)
