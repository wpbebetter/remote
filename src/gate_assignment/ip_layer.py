"""Differentiable LP layer for gate assignment LP relaxations."""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from torch.autograd import Function

from qpth.qp import pdipm_b


_LAST_IP_STATS: dict[str, Any] | None = None


@dataclass
class IPParams:
    """Hyperparameters for the interior-point layer."""

    max_iter: int = 30
    tol: float = 1e-9
    qp_regularization: float = 1e-6
    ratio_clamp: float = 1e-8
    verbose: int = 0
    fallback_to_highs: bool = True
    capture_warnings: bool = True


def _tensor_or_empty(t: Optional[torch.Tensor], n_cols: int, like: torch.Tensor) -> torch.Tensor:
    if t is None:
        return torch.zeros((0, n_cols), dtype=like.dtype, device=like.device)
    return t


def _vector_or_empty(t: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if t is None:
        return torch.zeros(0, dtype=like.dtype, device=like.device)
    return t


def _augment_inequalities(
    G: torch.Tensor,
    h: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Combine general inequalities with bound constraints."""

    n = G.shape[1]
    device = G.device
    dtype = G.dtype

    parts_G = []
    parts_h = []

    base_rows = G.shape[0]
    if base_rows > 0:
        parts_G.append(G)
        parts_h.append(h)

    eye = torch.eye(n, dtype=dtype, device=device)
    finite_ub = torch.isfinite(ub)
    if finite_ub.any():
        parts_G.append(eye[finite_ub])
        parts_h.append(ub[finite_ub])

    finite_lb = torch.isfinite(lb)
    if finite_lb.any():
        parts_G.append(-eye[finite_lb])
        parts_h.append(-lb[finite_lb])

    if parts_G:
        G_full = torch.cat(parts_G, dim=0)
        h_full = torch.cat(parts_h, dim=0)
    else:
        G_full = torch.zeros((0, n), dtype=dtype, device=device)
        h_full = torch.zeros(0, dtype=dtype, device=device)

    return G_full, h_full, base_rows


def solve_lp_highs_debug(
    c: torch.Tensor,
    A_eq: torch.Tensor,
    b_eq: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> np.ndarray:
    """Solve LP with SciPy HiGHS for forward reference."""

    bound_list = list(zip(lb.detach().cpu().numpy(), ub.detach().cpu().numpy()))
    res = linprog(
        c.detach().cpu().numpy(),
        A_eq=A_eq.detach().cpu().numpy() if A_eq.numel() else None,
        b_eq=b_eq.detach().cpu().numpy() if b_eq.numel() else None,
        A_ub=G.detach().cpu().numpy() if G.numel() else None,
        b_ub=h.detach().cpu().numpy() if h.numel() else None,
        bounds=bound_list,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"HiGHS failed: {res.message}")
    return res.x


class GateIPFunction(Function):
    """Interior-point layer with analytical gradients for constraint RHS."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        c: torch.Tensor,
        A_eq: torch.Tensor,
        b_eq: torch.Tensor,
        G: torch.Tensor,
        h: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
        params: Optional[IPParams] = None,
    ) -> torch.Tensor:
        if params is None:
            params = IPParams()

        dtype = c.dtype
        device = c.device
        n = c.shape[0]

        G_full, h_full, base_rows = _augment_inequalities(G, h, lb, ub)

        Q = torch.eye(n, dtype=dtype, device=device).mul(params.qp_regularization).unsqueeze(0)
        p = c.unsqueeze(0)
        G_batch = G_full.unsqueeze(0)
        h_batch = h_full.unsqueeze(0)
        if A_eq.numel():
            A_batch = A_eq.unsqueeze(0)
            b_batch = b_eq.unsqueeze(0)
        else:
            A_batch = torch.zeros((1, 0, n), dtype=dtype, device=device)
            b_batch = torch.zeros((1, 0), dtype=dtype, device=device)

        ctx.base_rows = base_rows
        ctx.total_ineq = G_full.shape[0]
        ctx.neq = A_batch.shape[1]
        ctx.params = params

        ctx.Q_LU, ctx.S_LU, ctx.R = pdipm_b.pre_factor_kkt(Q, G_batch, A_batch)

        capture_stdout = params.capture_warnings
        warning_text = ""
        start_time = time.perf_counter()
        if capture_stdout:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                x_hat, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                    Q,
                    p,
                    G_batch,
                    h_batch,
                    A_batch,
                    b_batch,
                    ctx.Q_LU,
                    ctx.S_LU,
                    ctx.R,
                    eps=params.tol,
                    verbose=params.verbose,
                    maxIter=params.max_iter,
                )
            warning_text = buffer.getvalue()
        else:
            x_hat, ctx.nus, ctx.lams, ctx.slacks = pdipm_b.forward(
                Q,
                p,
                G_batch,
                h_batch,
                A_batch,
                b_batch,
                ctx.Q_LU,
                ctx.S_LU,
                ctx.R,
                eps=params.tol,
                verbose=params.verbose,
                maxIter=params.max_iter,
            )
        elapsed = time.perf_counter() - start_time
        warning_flag = "qpth warning" in warning_text.lower()
        needs_fallback = warning_flag or not torch.isfinite(x_hat).all()
        fallback_used = False

        if warning_text and params.verbose > 0:
            print(warning_text.strip())

        x_out = x_hat.squeeze(0)
        ctx.use_qpth_backward = True

        if needs_fallback and params.fallback_to_highs:
            fallback_used = True
            highs_solution = solve_lp_highs_debug(
                c,
                A_eq,
                b_eq,
                G_full,
                h_full,
                lb,
                ub,
            )
            x_out = torch.from_numpy(highs_solution).to(dtype=dtype, device=device)
            ctx.use_qpth_backward = False

        stats = {
            "runtime_sec": elapsed,
            "warning_flag": warning_flag,
            "warning_text": warning_text.strip(),
            "fallback_used": fallback_used,
            "n_var": n,
            "n_eq": A_batch.shape[1],
            "n_ineq": G_full.shape[0],
            "max_iter": params.max_iter,
            "tol": params.tol,
        }
        _set_last_ip_stats(stats)

        ctx.save_for_backward(x_hat, Q, p, G_batch, h_batch, A_batch, b_batch)
        return x_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        (
            zhats,
            Q,
            p,
            G,
            h,
            A,
            b,
        ) = ctx.saved_tensors

        if not getattr(ctx, "use_qpth_backward", True) or ctx.total_ineq == 0:
            grad_h = torch.zeros(ctx.base_rows, dtype=h.dtype, device=h.device)
            return (None, None, None, None, grad_h, None, None, None)

        grad = grad_output.unsqueeze(0)
        clamp = ctx.params.ratio_clamp
        d = torch.clamp(ctx.lams, min=clamp) / torch.clamp(ctx.slacks, min=clamp)
        try:
            pdipm_b.factor_kkt(ctx.S_LU, ctx.R, d)

            zeros_nineq = torch.zeros_like(ctx.lams)
            if ctx.neq > 0:
                zeros_neq = torch.zeros((1, ctx.neq), dtype=G.dtype, device=G.device)
            else:
                zeros_neq = torch.zeros((1, 0), dtype=G.dtype, device=G.device)

            dx, _, dlam, _ = pdipm_b.solve_kkt(
                ctx.Q_LU,
                d,
                G,
                A,
                ctx.S_LU,
                grad,
                zeros_nineq,
                zeros_nineq,
                zeros_neq,
            )
        except RuntimeError as err:
            print(
                f"[GateIPFunction] backward KKT solve failed: {err}. "
                "Returning zero gradients for h."
            )
            grad_h = torch.zeros(ctx.base_rows, dtype=h.dtype, device=h.device)
            return (None, None, None, None, grad_h, None, None, None)

        grad_h_full = -dlam.squeeze(0)
        base_rows = ctx.base_rows
        grad_h = grad_h_full[:base_rows] if base_rows > 0 else torch.zeros(0, dtype=h.dtype, device=h.device)
        return (None, None, None, None, grad_h, None, None, None)


def gate_ip_solve(
    c: torch.Tensor,
    A_eq: Optional[torch.Tensor],
    b_eq: Optional[torch.Tensor],
    G: Optional[torch.Tensor],
    h: Optional[torch.Tensor],
    bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    params: Optional[IPParams] = None,
) -> torch.Tensor:
    """Unified interface for LP solving inside torch graphs."""

    n = c.shape[0]
    device = c.device
    dtype = c.dtype

    A_eq_tensor = _tensor_or_empty(A_eq, n, c)
    b_eq_tensor = _vector_or_empty(b_eq, c)
    G_tensor = _tensor_or_empty(G, n, c)
    h_tensor = _vector_or_empty(h, c)

    if bounds is None:
        lb = torch.zeros(n, dtype=dtype, device=device)
        ub = torch.full((n,), float("inf"), dtype=dtype, device=device)
    else:
        lb, ub = bounds
    return GateIPFunction.apply(c, A_eq_tensor, b_eq_tensor, G_tensor, h_tensor, lb, ub, params)


__all__ = [
    "IPParams",
    "GateIPFunction",
    "gate_ip_solve",
    "solve_lp_highs_debug",
    "get_last_ip_stats",
]


def _set_last_ip_stats(stats: dict[str, Any]) -> None:
    global _LAST_IP_STATS
    _LAST_IP_STATS = stats


def get_last_ip_stats() -> dict[str, Any] | None:
    """Return the most recent IP runtime stats recorded by GateIPFunction."""
    if _LAST_IP_STATS is None:
        return None
    return dict(_LAST_IP_STATS)
