"""Torch autograd wrapper around an LP solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from torch.autograd import Function


@dataclass
class IPParams:
    """Placeholder for future interior-point hyperparameters."""

    max_iter: int = 50
    tol: float = 1e-8


def _tensor_or_empty(t: Optional[torch.Tensor], n_cols: int, like: torch.Tensor) -> torch.Tensor:
    if t is None:
        return torch.zeros((0, n_cols), dtype=like.dtype, device=like.device)
    return t


def _vector_or_empty(t: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if t is None:
        return torch.zeros(0, dtype=like.dtype, device=like.device)
    return t


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
    """Temporary autograd stub: forward via HiGHS, backward returns zero grad."""

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
        x_np = solve_lp_highs_debug(c, A_eq, b_eq, G, h, lb, ub)
        x = torch.from_numpy(x_np).to(c)
        ctx.save_for_backward(h)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        (h_saved,) = ctx.saved_tensors
        grad_h = torch.zeros_like(h_saved)
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
]
