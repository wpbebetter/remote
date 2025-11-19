"""Differentiable LP layer using Gurobi forward solve and KKT backward gradients."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from scipy.optimize import linprog
from torch.autograd import Function

LOGGER = logging.getLogger(__name__)
_LAST_IP_STATS: dict[str, Any] | None = None


@dataclass
class IPParams:
    """Hyperparameters controlling numerical stability for the LP layer."""

    backward_damp: float = 1e-6
    verbose: int = 0


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
    """Combine explicit inequalities with bound constraints."""

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


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.numel() == 0:
        return np.zeros(t.shape, dtype=np.float64)
    return t.detach().cpu().numpy().astype(np.float64, copy=True)


def solve_lp_gurobi(
    c: np.ndarray,
    A: Optional[np.ndarray],
    b: Optional[np.ndarray],
    G: Optional[np.ndarray],
    h: Optional[np.ndarray],
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve ``min c^T x`` subject to ``Ax=b`` and ``Gx<=h`` using Gurobi."""

    n_x = c.shape[0]
    n_eq = A.shape[0] if A is not None and A.size else 0
    n_ineq = G.shape[0] if G is not None and G.size else 0

    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", verbose)
            env.start()
            with gp.Model(env=env) as model:
                model.setParam("Method", 2)
                model.setParam("Crossover", 0)
                model.setParam("BarConvTol", 1e-6)
                x = model.addMVar(shape=n_x, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
                model.setObjective(c @ x, GRB.MINIMIZE)

                eq_constr = None
                if n_eq > 0:
                    eq_constr = model.addMConstr(A, x, "=", b)

                ineq_constr = None
                if n_ineq > 0:
                    ineq_constr = model.addMConstr(G, x, "<=", h)

                model.optimize()

                status = model.Status
                if status != GRB.OPTIMAL:
                    LOGGER.warning("Gurobi LP solve failed with status %s", status)
                    return (
                        np.zeros(n_x),
                        np.zeros(n_eq),
                        np.zeros(n_ineq),
                        np.zeros(n_ineq),
                        status,
                    )

                x_val = x.X.copy()
                y_val = eq_constr.Pi.copy() if eq_constr is not None else np.zeros(0, dtype=np.float64)
                if ineq_constr is not None:
                    z_raw = -ineq_constr.Pi.copy()
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        s_val = h - G @ x_val
                else:
                    z_raw = np.zeros(0, dtype=np.float64)
                    s_val = np.zeros(0, dtype=np.float64)

                z_val = np.maximum(z_raw, 1e-8)
                s_val = np.maximum(s_val, 1e-8)
                return x_val, y_val, z_val, s_val, status
    except gp.GurobiError as exc:
        LOGGER.error("Gurobi raised an exception: %s", exc)
        return (
            np.zeros(n_x),
            np.zeros(n_eq),
            np.zeros(n_ineq),
            np.zeros(n_ineq),
            -1,
        )


def solve_kkt_backward(
    grad_output: torch.Tensor,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    A: Optional[np.ndarray],
    G: Optional[np.ndarray],
    damp: float,
) -> np.ndarray:
    """Solve for ``grad_h`` using the KKT implicit differentiation system."""

    del y
    dtype = np.float64
    grad_np = grad_output.detach().cpu().numpy().astype(dtype, copy=True)

    if G is None or G.size == 0:
        return np.zeros(0, dtype=dtype)

    n_x = grad_np.shape[0]
    if A is None or A.size == 0:
        A = np.zeros((0, n_x), dtype=dtype)
    if G is None or G.size == 0:
        G = np.zeros((0, n_x), dtype=dtype)

    n_eq = A.shape[0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        W = z / (s + 1e-9)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        H = G.T @ (W[:, None] * G)
    H += damp * np.eye(n_x, dtype=dtype)

    if n_eq > 0:
        top = np.concatenate([H, A.T], axis=1)
        bottom = np.concatenate([A, np.zeros((n_eq, n_eq), dtype=dtype)], axis=1)
        KKT = np.concatenate([top, bottom], axis=0)
        rhs = np.concatenate([grad_np, np.zeros(n_eq, dtype=dtype)])
    else:
        KKT = H
        rhs = grad_np

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    v_x = sol[:n_x]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        grad_h = W * (G @ v_x)
    return grad_h


class GateIPFunction(Function):
    """Custom torch Function wrapping the Gurobi solve + KKT backward."""

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

        c_np = _to_numpy(c)
        A_np = _to_numpy(A_eq) if A_eq.numel() else None
        b_np = _to_numpy(b_eq) if b_eq.numel() else None
        G_np = _to_numpy(G_full) if G_full.numel() else None
        h_np = _to_numpy(h_full) if h_full.numel() else None

        start = time.perf_counter()
        x_np, y_np, z_np, s_np, status = solve_lp_gurobi(c_np, A_np, b_np, G_np, h_np, verbose=params.verbose)
        elapsed = time.perf_counter() - start

        fallback_used = status != GRB.OPTIMAL
        if fallback_used:
            if params.verbose > 0:
                LOGGER.warning("Gurobi did not return OPTIMAL status (%s); falling back to HiGHS", status)
            c_ref = c.detach().cpu().double()
            A_ref = A_eq.detach().cpu().double()
            b_ref = b_eq.detach().cpu().double()
            G_ref = G_full.detach().cpu().double()
            h_ref = h_full.detach().cpu().double()
            lb_ref = lb.detach().cpu().double()
            ub_ref = ub.detach().cpu().double()
            x_fallback = solve_lp_highs_debug(c_ref, A_ref, b_ref, G_ref, h_ref, lb_ref, ub_ref)
            x_tensor = torch.from_numpy(x_fallback).to(dtype=dtype, device=device)
        else:
            x_tensor = torch.from_numpy(x_np).to(dtype=dtype, device=device)

        stats = {
            "runtime_sec": elapsed,
            "status": status,
            "fallback_used": fallback_used,
            "warning_flag": fallback_used,
            "n_var": n,
            "n_eq": A_eq.shape[0],
            "n_ineq": G_full.shape[0],
        }
        _set_last_ip_stats(stats)

        ctx.base_rows = base_rows
        ctx.total_ineq = G_full.shape[0]
        ctx.h_device = h.device
        ctx.h_dtype = h.dtype
        ctx.params = params
        ctx.backward_ready = (not fallback_used) and (ctx.total_ineq > 0) and (base_rows > 0)
        ctx.A_np = A_np
        ctx.G_np = G_np
        ctx.x_np = x_np
        ctx.y_np = y_np
        ctx.z_np = z_np
        ctx.s_np = s_np

        return x_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        if not getattr(ctx, "backward_ready", False):
            grad_h = torch.zeros(ctx.base_rows, dtype=ctx.h_dtype, device=ctx.h_device)
            return (None, None, None, None, grad_h, None, None, None)

        grad_h_full = solve_kkt_backward(
            grad_output,
            ctx.x_np,
            ctx.y_np,
            ctx.z_np,
            ctx.s_np,
            ctx.A_np,
            ctx.G_np,
            ctx.params.backward_damp,
        )
        grad_h_tensor = torch.from_numpy(grad_h_full).to(ctx.h_device, ctx.h_dtype)
        base_rows = ctx.base_rows
        if base_rows > 0:
            grad_h = grad_h_tensor[:base_rows]
        else:
            grad_h = torch.zeros(0, dtype=ctx.h_dtype, device=ctx.h_device)

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


def solve_lp_highs_debug(
    c: torch.Tensor,
    A_eq: torch.Tensor,
    b_eq: torch.Tensor,
    G: torch.Tensor,
    h: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> np.ndarray:
    """Reference solver using SciPy HiGHS for debugging scripts."""

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


__all__ = [
    "IPParams",
    "GateIPFunction",
    "gate_ip_solve",
    "solve_lp_gurobi",
    "solve_lp_highs_debug",
    "solve_kkt_backward",
    "get_last_ip_stats",
]


def _set_last_ip_stats(stats: dict[str, Any]) -> None:
    global _LAST_IP_STATS
    _LAST_IP_STATS = stats


def get_last_ip_stats() -> dict[str, Any] | None:
    if _LAST_IP_STATS is None:
        return None
    return dict(_LAST_IP_STATS)
