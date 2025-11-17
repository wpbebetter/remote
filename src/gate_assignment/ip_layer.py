"""Interior-point LP layer for gate assignment.

该模块借鉴 reference/code/NSP/ip_model_whole.py 的 IPOfunc 思路，
但先实现一个精简版：forward 通过 SciPy 的 interior-point LP 求解器得到松弛解，
backward 暂返回零梯度，后续将按照论文补充 KKT 求导。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog
from torch.autograd import Function

Tensor = torch.Tensor


def _to_numpy(tensor: Optional[Tensor]) -> Optional[np.ndarray]:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _bounds_from_arrays(lower: np.ndarray, upper: np.ndarray) -> Sequence[Tuple[float, float]]:
    if lower.shape != upper.shape:
        raise ValueError("lower/upper bounds shape mismatch")
    return list(zip(lower.tolist(), upper.tolist()))


class GateIPFunction(Function):
    """Torch autograd Function wrapping an LP interior-point solve."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        c: Tensor,
        b_eq: Tensor,
        h_ub: Tensor,
        A_eq: Optional[sparse.csr_matrix],
        G_ub: Optional[sparse.csr_matrix],
        bounds: Tuple[np.ndarray, np.ndarray],
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        method = "interior-point"
        options = {"tol": 1e-9}
        if solver_options:
            method = solver_options.get("method", method)
            options.update(solver_options.get("options", {}))

        c_np = _to_numpy(c).astype(float)
        b_eq_np = _to_numpy(b_eq).astype(float)
        h_ub_np = _to_numpy(h_ub).astype(float)
        bounds_seq = _bounds_from_arrays(bounds[0], bounds[1])

        res = linprog(
            c_np,
            A_ub=G_ub,
            b_ub=h_ub_np,
            A_eq=A_eq,
            b_eq=b_eq_np,
            bounds=bounds_seq,
            method=method,
            options=options,
        )
        if not res.success:
            raise RuntimeError(f"Interior-point solver failed: {res.message}")

        x = torch.from_numpy(res.x).to(c.dtype).to(c.device)
        ctx.save_for_backward(c, b_eq, h_ub)
        ctx.solver_meta = {
            "status": res.status,
            "message": res.message,
            "iterations": getattr(res, "nit", None),
        }
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:  # type: ignore[override]
        c, b_eq, h_ub = ctx.saved_tensors
        zeros_like = torch.zeros_like
        grad_c = zeros_like(c)
        grad_b = zeros_like(b_eq)
        grad_h = zeros_like(h_ub)
        return grad_c, grad_b, grad_h, None, None, None, None


def gate_ip_solve(
    c: Tensor,
    b_eq: Tensor,
    h_ub: Tensor,
    A_eq: Optional[sparse.csr_matrix],
    G_ub: Optional[sparse.csr_matrix],
    bounds: Tuple[np.ndarray, np.ndarray],
    solver_options: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Convenience wrapper around GateIPFunction.apply."""

    return GateIPFunction.apply(c, b_eq, h_ub, A_eq, G_ub, bounds, solver_options)

