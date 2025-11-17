"""LP relaxation utilities for gate assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy import sparse

from .data import GateAssignmentInstance
from .model_mip import find_potential_conflict_pairs


@dataclass
class Stage1LPMatrices:
    """Container for Stage1 LP matrices."""

    c: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    G_ub: sparse.csr_matrix
    h_ub: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    meta: Dict[str, int]


def _flatten_index(flight_idx: int, stand_idx: int, num_stands: int) -> int:
    return flight_idx * num_stands + stand_idx


def build_stage1_lp_matrices(
    inst: GateAssignmentInstance,
    arrival_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
) -> Stage1LPMatrices:
    """
    构造 Stage1 LP 松弛的矩阵形式（仅包含 x 变量）。

    约束：
        1. 每个航班恰好分配到一个机位；
        2. 若两航班占用区间重叠，则在任意同一机位上的指派和不超过 1；
        3. 兼容性通过变量上界体现，0 <= x <= compat。
    """

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    compat = inst.compat_matrix.astype(bool)
    num_vars = flights * stands

    # 目标向量
    c = inst.taxi_cost_matrix.astype(np.float64).reshape(-1)

    # 每个航班的等式约束
    eq_rows, eq_cols, eq_data = [], [], []
    for f in range(flights):
        for s in range(stands):
            idx = _flatten_index(f, s, stands)
            eq_rows.append(f)
            eq_cols.append(idx)
            eq_data.append(1.0)
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(flights, num_vars))
    b_eq = np.ones(flights, dtype=np.float64)

    # 时间冲突不等式
    ineq_rows, ineq_cols, ineq_data, h_vals = [], [], [], []
    row = 0
    potential_pairs = find_potential_conflict_pairs(arrival_min, turnaround_min, buffer_min)
    for i, k in potential_pairs:
        for s in range(stands):
            if compat[i, s] and compat[k, s]:
                idx_i = _flatten_index(i, s, stands)
                idx_k = _flatten_index(k, s, stands)
                ineq_rows.extend([row, row])
                ineq_cols.extend([idx_i, idx_k])
                ineq_data.extend([1.0, 1.0])
                h_vals.append(1.0)
                row += 1
    if h_vals:
        G_ub = sparse.csr_matrix((ineq_data, (ineq_rows, ineq_cols)), shape=(row, num_vars))
        h_ub = np.array(h_vals, dtype=np.float64)
    else:
        G_ub = sparse.csr_matrix((0, num_vars), dtype=np.float64)
        h_ub = np.zeros(0, dtype=np.float64)

    lower_bounds = np.zeros(num_vars, dtype=np.float64)
    upper_bounds = compat.reshape(-1).astype(np.float64)

    meta = {
        "num_vars": num_vars,
        "num_eq": flights,
        "num_ineq": int(row),
        "num_pairs": len(potential_pairs),
    }

    return Stage1LPMatrices(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        G_ub=G_ub,
        h_ub=h_ub,
        bounds=(lower_bounds, upper_bounds),
        meta=meta,
    )


def solve_stage1_lp_gurobi(
    inst: GateAssignmentInstance,
    arrival_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """使用与 LP 矩阵相同的建模方式在 Gurobi 中求解，供校验使用。"""

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    compat = inst.compat_matrix.astype(bool)
    pairs = find_potential_conflict_pairs(arrival_min, turnaround_min, buffer_min)

    model = gp.Model("gate_stage1_lp")
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    x_vars: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(flights):
        for j in range(stands):
            ub = 1.0 if compat[i, j] else 0.0
            x_vars[(i, j)] = model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    model.setObjective(
        gp.quicksum(inst.taxi_cost_matrix[i, j] * x_vars[(i, j)] for i in range(flights) for j in range(stands)),
        GRB.MINIMIZE,
    )

    for i in range(flights):
        model.addConstr(gp.quicksum(x_vars[(i, j)] for j in range(stands)) == 1.0, name=f"assign_{i}")

    for i, k in pairs:
        for j in range(stands):
            if compat[i, j] and compat[k, j]:
                model.addConstr(
                    x_vars[(i, j)] + x_vars[(k, j)] <= 1.0,
                    name=f"overlap_{i}_{k}_{j}",
                )

    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi LP failed with status {model.Status}")

    solution = np.zeros(flights * stands, dtype=np.float64)
    for i in range(flights):
        for j in range(stands):
            solution[_flatten_index(i, j, stands)] = float(x_vars[(i, j)].X)

    return solution, float(model.ObjVal)


__all__ = [
    "Stage1LPMatrices",
    "build_stage1_lp_matrices",
    "solve_stage1_lp_gurobi",
]

