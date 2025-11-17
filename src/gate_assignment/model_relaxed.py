"""LP relaxation utilities for gate assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from scipy import sparse

from .data import GateAssignmentInstance
from .ip_layer import IPParams, gate_ip_solve
from .model_mip import find_potential_conflict_pairs


@dataclass
class Stage1LPMatrices:
    """Stage1 LP matrices and metadata for dynamic RHS construction."""

    c: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    G_ub: sparse.csr_matrix
    h_base: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    meta: Dict[str, int]
    time_rows: np.ndarray  # shape (num_time_rows, 3): [row_idx, flight_before, flight_after]
    big_m: float
    service_time: float


@dataclass
class Stage2LPMatrices:
    c: np.ndarray
    A_eq: sparse.csr_matrix
    b_eq: np.ndarray
    G_ub: sparse.csr_matrix
    h_base: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]
    meta: Dict[str, int]
    time_rows: np.ndarray
    penalty_pos_rows: np.ndarray  # row indices for x - delta <= x1
    penalty_neg_rows: np.ndarray  # row indices for -x - delta <= -x1
    penalty_pos_x: np.ndarray
    penalty_neg_x: np.ndarray
    big_m: float
    service_time: float


def _flatten_index(flight_idx: int, stand_idx: int, num_stands: int) -> int:
    return flight_idx * num_stands + stand_idx


def _csr_to_dense_tensor(matrix: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    if matrix.shape == (0, 0):
        return torch.zeros((0, 0), dtype=torch.double, device=device)
    return torch.from_numpy(matrix.toarray()).double().to(device)


def _stage1_rhs_tensor(mats: Stage1LPMatrices, arrival_tensor: torch.Tensor) -> torch.Tensor:
    h = torch.from_numpy(mats.h_base).double().to(arrival_tensor.device)
    if mats.time_rows.size > 0:
        rows = torch.from_numpy(mats.time_rows[:, 0]).long().to(arrival_tensor.device)
        before_idx = torch.from_numpy(mats.time_rows[:, 1]).long().to(arrival_tensor.device)
        after_idx = torch.from_numpy(mats.time_rows[:, 2]).long().to(arrival_tensor.device)
        values = mats.big_m + arrival_tensor[after_idx] - arrival_tensor[before_idx] - mats.service_time
        h.index_copy_(0, rows, values)
    return h


def _stage2_rhs_tensor(
    mats: Stage2LPMatrices,
    arrival_tensor: torch.Tensor,
    x1_tensor: torch.Tensor,
) -> torch.Tensor:
    h = torch.from_numpy(mats.h_base).double().to(arrival_tensor.device)
    flat_x = x1_tensor.reshape(-1).to(arrival_tensor.device, dtype=torch.double)
    if mats.time_rows.size > 0:
        rows = torch.from_numpy(mats.time_rows[:, 0]).long().to(arrival_tensor.device)
        before_idx = torch.from_numpy(mats.time_rows[:, 1]).long().to(arrival_tensor.device)
        after_idx = torch.from_numpy(mats.time_rows[:, 2]).long().to(arrival_tensor.device)
        values = mats.big_m + arrival_tensor[after_idx] - arrival_tensor[before_idx] - mats.service_time
        h.index_copy_(0, rows, values)

    if mats.penalty_pos_rows.size > 0:
        pos_rows = torch.from_numpy(mats.penalty_pos_rows).long().to(arrival_tensor.device)
        pos_x = torch.from_numpy(mats.penalty_pos_x).long().to(arrival_tensor.device)
        h.index_copy_(0, pos_rows, flat_x[pos_x])

    if mats.penalty_neg_rows.size > 0:
        neg_rows = torch.from_numpy(mats.penalty_neg_rows).long().to(arrival_tensor.device)
        neg_x = torch.from_numpy(mats.penalty_neg_x).long().to(arrival_tensor.device)
        h.index_copy_(0, neg_rows, -flat_x[neg_x])

    return h


def build_stage1_lp_matrices(
    inst: GateAssignmentInstance,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    pair_reference_arrival: np.ndarray | None = None,
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
    n_x = flights * stands
    service_time = turnaround_min + buffer_min
    reference_arrival = pair_reference_arrival if pair_reference_arrival is not None else inst.arrival_true_min

    potential_pairs = find_potential_conflict_pairs(reference_arrival, turnaround_min, buffer_min)
    num_pairs = len(potential_pairs)
    n_y = num_pairs * 2 * stands
    total_vars = n_x + n_y

    # 目标向量
    c = inst.taxi_cost_matrix.astype(np.float64).reshape(-1)
    c = np.pad(c, (0, n_y), constant_values=0.0)

    # 每个航班的等式约束
    eq_rows, eq_cols, eq_data = [], [], []
    for f in range(flights):
        for s in range(stands):
            idx = _flatten_index(f, s, stands)
            eq_rows.append(f)
            eq_cols.append(idx)
            eq_data.append(1.0)
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(flights, total_vars))
    b_eq = np.ones(flights, dtype=np.float64)

    # 时间冲突不等式
    ineq_rows: List[int] = []
    ineq_cols: List[int] = []
    ineq_data: List[float] = []
    h_vals: List[float] = []
    time_rows: List[Tuple[int, int, int]] = []
    row = 0

    def y_index(pair_idx: int, orient: int, stand_idx: int) -> int:
        return n_x + (pair_idx * 2 + orient) * stands + stand_idx

    for pair_idx, (i, k) in enumerate(potential_pairs):
        for orient, (a, b) in enumerate(((i, k), (k, i))):
            for s in range(stands):
                if not (compat[a, s] and compat[b, s]):
                    continue
                y_idx = y_index(pair_idx, orient, s)
                x_a = _flatten_index(a, s, stands)
                x_b = _flatten_index(b, s, stands)
                # y <= x_a
                ineq_rows.extend([row, row])
                ineq_cols.extend([y_idx, x_a])
                ineq_data.extend([1.0, -1.0])
                h_vals.append(0.0)
                row += 1
                # y <= x_b
                ineq_rows.extend([row, row])
                ineq_cols.extend([y_idx, x_b])
                ineq_data.extend([1.0, -1.0])
                h_vals.append(0.0)
                row += 1
                # time feasibility row (placeholder h)
                ineq_rows.append(row)
                ineq_cols.append(y_idx)
                ineq_data.append(big_m)
                h_vals.append(0.0)
                time_rows.append((row, a, b))
                row += 1

        for s in range(stands):
            if not (compat[i, s] and compat[k, s]):
                continue
            y_forward = y_index(pair_idx, 0, s)
            y_backward = y_index(pair_idx, 1, s)
            x_i = _flatten_index(i, s, stands)
            x_k = _flatten_index(k, s, stands)
            ineq_rows.extend([row, row, row, row])
            ineq_cols.extend([x_i, x_k, y_forward, y_backward])
            ineq_data.extend([1.0, 1.0, -1.0, -1.0])
            h_vals.append(1.0)
            row += 1

    if h_vals:
        G_ub = sparse.csr_matrix((ineq_data, (ineq_rows, ineq_cols)), shape=(row, total_vars))
        h_base = np.array(h_vals, dtype=np.float64)
    else:
        G_ub = sparse.csr_matrix((0, total_vars), dtype=np.float64)
        h_base = np.zeros(0, dtype=np.float64)

    lower_bounds = np.zeros(total_vars, dtype=np.float64)
    upper_bounds = np.concatenate(
        [compat.reshape(-1).astype(np.float64), np.ones(n_y, dtype=np.float64)]
    )

    meta = {
        "num_vars": total_vars,
        "num_x": n_x,
        "num_y": n_y,
        "num_eq": flights,
        "num_ineq": int(row),
        "num_pairs": num_pairs,
        "y_offset": n_x,
    }

    return Stage1LPMatrices(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        G_ub=G_ub,
        h_base=h_base,
        bounds=(lower_bounds, upper_bounds),
        meta=meta,
        time_rows=np.array(time_rows, dtype=np.int64) if time_rows else np.zeros((0, 3), dtype=np.int64),
        big_m=big_m,
        service_time=service_time,
    )


def build_stage2_lp_matrices(
    inst: GateAssignmentInstance,
    change_penalty_gamma: float = 1000.0,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    pair_reference_arrival: np.ndarray | None = None,
) -> Stage2LPMatrices:
    """
    Stage2 LP 松弛矩阵，变量为 [x, delta]：
        min taxi_cost^T x + gamma * 1^T delta
        s.t.  航班唯一分配（等式）
              时间冲突约束（同 Stage1，但基于真实到达时间）
              x - delta <= x1_reference
             -x - delta <= -x1_reference
              0 <= x <= compat， 0 <= delta <= 1
    """

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    compat = inst.compat_matrix.astype(bool)
    service_time = turnaround_min + buffer_min
    reference_arrival = pair_reference_arrival if pair_reference_arrival is not None else inst.arrival_true_min

    n_x = flights * stands
    potential_pairs = find_potential_conflict_pairs(reference_arrival, turnaround_min, buffer_min)
    num_pairs = len(potential_pairs)
    n_y = num_pairs * 2 * stands
    n_delta = n_x
    total_vars = n_x + n_y + n_delta

    taxi_flat = inst.taxi_cost_matrix.astype(np.float64).reshape(-1)
    c = np.concatenate(
        [
            taxi_flat,
            np.zeros(n_y, dtype=np.float64),
            np.full(n_delta, change_penalty_gamma, dtype=np.float64),
        ]
    )

    # 等式约束 (sum_s x_{f,s} = 1)
    eq_rows, eq_cols, eq_data = [], [], []
    for f in range(flights):
        for s in range(stands):
            idx = _flatten_index(f, s, stands)
            eq_rows.append(f)
            eq_cols.append(idx)
            eq_data.append(1.0)
    A_eq = sparse.csr_matrix((eq_data, (eq_rows, eq_cols)), shape=(flights, total_vars))
    b_eq = np.ones(flights, dtype=np.float64)

    # 不等式集合
    ineq_rows: List[int] = []
    ineq_cols: List[int] = []
    ineq_data: List[float] = []
    h_vals: List[float] = []
    time_rows: List[Tuple[int, int, int]] = []
    penalty_pos_rows: List[int] = []
    penalty_neg_rows: List[int] = []
    penalty_pos_x: List[int] = []
    penalty_neg_x: List[int] = []
    row = 0

    def y_index(pair_idx: int, orient: int, stand_idx: int) -> int:
        return n_x + (pair_idx * 2 + orient) * stands + stand_idx

    def delta_index(x_idx: int) -> int:
        return n_x + n_y + x_idx

    for pair_idx, (i, k) in enumerate(potential_pairs):
        for orient, (a, b) in enumerate(((i, k), (k, i))):
            for s in range(stands):
                if not (compat[a, s] and compat[b, s]):
                    continue
                y_idx = y_index(pair_idx, orient, s)
                x_a = _flatten_index(a, s, stands)
                x_b = _flatten_index(b, s, stands)
                # y <= x_a
                ineq_rows.extend([row, row])
                ineq_cols.extend([y_idx, x_a])
                ineq_data.extend([1.0, -1.0])
                h_vals.append(0.0)
                row += 1
                # y <= x_b
                ineq_rows.extend([row, row])
                ineq_cols.extend([y_idx, x_b])
                ineq_data.extend([1.0, -1.0])
                h_vals.append(0.0)
                row += 1
                # time row placeholder
                ineq_rows.append(row)
                ineq_cols.append(y_idx)
                ineq_data.append(big_m)
                h_vals.append(0.0)
                time_rows.append((row, a, b))
                row += 1

        for s in range(stands):
            if not (compat[i, s] and compat[k, s]):
                continue
            y_forward = y_index(pair_idx, 0, s)
            y_backward = y_index(pair_idx, 1, s)
            x_i = _flatten_index(i, s, stands)
            x_k = _flatten_index(k, s, stands)
            ineq_rows.extend([row, row, row, row])
            ineq_cols.extend([x_i, x_k, y_forward, y_backward])
            ineq_data.extend([1.0, 1.0, -1.0, -1.0])
            h_vals.append(1.0)
            row += 1

    for idx in range(n_x):
        delta_idx = delta_index(idx)
        # x - delta <= x1_reference (placeholder)
        ineq_rows.extend([row, row])
        ineq_cols.extend([idx, delta_idx])
        ineq_data.extend([1.0, -1.0])
        h_vals.append(0.0)
        penalty_pos_rows.append(row)
        penalty_pos_x.append(idx)
        row += 1
        # -x - delta <= -x1_reference
        ineq_rows.extend([row, row])
        ineq_cols.extend([idx, delta_idx])
        ineq_data.extend([-1.0, -1.0])
        h_vals.append(0.0)
        penalty_neg_rows.append(row)
        penalty_neg_x.append(idx)
        row += 1

    if h_vals:
        G_ub = sparse.csr_matrix((ineq_data, (ineq_rows, ineq_cols)), shape=(row, total_vars))
        h_base = np.array(h_vals, dtype=np.float64)
    else:
        G_ub = sparse.csr_matrix((0, total_vars), dtype=np.float64)
        h_base = np.zeros(0, dtype=np.float64)

    lower_bounds = np.zeros(total_vars, dtype=np.float64)
    upper_bounds = np.concatenate(
        [
            compat.reshape(-1).astype(np.float64),
            np.ones(n_y, dtype=np.float64),
            np.ones(n_delta, dtype=np.float64),
        ]
    )

    meta = {
        "num_vars": total_vars,
        "num_x": n_x,
        "num_y": n_y,
        "num_delta": n_delta,
        "num_pairs": num_pairs,
        "y_offset": n_x,
        "delta_offset": n_x + n_y,
        "num_eq": flights,
        "num_ineq": int(row),
    }

    return Stage2LPMatrices(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        G_ub=G_ub,
        h_base=h_base,
        bounds=(lower_bounds, upper_bounds),
        meta=meta,
        time_rows=np.array(time_rows, dtype=np.int64) if time_rows else np.zeros((0, 3), dtype=np.int64),
        penalty_pos_rows=np.array(penalty_pos_rows, dtype=np.int64),
        penalty_neg_rows=np.array(penalty_neg_rows, dtype=np.int64),
        penalty_pos_x=np.array(penalty_pos_x, dtype=np.int64),
        penalty_neg_x=np.array(penalty_neg_x, dtype=np.int64),
        big_m=big_m,
        service_time=service_time,
    )


def solve_stage1_lp_gurobi(
    inst: GateAssignmentInstance,
    arrival_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """使用与 LP 矩阵相同的建模方式在 Gurobi 中求解，供校验使用。"""

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    compat = inst.compat_matrix.astype(bool)
    service_time = turnaround_min + buffer_min
    pairs = find_potential_conflict_pairs(inst.arrival_true_min, turnaround_min, buffer_min)

    model = gp.Model("gate_stage1_lp")
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0

    x_vars: Dict[Tuple[int, int], gp.Var] = {}
    for i in range(flights):
        for j in range(stands):
            ub = 1.0 if compat[i, j] else 0.0
            x_vars[(i, j)] = model.addVar(lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    y_vars: Dict[Tuple[int, int, int], gp.Var] = {}
    for pair_idx, (i, k) in enumerate(pairs):
        for orient, (a, b) in enumerate(((i, k), (k, i))):
            for s in range(stands):
                if not (compat[a, s] and compat[b, s]):
                    continue
                name = f"y_{pair_idx}_{orient}_{s}"
                y_vars[(pair_idx, orient, s)] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=name)

    model.setObjective(
        gp.quicksum(inst.taxi_cost_matrix[i, j] * x_vars[(i, j)] for i in range(flights) for j in range(stands)),
        GRB.MINIMIZE,
    )

    for i in range(flights):
        model.addConstr(gp.quicksum(x_vars[(i, j)] for j in range(stands)) == 1.0, name=f"assign_{i}")

    for pair_idx, (i, k) in enumerate(pairs):
        for s in range(stands):
            if not (compat[i, s] and compat[k, s]):
                continue
            y_forward = y_vars[(pair_idx, 0, s)]
            y_backward = y_vars[(pair_idx, 1, s)]
            x_i = x_vars[(i, s)]
            x_k = x_vars[(k, s)]
            model.addConstr(y_forward <= x_i, name=f"y_le_x_{pair_idx}_0_{s}_i")
            model.addConstr(y_forward <= x_k, name=f"y_le_x_{pair_idx}_0_{s}_k")
            model.addConstr(y_backward <= x_i, name=f"y_le_x_{pair_idx}_1_{s}_i")
            model.addConstr(y_backward <= x_k, name=f"y_le_x_{pair_idx}_1_{s}_k")
            model.addConstr(
                x_i + x_k - y_forward - y_backward <= 1.0,
                name=f"link_{pair_idx}_{s}",
            )
            # Time feasibility constraints depend on arrival_min
            rhs_forward = big_m + arrival_min[k] - arrival_min[i] - service_time
            rhs_backward = big_m + arrival_min[i] - arrival_min[k] - service_time
            model.addConstr(
                big_m * y_forward <= rhs_forward,
                name=f"time_forward_{pair_idx}_{s}",
            )
            model.addConstr(
                big_m * y_backward <= rhs_backward,
                name=f"time_backward_{pair_idx}_{s}",
            )

    model.optimize()
    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi LP failed with status {model.Status}")

    solution = np.zeros(flights * stands, dtype=np.float64)
    for i in range(flights):
        for j in range(stands):
            solution[_flatten_index(i, j, stands)] = float(x_vars[(i, j)].X)

    return solution, float(model.ObjVal)


def solve_stage1_relaxed_torch(
    inst: GateAssignmentInstance,
    arrival_min_tensor: torch.Tensor,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    ip_params: IPParams | None = None,
) -> torch.Tensor:
    """Torch 版 Stage1 LP 松弛求解。"""

    mats = build_stage1_lp_matrices(
        inst,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
    )
    arrival = arrival_min_tensor.to(dtype=torch.double)
    device = arrival.device
    c_tensor = torch.from_numpy(mats.c).double().to(device)
    A_eq_tensor = _csr_to_dense_tensor(mats.A_eq, device)
    b_tensor = torch.from_numpy(mats.b_eq).double().to(device)
    G_tensor = _csr_to_dense_tensor(mats.G_ub, device)
    h_tensor = _stage1_rhs_tensor(mats, arrival)
    lb_tensor = torch.from_numpy(mats.bounds[0]).double().to(device)
    ub_tensor = torch.from_numpy(mats.bounds[1]).double().to(device)
    solution = gate_ip_solve(
        c_tensor,
        A_eq_tensor,
        b_tensor,
        G_tensor,
        h_tensor,
        (lb_tensor, ub_tensor),
        ip_params,
    )
    n_x = mats.meta["num_x"]
    return solution[:n_x].reshape(len(inst.flight_ids), len(inst.stand_ids))


def solve_stage2_relaxed_ip(
    inst: GateAssignmentInstance,
    arrival_true_min: np.ndarray,
    x1_reference: np.ndarray,
    change_penalty_gamma: float = 1000.0,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
) -> np.ndarray:
    """使用 GateIPFunction 求解 Stage2 LP 松弛（numpy wrapper）。"""

    x1_tensor = torch.from_numpy(x1_reference).double()
    arrival_tensor = torch.from_numpy(arrival_true_min).double()
    x_torch = solve_stage2_relaxed_torch(
        inst,
        arrival_true_tensor=arrival_tensor,
        x1_tensor=x1_tensor,
        change_penalty_gamma=change_penalty_gamma,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
    )
    return x_torch.detach().cpu().numpy()


def solve_stage1_relaxed_ip(
    inst: GateAssignmentInstance,
    arrival_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
) -> np.ndarray:
    """numpy wrapper，供与 Gurobi 对比时使用。"""

    arrival_tensor = torch.from_numpy(arrival_min).double()
    x_torch = solve_stage1_relaxed_torch(
        inst,
        arrival_tensor,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
    )
    return x_torch.detach().cpu().numpy()


def solve_stage2_relaxed_torch(
    inst: GateAssignmentInstance,
    arrival_true_tensor: torch.Tensor,
    x1_tensor: torch.Tensor,
    change_penalty_gamma: float = 1000.0,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    ip_params: IPParams | None = None,
) -> torch.Tensor:
    """Torch 版 Stage2 LP 松弛求解。"""

    mats = build_stage2_lp_matrices(
        inst,
        change_penalty_gamma=change_penalty_gamma,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
    )
    arrival = arrival_true_tensor.to(dtype=torch.double)
    x1 = x1_tensor.to(dtype=torch.double)
    device = arrival.device
    c_tensor = torch.from_numpy(mats.c).double().to(device)
    A_eq_tensor = _csr_to_dense_tensor(mats.A_eq, device)
    b_tensor = torch.from_numpy(mats.b_eq).double().to(device)
    G_tensor = _csr_to_dense_tensor(mats.G_ub, device)
    h_tensor = _stage2_rhs_tensor(mats, arrival, x1)
    lb_tensor = torch.from_numpy(mats.bounds[0]).double().to(device)
    ub_tensor = torch.from_numpy(mats.bounds[1]).double().to(device)
    solution = gate_ip_solve(
        c_tensor,
        A_eq_tensor,
        b_tensor,
        G_tensor,
        h_tensor,
        (lb_tensor, ub_tensor),
        ip_params,
    )
    n_x = mats.meta["num_x"]
    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    return solution[:n_x].reshape(flights, stands)


__all__ = [
    "Stage1LPMatrices",
    "Stage2LPMatrices",
    "build_stage1_lp_matrices",
    "build_stage2_lp_matrices",
    "solve_stage1_lp_gurobi",
    "solve_stage1_relaxed_torch",
    "solve_stage2_relaxed_torch",
    "solve_stage1_relaxed_ip",
    "solve_stage2_relaxed_ip",
]
