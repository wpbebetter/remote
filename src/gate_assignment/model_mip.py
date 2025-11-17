"""单阶段/两阶段机位分配 MILP 模型。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data import GateAssignmentInstance


def find_potential_conflict_pairs(
    arrival_min: np.ndarray,
    turnaround_min: float,
    buffer_min: float,
) -> List[Tuple[int, int]]:
    """
    根据到达时间与固定周转时间，筛选潜在时间冲突的航班对。

    若两航班在区间 [A_i, A_i + turnaround_min + buffer_min] 内存在重叠，
    视为可能冲突，需要在同一机位上强制排序。
    """

    arrival = np.asarray(arrival_min, dtype=float)
    flights = len(arrival)
    if flights <= 1:
        return []

    occupy_end = arrival + turnaround_min + buffer_min
    potential_pairs: List[Tuple[int, int]] = []
    for i in range(flights):
        for k in range(i + 1, flights):
            start = max(arrival[i], arrival[k])
            end = min(occupy_end[i], occupy_end[k])
            if start < end:
                potential_pairs.append((i, k))
    return potential_pairs


def build_gate_assignment_mip(
    inst: GateAssignmentInstance,
    arrival_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    model_name: str = "gate_assignment",
    add_change_penalty: bool = False,
    x1_reference: np.ndarray | None = None,
    change_penalty_gamma: float = 0.0,
) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Optional[Dict[Tuple[int, int], gp.Var]]]:
    """
    构建“可接受任意到达时间向量”的机位分配 MILP。

    Args:
        inst: 日实例。
        arrival_min: 每个航班的占用起始时间（分钟）向量，可来自真实值或预测值。
        buffer_min: 相邻航班同机位的安全缓冲。
        turnaround_min: 飞机占用机位的服务时间。
        big_m: Big-M 常数。
        model_name: 模型命名。
        add_change_penalty: 是否加入 |x - x1| 的惩罚。
        x1_reference: 参考指派矩阵，add_change_penalty=True 时必填。
        change_penalty_gamma: 惩罚系数。

    Returns:
        (模型, 分配变量字典, 惩罚变量字典/None)。
    """

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    if flights == 0 or stands == 0:
        raise ValueError("实例中航班或机位数量为 0，无法构建 MILP")
    arrival_min = np.asarray(arrival_min, dtype=float)
    if arrival_min.shape[0] != flights:
        raise ValueError("arrival_min 长度与航班数不一致")

    departures = arrival_min + turnaround_min
    potential_pairs = find_potential_conflict_pairs(
        arrival_min=arrival_min,
        turnaround_min=turnaround_min,
        buffer_min=buffer_min,
    )

    if add_change_penalty:
        if x1_reference is None:
            raise ValueError("add_change_penalty=True 需要提供 x1_reference")
        if x1_reference.shape != (flights, stands):
            raise ValueError("x1_reference 维度与实例不匹配")

    model = gp.Model(model_name)
    x_vars: Dict[Tuple[int, int], gp.Var] = {}
    delta_vars: Dict[Tuple[int, int], gp.Var] | None = {} if add_change_penalty else None

    for i in range(flights):
        for j in range(stands):
            ub = 1.0 if inst.compat_matrix[i, j] else 0.0
            x_vars[(i, j)] = model.addVar(
                vtype=GRB.BINARY,
                ub=ub,
                name=f"x_{i}_{j}",
            )
            if delta_vars is not None:
                delta = model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0.0, name=f"delta_{i}_{j}"
                )
                delta_vars[(i, j)] = delta
    model.update()

    objective = gp.quicksum(
        inst.taxi_cost_matrix[i, j] * x_vars[(i, j)]
        for i in range(flights)
        for j in range(stands)
    )
    if delta_vars is not None and change_penalty_gamma > 0:
        objective += change_penalty_gamma * gp.quicksum(delta_vars.values())
    model.setObjective(objective, GRB.MINIMIZE)

    for i in range(flights):
        model.addConstr(
            gp.quicksum(x_vars[(i, j)] for j in range(stands)) == 1,
            name=f"assign_{i}",
        )

    y_vars: Dict[Tuple[int, int, int], gp.Var] = {}
    y_counter = 0
    for (i, k) in potential_pairs:
        for j in range(stands):
            if not (inst.compat_matrix[i, j] and inst.compat_matrix[k, j]):
                continue
            y_ik = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{j}")
            y_ki = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}_{j}")
            y_vars[(i, k, j)] = y_ik
            y_vars[(k, i, j)] = y_ki
            y_counter += 2

            model.addConstr(
                y_ik + y_ki >= x_vars[(i, j)] + x_vars[(k, j)] - 1,
                name=f"order_link_{i}_{k}_{j}",
            )
            model.addConstr(
                arrival_min[k]
                - departures[i]
                - buffer_min
                + big_m * (3 - y_ik - x_vars[(i, j)] - x_vars[(k, j)])
                >= 0,
                name=f"time_{i}_before_{k}_{j}",
            )
            model.addConstr(
                arrival_min[i]
                - departures[k]
                - buffer_min
                + big_m * (3 - y_ki - x_vars[(i, j)] - x_vars[(k, j)])
                >= 0,
                name=f"time_{k}_before_{i}_{j}",
            )

    model._potential_pair_count = len(potential_pairs)
    model._y_var_count = y_counter

    if delta_vars is not None:
        for (i, j), delta in delta_vars.items():
            x1_val = float(x1_reference[i, j])
            model.addConstr(
                delta >= x_vars[(i, j)] - x1_val,
                name=f"delta_pos_{i}_{j}",
            )
            model.addConstr(
                delta >= -(x_vars[(i, j)] - x1_val),
                name=f"delta_neg_{i}_{j}",
            )

    return model, x_vars, delta_vars


def _solve_model(
    model: gp.Model,
    x_vars: Dict[Tuple[int, int], gp.Var],
    flights: int,
    stands: int,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """运行 Gurobi 求解并返回 (解矩阵, 目标值)。"""
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"模型未找到可行解，status={model.Status}")

    x_solution = np.zeros((flights, stands), dtype=int)
    for (i, j), var in x_vars.items():
        x_solution[i, j] = int(round(var.X))
    return x_solution, float(model.ObjVal)


def solve_single_stage(
    inst: GateAssignmentInstance,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """直接使用真实到达时间的单阶段 MILP。"""

    model, x_vars, _ = build_gate_assignment_mip(
        inst,
        arrival_min=inst.arrival_true_min,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
        model_name="gate_single_stage",
    )
    return _solve_model(
        model,
        x_vars,
        flights=len(inst.flight_ids),
        stands=len(inst.stand_ids),
        time_limit=time_limit,
    )


def solve_stage1(
    inst: GateAssignmentInstance,
    arrival_pred_min: np.ndarray,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Stage1：使用预测到达时间求解，仅最小化滑行距离。
    """

    model, x_vars, _ = build_gate_assignment_mip(
        inst,
        arrival_min=arrival_pred_min,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
        model_name="gate_stage1",
    )
    return _solve_model(
        model,
        x_vars,
        flights=len(inst.flight_ids),
        stands=len(inst.stand_ids),
        time_limit=time_limit,
    )


def solve_stage2(
    inst: GateAssignmentInstance,
    arrival_true_min: np.ndarray,
    x1_solution: np.ndarray,
    change_penalty_gamma: float = 1000.0,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Stage2：使用真实到达时间并对 |x - x1| 进行惩罚。
    返回包含惩罚后的目标值。
    """

    model, x_vars, _ = build_gate_assignment_mip(
        inst,
        arrival_min=arrival_true_min,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
        model_name="gate_stage2",
        add_change_penalty=True,
        x1_reference=x1_solution,
        change_penalty_gamma=change_penalty_gamma,
    )
    return _solve_model(
        model,
        x_vars,
        flights=len(inst.flight_ids),
        stands=len(inst.stand_ids),
        time_limit=time_limit,
    )
