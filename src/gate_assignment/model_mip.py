"""单阶段机位分配 MILP 模型。"""

from __future__ import annotations

from typing import Dict, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .data import GateAssignmentInstance


def build_single_stage_mip(
    inst: GateAssignmentInstance,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    model_name: str = "gate_single_stage",
) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var]]:
    """
    构建“真实到达时间”版本的单阶段机位分配 MILP。

    Args:
        inst: 需要求解的日实例。
        buffer_min: 相邻两航班在同一机位上的安全缓冲分钟数。
        turnaround_min: 机位占用的周转时间长度。
        big_m: Big-M 常数，用于线性化时间冲突约束。
        model_name: Gurobi 模型名称。

    Returns:
        (model, x_vars) 元组，其中 x_vars[(i, j)] 表示航班 i 分配到机位 j 的决策变量。
    """

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    if flights == 0 or stands == 0:
        raise ValueError("实例中航班或机位数量为 0，无法构建 MILP")

    arrivals = np.asarray(inst.arrival_true_min, dtype=float)
    departures = arrivals + turnaround_min
    finish_times = departures + buffer_min

    model = gp.Model(model_name)
    x_vars: Dict[Tuple[int, int], gp.Var] = {}

    for i in range(flights):
        for j in range(stands):
            ub = 1.0 if inst.compat_matrix[i, j] else 0.0
            x_vars[(i, j)] = model.addVar(
                vtype=GRB.BINARY,
                ub=ub,
                name=f"x_{i}_{j}",
            )
    model.update()

    objective = gp.quicksum(
        inst.taxi_cost_matrix[i, j] * x_vars[(i, j)]
        for i in range(flights)
        for j in range(stands)
    )
    model.setObjective(objective, GRB.MINIMIZE)

    for i in range(flights):
        model.addConstr(
            gp.quicksum(x_vars[(i, j)] for j in range(stands)) == 1,
            name=f"assign_{i}",
        )

    y_vars: Dict[Tuple[int, int, int], gp.Var] = {}
    for j in range(stands):
        compatible_flights = [i for i in range(flights) if inst.compat_matrix[i, j]]
        for idx_a in range(len(compatible_flights)):
            i = compatible_flights[idx_a]
            for idx_b in range(idx_a + 1, len(compatible_flights)):
                k = compatible_flights[idx_b]
                if finish_times[i] <= arrivals[k] or finish_times[k] <= arrivals[i]:
                    # 若时间窗口天然不重叠，则无需强制排序。
                    continue
                # TODO: 只为真正存在潜在冲突的机位-航班组合建模，以减少变量规模
                y_ik = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}_{j}")
                y_ki = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}_{j}")
                y_vars[(i, k, j)] = y_ik
                y_vars[(k, i, j)] = y_ki

                model.addConstr(
                    y_ik + y_ki >= x_vars[(i, j)] + x_vars[(k, j)] - 1,
                    name=f"order_link_{i}_{k}_{j}",
                )
                model.addConstr(
                    arrivals[k]
                    - departures[i]
                    - buffer_min
                    + big_m * (3 - y_ik - x_vars[(i, j)] - x_vars[(k, j)])
                    >= 0,
                    name=f"time_{i}_before_{k}_{j}",
                )
                model.addConstr(
                    arrivals[i]
                    - departures[k]
                    - buffer_min
                    + big_m * (3 - y_ki - x_vars[(i, j)] - x_vars[(k, j)])
                    >= 0,
                    name=f"time_{k}_before_{i}_{j}",
                )

    return model, x_vars


def solve_single_stage(
    inst: GateAssignmentInstance,
    buffer_min: float = 15.0,
    turnaround_min: float = 60.0,
    big_m: float = 24 * 60.0,
    time_limit: float | None = None,
) -> Tuple[np.ndarray, float]:
    """
    使用单阶段 MILP 求解给定日实例。

    Args:
        inst: 待求解的 GateAssignmentInstance。
        buffer_min: 时间冲突缓冲（分钟）。
        turnaround_min: 停机位占用时间（分钟）。
        big_m: Big-M 常数。
        time_limit: 可选求解时间上限（秒）。

    Returns:
        (x_solution, obj_value)，其中 x_solution 为 (航班数, 机位数) 的 0/1 矩阵。
    """

    model, x_vars = build_single_stage_mip(
        inst,
        buffer_min=buffer_min,
        turnaround_min=turnaround_min,
        big_m=big_m,
    )
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.optimize()

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"模型未找到可行解，status={model.Status}")

    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    x_solution = np.zeros((flights, stands), dtype=int)
    for (i, j), var in x_vars.items():
        x_solution[i, j] = int(round(var.X))

    return x_solution, float(model.ObjVal)
