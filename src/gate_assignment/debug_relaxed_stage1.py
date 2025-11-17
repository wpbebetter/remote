"""Compare interior-point LP layer with Gurobi LP solution."""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import GateAssignmentInstance, build_daily_instances
from .model_relaxed import (
    build_stage1_lp_matrices,
    solve_stage1_lp_gurobi,
    solve_stage1_relaxed_torch,
)


def _trim_instance(inst: GateAssignmentInstance, max_flights: int) -> GateAssignmentInstance:
    if len(inst.flight_ids) <= max_flights:
        return inst
    idx = np.arange(max_flights)
    flights_subset = inst.flights.iloc[idx].reset_index(drop=True)
    return GateAssignmentInstance(
        date=inst.date,
        flights=flights_subset,
        stands=inst.stands,
        taxi_distances=inst.taxi_distances,
        flight_ids=inst.flight_ids[idx],
        stand_ids=inst.stand_ids,
        arrival_true_min=inst.arrival_true_min[idx],
        runway_codes=inst.runway_codes[idx],
        taxi_cost_matrix=inst.taxi_cost_matrix[idx, :],
        compat_matrix=inst.compat_matrix[idx, :],
        arrival_sched_min=inst.arrival_sched_min[idx],
        features=inst.features[idx, :],
    )


def _select_instance(instances: list[GateAssignmentInstance], max_flights: int) -> GateAssignmentInstance:
    for inst in instances:
        if len(inst.flight_ids) >= max_flights:
            return _trim_instance(inst, max_flights)
    return _trim_instance(instances[0], max_flights)


def main() -> None:
    instances = build_daily_instances(min_flights_per_day=20)
    if not instances:
        print("未能构造任何日实例。")
        return

    inst = _select_instance(instances, max_flights=25)
    arrival = inst.arrival_true_min

    matrices = build_stage1_lp_matrices(inst)
    print(
        f"测试日期: {inst.date.date()}, 航班数: {len(inst.flight_ids)}, "
        f"机位数: {len(inst.stand_ids)}"
    )
    print(
        f"变量数: {matrices.meta['num_vars']}, 等式: {matrices.meta['num_eq']}, "
        f"不等式: {matrices.meta['num_ineq']}, 潜在冲突对: {matrices.meta['num_pairs']}"
    )

    start_ip = time.time()
    arrival_tensor = torch.from_numpy(arrival).double()
    x_relaxed = solve_stage1_relaxed_torch(inst, arrival_tensor)
    ip_time = time.time() - start_ip
    x_relaxed_np = x_relaxed.detach().cpu().numpy()
    obj_ip = float((inst.taxi_cost_matrix * x_relaxed_np.reshape(inst.taxi_cost_matrix.shape)).sum())

    start_grb = time.time()
    x_lp_gurobi, obj_gurobi = solve_stage1_lp_gurobi(inst, arrival, time_limit=60.0)
    grb_time = time.time() - start_grb

    diff = float(np.max(np.abs(x_lp_gurobi - x_relaxed_np)))
    print(f"Interior-point obj: {obj_ip:.4f}, runtime: {ip_time:.2f}s")
    print(f"Gurobi LP obj:      {obj_gurobi:.4f}, runtime: {grb_time:.2f}s")
    print(f"最大分配差异 (L_inf): {diff:.3e}")


if __name__ == "__main__":
    main()
