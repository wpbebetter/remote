"""两阶段 MILP 流程调试脚本（大规模实例）。"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .data import GateAssignmentInstance, build_daily_instances
from .model_mip import (
    find_potential_conflict_pairs,
    solve_single_stage,
    solve_stage1,
    solve_stage2,
)

TURNAROUND_MIN = 60.0
BUFFER_MIN = 15.0


def _trim_instance(inst: GateAssignmentInstance, max_flights: int) -> GateAssignmentInstance:
    """裁剪航班数量，避免模型规模过大。"""
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
    )


def _select_instance(
    instances: list[GateAssignmentInstance],
    min_flights: int,
    max_flights: int,
) -> GateAssignmentInstance:
    """选出满足最小航班量的实例，并裁剪到 max_flights。"""
    for inst in instances:
        if len(inst.flight_ids) >= min_flights:
            return _trim_instance(inst, max_flights)
    return _trim_instance(instances[0], max_flights)


def _estimate_y_variables(inst: GateAssignmentInstance, pairs: list[Tuple[int, int]]) -> int:
    """估算需要创建的 y 变量数量。"""
    compat = inst.compat_matrix
    total = 0
    for i, k in pairs:
        shared = int(np.logical_and(compat[i], compat[k]).sum())
        total += 2 * shared
    return total


def _timed_call(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def main() -> None:
    """运行大规模两阶段 MILP，输出冲突对统计与 regret。"""
    min_flights = 40
    max_flights = 60
    instances = build_daily_instances(min_flights_per_day=min_flights)
    if not instances:
        print("没有可用的日实例，请检查数据。")
        return

    inst = _select_instance(instances, min_flights=min_flights, max_flights=max_flights)
    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    print("测试日期:", inst.date.date())
    print(f"航班数: {flights}, 机位数: {stands}")
    print(f"compat_matrix mean (可行比例): {inst.compat_matrix.mean():.3f}")

    pairs = find_potential_conflict_pairs(
        arrival_min=inst.arrival_true_min,
        turnaround_min=TURNAROUND_MIN,
        buffer_min=BUFFER_MIN,
    )
    y_vars_est = _estimate_y_variables(inst, pairs)
    print(f"潜在冲突对数量: {len(pairs)}")
    print(f"估计 y 变量数量: {y_vars_est}")
    print(f"x 变量总数: {flights * stands}")

    arrival_true = inst.arrival_true_min
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=0.0, scale=5.0, size=arrival_true.shape)
    arrival_pred = np.clip(arrival_true + noise, 0.0, 24 * 60 - 1e-3)

    (x_star, obj_star), t_star = _timed_call(
        solve_single_stage,
        inst,
        time_limit=120.0,
    )
    (x1, obj1), t_stage1 = _timed_call(
        solve_stage1,
        inst,
        arrival_pred_min=arrival_pred,
        time_limit=120.0,
    )
    gamma = 1000.0
    (x2, obj2), t_stage2 = _timed_call(
        solve_stage2,
        inst,
        arrival_true_min=arrival_true,
        x1_solution=x1,
        change_penalty_gamma=gamma,
        time_limit=120.0,
    )

    regret = obj2 - obj_star
    print(f"Oracle 目标: {obj_star:.2f}, runtime: {t_star:.2f}s")
    print(f"Stage1 目标: {obj1:.2f}, runtime: {t_stage1:.2f}s")
    print(f"Stage2 目标: {obj2:.2f}, runtime: {t_stage2:.2f}s, gamma={gamma}")
    print(f"Post-hoc regret: {regret:.2f}")

    for i, flight_id in enumerate(inst.flight_ids[:10]):
        stand_stage1 = inst.stand_ids[x1[i].argmax()]
        stand_stage2 = inst.stand_ids[x2[i].argmax()]
        stand_star = inst.stand_ids[x_star[i].argmax()]
        print(
            f"航班 {flight_id}: Stage1 -> {stand_stage1}, "
            f"Stage2 -> {stand_stage2}, Oracle -> {stand_star}"
        )


if __name__ == "__main__":
    main()
