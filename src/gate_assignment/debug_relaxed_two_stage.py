"""Compare integer two-stage vs relaxed two-stage pipeline."""

from __future__ import annotations

import time

import numpy as np

from .data import GateAssignmentInstance, build_daily_instances
from .model_mip import solve_single_stage, solve_stage1, solve_stage2
from .model_relaxed import (
    solve_stage1_relaxed_ip,
    solve_stage2_relaxed_ip,
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
    )


def _select_instance(instances: list[GateAssignmentInstance], target_flights: int) -> GateAssignmentInstance:
    for inst in instances:
        if len(inst.flight_ids) >= target_flights:
            return _trim_instance(inst, target_flights)
    return _trim_instance(instances[0], target_flights)


def _objective_from_assignment(inst: GateAssignmentInstance, x_matrix: np.ndarray) -> float:
    return float((inst.taxi_cost_matrix * x_matrix).sum())


def main() -> None:
    min_flights = 10
    max_flights = 10
    instances = build_daily_instances(min_flights_per_day=min_flights)
    if not instances:
        print("没有满足条件的实例。")
        return

    inst = _select_instance(instances, max_flights)
    flights = len(inst.flight_ids)
    stands = len(inst.stand_ids)
    print(f"测试日期: {inst.date.date()}, 航班数: {flights}, 机位数: {stands}")

    arrival_true = inst.arrival_true_min
    rng = np.random.default_rng(0)
    noise = rng.normal(loc=0.0, scale=5.0, size=arrival_true.shape)
    arrival_pred = np.clip(arrival_true + noise, 0.0, 24 * 60.0)
    gamma = 1000.0

    start = time.time()
    x_star_int, obj_star_int = solve_single_stage(inst, time_limit=60.0)
    t_oracle_int = time.time() - start

    start = time.time()
    x1_int, obj1_int = solve_stage1(inst, arrival_pred_min=arrival_pred, time_limit=60.0)
    t_stage1_int = time.time() - start

    start = time.time()
    x2_int, obj2_int = solve_stage2(
        inst,
        arrival_true_min=arrival_true,
        x1_solution=x1_int,
        change_penalty_gamma=gamma,
        time_limit=60.0,
    )
    t_stage2_int = time.time() - start
    regret_int = obj2_int - obj_star_int

    start = time.time()
    x1_rel = solve_stage1_relaxed_ip(inst, arrival_min=arrival_pred)
    t_stage1_rel = time.time() - start

    start = time.time()
    x2_rel = solve_stage2_relaxed_ip(
        inst,
        arrival_true_min=arrival_true,
        x1_reference=x1_rel,
        change_penalty_gamma=gamma,
    )
    t_stage2_rel = time.time() - start

    obj1_rel = _objective_from_assignment(inst, x1_rel)
    penalty_rel = gamma * np.abs(x2_rel - x1_rel).sum()
    obj2_rel = _objective_from_assignment(inst, x2_rel) + penalty_rel
    regret_rel = obj2_rel - obj_star_int

    diff_stage1 = float(np.max(np.abs(x1_int - x1_rel)))
    diff_stage2 = float(np.max(np.abs(x2_int - x2_rel)))

    print("--- 整数两阶段 ---")
    print(f"Oracle obj: {obj_star_int:.2f} (time {t_oracle_int:.2f}s)")
    print(f"Stage1 obj: {obj1_int:.2f} (time {t_stage1_int:.2f}s)")
    print(f"Stage2 obj: {obj2_int:.2f} (time {t_stage2_int:.2f}s) regret: {regret_int:.2f}")

    print("--- Relaxed 两阶段 ---")
    print(f"Stage1 obj: {obj1_rel:.2f} (time {t_stage1_rel:.2f}s)")
    print(f"Stage2 obj: {obj2_rel:.2f} (time {t_stage2_rel:.2f}s) regret: {regret_rel:.2f}")
    print(f"Stage1 diff (L_inf): {diff_stage1:.3e}")
    print(f"Stage2 diff (L_inf): {diff_stage2:.3e}")


if __name__ == "__main__":
    main()
