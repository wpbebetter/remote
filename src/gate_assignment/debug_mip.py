"""调试单阶段 MILP 求解流程。"""

from __future__ import annotations

from .data import GateAssignmentInstance, build_daily_instances
from .model_mip import solve_single_stage


def _limit_instance(inst: GateAssignmentInstance, max_flights: int) -> GateAssignmentInstance:
    """为快速调试裁剪航班数量，避免模型过大。"""
    if len(inst.flight_ids) <= max_flights:
        return inst
    idx = slice(0, max_flights)
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


def main() -> None:
    """加载一个实例并运行单阶段 MILP。"""
    instances = build_daily_instances(min_flights_per_day=5)
    if not instances:
        print("没有可用的日实例，请检查数据。")
        return

    inst = min(instances, key=lambda item: len(item.flight_ids))
    print("调试日期:", inst.date.date())
    print("原始航班数:", len(inst.flight_ids), "机位数:", len(inst.stand_ids))
    inst_small = _limit_instance(inst, max_flights=10)
    if inst_small is not inst:
        print("裁剪后航班数:", len(inst_small.flight_ids))

    x_sol, obj = solve_single_stage(inst_small, time_limit=60.0)
    print("单阶段最优总滑行距离:", obj)
    for i, flight_id in enumerate(inst_small.flight_ids[:10]):
        stand_idx = x_sol[i].argmax()
        stand_id = inst_small.stand_ids[stand_idx]
        print(f"航班 {flight_id} -> 机位 {stand_id}")


if __name__ == "__main__":
    main()
