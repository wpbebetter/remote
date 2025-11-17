"""调试 GateAssignmentInstance 构造逻辑。"""

from __future__ import annotations

from .data import build_daily_instances


def main() -> None:
    """构造若干日实例并打印基本统计。"""
    instances = build_daily_instances(min_flights_per_day=5)
    print(f"构造得到 {len(instances)} 个日实例")
    if not instances:
        return

    inst = min(instances, key=lambda item: len(item.flight_ids))
    print("示例日期:", inst.date.date())
    print("航班数:", len(inst.flight_ids), "机位数:", len(inst.stand_ids))
    print("taxi_cost_matrix shape:", inst.taxi_cost_matrix.shape)
    print("compat_matrix shape:", inst.compat_matrix.shape)
    print("compat_matrix mean (可行比例):", float(inst.compat_matrix.mean()))


if __name__ == "__main__":
    main()
