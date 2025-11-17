"""快速检查数据加载是否正常的调试脚本。"""

from __future__ import annotations

from .data import (
    load_candidate_flights,
    load_stands,
    load_taxi_distances,
)


def main() -> None:
    """加载三份核心数据并打印基本信息。"""
    taxi = load_taxi_distances()
    stands = load_stands()
    flights = load_candidate_flights()

    print("Taxi distances shape:", taxi.shape)
    print("Stands shape:", stands.shape)
    print("Candidate flights shape:", flights.shape)

    print("Taxi head:\n", taxi.head())
    print("Stands head:\n", stands.head())
    print("Flights head:\n", flights.head())


if __name__ == "__main__":
    main()

