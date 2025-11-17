"""Measure GateIPFunction runtime and warnings across larger instances."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch

from .data import build_daily_instances
from .ip_layer import IPParams, get_last_ip_stats
from .model_relaxed import solve_stage1_relaxed_torch, solve_stage2_relaxed_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GateIPFunction runtime profiler")
    parser.add_argument("--min_flights", type=int, default=20)
    parser.add_argument("--max_flights", type=int, default=60)
    parser.add_argument("--num_instances", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=5.0)
    return parser.parse_args()


def load_instances(args: argparse.Namespace):
    instances = build_daily_instances(min_flights_per_day=args.min_flights)
    filtered = [
        inst for inst in instances if len(inst.flight_ids) <= args.max_flights
    ]
    return filtered[: args.num_instances]


def record_stats(
    rows: List[Dict[str, object]],
    inst,
    stage: str,
    runtime: float,
    stats: Dict[str, object] | None,
    ip_params: IPParams,
) -> None:
    row = {
        "date": str(inst.date.date()),
        "flights": len(inst.flight_ids),
        "stands": len(inst.stand_ids),
        "stage": stage,
        "runtime_sec": runtime,
        "warning_flag": bool(stats["warning_flag"]) if stats else False,
        "fallback_used": bool(stats["fallback_used"]) if stats else False,
        "n_var": stats.get("n_var") if stats else None,
        "n_eq": stats.get("n_eq") if stats else None,
        "n_ineq": stats.get("n_ineq") if stats else None,
        "max_iter": ip_params.max_iter,
        "tol": ip_params.tol,
    }
    rows.append(row)


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    instances = load_instances(args)
    if not instances:
        print("未找到符合条件的实例。")
        return

    ip_params = IPParams(
        max_iter=args.max_iter,
        tol=args.tol,
        fallback_to_highs=True,
        capture_warnings=True,
    )

    rows: List[Dict[str, object]] = []
    warning_count = 0
    fallback_count = 0

    for inst in instances:
        flights = len(inst.flight_ids)
        arrival_true_np = inst.arrival_true_min.astype(np.float64)
        noise = rng.normal(0.0, args.noise_std, size=arrival_true_np.shape)
        arrival_pred_np = np.clip(arrival_true_np + noise, 0.0, 24 * 60.0)

        arrival_pred_tensor = torch.from_numpy(arrival_pred_np).double()
        arrival_true_tensor = torch.from_numpy(arrival_true_np).double()

        start = time.perf_counter()
        x1 = solve_stage1_relaxed_torch(
            inst,
            arrival_min_tensor=arrival_pred_tensor,
            ip_params=ip_params,
        )
        stage1_time = time.perf_counter() - start
        stats = get_last_ip_stats()
        warning_count += int(stats["warning_flag"]) if stats else 0
        fallback_count += int(stats["fallback_used"]) if stats else 0
        record_stats(rows, inst, "stage1", stage1_time, stats, ip_params)

        start = time.perf_counter()
        _ = solve_stage2_relaxed_torch(
            inst,
            arrival_true_tensor=arrival_true_tensor,
            x1_tensor=x1,
            change_penalty_gamma=args.gamma,
            ip_params=ip_params,
        )
        stage2_time = time.perf_counter() - start
        stats = get_last_ip_stats()
        warning_count += int(stats["warning_flag"]) if stats else 0
        fallback_count += int(stats["fallback_used"]) if stats else 0
        record_stats(rows, inst, "stage2", stage2_time, stats, ip_params)

        print(
            f"[{inst.date.date()}] flights={flights} "
            f"stage1={stage1_time:.2f}s stage2={stage2_time:.2f}s "
            f"warning={stats['warning_flag'] if stats else False}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, "ip_runtime_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "flights",
                "stands",
                "stage",
                "runtime_sec",
                "warning_flag",
                "fallback_used",
                "n_var",
                "n_eq",
                "n_ineq",
                "max_iter",
                "tol",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("===")
    print(f"写入统计到 {csv_path}")
    print(f"总调用数: {len(rows)}, warnings: {warning_count}, fallbacks: {fallback_count}")


if __name__ == "__main__":
    main()
