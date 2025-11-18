"""Evaluate trained models on large-scale gate assignment instances."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .data import GateAssignmentInstance, build_daily_instances
from .dataset import GateAssignmentDataset
from .model_mip import solve_single_stage, solve_stage1, solve_stage2
from .net import ArrivalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-scale integer two-stage evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--min_flights", type=int, default=40)
    parser.add_argument("--max_flights", type=int, default=80)
    parser.add_argument("--max_instances", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="runs/exp_real_scale_v1")
    parser.add_argument("--tag", type=str, default=None, help="Optional label used for saving CSV")
    return parser.parse_args()


def _infer_mode_label(
    checkpoint: str,
    explicit: str | None = None,
    ckpt_args: Dict[str, object] | None = None,
) -> str:
    if explicit:
        return explicit
    if ckpt_args and isinstance(ckpt_args.get("mode"), str):
        return str(ckpt_args["mode"])
    base = os.path.basename(checkpoint)
    if base.startswith("arrival_model_"):
        suffix = base[len("arrival_model_"):]
        return suffix.split("_")[0]
    return Path(base).stem


def _select_instances(args: argparse.Namespace) -> List[GateAssignmentInstance]:
    instances = build_daily_instances(min_flights_per_day=args.min_flights)
    filtered = [inst for inst in instances if len(inst.flight_ids) <= args.max_flights]
    if not filtered:
        return []
    if args.max_instances is not None and len(filtered) > args.max_instances:
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(filtered))
        rng.shuffle(indices)
        chosen = np.sort(indices[: args.max_instances])
        filtered = [filtered[i] for i in chosen]
    return filtered


def _load_model(
    checkpoint: str,
    input_dim: int,
    device: torch.device,
) -> tuple[ArrivalPredictor, Dict[str, object]]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = ArrivalPredictor(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    return model, ckpt_args


def evaluate_real_scale(args: argparse.Namespace) -> Dict[str, float]:
    """Run integer two-stage evaluation on large-scale instances.

    Args:
        args: CLI namespace describing checkpoint path and data filters.

    Returns:
        Dict summarizing aggregate regret and runtime statistics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instances = _select_instances(args)
    if not instances:
        raise RuntimeError("No instances satisfy the requested flight count range.")
    dataset = GateAssignmentDataset(instances, precompute_oracle=False)

    sample = dataset[0]
    input_dim = sample["features"].shape[1]
    model, ckpt_args = _load_model(args.checkpoint, input_dim, device)
    mode_label = _infer_mode_label(args.checkpoint, args.tag, ckpt_args)

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"real_scale_eval_{mode_label}.csv")

    rows: List[Dict[str, object]] = []
    regrets: List[float] = []
    stage1_costs: List[float] = []
    stage2_costs: List[float] = []
    oracle_costs: List[float] = []
    stage1_times: List[float] = []
    stage2_times: List[float] = []
    oracle_times: List[float] = []

    print(
        f"==== Real-scale evaluation ({mode_label}) ===="
        f"\ncheckpoint={args.checkpoint}\nmin/max flights={args.min_flights}/{args.max_flights} "
        f"instances={len(dataset)}"
    )

    for idx, inst in enumerate(dataset.instances):
        batch = dataset[idx]
        features = batch["features"].to(device)
        arrival_sched = batch["arrival_sched"].to(device)

        with torch.no_grad():
            delta = model(features)
            arrival_pred = torch.clamp(arrival_sched + delta, 0.0, 24 * 60.0)
            arrival_pred_np = arrival_pred.cpu().numpy()

        t_stage1 = time.time()
        x1, obj_stage1 = solve_stage1(
            inst,
            arrival_pred_min=arrival_pred_np,
            time_limit=args.time_limit,
        )
        stage1_runtime = time.time() - t_stage1

        t_stage2 = time.time()
        _, obj_stage2 = solve_stage2(
            inst,
            arrival_true_min=inst.arrival_true_min,
            x1_solution=x1,
            change_penalty_gamma=args.gamma,
            time_limit=args.time_limit,
        )
        stage2_runtime = time.time() - t_stage2

        t_oracle = time.time()
        _, obj_oracle = solve_single_stage(
            inst,
            time_limit=args.time_limit,
        )
        oracle_runtime = time.time() - t_oracle

        regret = obj_stage2 - obj_oracle

        rows.append(
            {
                "kind": "instance",
                "instance_idx": idx,
                "date": inst.date.strftime("%Y-%m-%d"),
                "num_flights": len(inst.flight_ids),
                "num_stands": len(inst.stand_ids),
                "oracle_cost": obj_oracle,
                "stage1_cost": obj_stage1,
                "stage2_cost": obj_stage2,
                "regret": regret,
                "stage1_runtime_sec": stage1_runtime,
                "stage2_runtime_sec": stage2_runtime,
                "oracle_runtime_sec": oracle_runtime,
                "total_runtime_sec": stage1_runtime + stage2_runtime + oracle_runtime,
            }
        )
        regrets.append(regret)
        stage1_costs.append(obj_stage1)
        stage2_costs.append(obj_stage2)
        oracle_costs.append(obj_oracle)
        stage1_times.append(stage1_runtime)
        stage2_times.append(stage2_runtime)
        oracle_times.append(oracle_runtime)

        print(
            f"[Instance {idx}] date={rows[-1]['date']} flights={rows[-1]['num_flights']} "
            f"oracle={obj_oracle:.1f} stage2={obj_stage2:.1f} regret={regret:.1f} "
            f"runtime(s1/s2)={stage1_runtime:.1f}/{stage2_runtime:.1f}"
        )

    regrets_np = np.array(regrets)
    summary = {
        "kind": "summary",
        "instances": len(rows),
        "mean_regret": float(np.mean(regrets_np)),
        "median_regret": float(np.median(regrets_np)),
        "std_regret": float(np.std(regrets_np)),
        "mean_stage1_cost": float(np.mean(stage1_costs)),
        "mean_stage2_cost": float(np.mean(stage2_costs)),
        "mean_oracle_cost": float(np.mean(oracle_costs)),
        "mean_stage1_runtime_sec": float(np.mean(stage1_times)),
        "mean_stage2_runtime_sec": float(np.mean(stage2_times)),
        "mean_oracle_runtime_sec": float(np.mean(oracle_times)),
    }

    rows.append(summary)
    fieldnames = [
        "kind",
        "instance_idx",
        "date",
        "num_flights",
        "num_stands",
        "oracle_cost",
        "stage1_cost",
        "stage2_cost",
        "regret",
        "stage1_runtime_sec",
        "stage2_runtime_sec",
        "oracle_runtime_sec",
        "total_runtime_sec",
        "instances",
        "mean_regret",
        "median_regret",
        "std_regret",
        "mean_stage1_cost",
        "mean_stage2_cost",
        "mean_oracle_cost",
        "mean_stage1_runtime_sec",
        "mean_stage2_runtime_sec",
        "mean_oracle_runtime_sec",
        "checkpoint",
        "mode",
        "min_flights",
        "max_flights",
        "max_instances",
        "gamma",
        "time_limit",
    ]

    for row in rows:
        row.setdefault("checkpoint", args.checkpoint)
        row.setdefault("mode", mode_label)
        row.setdefault("min_flights", args.min_flights)
        row.setdefault("max_flights", args.max_flights)
        row.setdefault("max_instances", args.max_instances)
        row.setdefault("gamma", args.gamma)
        row.setdefault("time_limit", args.time_limit)
        for key in fieldnames:
            row.setdefault(key, "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("==== Summary ====")
    print(
        f"instances={summary['instances']} mean_regret={summary['mean_regret']:.2f} "
        f"median_regret={summary['median_regret']:.2f} std_regret={summary['std_regret']:.2f}"
    )
    print(
        f"mean_stage2={summary['mean_stage2_cost']:.2f} "
        f"mean_oracle={summary['mean_oracle_cost']:.2f}"
    )
    print(
        f"runtime avg (s1/s2)={summary['mean_stage1_runtime_sec']:.1f}/"
        f"{summary['mean_stage2_runtime_sec']:.1f} s"
    )
    print(f"Results saved to {csv_path}")
    return summary


def main() -> None:
    args = parse_args()
    evaluate_real_scale(args)


if __name__ == "__main__":
    main()
