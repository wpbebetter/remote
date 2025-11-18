"""Run comparative experiments across different training loss configurations."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

from .evaluate_integer_two_stage import evaluate_checkpoint
from .train_relaxed import train as train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-Stage relaxed vs integer experiments")
    parser.add_argument("--modes", nargs="+", default=["mse_only", "regret_only", "combined"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--lambda_regret", type=float, default=0.5)
    parser.add_argument("--min_flights", type=int, default=10)
    parser.add_argument("--max_flights", type=int, default=20)
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--save_dir", type=str, default="runs/experiments")
    parser.add_argument("--max_stands", type=int, default=None)
    return parser.parse_args()


def train_one_mode(mode: str, args: argparse.Namespace) -> dict:
    train_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        lambda_regret=args.lambda_regret,
        mode=mode,
        min_flights=args.min_flights,
        max_flights=args.max_flights,
        max_instances=args.max_instances,
        save_dir=args.save_dir,
        seed=args.seed,
        max_stands=args.max_stands,
    )
    return train_model(train_args)


def evaluate_checkpoint_metrics(checkpoint: str, args: argparse.Namespace) -> Dict[str, float]:
    eval_args = argparse.Namespace(
        checkpoint=checkpoint,
        gamma=args.gamma,
        min_flights=args.min_flights,
        max_flights=args.max_flights,
        max_instances=args.max_instances,
        seed=args.seed,
        time_limit=args.time_limit,
        max_stands=args.max_stands,
    )
    return evaluate_checkpoint(eval_args)


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    results: List[Dict[str, float]] = []
    for mode in args.modes:
        print(f"=== Training mode: {mode} ===")
        train_result = train_one_mode(mode, args)
        ckpt_path = train_result["checkpoint"]
        print(f"Checkpoint saved to {ckpt_path}")
        metrics = evaluate_checkpoint_metrics(ckpt_path, args)
        metrics.update(
            {
                "mode": mode,
                "train_ip_calls": train_result["ip_stats"]["calls"],
                "train_ip_fallbacks": train_result["ip_stats"]["fallbacks"],
                "train_ip_warnings": train_result["ip_stats"]["warnings"],
                "train_instances": train_result["train_instances"],
            }
        )
        results.append(metrics)

    csv_path = os.path.join(args.save_dir, "experiments_summary.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "mode",
            "instances",
            "mean_int_regret",
            "median_int_regret",
            "std_int_regret",
            "mean_oracle_cost",
            "mean_stage2_cost",
            "train_ip_calls",
            "train_ip_warnings",
            "train_ip_fallbacks",
            "train_instances",
            "ip_calls",
            "ip_warnings",
            "ip_fallbacks",
            "min_flights",
            "max_flights",
            "max_instances",
            "epochs",
            "lr",
            "gamma",
            "lambda_regret",
            "seed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row.update(
                {
                    "min_flights": args.min_flights,
                    "max_flights": args.max_flights,
                    "max_instances": args.max_instances,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "lambda_regret": args.lambda_regret,
                    "seed": args.seed,
                }
            )
            writer.writerow(row)
    print(f"Wrote {len(results)} rows to {csv_path}")

    print("==== Integer Two-Stage Regret Comparison (test set) ====")
    print(f"{'mode':<12} {'mean':>10} {'median':>10} {'std':>10} {'train_fb%':>10} {'eval_fb%':>10}")
    for row in results:
        train_fb = (
            row["train_ip_fallbacks"] / row["train_ip_calls"] * 100.0
            if row["train_ip_calls"]
            else 0.0
        )
        eval_fb = (
            row["ip_fallbacks"] / row["ip_calls"] * 100.0 if row["ip_calls"] else 0.0
        )
        print(
            f"{row['mode']:<12} "
            f"{row['mean_int_regret']:>10.2f} "
            f"{row['median_int_regret']:>10.2f} "
            f"{row['std_int_regret']:>10.2f} "
            f"{train_fb:>10.1f} "
            f"{eval_fb:>10.1f}"
        )
    print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
