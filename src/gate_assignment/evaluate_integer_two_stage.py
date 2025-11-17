"""Evaluate trained arrival models using integer Two-Stage MILP."""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch

from .dataset import build_dataset_splits, make_dataset_from_instances
from .model_mip import solve_single_stage, solve_stage1, solve_stage2
from .net import ArrivalPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models via integer Two-Stage MILP")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--min_flights", type=int, default=10)
    parser.add_argument("--max_flights", type=int, default=20)
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=float, default=60.0)
    return parser.parse_args()


def build_test_dataset(args: argparse.Namespace):
    _, _, test_list = build_dataset_splits(
        min_flights_per_day=args.min_flights,
        max_flights_per_day=args.max_flights,
        max_instances=args.max_instances,
        seed=args.seed,
    )
    test_ds = make_dataset_from_instances(test_list, precompute_oracle=False)
    return test_ds


def load_model(checkpoint: str, input_dim: int, device: torch.device) -> ArrivalPredictor:
    ckpt = torch.load(checkpoint, map_location=device)
    model = ArrivalPredictor(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = build_test_dataset(args)
    if len(test_ds) == 0:
        raise RuntimeError("No test instances available for evaluation.")

    sample = test_ds[0]
    input_dim = sample["features"].shape[1]
    model = load_model(args.checkpoint, input_dim, device)

    regrets: List[float] = []
    stage2_costs: List[float] = []
    oracle_costs: List[float] = []

    for idx, inst in enumerate(test_ds.instances):
        batch = test_ds[idx]
        features = batch["features"].to(device)
        arrival_sched = batch["arrival_sched"].to(device)

        with torch.no_grad():
            delta = model(features)
            arrival_pred = torch.clamp(arrival_sched + delta, 0.0, 24 * 60.0)
            arrival_pred_np = arrival_pred.cpu().numpy()

        _, obj_star = solve_single_stage(
            inst,
            time_limit=args.time_limit,
        )
        stage1_sol, _ = solve_stage1(
            inst,
            arrival_pred_min=arrival_pred_np,
            time_limit=args.time_limit,
        )
        _, obj_stage2 = solve_stage2(
            inst,
            arrival_true_min=inst.arrival_true_min,
            x1_solution=stage1_sol,
            change_penalty_gamma=args.gamma,
            time_limit=args.time_limit,
        )
        regret = obj_stage2 - obj_star
        regrets.append(regret)
        stage2_costs.append(obj_stage2)
        oracle_costs.append(obj_star)

        print(
            f"[Test {idx}] flights={len(inst.flight_ids)} oracle={obj_star:.2f} "
            f"stage2={obj_stage2:.2f} regret={regret:.2f}"
        )

    regrets_np = np.array(regrets)
    metrics = {
        "instances": len(test_ds),
        "mean_oracle_cost": float(np.mean(oracle_costs)),
        "mean_stage2_cost": float(np.mean(stage2_costs)),
        "mean_int_regret": float(np.mean(regrets_np)),
        "median_int_regret": float(np.median(regrets_np)),
        "std_int_regret": float(np.std(regrets_np)),
    }
    print("==== Integer evaluation summary ====")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    return metrics


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
