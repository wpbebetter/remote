"""Training skeleton using relaxed two-stage gate assignment."""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import GateAssignmentDataset, build_default_dataset
from .net import ArrivalPredictor
from .model_relaxed import solve_stage1_relaxed_ip, solve_stage2_relaxed_ip


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def compute_two_stage_relaxed(
    inst,
    arrival_pred: np.ndarray,
    oracle_cost: float,
    gamma: float = 1000.0,
) -> Tuple[float, float, float]:
    x1_rel = solve_stage1_relaxed_ip(inst, arrival_min=arrival_pred)
    cost1 = float((x1_rel * inst.taxi_cost_matrix).sum())
    x2_rel = solve_stage2_relaxed_ip(
        inst,
        arrival_true_min=inst.arrival_true_min,
        x1_reference=x1_rel,
        change_penalty_gamma=gamma,
    )
    taxi_cost2 = float((x2_rel * inst.taxi_cost_matrix).sum())
    penalty = float(np.abs(x2_rel - x1_rel).sum() * gamma)
    cost2 = taxi_cost2 + penalty
    regret = cost2 - oracle_cost
    return regret, cost1, cost2


def train(
    num_epochs: int = 5,
    lr: float = 1e-3,
    gamma: float = 1000.0,
    lambda_regret: float = 0.5,
    min_flights: int = 20,
    max_flights: int = 30,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_default_dataset(
        min_flights_per_day=min_flights,
        max_flights_per_day=max_flights,
        precompute_oracle=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("No instances available for training.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    first_sample = dataset[0]
    input_dim = first_sample["features"].shape[1]
    model = ArrivalPredictor(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_regret = 0.0
        t0 = time.time()

        for batch in dataloader:
            features = batch["features"][0].to(device)
            arrival_true = batch["arrival_true"][0].to(device)
            arrival_sched = batch["arrival_sched"][0].to(device)
            idx = int(batch["instance_index"][0].item())
            oracle_cost = float(batch["oracle_cost"][0].item())

            delta = model(features)
            arrival_pred = arrival_sched + delta
            arrival_pred = torch.clamp(arrival_pred, 0.0, 24 * 60.0)

            mse_loss = mse_loss_fn(arrival_pred, arrival_true)

            arrival_pred_np = tensor_to_np(arrival_pred)
            inst = dataset.instances[idx]
            regret_val, _, _ = compute_two_stage_relaxed(
                inst,
                arrival_pred=arrival_pred_np,
                oracle_cost=oracle_cost,
                gamma=gamma,
            )
            regret_tensor = torch.tensor(regret_val, dtype=torch.float32, device=device)

            loss = lambda_regret * regret_tensor + (1.0 - lambda_regret) * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            total_regret += float(regret_val)

        duration = time.time() - t0
        n = len(dataset)
        print(
            f"[Epoch {epoch}/{num_epochs}] loss={total_loss/n:.4f} "
            f"mse={total_mse/n:.4f} avg_relaxed_regret={total_regret/n:.2f} "
            f"time={duration:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train relaxed two-stage model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--lambda_regret", type=float, default=0.5)
    parser.add_argument("--min_flights", type=int, default=20)
    parser.add_argument("--max_flights", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        num_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        lambda_regret=args.lambda_regret,
        min_flights=args.min_flights,
        max_flights=args.max_flights,
    )


if __name__ == "__main__":
    main()
