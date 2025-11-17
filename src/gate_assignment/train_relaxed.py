"""Training skeleton using relaxed two-stage gate assignment."""

from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import GateAssignmentDataset, build_default_dataset
from .net import ArrivalPredictor
from .model_relaxed import (
    solve_stage1_relaxed_torch,
    solve_stage2_relaxed_torch,
)
from .ip_layer import IPParams


def train(
    num_epochs: int = 5,
    lr: float = 1e-3,
    gamma: float = 1000.0,
    lambda_regret: float = 0.5,
    min_flights: int = 20,
    max_flights: int = 30,
    max_instances: int | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_default_dataset(
        min_flights_per_day=min_flights,
        max_flights_per_day=max_flights,
        precompute_oracle=True,
    )
    if max_instances is not None and len(dataset) > max_instances:
        dataset.instances = dataset.instances[:max_instances]
        if dataset.oracle_costs is not None:
            dataset.oracle_costs = dataset.oracle_costs[:max_instances]

    if len(dataset) == 0:
        raise RuntimeError("No instances available for training.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    first_sample = dataset[0]
    input_dim = first_sample["features"].shape[1]
    model = ArrivalPredictor(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    ip_params = IPParams(max_iter=30, eps=1e-9)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_regret = 0.0
        grad_norms = []
        t0 = time.time()

        for batch in dataloader:
            features = batch["features"][0].to(device)
            arrival_true = batch["arrival_true"][0].to(device)
            arrival_sched = batch["arrival_sched"][0].to(device)
            idx = int(batch["instance_index"][0].item())
            oracle_cost = float(batch["oracle_cost"][0].item())
            inst = dataset.instances[idx]

            delta = model(features)
            delta.retain_grad()
            arrival_pred = arrival_sched + delta
            arrival_pred = torch.clamp(arrival_pred, 0.0, 24 * 60.0)

            mse_loss = mse_loss_fn(arrival_pred, arrival_true)

            x1 = solve_stage1_relaxed_torch(
                inst,
                arrival_min_tensor=arrival_pred,
                ip_params=ip_params,
            )
            x2 = solve_stage2_relaxed_torch(
                inst,
                arrival_true_tensor=arrival_true,
                x1_tensor=x1,
                change_penalty_gamma=gamma,
                ip_params=ip_params,
            )
            cost_matrix = torch.from_numpy(inst.taxi_cost_matrix).to(device=device, dtype=torch.double)
            taxi_cost1 = (x1 * cost_matrix).sum()
            taxi_cost2 = (x2 * cost_matrix).sum()
            penalty = gamma * torch.abs(x2 - x1).sum()

            oracle_cost_tensor = torch.tensor(oracle_cost, dtype=torch.double, device=device)
            regret_tensor = (taxi_cost2 + penalty - oracle_cost_tensor).to(dtype=torch.float32)

            loss = lambda_regret * regret_tensor + (1.0 - lambda_regret) * mse_loss
            optimizer.zero_grad()
            loss.backward()
            if delta.grad is not None:
                grad_norms.append(float(delta.grad.norm().item()))
            optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            total_regret += float(regret_tensor.item())

        duration = time.time() - t0
        n = len(dataset)
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        print(
            f"[Epoch {epoch}/{num_epochs}] loss={total_loss/n:.4f} "
            f"mse={total_mse/n:.4f} avg_relaxed_regret={total_regret/n:.2f} "
            f"time={duration:.1f}s grad_norm(delta)={avg_grad:.3e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train relaxed two-stage model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--lambda_regret", type=float, default=0.5)
    parser.add_argument("--min_flights", type=int, default=20)
    parser.add_argument("--max_flights", type=int, default=30)
    parser.add_argument("--max_instances", type=int, default=3)
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
        max_instances=args.max_instances,
    )


if __name__ == "__main__":
    main()
