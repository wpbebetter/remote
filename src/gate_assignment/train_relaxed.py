"""Training skeleton using relaxed two-stage gate assignment."""

from __future__ import annotations

import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_default_datasets
from .net import ArrivalPredictor
from .model_relaxed import (
    solve_stage1_relaxed_torch,
    solve_stage2_relaxed_torch,
)
from .ip_layer import IPParams


def train(args: argparse.Namespace) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = build_default_datasets(
        min_flights_per_day=args.min_flights,
        max_flights_per_day=args.max_flights,
        max_instances=args.max_instances,
        seed=args.seed,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training instances available.")
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    first_sample = train_ds[0]
    input_dim = first_sample["features"].shape[1]
    model = ArrivalPredictor(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_loss_fn = nn.MSELoss()
    ip_params = IPParams(max_iter=30, eps=1e-9)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_regret = 0.0
        grad_norms: list[float] = []
        t0 = time.time()

        for batch in dataloader:
            features = batch["features"][0].to(device)
            arrival_true = batch["arrival_true"][0].to(device)
            arrival_sched = batch["arrival_sched"][0].to(device)
            idx = int(batch["instance_index"][0].item())
            oracle_cost = float(batch["oracle_cost"][0].item())
            inst = train_ds.instances[idx]

            delta = model(features)
            delta.retain_grad()
            arrival_pred = torch.clamp(arrival_sched + delta, 0.0, 24 * 60.0)

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
                change_penalty_gamma=args.gamma,
                ip_params=ip_params,
            )
            cost_matrix = torch.from_numpy(inst.taxi_cost_matrix).to(device=device, dtype=torch.double)
            taxi_cost2 = (x2 * cost_matrix).sum()
            penalty = args.gamma * torch.abs(x2 - x1).sum()
            oracle_cost_tensor = torch.tensor(oracle_cost, dtype=torch.double, device=device)
            regret_tensor = (taxi_cost2 + penalty - oracle_cost_tensor).to(dtype=torch.float32)

            if args.mode == "mse_only":
                loss = mse_loss
            elif args.mode == "regret_only":
                loss = regret_tensor
            else:
                loss = args.lambda_regret * regret_tensor + (1.0 - args.lambda_regret) * mse_loss

            optimizer.zero_grad()
            loss.backward()
            if delta.grad is not None:
                grad_norms.append(float(delta.grad.norm().item()))
            optimizer.step()

            total_loss += float(loss.item())
            total_mse += float(mse_loss.item())
            total_regret += float(regret_tensor.item())

        duration = time.time() - t0
        n = len(train_ds)
        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        print(
            f"[Epoch {epoch}/{args.epochs}] loss={total_loss/n:.4f} "
            f"mse={total_mse/n:.4f} avg_relaxed_regret={total_regret/n:.2f} "
            f"time={duration:.1f}s grad_norm(delta)={avg_grad:.3e}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.save_dir,
        f"arrival_model_{args.mode}_mf{args.min_flights}_Mf{args.max_flights}_ni{args.max_instances}.pt",
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train relaxed two-stage model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1000.0)
    parser.add_argument("--lambda_regret", type=float, default=0.5)
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["mse_only", "regret_only", "combined"],
    )
    parser.add_argument("--min_flights", type=int, default=10)
    parser.add_argument("--max_flights", type=int, default=20)
    parser.add_argument("--max_instances", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
