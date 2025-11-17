"""PyTorch dataset wrapper for gate assignment instances."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import GateAssignmentInstance, build_daily_instances
from .model_relaxed import solve_stage1_relaxed_ip


class GateAssignmentDataset(Dataset):
    """Treat each day (all flights) as one training sample."""

    def __init__(
        self,
        instances: Sequence[GateAssignmentInstance],
        precompute_oracle: bool = True,
    ) -> None:
        self.instances = list(instances)
        self.precompute_oracle = precompute_oracle

        self.oracle_costs: np.ndarray | None = None
        if precompute_oracle and self.instances:
            costs: List[float] = []
            for inst in self.instances:
                x_star = solve_stage1_relaxed_ip(inst, arrival_min=inst.arrival_true_min)
                cost_star = float((x_star * inst.taxi_cost_matrix).sum())
                costs.append(cost_star)
            self.oracle_costs = np.asarray(costs, dtype=float)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        inst = self.instances[idx]
        features = torch.from_numpy(inst.features.astype(np.float32))
        arrival_true = torch.from_numpy(inst.arrival_true_min.astype(np.float32))
        arrival_sched = torch.from_numpy(inst.arrival_sched_min.astype(np.float32))
        oracle_cost = float(self.oracle_costs[idx]) if self.oracle_costs is not None else 0.0
        return {
            "features": features,
            "arrival_true": arrival_true,
            "arrival_sched": arrival_sched,
            "instance_index": torch.tensor(idx, dtype=torch.long),
            "oracle_cost": torch.tensor(oracle_cost, dtype=torch.float32),
        }


def build_default_dataset(
    min_flights_per_day: int = 20,
    max_flights_per_day: int = 40,
    precompute_oracle: bool = True,
) -> GateAssignmentDataset:
    instances = build_daily_instances(min_flights_per_day=min_flights_per_day)
    filtered = [inst for inst in instances if len(inst.flight_ids) <= max_flights_per_day]
    return GateAssignmentDataset(filtered, precompute_oracle=precompute_oracle)


__all__ = [
    "GateAssignmentDataset",
    "build_default_dataset",
]
