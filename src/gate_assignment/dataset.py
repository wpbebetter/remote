"""PyTorch dataset wrapper for gate assignment instances."""

from __future__ import annotations

from typing import List, Sequence, Tuple

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


def build_dataset_splits(
    min_flights_per_day: int = 20,
    max_flights_per_day: int = 40,
    max_instances: int = 30,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
    max_stands: int | None = None,
) -> Tuple[
    List[GateAssignmentInstance],
    List[GateAssignmentInstance],
    List[GateAssignmentInstance],
]:
    """Construct GateAssignmentInstance list and split into train/val/test."""

    instances = build_daily_instances(min_flights_per_day=min_flights_per_day)
    filtered = [inst for inst in instances if len(inst.flight_ids) <= max_flights_per_day]
    if max_stands is not None:
        filtered = [_trim_instance_stands(inst, max_stands) for inst in filtered]
    if max_instances is not None and len(filtered) > max_instances:
        filtered = filtered[:max_instances]

    n = len(filtered)
    if n == 0:
        return [], [], []

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = max(1, int(n * train_ratio)) if n >= 1 else 0
    n_val = max(0, int(n * val_ratio))
    if n_train + n_val >= n and n > 1:
        n_val = max(0, n - n_train - 1)
    n_train = min(n_train, max(1, n - 1)) if n > 1 else n_train
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 1:
        if n_val > 0:
            n_val -= 1
            n_test = 1
        elif n_train > 1:
            n_train -= 1
            n_test = 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_list = [filtered[i] for i in train_idx]
    val_list = [filtered[i] for i in val_idx]
    test_list = [filtered[i] for i in test_idx]
    return train_list, val_list, test_list


def make_dataset_from_instances(
    instances: List[GateAssignmentInstance],
    precompute_oracle: bool = True,
) -> GateAssignmentDataset:
    return GateAssignmentDataset(instances, precompute_oracle=precompute_oracle)


def build_default_datasets(
    min_flights_per_day: int = 20,
    max_flights_per_day: int = 40,
    max_instances: int = 30,
    seed: int | None = 0,
    max_stands: int | None = None,
) -> Tuple[GateAssignmentDataset, GateAssignmentDataset, GateAssignmentDataset]:
    train_list, val_list, test_list = build_dataset_splits(
        min_flights_per_day=min_flights_per_day,
        max_flights_per_day=max_flights_per_day,
        max_instances=max_instances,
        seed=0 if seed is None else seed,
        max_stands=max_stands,
    )
    train_ds = GateAssignmentDataset(train_list, precompute_oracle=True)
    val_ds = GateAssignmentDataset(val_list, precompute_oracle=True)
    test_ds = GateAssignmentDataset(test_list, precompute_oracle=True)
    return train_ds, val_ds, test_ds


__all__ = [
    "GateAssignmentDataset",
    "build_dataset_splits",
    "build_default_datasets",
    "make_dataset_from_instances",
    "_trim_instance_stands",
]


def _trim_instance_stands(inst: GateAssignmentInstance, max_stands: int) -> GateAssignmentInstance:
    if max_stands is None or len(inst.stand_ids) <= max_stands:
        return inst
    idx = np.arange(max_stands)
    stands_subset = inst.stands.iloc[idx].reset_index(drop=True)
    compat_trim = inst.compat_matrix[:, :max_stands].copy()
    row_zero = compat_trim.sum(axis=1) == 0
    if row_zero.any():
        compat_trim[row_zero, :] = 1

    return GateAssignmentInstance(
        date=inst.date,
        flights=inst.flights,
        stands=stands_subset,
        taxi_distances=inst.taxi_distances,
        flight_ids=inst.flight_ids,
        stand_ids=inst.stand_ids[:max_stands],
        arrival_true_min=inst.arrival_true_min,
        runway_codes=inst.runway_codes,
        taxi_cost_matrix=inst.taxi_cost_matrix[:, :max_stands],
        compat_matrix=compat_trim,
        arrival_sched_min=inst.arrival_sched_min,
        features=inst.features,
    )
