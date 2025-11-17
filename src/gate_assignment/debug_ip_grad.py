"""Finite-difference check for GateIPFunction gradients."""

from __future__ import annotations

import torch

from .ip_layer import IPParams, gate_ip_solve


def compute_loss(h_vec: torch.Tensor) -> torch.Tensor:
    c = torch.tensor([1.0, 2.5], dtype=torch.double)
    A_eq = torch.tensor([[1.0, 1.0]], dtype=torch.double)
    b_eq = torch.tensor([1.0], dtype=torch.double)
    G = torch.eye(2, dtype=torch.double)
    lb = torch.zeros(2, dtype=torch.double)
    ub = torch.ones(2, dtype=torch.double) * 5.0
    x = gate_ip_solve(c, A_eq, b_eq, G, h_vec, (lb, ub), IPParams(max_iter=20))
    return (x ** 2).sum()


def finite_difference(h_init: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    grads = []
    for i in range(h_init.numel()):
        h_plus = h_init.clone()
        h_minus = h_init.clone()
        h_plus[i] += eps
        h_minus[i] -= eps
        loss_plus = compute_loss(h_plus).item()
        loss_minus = compute_loss(h_minus).item()
        grads.append((loss_plus - loss_minus) / (2 * eps))
    return torch.tensor(grads, dtype=torch.double)


def main() -> None:
    h = torch.tensor([0.8, 0.6], dtype=torch.double, requires_grad=True)
    loss = compute_loss(h)
    loss.backward()
    print("Autograd grad_h:", h.grad)
    fd = finite_difference(h.detach())
    print("Finite diff grad_h:", fd)
    print("Difference:", (h.grad - fd).abs())


if __name__ == "__main__":
    main()
