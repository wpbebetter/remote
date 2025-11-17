"""Debug script for GateIPFunction forward/backward signatures."""

from __future__ import annotations

import torch

from .ip_layer import gate_ip_solve, solve_lp_highs_debug, IPParams


def main() -> None:
    # min x1 + 2x2 subject to x1 + x2 = 1, x >=0
    c = torch.tensor([1.0, 2.0], dtype=torch.double, requires_grad=True)
    A_eq = torch.tensor([[1.0, 1.0]], dtype=torch.double)
    b_eq = torch.tensor([1.0], dtype=torch.double)
    G = torch.empty((0, 2), dtype=torch.double)
    h = torch.empty((0,), dtype=torch.double)
    lb = torch.zeros(2, dtype=torch.double)
    ub = torch.ones(2, dtype=torch.double) * 10.0

    x = gate_ip_solve(c, A_eq, b_eq, G, h, (lb, ub), IPParams())
    print("gate_ip_solve x:", x)

    # Backward should work even though grad is zero placeholder
    loss = (x ** 2).sum()
    loss.backward()
    print("grad wrt c (may be zero)", c.grad)

    x_highs = solve_lp_highs_debug(c.detach(), A_eq, b_eq, G, h, lb, ub)
    print("HiGHS x:", x_highs)


if __name__ == "__main__":
    main()
