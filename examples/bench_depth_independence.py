"""Depth-independence: on each network, compare Jacobi k=1..8 against the exact
wavefront on the SAME network, on a flood transient, at three runoff levels that
bracket the operational Courant regime. If the k needed for a given accuracy is
~constant while the level count varies, depth-independence holds.

Usage:
    python examples/bench_depth_independence.py static/routing_static_*.npz
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter

DEV, DT = "cuda", torch.float32
SPIN_DAYS = 120
print("torch", torch.__version__, torch.cuda.get_device_name(0))

for npz in sys.argv[1:]:
    net = RoutingNetwork(npz)
    router = TorchRouter(net, device=DEV, dtype=DT)
    N, nsub = net.n_cells, net.noRoutingSteps
    print("\n=== %s : %d cells  %d levels ===" % (os.path.basename(npz), N, net.n_levels))
    cellArea = torch.as_tensor(net.cellArea, dtype=DT, device=DEV)
    dx = torch.as_tensor(net.chanLength, dtype=DT, device=DEV)
    for rmm in (0.2, 1.0, 5.0):
        q_side = torch.nan_to_num(rmm / 1000 / 86400 * cellArea / dx).clamp_min(0)
        q = torch.zeros(N, dtype=DT, device=DEV)
        for _ in range(nsub * SPIN_DAYS):
            q = router.substep_jacobi(q, q_side, sweeps=3, n_newton=4)
        torch.cuda.synchronize()
        Q = q.clamp_min(1e-6)
        c = 1.0 / (router.alpha * router.beta * Q ** (router.beta - 1))
        Cr = (c * router.dt / dx).cpu().numpy()
        q_flood = q_side * 5.0  # step the inflow up: a transient extra sweeps must carry
        qref = router.substep_wavefront(q, q_flood, n_newton=8)
        print("  runoff %.1f mm/d: Cr med/p90/p99 %.2f/%.2f/%.2f"
              % (rmm, np.median(Cr), np.percentile(Cr, 90), np.percentile(Cr, 99)))
        for k in (1, 2, 3, 5, 8):
            qk = router.substep_jacobi(q, q_flood, sweeps=k, n_newton=6)
            rel = ((qk - qref).abs() / qref.clamp_min(1e-3)).detach().cpu().numpy()
            print("     k=%d  median %.2e  p99 %.2e" % (k, np.median(rel), np.percentile(rel, 99)))
