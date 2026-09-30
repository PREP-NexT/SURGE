"""Single-run throughput per routing substep: A100 eager vs torch.compile-fused
the parallel scheme (k=3), for one or more routing-network .npz files. Reproduces the
resolution-scaling result (speedup grows with network size).

Usage:
    python examples/bench_resolution_scaling.py static/routing_static_*.npz
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter

print("torch", torch.__version__, torch.cuda.get_device_name(0))
for path in sys.argv[1:]:
    net = RoutingNetwork(path)
    r = TorchRouter(net, device="cuda", dtype=torch.float32)
    N = net.n_cells
    q = torch.zeros(N, device="cuda")
    qd = torch.full((N,), 1e-4, device="cuda")

    for _ in range(10):
        q = r.substep_parallel(q, qd, iterations=3, n_newton=4)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(60):
        q = r.substep_parallel(q, qd, iterations=3, n_newton=4)
    torch.cuda.synchronize()
    eager = (time.time() - t0) / 60 * 1e3

    step = torch.compile(lambda a, b: r.substep_parallel(a, b, iterations=3, n_newton=4))
    for _ in range(30):
        q = step(q, qd)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(60):
        q = step(q, qd)
    torch.cuda.synchronize()
    fused = (time.time() - t0) / 60 * 1e3

    print("%-28s N=%d levels=%d  eager %.3f ms  fused %.3f ms"
          % (os.path.basename(path), N, net.n_levels, eager, fused))
