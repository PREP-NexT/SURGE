"""Time the barrier-free Jacobi solver on the CPU (the paper's CPU-parallel).

The same fixed-point solver used on the GPU is run multi-threaded through
torch-CPU, giving a same-algorithm CPU baseline for the GPU speedup. The paper's
"CPU-parallel" column is the multi-thread run, and its "two to seven times"
figure is CPU-serial (the 1-core C kernel) divided by this per-substep time.

Usage:
    python examples/bench_cpu_parallel.py routing_static.npz [n_threads]

Reports per-substep wall-clock (k=3, 4 Newton iterations) at 1 thread and at
n_threads (default 24), after a short spin-up, timed with the same
warm-up-then-average protocol as the GPU benchmarks.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter

NPZ = sys.argv[1]
NTHREADS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
net = RoutingNetwork(NPZ)
N, nsub = net.n_cells, net.noRoutingSteps
area = np.asarray(net.cellArea, np.float64)
dx = np.asarray(net.chanLength, np.float64)
q_lat = np.nan_to_num(1.0e-3 * area / dx / 1000.0)          # a nonzero lateral inflow so Newton does real work


def bench(nthreads, spin=40, warm=3, reps=20):
    torch.set_num_threads(nthreads)
    r = TorchRouter(net, device="cpu", dtype=torch.float32)
    q = torch.zeros(N, dtype=torch.float32)
    ql = torch.as_tensor(q_lat, dtype=torch.float32)
    for _ in range(spin):                                    # reach a flowing steady state
        q = r.substep_jacobi(q, ql, sweeps=3, n_newton=4)
    for _ in range(warm):
        r.substep_jacobi(q, ql, sweeps=3, n_newton=4)
    t0 = time.time()
    for _ in range(reps):
        q = r.substep_jacobi(q, ql, sweeps=3, n_newton=4)
    return (time.time() - t0) / reps * 1e3                    # ms per substep


print("net %d cells  %d levels  (torch %s, %d cores available)"
      % (N, net.n_levels, torch.__version__, os.cpu_count()))
t1 = bench(1)
tn = bench(NTHREADS)
print("   1 thread : %.3f ms/substep" % t1)
print("  %2d threads: %.3f ms/substep  (%.1fx over 1 thread)" % (NTHREADS, tn, t1 / tn))
