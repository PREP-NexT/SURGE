"""Bit-level accuracy oracle: compare the GPU parallel scheme (k=1..8) and the exact
ordered scheme against CWatM's serial C kernel, on the same network geometry, driven by
a real daily-runoff series. Isolates the numerical scheme (same discretisation on
both sides), so it measures exactly what the truncation at k iterations costs.

Requires the CWatM serial kernel: set GPU_KWR_T5_SO (see surge/serial_ref.py).
Without it, the GPU ordered scheme is itself a machine-precision reference.

Usage:
    export GPU_KWR_T5_SO=/path/to/t5_linux.so
    python experiments/bench_accuracy_vs_serial.py routing_static_global.npz runoff_2010.npy
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter
from surge.serial_ref import kinematic_serial

NPZ, RUNOFF = sys.argv[1], sys.argv[2]
net = RoutingNetwork(NPZ)
N, nsub = net.n_cells, net.noRoutingSteps
runoff = np.load(RUNOFF).astype(np.float64)          # (N, ndays), mm/day
ndays = runoff.shape[1]
factor = net.cellArea / net.chanLength / 1000.0
print("net %d cells %d levels  %d days" % (N, net.n_levels, ndays))

router = TorchRouter(net, device="cuda", dtype=torch.float64)
r_g = torch.as_tensor(runoff, device="cuda", dtype=torch.float64)
fac_g = torch.as_tensor(factor, device="cuda", dtype=torch.float64)


def route_gpu(iterations, q0):
    q = q0.clone()
    daily = np.zeros((ndays, N), np.float32)
    for d in range(ndays):
        qd = r_g[:, d] * fac_g
        acc = torch.zeros(N, device="cuda", dtype=torch.float64)
        for _ in range(nsub):
            q = router.substep_parallel(q, qd, iterations=iterations, n_newton=4)
            acc += q
        daily[d] = (acc / nsub).cpu().numpy()
    return q, daily


# common spun-up start state (near-exact k=8), shared by serial-C and the tests
q0g, _ = route_gpu(8, torch.zeros(N, device="cuda", dtype=torch.float64))
q0 = q0g.cpu().numpy()

print("serial-C reference year (%d substeps)..." % (ndays * nsub))
t0 = time.time()
q = q0.copy()
ref = np.zeros((ndays, N), np.float32)
for d in range(ndays):
    sf = runoff[:, d] * factor
    acc = np.zeros(N)
    for _ in range(nsub):
        q = kinematic_serial(q, sf, net)
        acc += q
    ref[d] = (acc / nsub).astype(np.float32)
print("  serial-C year done (%.0fs)" % (time.time() - t0))

ann = ref.mean(0)
big = ann > 100.0
ref_pk, ref_tp = ref.max(0), ref.argmax(0)
print("big-river cells (>100 m3/s): %d" % int(big.sum()))
for k in (1, 2, 3, 5, 8):
    _, day = route_gpu(k, q0g)
    err = np.sqrt(((day - ref) ** 2).mean(0)) / np.maximum(ann, 1e-3)
    amp = np.abs(day.max(0) - ref_pk) / np.maximum(ref_pk, 1e-3)
    tim = np.abs(day.argmax(0) - ref_tp).astype(float)
    num = ((day[:, big] - ref[:, big]) ** 2).sum(0)
    den = ((ref[:, big] - ref[:, big].mean(0)) ** 2).sum(0)
    nse = 1.0 - num / np.maximum(den, 1e-9)
    print("k=%d  RMSE med/p99 %.2e/%.2e  NSE %.5f  peak-amp med %.2e  peak-timing med %.0f d"
          % (k, np.median(err[big]), np.percentile(err[big], 99),
             np.median(nse), np.median(amp[big]), np.median(tim[big])))
