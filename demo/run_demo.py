"""A thirty-day run on the global 30 arc-minute river network, small enough to finish
on a laptop CPU and large enough to show what SURGE does.

    python demo/run_demo.py

It routes the runoff beside it with the parallel scheme at the two iterations this
network needs, and with the ordered scheme, which returns the serial solution. The two
are then compared, which is the comparison the paper makes at every resolution.

Expected output is printed at the end of demo/README.md.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from surge import RoutingNetwork, TorchRouter

NET = os.path.join(HERE, '..', 'paper', 'figure_data', 'routing_static_30min.npz')
RUNOFF = os.path.join(HERE, 'runoff_30day.npy')
K = 2                     # what the 30 arc-minute network needs (Table 6 of the paper)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
net = RoutingNetwork(NET)
runoff = np.load(RUNOFF)                       # (cells, days), kg m-2 s-1
ndays = runoff.shape[1]
print('network %s cells, %d levels, %d substeps a day, on the %s'
      % (format(net.n_cells, ','), net.n_levels, net.noRoutingSteps, dev.upper()))

router = TorchRouter(net, device=dev, dtype=torch.float64)
factor = torch.as_tensor(net.cellArea / net.chanLength / 1000.0, device=dev, dtype=torch.float64)
runoff_t = torch.as_tensor(runoff, device=dev, dtype=torch.float64)


def route(step, label):
    """One run of the whole series, returning the daily mean discharge of every cell."""
    q = torch.zeros(net.n_cells, device=dev, dtype=torch.float64)
    daily = np.empty((ndays, net.n_cells), np.float64)
    t0 = time.time()
    for day in range(ndays):
        qd = runoff_t[:, day] * factor
        acc = torch.zeros_like(q)
        for _ in range(net.noRoutingSteps):
            q = step(q, qd)
            acc += q
        daily[day] = (acc / net.noRoutingSteps).cpu().numpy()
    print('  %-28s %5.1f s' % (label, time.time() - t0))
    return daily


print('routing %d days twice' % ndays)
par = route(lambda q, qd: router.substep_parallel(q, qd, iterations=K, n_newton=4),
            'parallel scheme, k = %d' % K)
ordered = route(lambda q, qd: router.substep_ordered(q, qd, n_newton=8),
                'ordered scheme (the reference)')

# The paper evaluates on cells above 100 m3/s, but thirty days from an empty network is
# not long enough to fill a large river, so this demo takes the rivers it does fill.
ann = ordered.mean(0)
big = ann > 10.0
rel = np.abs(par[:, big] - ordered[:, big]) / np.maximum(ordered[:, big], 1e-3)
print('\nevaluated on %s cells above 10 m3/s' % format(int(big.sum()), ','))
print('  median relative difference  %.2e' % np.median(rel))
print('  99th percentile             %.2e' % np.percentile(rel, 99))
print('  largest discharge           %.0f m3/s (parallel)  %.0f m3/s (ordered)'
      % (par.max(), ordered.max()))
out = os.path.join(HERE, 'discharge_30day.npy')
np.save(out, par.astype(np.float32))
print('\nwrote %s  %s' % (out, (ndays, net.n_cells)))
