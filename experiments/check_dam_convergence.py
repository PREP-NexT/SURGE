"""Does the fixed point converge as fast with reservoirs as without, in a single time step?

A supplementary check of one flood step, which corroborates the year-long result of
Figure 9b: from a common state, one step of SURGE at k iterations against the
level-by-level solution of the same system, with and without the GRanD dams, on the
5 arc-minute network.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter
from surge.reservoir import ResRouter

DATA = os.environ.get('SURGE_DATA', '.')     # the directory holding the network,
                                            # the reservoirs and the steady state
net = RoutingNetwork(DATA + '/routing_05min.npz')
dev, dt = 'cuda', torch.float32
cellArea = torch.as_tensor(net.cellArea, dtype=dt, device=dev)
dx = torch.as_tensor(net.chanLength, dtype=dt, device=dev)
q_side = torch.nan_to_num(1.0 / 1000 / 86400 * cellArea / dx).clamp_min(0)

for dams in (False, True):
    if dams:
        r = ResRouter(net, device=dev, dtype=dt)
        z = np.load(DATA + '/res_grand_05min.npz')
        r.set_reservoirs(z['res_cell'], z['cap_m3'], z['rmean'])
        jac = lambda a, b, k: r.parallel_res(a, b, iterations=k, n_newton=6)
        wave = lambda a, b: r.ordered_res(a, b, n_newton=8)
        is_res = r.is_res
    else:
        r = TorchRouter(net, device=dev, dtype=dt)
        jac = lambda a, b, k: r.substep_parallel(a, b, iterations=k, n_newton=6)
        wave = lambda a, b: r.substep_ordered(a, b, n_newton=8)
        is_res = torch.zeros(net.n_cells, dtype=torch.bool, device=dev)
    q = torch.zeros(net.n_cells, device=dev, dtype=dt)
    for _ in range(net.noRoutingSteps * 60):          # two months to a steady state
        q = jac(q, q_side, 8)
    if dams:
        S_keep = r.S.clone()
    q_flood = q_side * 5.0                            # the flood step both sides take
    if dams:
        r.S = S_keep.clone()
    qref = wave(q, q_flood)
    print('\n%s reservoirs' % ('WITH' if dams else 'NO'))
    for k in (1, 2, 3, 5, 8, 12):
        if dams:
            r.S = S_keep.clone()
        qk = jac(q, q_flood, k)
        rel = ((qk - qref).abs() / qref.clamp_min(1e-3)).detach()
        line = '  k=%-2d median %.2e  p99 %.2e' % (k, rel.median(), rel.quantile(0.99))
        if dams:
            line += '  | reservoir cells median %.2e' % rel[is_res].median()
        print(line, flush=True)
