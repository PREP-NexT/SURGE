"""Does the clipped release keep the storage inside [0, cap] and conserve water?

The release is clipped to what the storage bounds allow before it is applied, so
S + (I - R) dt should land inside [0, cap] on its own, with no clamp.
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork
from surge.reservoir import ResRouter

DATA = os.environ.get('SURGE_DATA', '.')     # the directory holding the network,
                                            # the reservoirs and the steady state
dev, dt = 'cuda', torch.float32
net = RoutingNetwork(DATA + '/routing_05min.npz')
cellArea = torch.as_tensor(net.cellArea, dtype=dt, device=dev)
dx = torch.as_tensor(net.chanLength, dtype=dt, device=dev)
ql = torch.nan_to_num(1.0 / 1000 / 86400 * cellArea / dx).clamp_min(0)

out = {}
for name, use_ordered in (('parallel', False), ('ordered', True)):
    r = ResRouter(net, device=dev, dtype=dt)
    z = np.load(DATA + '/res_grand_05min.npz')
    r.set_reservoirs(z['res_cell'], z['cap_m3'], z['rmean'])
    S0 = r.S.clone()
    q = torch.zeros(net.n_cells, device=dev, dtype=dt)
    below, above, vin, vout = 0, 0, 0.0, 0.0
    DAYS = 30 if not use_ordered else 3          # the ordered reference is far slower
    for d in range(DAYS):
        for _ in range(net.noRoutingSteps):
            Sbefore = r.S.clone()
            q = r.ordered_res(q, ql, n_newton=8) if use_ordered else r.parallel_res(q, ql, iterations=7, n_newton=4)
            tol = torch.clamp(1e-6 * r.cap, min=1.0)       # 1 m3, or a millionth of the capacity
            below += int((r.S < -tol).sum())
            above += int((r.S > r.cap + tol).sum())
            I = r._inflow(q, ql); R = q.index_select(-1, r.res_cell)
            vin += float((I * r.dt).sum()); vout += float((R * r.dt).sum())
    dS = float((r.S - S0).sum())
    out[name] = dict(days=DAYS, steps=DAYS * net.noRoutingSteps,
                     cells_below_zero=below, cells_above_cap=above,
                     inflow_m3=vin, release_m3=vout, storage_change_m3=dS,
                     balance_residual_rel=(vin - vout - dS) / max(vin, 1.0),
                     S_min_over_cap=float((r.S / r.cap).min()), S_max_over_cap=float((r.S / r.cap).max()),
                     worst_overshoot_m3=float(torch.clamp(r.S - r.cap, min=0).max()),
                     worst_undershoot_m3=float(torch.clamp(-r.S, min=0).max()))
    print(name, json.dumps(out[name], indent=2))
json.dump(out, open('reservoir_massbalance.json', 'w'), indent=2)
