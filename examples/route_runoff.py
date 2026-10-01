"""Route a daily runoff series to discharge on a routing network, the basic end-use
of the solver. Saves annual-mean discharge per cell.

    python examples/route_runoff.py routing_05min.npz runoff_2010.npy out.npy [k]

k is the number of iterations each time step takes. A network needs as many as its
Courant number asks for, which the paper measures as two at 30 arc-minutes, three at
15, seven at 5, eleven at 3 and thirty at 1 arc-minute. Passing too few leaves the
discharge short of the serial solution, so the default here is the thirty that covers
every network of the paper, and a coarse grid can take far fewer.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter

NPZ, RUNOFF = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else "annual_mean_discharge.npy"
K = int(sys.argv[4]) if len(sys.argv) > 4 else 30

net = RoutingNetwork(NPZ)
router = TorchRouter(net, device="cuda", dtype=torch.float32)
N, nsub = net.n_cells, net.noRoutingSteps
runoff = torch.as_tensor(np.load(RUNOFF), device="cuda", dtype=torch.float32)  # (N, ndays)
ndays = runoff.shape[1]
factor = (torch.as_tensor(net.cellArea, device="cuda", dtype=torch.float32)
          / torch.as_tensor(net.chanLength, device="cuda", dtype=torch.float32) / 1000.0)
print("net %d cells %d levels, routing %d days" % (N, net.n_levels, ndays))


def route(q, record=False):
    acc = torch.zeros(N, device="cuda") if record else None
    for d in range(ndays):
        qd = runoff[:, d] * factor
        for _ in range(nsub):
            q = router.substep_parallel(q, qd, iterations=K, n_newton=4)
            if record:
                acc += q
    return q, (acc / (ndays * nsub) if record else None)


t0 = time.time()
q, _ = route(torch.zeros(N, device="cuda"))          # spin-up
_, ann = route(q, record=True)                        # recording pass
torch.cuda.synchronize()
np.save(OUT, ann.cpu().numpy())
print("routed %d model-days in %.1fs -> %s" % (2 * ndays, time.time() - t0, OUT))
