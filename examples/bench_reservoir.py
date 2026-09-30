"""Three schemes WITH real GRanD reservoirs (paper Section 3.6, Fig. 7): per-substep
cost of ordered_res vs parallel_res (eager & fused), and whether reservoirs change
the parallel-vs-exact agreement. Needs a res_*.npz from map_grand_dams.py.

    python examples/bench_reservoir.py net.npz res.npz
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork
from surge.reservoir import ResRouter
print('torch', torch.__version__, torch.cuda.get_device_name(0))

def timeit(fn, warm, reps):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps * 1e3

net = RoutingNetwork(sys.argv[1]); z = np.load(sys.argv[2])
r = ResRouter(net, device='cuda', dtype=torch.float32)
r.set_reservoirs(z['res_cell'], z['cap_m3'], z['rmean'])
N, L = net.n_cells, net.n_levels
area = torch.as_tensor(net.cellArea, dtype=torch.float32, device='cuda')
q = (0.02 * area / 86400.).float(); qref = torch.zeros(N, device='cuda')
reps = 60 if L < 300 else 30 if L < 1000 else 8 if L < 2000 else 3
r.reset_storage(); wf = timeit(lambda: r.ordered_res(qref, q, n_newton=4), max(2, reps // 5), reps)
r.reset_storage(); je = timeit(lambda: r.parallel_res(qref, q, iterations=3, n_newton=4), 10, 60)
print('N=%d L=%d nres=%d  ordered_res %.3f ms  parallel_res %.3f ms  (gap %.0fx)'
      % (N, L, len(z['res_cell']), wf, je, wf / je))

# accuracy: parallel_res(k) vs exact ordered_res at identical state.
# NOTE: use TIME-VARYING forcing -- constant forcing settles to a steady state
# where every scheme returns the same field and the error is trivially zero.
big = torch.as_tensor(net.channelAlpha, dtype=torch.float32, device='cuda') > np.percentile(net.channelAlpha, 90)
def relerr(A, B):
    e = ((A - B).abs() / B.abs().clamp_min(1e-6))[big]; e = e[torch.isfinite(e)]; return float(e.median())
def sf(t):
    return (0.02 * (1.0 + 0.5 * np.sin(t / 15.0)) * area / 86400.).float()
r.reset_storage(); qref = torch.zeros(N, device='cuda'); acc = {1: [], 2: [], 3: [], 5: []}
for t in range(120):
    qt = sf(t)
    S0 = r.S.clone(); W = r.ordered_res(qref, qt, n_newton=4); Sa = r.S.clone()
    for k in acc:
        r.S = S0.clone(); J = r.parallel_res(qref, qt, iterations=k, n_newton=4); acc[k].append(relerr(J, W))
    r.S = Sa; qref = W
print('parallel_res vs exact ordered_res (big-river median rel err):',
      {k: round(float(np.mean(v[-20:])), 4) for k, v in acc.items()})
