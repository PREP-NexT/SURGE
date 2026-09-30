"""Per-substep wall-clock of the three schemes (paper Section 3.5, Fig. 6):
the ordered scheme (O(L) level barriers) vs the parallel scheme k=3 (eager & fused), fp32.
Same protocol as bench_resolution_scaling.py.

    python examples/bench_ordered.py static/routing_static_*.npz
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter
print('torch', torch.__version__, torch.cuda.get_device_name(0))

def timeit(fn, warm, reps):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps): fn()
    torch.cuda.synchronize(); return (time.time() - t0) / reps * 1e3

for p in sys.argv[1:]:
    net = RoutingNetwork(p); r = TorchRouter(net, device='cuda', dtype=torch.float32)
    N, L = net.n_cells, net.n_levels
    q = torch.zeros(N, device='cuda'); qd = torch.full((N,), 1e-4, device='cuda')
    reps = 60 if L < 300 else 30 if L < 1000 else 8 if L < 2000 else 3   # ordered cost ~ O(L)
    wf = timeit(lambda: r.substep_ordered(q, qd, n_newton=4), max(2, reps // 5), reps)
    je = timeit(lambda: r.substep_parallel(q, qd, iterations=3, n_newton=4), 10, 60)
    try:                                    # torch.compile needs Python <= 3.11
        step = torch.compile(lambda a, b: r.substep_parallel(a, b, iterations=3, n_newton=4))
        jf = timeit(lambda: step(q, qd), 20, 60)
    except Exception as e:
        jf = float('nan'); print('  (fused skipped:', e, ')')
    print('%-26s N=%d L=%d  ordered %.3f ms  parallel_eager %.3f ms  parallel_fused %.3f ms'
          % (os.path.basename(p), N, L, wf, je, jf))
