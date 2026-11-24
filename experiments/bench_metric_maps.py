"""Per-cell KGE, mass bias, NSE and rRMSE at k = 1, 3, 7, 20, against the CWatM
serial-C kernel. Area-proxy geometry, 2010 daily runoff, one-year k=8 spin-up shared
by both sides, fp32 on the GPU. The serial reference year is the expensive part; every
metric for every k comes out of the one run.

    python experiments/bench_metric_maps.py routing_05min.npz runoff_05min.npy out.npz

The serial stage calls the CWatM kernel, which surge.serial_ref finds through
GPU_KWR_T5_SO.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork, TorchRouter
from surge.serial_ref import kinematic_serial

NPZ, RUNOFF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KS = tuple(int(x) for x in os.environ.get('METRIC_KS', '1,3,7,20').split(','))

net = RoutingNetwork(NPZ)
N, nsub = net.n_cells, net.noRoutingSteps
runoff = np.load(RUNOFF).astype(np.float64)
ndays = runoff.shape[1]
factor = net.cellArea / net.chanLength / 1000.0
print('net %d cells %d days' % (N, ndays), flush=True)

dev, dt = 'cuda', torch.float32
router = TorchRouter(net, device=dev, dtype=dt)
r_g = torch.as_tensor(runoff, device=dev, dtype=dt)
fac = torch.as_tensor(factor, device=dev, dtype=dt)


def route_gpu(iterations, q0, out=None):
    q = q0.clone()
    for d in range(ndays):
        qd = r_g[:, d] * fac
        acc = torch.zeros(N, device=dev, dtype=dt)
        for _ in range(nsub):
            q = router.substep_parallel(q, qd, iterations=iterations, n_newton=4)
            acc += q
        if out is not None:
            out[d] = (acc / nsub).cpu().numpy()
    return q


t0 = time.time()
q0g = route_gpu(8, torch.zeros(N, device=dev, dtype=dt))
q0 = q0g.cpu().numpy().astype(np.float64)
print('spin-up done (%.0fs)' % (time.time() - t0), flush=True)

print('serial-C reference year...', flush=True)
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
    if d % 60 == 0:
        print('  day %3d/%d (%.0fs)' % (d, ndays, time.time() - t0), flush=True)
print('serial-C year done (%.0fs)' % (time.time() - t0), flush=True)

ann = ref.mean(0)
big = ann > 100.0
print('cells > 100 m3/s: %d' % int(big.sum()), flush=True)
R = ref[:, big].astype(np.float64)
Rm = R.mean(0); Rs = R.std(0)

out = {}
day = np.zeros((ndays, N), np.float32)
for k in KS:
    route_gpu(k, q0g, out=day)
    G = day[:, big].astype(np.float64)
    Gm = G.mean(0); Gs = G.std(0)
    r = ((G - Gm) * (R - Rm)).mean(0) / np.maximum(Gs * Rs, 1e-12)
    bias = Gm / np.maximum(Rm, 1e-9)
    sdr = Gs / np.maximum(Rs, 1e-12)
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (sdr - 1) ** 2 + (bias - 1) ** 2)
    nse = 1.0 - ((G - R) ** 2).sum(0) / np.maximum(((R - Rm) ** 2).sum(0), 1e-12)
    rmse = np.sqrt(((G - R) ** 2).mean(0)) / np.maximum(Rm, 1e-3)
    for nm, v in (('kge', kge), ('bias', bias), ('nse', nse), ('rrmse', rmse),
                  ('r', r), ('sdr', sdr)):
        out['%s_k%d' % (nm, k)] = v.astype(np.float32)
    print('k=%-2d KGE med %.5f (p5 %.4f)  NSE med %.5f  bias med %+.5f  rRMSE med %.5f'
          % (k, np.median(kge), np.percentile(kge, 5), np.median(nse),
             np.median(bias) - 1.0, np.median(rmse)), flush=True)

d = np.load(NPZ)
rows, cols = np.where(~d['mask2d'])
RES = 360.0 / d['mask2d'].shape[1]   # the maps span 360 deg of longitude
np.savez_compressed(OUT,
                    lon=(-180.0 + (cols + 0.5) * RES)[big].astype(np.float32),
                    lat=(90.0 - (rows + 0.5) * RES)[big].astype(np.float32),
                    ann=ann[big].astype(np.float32),
                    level=net.level[big].astype(np.int32),
                    note='per-cell KGE / bias ratio / NSE / rRMSE at k=1,3,7,20 against the '
                         'CWatM serial-C kernel; area-proxy geometry; 2010 daily runoff; fp32 GPU',
                    **out)
print('wrote %s (%d cells)' % (OUT, int(big.sum())), flush=True)
