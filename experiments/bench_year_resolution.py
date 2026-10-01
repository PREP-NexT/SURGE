"""One year of 2010 routing at one resolution: the CWatM serial kernel as the reference
and SURGE at each iteration count asked for, compared cell by cell, so the match can be
compared across the five global river networks under the same runoff.

Every simulated field is kept: the daily mean of every cell for the spin-up year, the
serial year and each SURGE year, as (365, cells) float32 arrays. They are what the
metrics stage reads, and keeping them is why a later question, such as another
iteration count or another evaluation threshold, costs a metrics run and not a model
run.

Stages, in this order (serial and gpu can run at the same time):
  spin     GPU: one year at k = 8 from an empty network -> q0.npy, the start both sides
           share, and the daily means of the spin-up year
  serial   CPU: the serial kernel for one year from q0 -> daily means of every cell;
           checkpointed every CKPT_DAYS days, so a stopped run resumes
  gpu      GPU: SURGE at each k for one year from q0 -> daily means of every cell
  ordered  GPU: the level-by-level solution, the reference where the serial kernel
           cannot run, as with the reservoirs
  metrics  KGE, NSE, rRMSE and bias over C = {serial annual mean > C_MIN}, with the
           topological level, upstream river length [km] and Courant number of each cell;
           daily series kept for the cells above BIG_MIN

The iteration counts come from the third argument or from YEAR_KS, and the paper swept
them densely enough to read the crossing off each curve: 1 to 10 and 20 on the 30, 15
and 5 arc-minute networks, out to 14, 16, 18, 20 at 3 arc-minutes, and out to 32 at
1 arc-minute, where the network needs thirty.

Runoff: CWatM qtot of 2010 at 0.5 deg, read at the 0.5-deg cell holding each routing
cell, as examples/regrid_runoff.py does. The channel geometry is the one the
routing_<tag>.npz file carries. fp32 on the GPU, fp64 in the serial kernel.

    export SURGE_DATA=/path/to/the/networks/and/runoff
    python experiments/bench_year_resolution.py 30min spin
    python experiments/bench_year_resolution.py 30min gpu 1,2,3,4,5,6,7,8,9,10,20
"""
import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surge import RoutingNetwork
# the serial stage calls the CWatM kernel, which surge.serial_ref finds through
# GPU_KWR_T5_SO; the other stages need nothing beyond this repository

DATA = os.environ.get('SURGE_DATA', '.')   # the directory holding the networks,
                                           # the reservoirs and the runoff
TAG, STAGE = sys.argv[1], sys.argv[2]
K_ARG = sys.argv[3] if len(sys.argv) > 3 else None
NPZ = '%s/routing_%s.npz' % (DATA, TAG)
OUT = os.environ.get('YEAR_OUT', '%s/year_resolution' % DATA) + '/' + TAG
os.makedirs(OUT, exist_ok=True)
KS = tuple(int(x) for x in os.environ.get('YEAR_KS', '1,2,3,4,5,6,7,8,9,10,20').split(','))
# the counts the metrics stage reads back; give the gpu stage the same list
PART = os.environ.get('METRICS_PART', '')   # a non-empty name writes this part of the k list
# on its own, as summary_<part>.json and metrics_<part>.npz, for merge_metrics_parts.py
C_MIN, BIG_MIN = 100.0, 10000.0
CKPT_DAYS = int(os.environ.get('SERIAL_CKPT_DAYS', 10))   # a 1 arc-minute day costs about
# twelve minutes, so a wide checkpoint interval throws away hours when a job hits its walltime
NEWTON = 4
RES_FILE = os.environ.get('YEAR_RES', '')   # a GRanD file turns the reservoirs on, for the
REF = os.environ.get('YEAR_REF', 'serial')  # dam experiment; its reference is the ordered
YEAR_DAYS = int(os.environ.get('YEAR_DAYS', 0))   # GPU solution rather than the serial kernel,
# which has no reservoirs. A day count shortens a smoke test.

net = RoutingNetwork(NPZ)
N, nsub = net.n_cells, net.noRoutingSteps
_d = np.load(NPZ)
mask = _d['mask2d']
RES = 360.0 / mask.shape[1]           # the maps span 360 deg of longitude, not always 180 of latitude
rows, cols = np.where(~mask)
lat = 90.0 - (rows + 0.5) * RES
lon = -180.0 + (cols + 0.5) * RES
del rows, cols, mask
QT = np.load(DATA + '/cwatm_qtot_2010_05deg.npz')['qtot']     # (365, 360, 720), kg m-2 s-1
ndays = QT.shape[0]
qflat = (np.clip(((90.0 - lat) / 0.5).astype(np.int64), 0, QT.shape[1] - 1) * QT.shape[2]
         + np.clip(((lon + 180.0) / 0.5).astype(np.int64), 0, QT.shape[2] - 1))
factor = net.cellArea / net.chanLength / 1000.0
ndays = min(ndays, YEAR_DAYS) if YEAR_DAYS else ndays
print('%s stage %s: %d cells, %d levels, %d substeps a day, %.4f deg, %d days'
      % (TAG, STAGE, N, net.n_levels, nsub, RES, ndays), flush=True)


def sideflow_np(day):
    return np.nan_to_num(QT[day].reshape(-1)[qflat], nan=0.0).astype(np.float64) * factor


def gpu_setup():
    import torch
    dev, dt = 'cuda', torch.float32
    if RES_FILE:
        from surge.reservoir import ResRouter
        router = ResRouter(net, device=dev, dtype=dt)
        z = np.load(RES_FILE)
        router.set_reservoirs(z['res_cell'], z['cap_m3'], z['rmean'])
        print('reservoirs: %d from %s' % (len(z['res_cell']), os.path.basename(RES_FILE)), flush=True)
    else:
        from surge import TorchRouter
        router = TorchRouter(net, device=dev, dtype=dt)
    qt_g = torch.as_tensor(QT.reshape(ndays, -1), device=dev, dtype=dt)
    idx_g = torch.as_tensor(qflat, device=dev)
    fac_g = torch.as_tensor(factor, device=dev, dtype=dt)

    def sideflow(day):
        return torch.nan_to_num(qt_g[day][idx_g], nan=0.0) * fac_g
    jac = ((lambda q, qd, k: router.parallel_res(q, qd, iterations=k, n_newton=NEWTON)) if RES_FILE
           else (lambda q, qd, k: router.substep_parallel(q, qd, iterations=k, n_newton=NEWTON)))
    wave = ((lambda q, qd: router.ordered_res(q, qd, n_newton=NEWTON)) if RES_FILE
            else (lambda q, qd: router.substep_ordered(q, qd, n_newton=NEWTON)))
    print('GPU %s' % torch.cuda.get_device_name(0), flush=True)
    return torch, router, sideflow, dev, dt, jac, wave


t00 = time.time()
if STAGE == 'spin':
    torch, router, sideflow, dev, dt, jac, wave = gpu_setup()
    q = torch.zeros(N, device=dev, dtype=dt)
    mm = np.lib.format.open_memmap(OUT + '/spin_k8_daily.npy', mode='w+', dtype=np.float32, shape=(ndays, N))
    for day in range(ndays):
        qd = sideflow(day)
        acc = torch.zeros(N, device=dev, dtype=dt)
        for _ in range(nsub):
            q = jac(q, qd, 8)
            acc += q
        mm[day] = (acc / nsub).cpu().numpy()
        if day % 60 == 0:
            print('  spin day %3d (%.0fs)' % (day, time.time() - t00), flush=True)
    del mm
    np.save(OUT + '/q0.npy', q.cpu().numpy().astype(np.float64))
    np.save(OUT + '/lonlat.npy', np.stack([lon, lat]).astype(np.float32))
    open(OUT + '/spin_k8_daily.npy.done', 'w').close()
    print('spin done (%.0fs)' % (time.time() - t00), flush=True)

elif STAGE == 'serial':
    from surge.serial_ref import kinematic_serial
    path = OUT + '/serial_daily.npy'
    ck = OUT + '/serial_checkpoint.npz'
    if os.path.exists(ck):
        z = np.load(ck); q, ann, start = z['q'], z['ann'], int(z['day'])
        mm = np.lib.format.open_memmap(path, mode='r+')
        print('resuming at day %d' % start, flush=True)
    else:
        q, ann, start = np.load(OUT + '/q0.npy'), np.zeros(N), 0
        mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(ndays, N))
    t0 = time.time()
    for day in range(start, ndays):
        sf = sideflow_np(day)
        acc = np.zeros(N)
        for _ in range(nsub):
            q = kinematic_serial(q, sf, net)
            acc += q
        mm[day] = (acc / nsub).astype(np.float32)
        ann += acc
        if (day + 1) % CKPT_DAYS == 0 or day == ndays - 1:
            mm.flush()
            np.savez(ck + '.tmp.npz', q=q, ann=ann, day=day + 1)
            os.replace(ck + '.tmp.npz', ck)
            el = time.time() - t0
            print('  serial day %3d (%.0fs, about %.1f h left)'
                  % (day + 1, el, el / (day + 1 - start) * (ndays - day - 1) / 3600), flush=True)
    np.save(OUT + '/serial_annual_mean.npy', (ann / (ndays * nsub)).astype(np.float32))
    del mm
    open(path + '.done', 'w').close()
    print('serial done (%.0fs)' % (time.time() - t00), flush=True)

elif STAGE == 'gpu':
    torch, router, sideflow, dev, dt, jac, wave = gpu_setup()
    q0 = torch.as_tensor(np.load(OUT + '/q0.npy'), device=dev, dtype=dt)
    for k in (KS if K_ARG is None else [int(x) for x in K_ARG.split(',')]):
        path = OUT + '/gpu_k%d_daily.npy' % k
        if os.path.exists(path + '.done'):
            continue
        mm = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(ndays, N))
        q = q0.clone()
        if RES_FILE:
            router.reset_storage(0.6)   # every k starts from the same reservoir storage, as the
        # ordered reference does; without this each k inherited the storage the last one ended with
        t0 = time.time()
        for day in range(ndays):
            qd = sideflow(day)
            acc = torch.zeros(N, device=dev, dtype=dt)
            for _ in range(nsub):
                q = jac(q, qd, k)
                acc += q
            mm[day] = (acc / nsub).cpu().numpy()
        del mm
        open(path + '.done', 'w').close()
        print('  k=%-2d year done (%.0fs)' % (k, time.time() - t0), flush=True)
    print('gpu done (%.0fs)' % (time.time() - t00), flush=True)

elif STAGE == 'ordered':
    # the level-by-level GPU solution, which keeps the upstream-to-downstream order and so
    # serves as the reference where the serial kernel cannot run, as with reservoirs
    torch, router, sideflow, dev, dt, jac, wave = gpu_setup()
    q = torch.as_tensor(np.load(OUT + '/q0.npy'), device=dev, dtype=dt)
    mm = np.lib.format.open_memmap(OUT + '/ordered_daily.npy', mode='w+', dtype=np.float32, shape=(ndays, N))
    ann = np.zeros(N)
    for day in range(ndays):
        qd = sideflow(day)
        acc = torch.zeros(N, device=dev, dtype=dt)
        for _ in range(nsub):
            q = wave(q, qd)
            acc += q
        d = (acc / nsub).cpu().numpy()
        mm[day] = d
        ann += d
        if day % 30 == 0:
            print('  ordered day %3d (%.0fs)' % (day, time.time() - t00), flush=True)
    del mm
    np.save(OUT + '/ordered_annual_mean.npy', (ann / ndays).astype(np.float32))
    open(OUT + '/ordered_daily.npy.done', 'w').close()
    print('ordered done (%.0fs)' % (time.time() - t00), flush=True)

elif STAGE == 'metrics':
    ann_all = np.load(OUT + '/%s_annual_mean.npy' % REF).astype(np.float64)
    C = np.flatnonzero(ann_all > C_MIN)
    print('evaluation cells %d above %g m3/s' % (C.size, C_MIN), flush=True)

    # longest river length upstream of each cell, including the cell itself
    lv, dx = net.level.astype(np.int64), net.chanLength.astype(np.float64)
    src, dst = net.edge_src, net.edge_dst
    assert np.all(lv[dst] > lv[src]), 'edges must run from lower to higher level'
    eo = np.argsort(lv[src], kind='stable'); src, dst = src[eo], dst[eo]
    co = np.argsort(lv, kind='stable')
    levels = np.arange(lv.min(), lv.max() + 2)
    eb = np.searchsorted(lv[src], levels); cb = np.searchsorted(lv[co], levels)
    upmax = np.zeros(N); L = np.zeros(N)
    for i in range(levels.size - 1):
        cells = co[cb[i]:cb[i + 1]]
        L[cells] = dx[cells] + upmax[cells]
        np.maximum.at(upmax, dst[eb[i]:eb[i + 1]], L[src[eb[i]:eb[i + 1]]])
    Cr = (1.0 / (net.channelAlpha[C] * net.beta * ann_all[C] ** (net.beta - 1.0))) * net.dtRouting / dx[C]

    def read_cells(path):
        """The cells of C, read in row blocks so the file is walked once, front to back.
        The blocks are read with plain file reads rather than through a memory map, because
        a map faults one page at a time and crawls at about 20 MB/s on this file system."""
        m = np.load(path, mmap_mode='r')
        nd, nc, off = m.shape[0], m.shape[1], m.offset
        assert m.dtype == np.float32, m.dtype
        del m
        blk = max(1, (2 << 30) // (nc * 4))            # about 2 GB a read
        out = np.empty((nd, C.size), np.float32)
        buf = np.empty(blk * nc, np.float32)
        with open(path, 'rb') as fh:
            fh.seek(off)
            for d0 in range(0, nd, blk):
                n = min(blk, nd - d0)
                mv = memoryview(buf[:n * nc]).cast('B')
                got = 0
                while got < len(mv):
                    r = fh.readinto(mv[got:])
                    if not r:
                        raise IOError('short read on %s at day %d' % (path, d0))
                    got += r
                out[d0:d0 + n] = buf[:n * nc].reshape(n, nc)[:, C]
        return out.astype(np.float64)

    R = read_cells(OUT + '/%s_daily.npy' % REF)
    Rm, Rs = R.mean(0), R.std(0)
    den = ((R - Rm) ** 2).sum(0)
    big = np.flatnonzero(ann_all[C] > BIG_MIN)
    res = dict(lon=lon[C].astype(np.float32), lat=lat[C].astype(np.float32),
               ann=ann_all[C].astype(np.float32), level=lv[C].astype(np.int32),
               upstream_km=(L[C] / 1000.0).astype(np.float32), dx_km=(dx[C] / 1000.0).astype(np.float32),
               courant=Cr.astype(np.float32))
    series = dict(lon=lon[C][big].astype(np.float32), lat=lat[C][big].astype(np.float32),
                  ann=ann_all[C][big].astype(np.float32), upstream_km=(L[C][big] / 1000.0).astype(np.float32),
                  serial=R[:, big].astype(np.float32))
    summary = dict(tag=TAG, reference=REF, reservoirs=os.path.basename(RES_FILE) if RES_FILE else None, cells=N, levels=int(net.n_levels), res_deg=RES, eval_cells=int(C.size),
                   big_cells=int(big.size), k={})
    for k in KS:
        G = read_cells(OUT + '/gpu_k%d_daily.npy' % k)
        Gm, Gs = G.mean(0), G.std(0)
        r = ((G - Gm) * (R - Rm)).mean(0) / np.maximum(Gs * Rs, 1e-12)
        bias = Gm / np.maximum(Rm, 1e-9)
        sdr = Gs / np.maximum(Rs, 1e-12)
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (sdr - 1) ** 2 + (bias - 1) ** 2)
        nse = 1.0 - ((G - R) ** 2).sum(0) / np.maximum(den, 1e-12)
        rrmse = np.sqrt(((G - R) ** 2).mean(0)) / np.maximum(Rm, 1e-3)
        for nm, v in (('kge', kge), ('nse', nse), ('rrmse', rrmse), ('bias', bias), ('r', r), ('sdr', sdr)):
            res['%s_k%d' % (nm, k)] = v.astype(np.float32)
        series['k%d' % k] = G[:, big].astype(np.float32)
        summary['k'][str(k)] = dict(kge_median=float(np.median(kge)), kge_p5=float(np.percentile(kge, 5)),
                                    kge_min=float(kge.min()), nse_median=float(np.median(nse)),
                                    rrmse_median=float(np.median(rrmse)), rrmse_p95=float(np.percentile(rrmse, 95)),
                                    bias_absmax=float(np.abs(bias - 1).max()))
        print('k=%-2d KGE median %.4f p5 %.4f min %.3f | NSE median %.4f | rRMSE median %.4f p95 %.3f | max|bias| %.4f'
              % (k, np.median(kge), np.percentile(kge, 5), kge.min(), np.median(nse),
                 np.median(rrmse), np.percentile(rrmse, 95), np.abs(bias - 1).max()), flush=True)
        del G
    sfx = ('_' + PART) if PART else ''
    np.save(OUT + '/eval_index.npy', C)
    np.savez_compressed(OUT + '/metrics%s.npz' % sfx, note='one year of 2010 routing, SURGE against the CWatM serial '
                        'kernel over cells with serial annual mean > %g m3/s; bench_year_resolution.py' % C_MIN, **res)
    np.savez_compressed(OUT + '/big_rivers_daily%s.npz' % sfx, **series)
    json.dump(summary, open(OUT + '/summary%s.json' % sfx, 'w'), indent=2)
    print('metrics written (%.0fs)' % (time.time() - t00), flush=True)
else:
    sys.exit('unknown stage ' + STAGE)
