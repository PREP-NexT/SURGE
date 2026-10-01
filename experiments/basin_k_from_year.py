"""Iterations each of the 26 benchmark basins needs, from the 1 arc-minute year already run.

Flow moves downstream only, so a basin's cells carry the same solution whether the basin is
routed on its own or inside the global network. The per-cell KGE of the global 1 arc-minute
year therefore gives each basin its own convergence, under the criterion of section 4.1:
the smallest k at which 95% of the basin's high-discharge cells reach KGE >= 0.99.

    python basin_k_from_year.py            # writes basin_k_courant.json next to metrics.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subset_basin import cell_coords, downstream_pointer, drainage_area, collect_upstream
from bench_basins import BASINS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SURGE_DATA', '.')   # the directory holding the network
YEAR = sys.argv[1] if len(sys.argv) > 1 else '.'   # the 1 arc-minute year of
                                                   # bench_year_resolution.py
GRAPH = os.path.join(DATA, 'routing_01min.npz')
THRESH = 0.99

M = np.load(os.path.join(YEAR, 'metrics.npz'))
KS = sorted(int(k[5:]) for k in M.files if k.startswith('kge_k'))
evi = np.load(os.path.join(YEAR, 'eval_index.npy'))
courant = M['courant'].astype(float)
kge = {k: M['kge_k%d' % k].astype(float) for k in KS}
print('%d evaluation cells, k = %s' % (evi.size, ','.join(map(str, KS))), flush=True)

print('loading the 1 arc-minute graph ...', flush=True)
g = {k: v for k, v in np.load(GRAPH).items()}
clat, clon = cell_coords(g['mask2d'])
ptr = downstream_pointer(g)
print('accumulating drainage ...', flush=True)
acc = drainage_area(g, ptr)
dirupLen, dirupID = g['dirupLen'].astype(np.int64), g['dirupID'].astype(np.int64)

pos = np.full(ptr.size, -1, np.int64)      # global cell -> row of the metrics arrays
pos[evi] = np.arange(evi.size)

rows = []
for name, lat, lon in BASINS:
    d = np.hypot(clat - lat, (clon - lon) * np.cos(np.radians(lat)))
    near = np.flatnonzero(d < 0.5)
    outlet = int(near[np.argmax(acc[near])]) if near.size else int(np.argmin(d))
    cells = collect_upstream(dirupLen, dirupID, outlet)
    idx = pos[cells]
    idx = idx[idx >= 0]                    # the basin's own high-discharge cells
    cr = float(np.median(courant[idx]))
    crq = {str(q): float(np.percentile(courant[idx], q)) for q in (50, 75, 90, 95, 99)}
    upq = {str(q): float(np.percentile(M['upstream_km'].astype(float)[idx], q)) for q in (50, 95, 100)}
    lvmax = int(M['level'].astype(int)[idx].max())
    p5 = {k: float(np.percentile(kge[k][idx], 5)) for k in KS}
    need = next((k for k in KS if p5[k] >= THRESH), None)
    # each iteration multiplies what is left by about the same factor, so the shortfall of the
    # 5th percentile falls log-linearly in k; fitting its tail gives the crossing as a number,
    # and carries the basins that need more iterations than the year was run for
    ks = np.array([k for k in KS if k >= 8], float)
    y = np.log10(np.maximum([1.0 - p5[int(k)] for k in ks], 1e-9))
    a, b = np.polyfit(ks, y, 1)
    k_fit = (np.log10(1.0 - THRESH) - b) / a
    rows.append(dict(name=name, cells=int(cells.size), eval_cells=int(idx.size),
                     courant_median=cr, k=need, k_fit=float(k_fit), kge_p5=p5.get(need),
                     kge_median=float(np.median(kge[need][idx])) if need else None,
                     p5_curve={str(k): p5[k] for k in KS}, area_km2=float(acc[outlet] / 1e6),
                     courant_q=crq, upstream_km_q=upq, level_max=lvmax))
    print('%-12s %9d cells %7d evaluated  Cr %5.2f  k %-4s  k from the tail %5.1f' %
          (name, cells.size, idx.size, cr, need, k_fit), flush=True)

out = os.path.join(YEAR, 'basin_k_courant.json')
json.dump(dict(note='iterations each benchmark basin needs at 1 arc-minute, from the 2010 year; '
               'criterion KGE_c >= %g at 95%% of the basin cells with mean discharge above '
               '100 m3/s; basin_k_from_year.py' % THRESH, ks_available=KS, rows=rows),
          open(out, 'w'), indent=1)
print('wrote', out)
