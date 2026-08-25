"""Upscale the 5-arcmin ldd to coarser resolutions with pyflwdir (IHU) and build a
routing-graph npz at each, in the format surge.RoutingNetwork expects.
Gives a true resolution-vs-speed series (same MERIT/Eilander source)."""
import sys, numpy as np, xarray as xr, pyflwdir
from affine import Affine

LDD_NC = sys.argv[1]
OUTDIR = sys.argv[2]
BETA, MANNING, SLOPE, DAY = 0.6, 0.03, 1e-3, 86400.0

d = xr.open_dataset(LDD_NC)
_raw = np.nan_to_num(np.asarray(d['ldd'].values).astype('float64'), nan=0).astype(np.int64)
ldd0 = np.where((_raw >= 1) & (_raw <= 9), _raw, 255).astype(np.uint8)   # pyflwdir mv=255
ny, nx = ldd0.shape
RES0 = 1.0 / 12.0
transform0 = Affine(RES0, 0, -180.0, 0, -RES0, 90.0)
flw0 = pyflwdir.from_array(ldd0, ftype='ldd', transform=transform0, latlon=True, cache=True)


def build_graph(ldd, lat_c, lon_c, ups, res_deg, nsub, out):
    ny, nx = ldd.shape
    ldd = ldd.astype(np.int64)
    valid = (ldd >= 1) & (ldd <= 9)
    N = int(valid.sum())
    idx2d = -np.ones((ny, nx), np.int64); idx2d[valid] = np.arange(N)
    rows, cols = np.where(valid); codes = ldd[rows, cols]
    drow = np.array([0, 1, 1, 1, 0, 0, 0, -1, -1, -1])[codes]
    dcol = np.array([0, -1, 0, 1, -1, 0, 1, -1, 0, 1])[codes]
    nr, ncl = rows + drow, cols + dcol
    inb = (nr >= 0) & (nr < ny) & (ncl >= 0) & (ncl < nx)
    down = np.full(N, -1, np.int64)
    di = idx2d[np.clip(nr, 0, ny-1), np.clip(ncl, 0, nx-1)]
    ok = inb & (codes != 5) & (di >= 0); down[ok] = di[ok]
    # upstream CSR
    child = np.arange(N)[down >= 0]; parent = down[down >= 0]
    o = np.argsort(parent, kind='stable')
    dirupID = child[o].astype(np.int64)
    counts = np.bincount(parent[o], minlength=N)
    dirupLen = np.zeros(N+1, np.int64); np.cumsum(counts, out=dirupLen[1:])
    # levels (Kahn waves)
    indeg = counts.copy(); level = np.zeros(N, np.int64); done = np.zeros(N, bool)
    cur = np.where(indeg == 0)[0]; r = 0
    while cur.size:
        level[cur] = r; done[cur] = True
        dn = down[cur]; dn = dn[dn >= 0]
        if dn.size:
            indeg -= np.bincount(dn, minlength=N)
            cur = np.where((indeg <= 0) & (~done))[0]
        else:
            cur = np.array([], np.int64)
        r += 1
    dirDown = np.argsort(level, kind='stable').astype(np.int64)
    edge_dst = np.repeat(np.arange(N), np.diff(dirupLen)).astype(np.int64)
    edge_src = dirupID.copy()
    # geometry (same synthetic recipe as global build, scaled by resolution)
    ups_c = np.maximum(ups[valid].astype(np.float64), 1.0)
    width = 3.0 * np.sqrt(ups_c)
    dx_ew = 111320.0 * np.cos(np.deg2rad(lat_c[rows])) * res_deg
    chanLength = np.where(np.isin(codes, [1, 3, 7, 9]), np.sqrt(2.0), 1.0) * np.abs(dx_ew)
    chanLength = np.maximum(chanLength, 1000.0)
    P = width + 2.0
    channelAlpha = (MANNING * P ** (2.0/3.0) / np.sqrt(SLOPE)) ** BETA
    cellArea = np.maximum(np.abs(dx_ew) * 111320.0 * res_deg, 1.0)
    mask2d = ~valid
    np.savez(out, dirDown=dirDown, dirupLen=dirupLen, dirupID=dirupID,
             lendirDown=np.int64(N), channelAlpha=channelAlpha, beta=np.float64(BETA),
             chanLength=chanLength, dtRouting=np.float64(DAY/nsub), noRoutingSteps=np.int64(nsub),
             cellArea=cellArea, mask2d=mask2d, mapC=np.int64(N),
             waterBodyID=np.zeros(N, np.int64), level=level, edge_src=edge_src, edge_dst=edge_dst)
    print('  %-7s N=%d levels=%d -> %s' % ('%gdeg' % res_deg, N, int(level.max())+1, out))
    return N


# native 5' (from flw0 directly) + upscaled 15', 30'
for scale, tag in [(1, '05min'), (3, '15min'), (6, '30min')]:
    if scale == 1:
        flw, res = flw0, RES0
        lddc = ldd0
        ups = flw0.upstream_area(unit='cell')
    else:
        flw, idxs = flw0.upscale(scale, method='ihu')
        res = RES0 * scale
        lddc = flw.to_array(ftype='ldd')
        ups = flw.upstream_area(unit='cell')
    nyc = lddc.shape[0]
    lat_c = 90.0 - (np.arange(nyc) + 0.5) * res
    lon_c = -180.0 + (np.arange(lddc.shape[1]) + 0.5) * res
    build_graph(lddc, lat_c, lon_c, ups, res, 20, f'{OUTDIR}/routing_static_{tag}.npz')
