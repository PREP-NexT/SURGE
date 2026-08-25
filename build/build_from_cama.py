"""Build a routing-graph npz from a CaMa-Flood map (nextxy.bin), same format as
surge.RoutingNetwork. Synthetic geometry (same recipe as the 5'/15'/30'
builds) so the resolution-speed series is apples-to-apples."""
import sys, os, numpy as np

MAPDIR, OUT, NSUB = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20
BETA, MANNING, SLOPE, DAY = 0.6, 0.03, 1e-3, 86400.0

# params.txt: nx, ny, nflp, gsize, west, east, south, north
P = [l.split('!')[0].strip() for l in open(os.path.join(MAPDIR, 'params.txt'))]
nx, ny = int(P[0]), int(P[1]); gsize = float(P[3]); west = float(P[4]); north = float(P[7])
a = np.fromfile(os.path.join(MAPDIR, 'nextxy.bin'), dtype=np.int32)
nextx = a[:nx*ny].reshape(ny, nx); nexty = a[nx*ny:].reshape(ny, nx)
valid = (nextx > 0) | (nextx == -9) | (nextx == -10)
N = int(valid.sum())
print(f'{MAPDIR}: {nx}x{ny} gsize={gsize}  valid cells N={N:,}')

idx2d = -np.ones((ny, nx), np.int64); idx2d[valid] = np.arange(N)
rows, cols = np.where(valid)
nxc, nyc = nextx[rows, cols], nexty[rows, cols]
down = np.full(N, -1, np.int64)
flow = nxc > 0
ty, tx = nyc[flow] - 1, nxc[flow] - 1
inb = (ty >= 0) & (ty < ny) & (tx >= 0) & (tx < nx)
tgt = np.full(flow.sum(), -1, np.int64)
tgt[inb] = idx2d[np.clip(ty, 0, ny-1), np.clip(tx, 0, nx-1)]
down[np.where(flow)[0]] = tgt
down[down < 0] = -1
print(f'  outlets/mouths: {int((down<0).sum()):,}')

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
print(f'  levels: {int(level.max())+1}')
dirDown = np.argsort(level, kind='stable').astype(np.int64)
edge_dst = np.repeat(np.arange(N), np.diff(dirupLen)).astype(np.int64)
edge_src = dirupID.copy()
# ups (upstream cell count) via level-ordered downstream scatter accumulation
ups = np.ones(N, np.float64)
maxlv = int(level.max())
cells_by_lv = [np.where(level == L)[0] for L in range(maxlv+1)]
for L in range(maxlv+1):
    c = cells_by_lv[L]
    dc = down[c]; m = dc >= 0
    if m.any():
        np.add.at(ups, dc[m], ups[c[m]])
# geometry (synthetic, same recipe as other resolutions)
lat = north - (rows + 0.5) * gsize
width = 3.0 * np.sqrt(np.maximum(ups, 1.0))
dx_ew = 111320.0 * np.cos(np.deg2rad(lat)) * gsize
chanLength = np.maximum(np.abs(dx_ew), 1000.0)
P_ = width + 2.0
channelAlpha = (MANNING * P_ ** (2.0/3.0) / np.sqrt(SLOPE)) ** BETA
cellArea = np.maximum(np.abs(dx_ew) * 111320.0 * gsize, 1.0)
mask2d = ~valid
np.savez(OUT, dirDown=dirDown, dirupLen=dirupLen, dirupID=dirupID, lendirDown=np.int64(N),
         channelAlpha=channelAlpha, beta=np.float64(BETA), chanLength=chanLength,
         dtRouting=np.float64(DAY/NSUB), noRoutingSteps=np.int64(NSUB), cellArea=cellArea,
         mask2d=mask2d, mapC=np.int64(N), waterBodyID=np.zeros(N, np.int64),
         level=level, edge_src=edge_src, edge_dst=edge_dst)
print(f'  wrote {OUT}  ({N:,} cells, {edge_src.size:,} edges, nsub={NSUB})')
