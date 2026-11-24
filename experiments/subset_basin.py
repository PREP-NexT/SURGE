"""Cut a sub-network out of a global routing graph.

Takes everything upstream of one outlet cell and rewrites the graph arrays so the
result loads with ``surge.RoutingNetwork``. This puts individual river basins on the
same speedup-versus-domain-size axis as the global runs: the global points vary the
cell count by refining the grid, these vary it by shrinking the domain.

    python subset_basin.py <global.npz> <lat> <lon> <out.npz> [name]

The outlet is the highest-drainage cell within half a degree of (lat, lon), so a
river mouth or a gauge position both work. Cell coordinates are recovered from
``mask2d``, which the builders store as the land mask of a global regular grid.
"""
import sys
import numpy as np


def cell_coords(mask2d):
    """Latitude and longitude of every cell, in the order the graph numbers them."""
    ny, nx = mask2d.shape
    gsize = 360.0 / nx                      # global in longitude; latitude may be clipped
    rows, cols = np.where(~mask2d)          # graph order is row-major over land cells
    return 90.0 - (rows + 0.5) * gsize, -180.0 + (cols + 0.5) * gsize


def downstream_pointer(g):
    """One downstream cell per cell, -1 at mouths, from the stored edge list."""
    n = int(g['mapC'])
    ptr = np.full(n, -1, np.int64)
    ptr[g['edge_src'].astype(np.int64)] = g['edge_dst'].astype(np.int64)
    return ptr


def drainage_area(g, ptr):
    """Accumulate cell area downstream, one topological level at a time."""
    level = g['level'].astype(np.int64)
    acc = g['cellArea'].astype(np.float64).copy()
    order = np.argsort(level, kind='stable')
    bounds = np.flatnonzero(np.r_[True, np.diff(level[order]) != 0])
    for s, e in zip(bounds, np.r_[bounds[1:], len(order)]):
        c = order[s:e]
        d = ptr[c]
        m = d >= 0
        if m.any():
            np.add.at(acc, d[m], acc[c[m]])
    return acc


def collect_upstream(dirupLen, dirupID, outlet):
    """Every cell that drains through `outlet`, including it."""
    keep = np.zeros(len(dirupLen) - 1, bool)
    keep[outlet] = True
    frontier = np.array([outlet], np.int64)
    while frontier.size:
        lo, hi = dirupLen[frontier], dirupLen[frontier + 1]
        nxt = np.concatenate([dirupID[a:b] for a, b in zip(lo, hi)]) if frontier.size else frontier
        nxt = nxt[~keep[nxt]] if nxt.size else nxt
        keep[nxt] = True
        frontier = nxt
    return np.flatnonzero(keep)


def levels_from(down, n):
    """Kahn waves over the subset, touching only the active frontier each round."""
    indeg = np.bincount(down[down >= 0], minlength=n)
    level = np.zeros(n, np.int64)
    cur = np.flatnonzero(indeg == 0)
    seen, r = 0, 0
    while cur.size:
        level[cur] = r
        seen += cur.size
        d = down[cur]
        d = d[d >= 0]
        if d.size:
            dec = np.bincount(d, minlength=n)
            indeg -= dec
            cur = np.flatnonzero((dec > 0) & (indeg == 0))
        else:
            cur = np.array([], np.int64)
        r += 1
    if seen != n:
        raise SystemExit('level sweep reached %d of %d cells; the subset is not a tree' % (seen, n))
    return level


def rebuild(g, cells, ptr):
    n = len(cells)
    full = g['mask2d']
    sub_mask = np.ones_like(full, dtype=bool)      # True = not a routing cell
    rr, cc = np.where(~full)
    sub_mask[rr[cells], cc[cells]] = False
    old2new = -np.ones(int(g['mapC']), np.int64)
    old2new[cells] = np.arange(n)
    down = ptr[cells]
    down = np.where(down >= 0, old2new[np.clip(down, 0, None)], -1)

    child = np.flatnonzero(down >= 0)
    parent = down[child]
    o = np.argsort(parent, kind='stable')
    dirupID = child[o].astype(np.int64)
    counts = np.bincount(parent[o], minlength=n)
    dirupLen = np.zeros(n + 1, np.int64)
    np.cumsum(counts, out=dirupLen[1:])

    level = levels_from(down, n)
    out = {
        'dirDown': np.argsort(level, kind='stable').astype(np.int64),
        'dirupLen': dirupLen, 'dirupID': dirupID,
        'lendirDown': np.int64(n), 'mapC': np.int64(n),
        'beta': g['beta'], 'dtRouting': g['dtRouting'], 'noRoutingSteps': g['noRoutingSteps'],
        'channelAlpha': g['channelAlpha'][cells], 'chanLength': g['chanLength'][cells],
        'cellArea': g['cellArea'][cells], 'level': level,
        'edge_src': dirupID, 'edge_dst': np.repeat(np.arange(n), counts).astype(np.int64),
        'mask2d': sub_mask,
        'waterBodyID': np.zeros(n, np.int64),
    }
    return out


def main():
    src, lat, lon, dst = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    name = sys.argv[5] if len(sys.argv) > 5 else dst
    g = {k: v for k, v in np.load(src).items()}
    clat, clon = cell_coords(g['mask2d'])
    ptr = downstream_pointer(g)
    acc = drainage_area(g, ptr)

    d = np.hypot(clat - lat, (clon - lon) * np.cos(np.radians(lat)))
    near = np.flatnonzero(d < 0.5)
    outlet = int(near[np.argmax(acc[near])]) if near.size else int(np.argmin(d))

    cells = collect_upstream(g['dirupLen'].astype(np.int64), g['dirupID'].astype(np.int64), outlet)
    sub = rebuild(g, cells, ptr)
    np.savez_compressed(dst, **sub)
    print('%-14s outlet=(%.2f,%.2f) cells=%d levels=%d area=%.0f km2 -> %s'
          % (name, clat[outlet], clon[outlet], len(cells),
             int(sub['level'].max()) + 1, acc[outlet] / 1e6, dst))


if __name__ == '__main__':
    main()
