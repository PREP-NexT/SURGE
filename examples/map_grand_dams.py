"""Map the GRanD v1.3 dam set onto a routing network (paper Section 3.6).

Dam locations/capacity/mean-discharge come from GRanD v1.3, georeferenced by GeoDAR
(Wang et al. 2022, ESSD; CC-BY). Download GeoDAR_v10_v11.zip from
https://zenodo.org/records/6163413 and point --grand at its GRanD_v13_issues.csv.

Each dam is snapped to the largest-river cell (max channelAlpha) in a small window;
dams sharing a cell are merged. Grid geometry (row 0 = north) is passed explicitly.

    python examples/map_grand_dams.py net.npz GRanD_v13_issues.csv out.npz \
        --lat0 89.9583 --dlat 0.083333 --lon0 -179.9583 --dlon 0.083333
"""
import argparse, csv, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('net'); ap.add_argument('grand'); ap.add_argument('out')
    ap.add_argument('--lat0', type=float, required=True); ap.add_argument('--dlat', type=float, required=True)
    ap.add_argument('--lon0', type=float, required=True); ap.add_argument('--dlon', type=float, required=True)
    ap.add_argument('--win', type=int, default=2)
    a = ap.parse_args()
    dams = []
    with open(a.grand, encoding='utf-8-sig') as f:
        for d in csv.DictReader(f):
            try:
                lat, lon, cap = float(d['LAT_DD']), float(d['LONG_DD']), float(d['CAP_MCM'])
            except (KeyError, ValueError):
                continue
            try: dis = float(d.get('DIS_AVG_LS', -99))
            except ValueError: dis = -99
            if np.isfinite(lat) and np.isfinite(lon):
                dams.append((lat, lon, cap, dis))
    z = np.load(a.net); mask2d = z['mask2d']; ny, nx = mask2d.shape; valid = ~mask2d
    idx2d = -np.ones((ny, nx), np.int64); idx2d[valid] = np.arange(int(z['mapC']))
    a2d = -np.ones((ny, nx)); a2d[valid] = z['channelAlpha']
    cell_of = {}
    for lat, lon, cap, dis in dams:
        r = int(round((a.lat0 - lat) / a.dlat)); c = int(round((lon - a.lon0) / a.dlon))
        if not (0 <= r < ny and 0 <= c < nx): continue
        r0, r1 = max(0, r - a.win), min(ny, r + a.win + 1)
        c0, c1 = max(0, c - a.win), min(nx, c + a.win + 1)
        sub = a2d[r0:r1, c0:c1]
        if sub.max() < 0: continue
        rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
        cell = int(idx2d[r0 + rr, c0 + cc]); cm = max(cap, 0.) * 1e6; rm = dis / 1000. if dis > 0 else 0.
        if cell in cell_of: cell_of[cell][0] += cm; cell_of[cell][1] += rm
        else: cell_of[cell] = [cm, rm]
    cells = np.array(sorted(cell_of), np.int64)
    cap_m3 = np.array([cell_of[c][0] for c in cells]); rmean = np.array([cell_of[c][1] for c in cells])
    bad = rmean <= 0; rmean[bad] = np.maximum(cap_m3[bad] / (365 * 86400.), 1e-3)
    np.savez(a.out, res_cell=cells, cap_m3=cap_m3, rmean=rmean, n_cells=int(z['mapC']))
    print('%d dams -> %d reservoir cells (%.0f km3)' % (len(dams), len(cells), cap_m3.sum() / 1e9))

if __name__ == '__main__':
    main()
