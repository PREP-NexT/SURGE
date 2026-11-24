"""Figure 8: the 26 large river basins routed on their own. (a) Map in the Robinson projection,
each basin coloured by its speedup and named by a leader from a dot at its interior point,
the point furthest from its edge. (b) Speedup against the river cells of the routed domain,
with a few basins named. (c) Wall-clock of SURGE per routing time step against the same cells.
Panel titles are centred over their panels.

Labels are placed by search, not by hand, on a fixed set of rows with leader ends on a 5
degree grid, so they line up in stacks. A label is refused if it leaves the map, overlaps
a label or a dot already placed, or if its leader crosses another label or leader. Among
the rest the cheapest is taken, the cost rising with leader length and with how much of
the label or leader lies over a painted basin. The most crowded basins are placed first,
while the most room is left around them.

The figure is built at the 166 mm text width, so its text prints at the size set here.

    python fig_basin_speedup.py [bench.json basin_map.npz coarse.npz out_stem]
"""
import json

SHOW_GLOBAL = True    # draw the whole 1 arc-minute network as a star in panel b
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_style import setup_paper1_style  # noqa: E402
setup_paper1_style()
# Fig 1 size tier: letter 11, title 10.5, labels and annotations 9, tick numbers stay 8
FS_LETTER, FS_TITLE = 11.0, 10.5
plt.rcParams.update({'axes.titlesize': FS_TITLE, 'axes.labelsize': 9.0,
                     'legend.fontsize': 9.0, 'font.size': 9.0})

DATA = os.path.join(HERE, '..', 'figure_data')
DEFAULTS = [os.path.join(DATA, 'bench_basins_display.json'),
            os.path.join(DATA, 'basin_map_01min.npz'),
            os.path.join(DATA, 'routing_static_30min.npz'),
            os.path.join(HERE, '..', 'figures', 'fig_basin_speedup')]

CMAP = plt.get_cmap('RdYlBu_r')        # blue for the slow end, red for the fast end
LAND, OUTLINE = '#E6E8EA', '#1A1A1A'
FS = 9.0
# names as printed. The data keep the names of Nijssen et al. (2001); where that name is not
# the one English-language readers know, the label uses the common one
DISPLAY = {'Changjiang': 'Yangtze', 'S.Dvina': 'Northern Dvina', 'Parana': 'Paraná', 'Xi': 'Pearl'}
# basins whose leader must be straight, and basins held to one direction
STRAIGHT_ONLY = {'Olenek', 'Yenisei', 'Kolyma'}
ONLY_UP = {'Pechora'}
# the crowded northern Eurasian basins are placed together by an exact search over short
# straight leaders; every other basin is placed one at a time afterwards
# two crowded groups are placed by exact search, north first, then East Asia around it
CLUSTERS = [['S.Dvina', 'Pechora', 'Volga', 'Ob', 'Yenisei', 'Olenek', 'Lena', 'Yana',
             'Indigirka', 'Kolyma'],
            ['Amur', 'Yellow', 'Changjiang', 'Xi', 'Irrawaddy', 'Brahmaputra', 'Mekong']]
CLUSTER_MAX = 30.0          # longest leader tried in the cluster [degrees]
HALO = [pe.withStroke(linewidth=2.4, foreground='white')]

# the grid of the figure, in millimetres from its top left corner
W = 166.0
MAP_X, MAP_Y, MAP_W = 2.0, 8.0, 162.0
# the map is drawn in the Robinson projection, scaled so one unit is one degree of
# longitude at the equator; every placement rule below works in these projected units
import pyproj
_R = 180.0 / (0.8487 * np.pi)
_ROBIN = pyproj.Proj('+proj=robin +R=%r' % _R)
def P(lon, lat):
    x, y = _ROBIN(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)
def Pinv(x, y):
    lon, lat = _ROBIN(np.asarray(x, float), np.asarray(y, float), inverse=True)
    return np.asarray(lon), np.asarray(lat)
LAT_S = -58.0                                 # the map stops south of Tierra del Fuego
Y_S, Y_N = float(P(0, LAT_S)[1]), float(P(0, 90)[1])
XLIM, YLIM = (-186.0, 186.0), (Y_S - 1.0, Y_N + 5.0)
PRES = 0.1                                    # cell of the projected rasters [units]
MAP_H = MAP_W * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])
SC_X, SC_GAP, SC_W, SC_H, SC_BOTTOM = 18.0, 9.0, 144.0, 40.0, 11.0   # b 8 mm lower so the figure and its caption fit the page
SC_Y = MAP_Y + MAP_H + SC_GAP
C_GAP, C_H = 12.0, 38.0           # panel c sits under b and shares its axis of cells
C_Y = SC_Y + SC_H + C_GAP
H = C_Y + C_H + SC_BOTTOM
MM = 1 / 25.4


def mmax(fig, x, y, w, h):
    return fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])


def panel_letter(fig, x, y, letter, title=None, gap_mm=2.6, centre=None):
    """The bold letter at the left, and the panel title centred over its panel (centre, in
    mm from the left), on the same baseline, as in the other figures"""
    fig.text(x / W, 1.0 - y / H, letter, fontsize=FS_LETTER, fontweight='bold', ha='left', va='bottom')
    if title:
        fig.text(centre / W, 1.0 - y / H, title, fontsize=plt.rcParams['axes.titlesize'],
                 ha='center', va='bottom')


def land_backdrop(path):
    mask = np.load(path)['mask2d']
    ny, nx = mask.shape
    gsize = 360.0 / nx
    return ~mask, (-180.0, 180.0, 90.0 - ny * gsize, 90.0), gsize


def inside_point(mask):
    """The interior cell furthest from the basin edge"""
    d = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    r, c = np.unravel_index(int(np.argmax(d)), mask.shape)
    return int(r), int(c)


def ink(rgba):
    r, g, b = rgba[:3]
    return '#FFFFFF' if 0.299 * r + 0.587 * g + 0.114 * b < 0.62 else '#101010'


def main():
    args = sys.argv[1:5] if len(sys.argv) >= 5 else DEFAULTS
    bench, mapnpz, coarse, stem = args
    rows = json.load(open(bench))['rows']
    basins = sorted([r for r in rows if r['name'] != 'global'], key=lambda r: r['cells'])
    glob = next((r for r in rows if r['name'] == 'global'), None)
    if not SHOW_GLOBAL:          # the whole network is quoted in the text instead, and
        glob = None              # its point stretches the axis over the 26 basins
    num = {r['name']: i + 1 for i, r in enumerate(basins)}

    m = np.load(mapnpz)
    grid, deg = m['grid'], float(m['deg'])
    painted = {n: s for n, s in zip(m['names'].tolist(), m['speedup'].tolist())}
    for r in basins:
        if abs(painted[r['name']] - r['speedup']) > 0.05:
            raise SystemExit('raster is out of date for %s; re-run map_basins.py' % r['name'])
    # the basin raster and the land mask are resampled once onto a regular grid in the
    # projected plane, nearest neighbour, so the placement rules can index them directly
    _xs = np.arange(XLIM[0], XLIM[1], PRES) + PRES / 2
    _ys = np.arange(YLIM[1], YLIM[0], -PRES) - PRES / 2
    _X, _Y = np.meshgrid(_xs, _ys)
    _lon, _lat = Pinv(_X, _Y)
    _ok = np.isfinite(_lon) & np.isfinite(_lat) & (np.abs(_lon) <= 180) & (_lat >= LAT_S)
    _bx, _by = P(_lon[_ok], _lat[_ok])
    _ok2 = np.zeros_like(_ok); _ok2[_ok] = (np.abs(_bx - _X[_ok]) < 0.05) & (np.abs(_by - _Y[_ok]) < 0.05)
    _ok = _ok2
    _r = np.clip(((90.0 - _lat[_ok]) / deg).astype(int), 0, grid.shape[0] - 1)
    _c = np.clip(((_lon[_ok] + 180.0) / deg).astype(int), 0, grid.shape[1] - 1)
    grid_p = np.full(_X.shape, np.nan); grid_p[_ok] = grid[_r, _c]
    PEXT = (XLIM[0], XLIM[0] + _xs.size * PRES, YLIM[1] - _ys.size * PRES, YLIM[1])
    painted_any = np.isfinite(grid_p)
    vals = np.array([r['speedup'] for r in basins])
    norm = matplotlib.colors.Normalize(vals.min(), vals.max())
    colour = {r['name']: CMAP(norm(r['speedup'])) for r in basins}

    # axes are placed on the millimetre grid, so the constrained layout of the style is off
    fig = plt.figure(figsize=(W * MM, H * MM), layout='none')

    # ---------------------------------------------------------------- a: the map ------
    ax = mmax(fig, MAP_X, MAP_Y, MAP_W, MAP_H)
    land0, _, _ = land_backdrop(coarse)
    _g = 360.0 / land0.shape[1]
    _r = np.clip(((90.0 - _lat[_ok]) / _g).astype(int), 0, land0.shape[0] - 1)
    _c = np.clip(((_lon[_ok] + 180.0) / _g).astype(int), 0, land0.shape[1] - 1)
    land = np.zeros(_X.shape, bool); land[_ok] = land0[_r, _c]
    ax.imshow(land, extent=PEXT, origin='upper', interpolation='nearest', zorder=1,
              cmap=matplotlib.colors.ListedColormap([(0, 0, 0, 0), LAND]))
    ax.imshow(grid_p, extent=PEXT, origin='upper', cmap=CMAP, norm=norm,
              interpolation='nearest', zorder=2)
    for name, val in painted.items():
        ax.contour((grid_p == val).astype(float), levels=[0.5], colors=[OUTLINE],
                   linewidths=.35, extent=PEXT, origin='upper', zorder=3)
    # the outline of the projected globe: the two edge meridians and the cut parallels
    _la = np.linspace(LAT_S, 90, 200)
    _ex, _ey = P(np.r_[np.full(200, 180.0), np.linspace(180, -180, 200), np.full(200, -180.0), np.linspace(-180, 180, 200)],
                 np.r_[_la, np.full(200, 90.0), _la[::-1], np.full(200, LAT_S)])
    ax.plot(_ex, _ey, color='#1A1A1A', lw=0.6, zorder=4)
    OUTLINE_XY = np.c_[_ex, _ey]
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    panel_letter(fig, 1.0, MAP_Y - 1.0, 'a', 'Speedup of each basin routed on its own', centre=MAP_X + MAP_W / 2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted()

    def text_size(s):
        t = ax.text(0, 0, s, fontsize=FS)
        bb = t.get_window_extent(renderer)
        t.remove()
        (x0, y0), (x1, y1) = to_data.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0, y1 - y0

    def painted_fraction(x0, y0, x1, y1, own=None):
        r0 = int(np.clip((PEXT[3] - y1) / PRES, 0, grid_p.shape[0]))
        r1 = int(np.clip((PEXT[3] - y0) / PRES, 0, grid_p.shape[0]))
        c0 = int(np.clip((x0 - PEXT[0]) / PRES, 0, grid_p.shape[1]))
        c1 = int(np.clip((x1 - PEXT[0]) / PRES, 0, grid_p.shape[1]))
        area = max((r1 - r0) * (c1 - c0), 1)
        if r1 <= r0 or c1 <= c0:
            return 0.0
        sub = painted_any[r0:r1, c0:c1]
        if own is not None:
            sub = sub & (grid_p[r0:r1, c0:c1] != own)
        return float(sub.sum()) / area

    # the colour key sits in the open Indian Ocean below Africa; its space is reserved first, and the
    # bar is made exactly as wide as the label that sits above it
    CB_LABEL = 'Speedup over CPU-serial'
    _t = ax.text(0, 0, CB_LABEL, fontsize=FS)
    _bb = _t.get_window_extent(fig.canvas.get_renderer()); _t.remove()
    _kx = ax.transData.inverted().transform([(0.0, 0.0), (_bb.width, 0.0)])[:, 0]
    KEY = (37.0, -50.0, 37.0 + (_kx[1] - _kx[0]), -47.5)   # open Indian Ocean below Africa, inside the globe
    taken = [(KEY[0] - 2, KEY[1] - 9, KEY[2] + 2, KEY[3] + 8)]

    dots = {}
    for r in basins:
        rr, cc = inside_point(grid == painted[r['name']])
        dots[r['name']] = np.array([float(v) for v in P(-180.0 + (cc + 0.5) * deg, 90.0 - (rr + 0.5) * deg)])
    allpts = np.array(list(dots.values()))

    crowd = {n: int((np.hypot(*(allpts - p).T) < 30).sum()) for n, p in dots.items()}
    order = sorted(dots, key=lambda n: (-crowd[n], num[n]))
    # the constrained basins choose first, while their few allowed positions are still free
    order = [n for n in order if n in ONLY_UP | STRAIGHT_ONLY] + [n for n in order if n not in ONLY_UP | STRAIGHT_ONLY]

    # Labels sit on a fixed set of rows and their leader ends on a 5 degree grid, so
    # neighbouring labels share baselines and edges and read as tidy stacks. Each leader
    # leaves its dot vertically and meets its label horizontally along one smooth curve,
    # and each label hangs off the end of its leader to the left or right, never at an angle
    ROW = 8.0
    rows_y = np.arange(Y_S + 6.0, YLIM[1] - 4.0 + 1e-9, ROW)
    GAP, PAD = 1.2, 1.0

    def curve(p, q, n=16):
        c = np.array([p[0], q[1]])
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - t) ** 2 * p + 2 * (1 - t) * t * c + t ** 2 * q

    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) -
                       (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    def poly_cross(A, B):
        if (A[:, 0].max() < B[:, 0].min() or B[:, 0].max() < A[:, 0].min() or
                A[:, 1].max() < B[:, 1].min() or B[:, 1].max() < A[:, 1].min()):
            return False
        p1, p2 = A[:-1, None, :], A[1:, None, :]
        q1, q2 = B[None, :-1, :], B[None, 1:, :]
        return bool(np.any((orient(p1, p2, q1) * orient(p1, p2, q2) < 0) &
                           (orient(q1, q2, p1) * orient(q1, q2, p2) < 0)))

    def in_box(pts, b):
        return bool(((pts[:, 0] > b[0]) & (pts[:, 0] < b[2]) &
                     (pts[:, 1] > b[1]) & (pts[:, 1] < b[3])).any())

    # ---- the northern Eurasian cluster: exact search over short straight leaders ----------
    import time

    def cluster_candidates(n, max_len, box_paint):
        p = dots[n]
        label = DISPLAY.get(n, n)
        w, h = text_size(label)
        own = painted[n]
        out = []

        def push(length, q, box, anchor, group):
            if box[0] < XLIM[0] or box[2] > XLIM[1] or box[1] < YLIM[0] or box[3] > YLIM[1]:
                return
            if painted_fraction(*box) > box_paint:
                return
            if any(box[0] - PAD < b[2] and b[0] < box[2] + PAD and
                   box[1] - PAD < b[3] and b[1] < box[3] + PAD for b in taken):
                return
            if any(box[0] - 1.5 < d[0] < box[2] + 1.5 and box[1] - 1.5 < d[1] < box[3] + 1.5
                   for d in allpts):
                return
            pts = np.linspace(p, q, 16)
            if any(in_box(pts[1:], b) for b in taken):
                return
            if any(poly_cross(pts, c_) or in_box(c_, box) for c_ in cluster_curves):
                return
            if any(np.hypot(*(pts[2:] - d).T).min() < 1.5 for k, d in dots.items() if k != n):
                return
            lp = np.mean([painted_fraction(x - .1, y - .1, x + .1, y + .1, own=own) for x, y in pts[2:]])
            # a label lying across the outline of the globe is penalised, so at 9 pt it sits
            # wholly inside or wholly outside the frame
            _bx = np.array([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]), (box[0], box[1])])
            edge = 40.0 if poly_cross(_bx, OUTLINE_XY) else 0.0
            out.append((length + 150.0 * lp + edge, q, box, pts, anchor, label, group))

        steps = np.arange(3.0, max_len + 1e-9, 1.5)
        allowed_h = n not in ONLY_UP
        for d in steps:
            for sgn in ((1, -1) if allowed_h else ()):
                q = np.array([p[0] + sgn * d, p[1]])
                x0 = q[0] + GAP if sgn > 0 else q[0] - GAP - w
                push(d, q, (x0, p[1] - h / 2, x0 + w, p[1] + h / 2), (x0, p[1], 'left', 'center'),
                     ('h', sgn, 0))
            for sgn in ((1,) if n in ONLY_UP else (1, -1)):
                q = np.array([p[0], p[1] + sgn * d])
                y0 = q[1] + GAP * 0.5 if sgn > 0 else q[1] - GAP * 0.5 - h
                push(d, q, (p[0] - w / 2, y0, p[0] + w / 2, y0 + h), (p[0], y0 + h / 2, 'center', 'center'),
                     ('v', sgn, 0))
                for side in (1, -1):
                    x0 = q[0] + GAP if side > 0 else q[0] - GAP - w
                    push(d + 1.0, q, (x0, q[1] - h / 2, x0 + w, q[1] + h / 2), (x0, q[1], 'left', 'center'),
                         ('v', sgn, side))
        out.sort(key=lambda c: c[0])
        keep, per = [], {}
        for c in out:
            if per.get(c[6], 0) < 6:
                keep.append(c)
                per[c[6]] = per.get(c[6], 0) + 1
        return keep

    def clashes(a, b):
        _, qa, ba, pa, _, _, _ = a
        _, qb, bb, pb, _, _, _ = b
        if ba[0] < bb[2] + PAD and bb[0] < ba[2] + PAD and ba[1] < bb[3] + PAD and bb[1] < ba[3] + PAD:
            return True
        return in_box(pa[1:], bb) or in_box(pb[1:], ba) or poly_cross(pa, pb)

    def solve_cluster(members, max_len, box_paint, limit=90.0):
        names = sorted(members, key=lambda n: dots[n][0])
        cand = {n: cluster_candidates(n, max_len, box_paint) for n in names}
        if any(not cand[n] for n in names):
            return None, cand
        bad = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for ia, ca in enumerate(cand[a]):
                    for ib, cb in enumerate(cand[b]):
                        if clashes(ca, cb):
                            bad.add((a, ia, b, ib))
                            bad.add((b, ib, a, ia))
        rest = np.cumsum([cand[n][0][0] for n in names][::-1])[::-1].tolist() + [0.0]
        best = [np.inf, None]
        t0 = time.time()

        def dfs(k, chosen, cost):
            if time.time() - t0 > limit or cost + rest[k] >= best[0]:
                return
            if k == len(names):
                best[0], best[1] = cost, dict(chosen)
                return
            n = names[k]
            for i, c in enumerate(cand[n]):
                if any((n, i, m, j) in bad for m, j in chosen.items()):
                    continue
                chosen[n] = i
                dfs(k + 1, chosen, cost + c[0])
                del chosen[n]

        dfs(0, {}, 0.0)
        return best, cand

    cluster_placed, cluster_curves = {}, []
    for members in CLUSTERS:
        for max_len, box_paint in ((CLUSTER_MAX, 0.02), (40.0, 0.02), (50.0, 0.06)):
            best, cand = solve_cluster(members, max_len, box_paint)
            if best is not None and best[1] is not None:
                print('  %s ... placed with leaders up to %.0f degrees, total cost %.0f'
                      % (members[0], max_len, best[0]))
                for n, i in best[1].items():
                    c, q, box, pts, anchor, label, _ = cand[n][i]
                    cluster_placed[n] = (q, box, label, pts, anchor)
                    taken.append(box)
                    cluster_curves.append(pts)
                break
            print('  %s ...: no placement with leaders up to %.0f degrees' % (members[0], max_len))
    order = [n for n in order if n not in cluster_placed]

    base_taken = list(taken)
    for attempt in range(8):
      taken = list(base_taken)
      placed = dict(cluster_placed)
      curves = list(cluster_curves)
      failed = []
      for n in order:
        p = dots[n]
        label = DISPLAY.get(n, n)
        w, h = text_size(label)
        own = painted[n]
        cands = []

        def add(cost0, q, box, pts, anchor):
            if box[0] < XLIM[0] or box[2] > XLIM[1] or box[1] < YLIM[0] or box[3] > YLIM[1]:
                return
            if painted_fraction(*box) > 0.02:
                return
            # a label over another basin hides it;
            # a leader over another basin reads as pointing at it
            lp = np.mean([painted_fraction(x - .1, y - .1, x + .1, y + .1, own=own)
                          for x, y in pts[2:]])
            cost = cost0 + 600.0 * painted_fraction(*box) + 250.0 * lp
            cands.append((cost, q, box, pts, anchor))

        # straight leaders first: horizontal, with the label beside its end, or vertical, with
        # the label centred over or under it. They cost less than a curve of the same length,
        # so a curve is used only where no straight leader finds room
        STRAIGHT = 0.6
        for dx in (range(3, 60, 2) if n not in ONLY_UP else ()):
            for sgn in (1, -1):
                q = np.array([p[0] + sgn * dx, p[1]])
                x0 = q[0] + GAP if sgn > 0 else q[0] - GAP - w
                add(STRAIGHT * dx, q, (x0, p[1] - h / 2, x0 + w, p[1] + h / 2),
                    np.linspace(p, q, 16), (x0, p[1], 'left', 'center'))
        for dy in range(3, 50, 2):
            for sgn in ((1,) if n in ONLY_UP else (1, -1)):
                q = np.array([p[0], p[1] + sgn * dy])
                y0 = q[1] + GAP * 0.5 if sgn > 0 else q[1] - GAP * 0.5 - h
                add(STRAIGHT * dy, q, (p[0] - w / 2, y0, p[0] + w / 2, y0 + h),
                    np.linspace(p, q, 16), (p[0], y0 + h / 2, 'center', 'center'))
        # curves: leave the dot vertically and meet the label horizontally, the label on a
        # fixed row and the leader end on a 5 degree grid so neighbouring labels line up
        for yr in (rows_y if n not in STRAIGHT_ONLY | ONLY_UP else ()):
            if abs(yr - p[1]) > 60 or abs(yr - p[1]) < 3:
                continue
            for dx in range(3, 100, 4):
                for sgn in (1, -1):
                    qx = round((p[0] + sgn * dx) / 5.0) * 5.0
                    if sgn * (qx - p[0]) < 2.5:
                        continue
                    q = np.array([qx, yr])
                    x0 = qx + GAP if sgn > 0 else qx - GAP - w
                    add(abs(qx - p[0]) + abs(yr - p[1]), q, (x0, yr - h / 2, x0 + w, yr + h / 2),
                        curve(p, q), (x0, yr, 'left', 'center'))
        cands.sort(key=lambda c: c[0])
        best = None
        for cost, q, box, pts, anchor in cands:
            if any(box[0] < b[2] + PAD and b[0] < box[2] + PAD and
                   box[1] < b[3] + PAD and b[1] < box[3] + PAD for b in taken):
                continue
            if any(box[0] - 1.5 < d[0] < box[2] + 1.5 and box[1] - 1.5 < d[1] < box[3] + 1.5
                   for d in allpts):
                continue
            if any(in_box(pts[1:], b) for b in taken):
                continue
            if any(in_box(c_, box) for c_ in curves):
                continue
            if any(poly_cross(pts, c_) for c_ in curves):
                continue
            if any(np.hypot(*(pts[2:] - d).T).min() < 1.5 for k, d in dots.items() if k != n):
                continue
            best = (q, box, pts, anchor)
            break
        if best is None:
            failed.append(n)
            continue
        q, box, pts, anchor = best
        placed[n] = (q, box, label, pts, anchor)
        taken.append(box)
        curves.append(pts)
      if not failed:
        break
      order = failed + [n for n in order if n not in failed]
    for n in failed:
        print('  could not place %s' % n)

    # ---- touch-ups after placement: only the named labels move, everything else stays -----
    # a vertical leader is shortened to the lowest end whose label still clears every other
    # label, leader, dot and basin
    LOWER = ['Lena']
    for n in LOWER:
        if n not in placed:
            continue
        q, box, label, pts, anc = placed[n]
        p = dots[n]
        if abs(q[0] - p[0]) > 1e-9:
            continue
        sgn = np.sign(q[1] - p[1])
        w, h = box[2] - box[0], box[3] - box[1]
        others = [v for k, v in placed.items() if k != n]
        for dy in np.arange(3.0, abs(q[1] - p[1]) + 1e-9, 0.5):
            q2 = np.array([p[0], p[1] + sgn * dy])
            if anc[2] == 'center':
                y0 = q2[1] + GAP * 0.5 if sgn > 0 else q2[1] - GAP * 0.5 - h
                b2 = (box[0], y0, box[2], y0 + h)
                a2 = (anc[0], y0 + h / 2, 'center', 'center')
            else:
                b2 = (box[0], q2[1] - h / 2, box[2], q2[1] + h / 2)
                a2 = (anc[0], q2[1], anc[2], anc[3])
            pts2 = np.linspace(p, q2, 16)
            if painted_fraction(*b2) > 0.02:
                continue
            if any(b2[0] < o[1][2] + PAD and o[1][0] < b2[2] + PAD and
                   b2[1] < o[1][3] + PAD and o[1][1] < b2[3] + PAD for o in others):
                continue
            if any(b2[0] < k_[2] + PAD and k_[0] < b2[2] + PAD and
                   b2[1] < k_[3] + PAD and k_[1] < b2[3] + PAD for k_ in taken[:1]):
                continue
            if any(b2[0] - 1.5 < d[0] < b2[2] + 1.5 and b2[1] - 1.5 < d[1] < b2[3] + 1.5 for d in allpts):
                continue
            if any(in_box(o[3], b2) or in_box(pts2[1:], o[1]) or poly_cross(pts2, o[3]) for o in others):
                continue
            print('  %s lowered: leader %.1f -> %.1f degrees' % (n, abs(q[1] - p[1]), dy))
            placed[n] = (q2, b2, label, pts2, a2)
            break

    for n, (q, box, label, pts, anchor) in placed.items():
        p = dots[n]
        straight = abs(q[0] - p[0]) < 1e-9 or abs(q[1] - p[1]) < 1e-9
        line = np.linspace(p, q, 2) if straight else curve(p, q, n=60)
        ax.plot(line[:, 0], line[:, 1], lw=0.6, color='#1A1A1A', solid_capstyle='butt', zorder=5)
        ax.text(anchor[0], anchor[1], label, fontsize=FS, ha=anchor[2], va=anchor[3],
                color='#111111', zorder=7, path_effects=HALO)
    for n, p in dots.items():
        ax.plot([p[0]], [p[1]], 'o', ms=2.6, mfc='k', mec='white', mew=0.5, zorder=6)

    cax = fig.add_axes([0, 0, 1, 1])
    (fx0, fy0), (fx1, fy1) = fig.transFigure.inverted().transform(
        ax.transData.transform([(KEY[0], KEY[1]), (KEY[2], KEY[3])]))
    cax.set_position([fx0, fy0, fx1 - fx0, fy1 - fy0])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation='horizontal')
    cb.outline.set_visible(False)
    cax.set_xticks([])
    cax.text(0.0, 1.6, CB_LABEL, transform=cax.transAxes,
             ha='left', va='bottom', fontsize=FS, color='#333333')
    cax.text(0.0, -0.7, '%d$\\times$' % round(vals.min()), transform=cax.transAxes,
             ha='left', va='top', fontsize=FS, color=CMAP(0.06), weight='bold')
    cax.text(1.0, -0.7, '%d$\\times$' % round(vals.max()), transform=cax.transAxes,
             ha='right', va='top', fontsize=FS, color=CMAP(0.94), weight='bold')

    # ------------------------------------------------------------ b: the scatter ------
    ax2 = mmax(fig, SC_X, SC_Y, SC_W, SC_H)
    ax2.set_xscale('log')
    ax2.set_yscale('linear')   # the speedups span 39 to 137, which reads better without exponents
    ax2.set_xlim(8.0e4, 1.2e8) if SHOW_GLOBAL else ax2.set_xlim(9.0e4, 2.6e6)
    _sp = [r['speedup'] for r in rows]          # the axis follows the data, which moved when
    ax2.set_ylim(0, max(_sp) * 1.30)               # headroom for the whole-globe label over its star
    ax2.set_yticks([0, 50, 100, 150])
    MS = 7.0
    for r in basins:
        ax2.plot(r['cells'], r['speedup'], 'o', ms=MS, color=colour[r['name']],
                 mec=OUTLINE, mew=.5, zorder=3)
    if glob:
        ax2.plot(glob['cells'], glob['speedup'], '*', ms=18, color='#FFFFFF',
                 mec=OUTLINE, mew=.8, zorder=3)
    ax2.tick_params(labelbottom=True)   # the same tick numbers as panel c
    ax2.set_ylabel('Speedup over CPU-serial')
    ax2.grid(False, which='both')
    panel_letter(fig, 1.0, SC_Y - 2.5, 'b', 'Speedup against the size of the routed domain', centre=SC_X + SC_W / 2)
    fig.canvas.draw()

    fx = 1.0 / SC_W
    fy = 1.0 / SC_H
    mk = MS * 25.4 / 72 / 2 + 0.4                     # marker radius [mm]

    def frac(x, y):
        return ax2.transLimits.transform(ax2.transScale.transform((x, y)))

    def label_wh(s):
        t = ax2.text(0, 0, s, fontsize=FS, transform=ax2.transAxes)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        t.remove()
        return bb.width / fig.dpi * 25.4 * fx, bb.height / fig.dpi * 25.4 * fy

    taken2 = []
    for r in basins:
        px, py = frac(r['cells'], r['speedup'])
        taken2.append((px - mk * fx, py - mk * fy, px + mk * fx, py + mk * fy))
    if glob:
        gx, gy = frac(glob['cells'], glob['speedup'])
        taken2 += [(gx - 3.2 * fx, gy - 3.2 * fy, gx + 3.2 * fx, gy + 3.2 * fy)]

    bands = []                 # corridors a leader has already used, kept clear of labels

    def free(box, pad=0.6, extra=()):
        x0, y0, x1, y1 = box
        for a0, b0, a1, b1 in list(taken2) + list(extra):
            if x0 < a1 + pad * fx and a0 < x1 + pad * fx and y0 < b1 + pad * fy and b0 < y1 + pad * fy:
                return False
        return 0 <= x0 and x1 <= 1 and 0 <= y0 and y1 <= 1

    def leader(px, py, tx, ty):
        # the line starts at the middle of the marker, on the dot, as the map panel does
        xs, ys = [px, tx], [py, ty]
        ax2.add_line(Line2D(xs, ys, transform=ax2.transAxes, lw=0.6, color='#1A1A1A',
                            solid_capstyle='butt', zorder=6))   # above the markers, under the dots

    # one straight leader a label, horizontal or vertical, and every leader reserves the
    # strip it crosses so the next label cannot sit on it. Neighbouring points start from
    # opposite sides, which keeps a cluster from stringing its labels along one line
    def reserve_leader(px, py, tx, ty):
        x0, x1 = sorted((px, tx))
        y0, y1 = sorted((py, ty))
        wide = 1.4                             # a band across the leader, so two neighbours do
        if abs(ty - py) < 1e-9:                # not both reach out on the same line
            bands.append((x0 - 0.3 * fx, y0 - wide * fy, x1 + 0.3 * fx, y1 + wide * fy))
        else:
            bands.append((x0 - wide * fx, y0 - 0.3 * fy, x1 + wide * fx, y1 + 0.3 * fy))

    FORCE_B = {1: 'd', 2: 'r2', 3: 'l', 4: 'd', 5: 'l2', 6: 'l', 7: 'l', 8: 'd', 9: 'd2',
               10: 'u', 11: 'u', 12: 'l', 13: 'u2', 14: 'd', 15: 'l', 16: 'd', 17: 'u',
               18: 'd', 19: 'u2', 20: 'u', 21: 'd', 22: 'u2', 23: 'd', 24: 'u', 25: 'd2',
               26: 'r'}      # every label is placed by hand now, so a change to one of
    # them does not send the automatic search off to rearrange the rest
    order = sorted(basins, key=lambda r: (num[r['name']] not in FORCE_B, -r['cells']))
    # the ones placed by hand go first, so their side is still free when they ask for it
    # only the basins the text or the eye needs are named: the two smallest, the largest,
    # and a few well-known rivers spread along the curve
    LABEL_B = {'Irrawaddy', 'Xi', 'Danube', 'Mekong', 'Mississippi', 'Congo', 'Amazon'}
    for i, r in enumerate(order):
        if r['name'] not in LABEL_B:
            continue
        px, py = frac(r['cells'], r['speedup'])
        label = DISPLAY.get(r['name'], r['name'])
        w, h = label_wh(label)
        dirs = ['u', 'r', 'd', 'l'] if i % 2 == 0 else ['d', 'l', 'u', 'r']
        _f = FORCE_B.get(num[r['name']])
        if _f:
            dirs = [_f] + [k for k in dirs if k != _f]
        steps = [(mk + g) for g in (1.0, 2.4, 4.0, 5.8, 7.6, 9.6, 12.0, 15.0, 18.0)]
        if _f:                      # a side asked for by hand goes exactly there, at the
            _si = int(_f[1:]) if len(_f) > 1 else 0   # an optional second digit pushes the
            _f = _f[0]                                # label further out, to clear a neighbour
            d = steps[_si]
            dx, dy = d * fx, d * fy
            cands = {'r': (px + dx, py - h / 2, px + dx, py),
                     'l': (px - dx - w, py - h / 2, px - dx, py),
                     'u': (px - w / 2, py + dy, px, py + dy),
                     'd': (px - w / 2, py - dy - h, px, py - dy)}
            x0, y0, tx, ty = cands[_f]
            ax2.text(x0 + w / 2, y0 + h / 2, label, transform=ax2.transAxes, ha='center',
                     va='center', fontsize=FS, color='#1A1A1A', zorder=5, path_effects=HALO)
            taken2.append((x0, y0, x0 + w, y0 + h))
            leader(px, py, tx, ty)
            reserve_leader(px, py, tx, ty)
            continue
        plan = [(d, k) for d in steps for k in dirs]
        for d, k in plan:
            dx, dy = d * fx, d * fy
            placed = False
            if True:
                cands = {'r': (px + dx, py - h / 2, px + dx, py),
                         'l': (px - dx - w, py - h / 2, px - dx, py),
                         'u': (px - w / 2, py + dy, px, py + dy),
                         'd': (px - w / 2, py - dy - h, px, py - dy)}
                x0, y0, tx, ty = cands[k]
                lx0, lx1 = sorted((px, tx)); ly0, ly1 = sorted((py, ty))
                path = (lx0 + 0.2 * fx, ly0 + 0.2 * fy, lx1 - 0.2 * fx, ly1 - 0.2 * fy)
                # the label keeps off markers, labels and the corridors other leaders use,
                # and its own line may not cross a marker or a label
                if free((x0, y0, x0 + w, y0 + h), extra=bands) and free(path, pad=0.0):
                    ax2.text(x0 + w / 2, y0 + h / 2, label, transform=ax2.transAxes, ha='center',
                             va='center', fontsize=FS, color='#1A1A1A', zorder=5, path_effects=HALO)
                    taken2.append((x0, y0, x0 + w, y0 + h))
                    if abs(tx - px) > 2.6 * fx or abs(ty - py) > 2.6 * fy:
                        leader(px, py, tx, ty)
                    reserve_leader(px, py, tx, ty)   # also for a label that needs no line, so
                    # the next one does not line up beside it
                    placed = True
            if placed:
                break
        else:
            # two of the smallest basins sit almost on top of each other at the corner, so the
            # last resort drops the corridor rule and keeps only the label-on-label check
            for d, k in plan:
                dx, dy = d * fx, d * fy
                if True:
                    cands = {'r': (px + dx, py - h / 2, px + dx, py),
                             'l': (px - dx - w, py - h / 2, px - dx, py),
                             'u': (px - w / 2, py + dy, px, py + dy),
                             'd': (px - w / 2, py - dy - h, px, py - dy)}
                    x0, y0, tx, ty = cands[k]
                    if free((x0, y0, x0 + w, y0 + h)):
                        ax2.text(x0 + w / 2, y0 + h / 2, label, transform=ax2.transAxes,
                                 ha='center', va='center', fontsize=FS, color='#1A1A1A',
                                 zorder=5, path_effects=HALO)
                        taken2.append((x0, y0, x0 + w, y0 + h))
                        if abs(tx - px) > 2.6 * fx or abs(ty - py) > 2.6 * fy:
                            leader(px, py, tx, ty)
                        reserve_leader(px, py, tx, ty)
                        print('  %s placed without the corridor rule' % r['name'])
                        break
    for r in basins:            # the leader ends on the same black dot in a white ring the map
        if r['name'] not in LABEL_B:      # only the named points carry the dot a leader ends on
            continue
        ax2.plot([r['cells']], [r['speedup']], 'o', ms=2.6, mfc='k', mec='white', mew=0.5,
                 zorder=7)       # panel uses, drawn above everything else

    if glob:
        ax2.text(gx, gy + 4.2 * fy, 'Whole globe, %.0f\u00d7' % glob['speedup'], transform=ax2.transAxes, ha='center',
                 va='bottom', fontsize=FS, color='#1A1A1A', zorder=5)

    # ------------------------------------------- c: the wall-clock of SURGE ------
    ax3 = mmax(fig, SC_X, C_Y, SC_W, C_H)
    ax3.set_xscale('log'); ax3.set_yscale('log')
    ax3.set_xlim(*ax2.get_xlim())
    cells = np.array([r['cells'] for r in basins], float)
    tms = np.array([r['surge_ms'] for r in basins], float)
    t0 = float(np.median(tms[cells < 2.0e5]))                 # the floor of the small basins
    big = cells >= 2.0e5
    per = float(cells[big] @ tms[big] / (cells[big] @ cells[big]))   # least squares through the origin, ms per cell once the GPU is full
    xs = np.logspace(np.log10(ax3.get_xlim()[0]), np.log10(ax3.get_xlim()[1]), 50)
    GUIDE = '#8A8F96'
    ax3.plot(xs, np.full_like(xs, t0), ls=(0, (4, 2)), lw=0.8, color=GUIDE, zorder=1)
    ax3.plot(xs, per * xs, ls=(0, (4, 2)), lw=0.8, color=GUIDE, zorder=1)
    for r in basins:
        ax3.plot(r['cells'], r['surge_ms'], 'o', ms=MS * 0.8, color=colour[r['name']],
                 mec=OUTLINE, mew=.5, zorder=3)
    if glob:
        ax3.plot(glob['cells'], glob['surge_ms'], '*', ms=15, color='#FFFFFF', mec=OUTLINE,
                 mew=.8, zorder=3)
        # labeled as in b, centered above the star
        ax3.annotate('Whole globe', (glob['cells'], glob['surge_ms']), textcoords='offset points',
                     xytext=(0, 9), ha='center', va='bottom', fontsize=FS, color='#1A1A1A', zorder=5)
    ax3.set_ylim(0.25, 600)
    # written along the sloping line, in the empty stretch between the largest basin and the globe
    _xa, _xb = 2.0e6, 2.0e7
    _pa, _pb = ax3.transData.transform([(_xa, per * _xa), (_xb, per * _xb)])
    _ang = np.degrees(np.arctan2(_pb[1] - _pa[1], _pb[0] - _pa[0]))
    _xm = 7.0e6      # centre of the label, clear of the basins and of the star of the whole globe
    ax3.annotate('Nearly proportional, %.2f ms per million river cells' % (per * 1e6), (_xm, per * _xm),
                 textcoords='offset points', xytext=(-2.0 * np.sin(np.radians(_ang)), 2.0 * np.cos(np.radians(_ang))),
                 ha='center', va='bottom', rotation=_ang, rotation_mode='anchor',
                 fontsize=FS, color='#555555')
    # two regimes split where the wall-clock per cell stops falling, shaded in b and c alike
    SPLIT, TINT_L, TINT_R = t0 / per, '#F2EFE8', '#E9EFF4'     # split where the two lines cross
    for a in (ax2, ax3):
        x0, x1 = a.get_xlim()
        a.axvspan(x0, SPLIT, color=TINT_L, lw=0, zorder=0)
        a.axvspan(SPLIT, x1, color=TINT_R, lw=0, zorder=0)
        a.set_xlim(x0, x1)
    ax3.text(np.sqrt(ax3.get_xlim()[0] * SPLIT), 150, 'Most time spent\nstarting each\nupdate, GPU not\nfully used', ha='center', va='center', linespacing=1.1,
             fontsize=FS, color='#555555')
    ax3.text(SPLIT * 2.8, 150, 'Most time spent\nupdating the\nriver cells, GPU\nmostly busy', ha='center', va='center', linespacing=1.1,
             fontsize=FS, color='#555555')
    nx = t0 / per                                         # where the cells cost as much as the floor
    # the label of the horizontal line, inside the left tint, with an arrow down to the line
    _xc = np.sqrt(ax3.get_xlim()[0] * SPLIT)
    ax3.text(_xc, 2.9, 'Nearly\nindependent of\nriver cells, %.2f ms' % t0, ha='center', va='bottom',
             linespacing=1.1, fontsize=FS, color='#555555')
    ax3.annotate('', xy=(_xc, t0 * 1.12), xytext=(_xc, 2.6),
                 arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#555555', mutation_scale=7,
                                 shrinkA=0, shrinkB=0), zorder=4)
    ax3.plot([nx, nx], [0.25, t0], ls=(0, (1, 1.5)), lw=0.7, color='#555555', zorder=1)
    ax3.text(nx * 1.07, 0.34, 'Lines cross at about %.1f million river cells' % (nx / 1e6), ha='left', va='center',
             fontsize=FS, color='#555555')
    ax3.set_xlabel('River cells in the routed domain')
    ax3.set_ylabel('SURGE wall-clock\nper time step [ms]')
    ax3.grid(False, which='both')
    panel_letter(fig, 1.0, C_Y - 2.5, 'c', 'Wall-clock of SURGE per time step', centre=SC_X + SC_W / 2)
    print('floor %.3f ms, %.3f ns per cell, crossing at %.2e cells' % (t0, per * 1e6, t0 / per))

    from paper_style import myriad_ticks; myriad_ticks(fig)   # tick numbers in Myriad Pro, symbols in STIX

    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=600)
    print('-> %s.{pdf,png}  (%.0f x %.0f mm)' % (stem, W, H))


if __name__ == '__main__':
    main()
