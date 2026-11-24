"""Figure 3 of the manuscript: how the parallel iteration closes the gap along one rain path, drawn
on the river network of Figure 1 with relief. An oblique slab with faceted
side walls, a curved arrow carrying the water into the upstream end of the headwater reach, the colours, opacities and widths of the
reaches (red fading as 0.25 + 0.75 * 0.5^n with the position n of the reach on the path, each reach keeping its shade until the correction reaches it, width growing
downstream), the dots at the confluences on the rain path, the white outlet, the panel titles,
key and note. The network, reaches and basin outline come from fig_method_seine.py.
The relief is built from the network: the river bed rises with the flow distance to the outlet,
and the land rises away from the nearest river towards the divide, so every reach lies in a valley.
Run:  SURGE_DATA=<folder with routing_static_05min.npz> python fig_isometric_terrain.py
Output: paper/figures/<FIG_OUT or fig_isometric>.{png,pdf}; copied to SURGE_FIGURE_COPY when set
"""
import os, sys, shutil
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'fig_method_seine.py')).read()
exec(src[:src.index("cmap = plt.get_cmap(")])        # network, reaches, basin outline of Figure 1
plt.close('all')

# ---- colours
BLUE = (0.184, 0.427, 0.710); RED = (0.753, 0.224, 0.169); LIGHT = (0.478, 0.639, 0.812)
RAIN = (0.231, 0.373, 0.561); TOPF = np.array([0.89, 0.925, 0.875]); EDGE = (0.49, 0.529, 0.49)
WALL = np.array([0.546, 0.59, 0.546]); INK = '#333333'

# ---- reaches as smooth lines, the rain path, flow distance to the outlet
lines = [chaikin(simplify(np.c_[X[ch], Y[ch]]), 3) for ch in reaches]
down_of = {i: j for j, ups in enumerate(ups_of) for i in ups}
def path_from(i):
    p = [i]
    while p[-1] in down_of:
        p.append(down_of[p[-1]])
    return p
path = max((path_from(h) for h in range(R) if not ups_of[h]), key=len)
pos = {r: k for k, r in enumerate(path)}
seglen = [np.hypot(*np.diff(l, axis=0).T).sum() for l in lines]
dist_end = {}
def dend(i):                                                  # flow distance from the end of reach i to the outlet
    if i not in dist_end:
        dist_end[i] = 0.0 if i not in down_of else seglen[down_of[i]] + dend(down_of[i])
    return dist_end[i]
rp, rz = [], []
for i, l in enumerate(lines):
    along = np.r_[0, np.cumsum(np.hypot(*np.diff(l, axis=0).T))]
    rp.append(l); rz.append(dend(i) + seglen[i] - along)       # flow distance at every point of the reach
rp = np.vstack(rp); fd = np.clip(np.concatenate(rz), 0, None)
tree = cKDTree(rp)

Wd = bX.max() - bX.min()
HILL_A, HILL_L = float(os.environ.get('FIG3_HA', '0.10')), float(os.environ.get('FIG3_HL', '0.12'))
FDMAX = fd.max()
def zfun(x, y):
    """Elevation: river bed rising with flow distance (concave profile), land rising away from the river."""
    d, k = tree.query(np.c_[np.ravel(x), np.ravel(y)])
    bed = 0.10 * Wd * (fd[k] / FDMAX) ** 0.8
    rough = 1 + 0.25 * np.sin(5.1 * np.ravel(x) / Wd * np.pi + 1.3) * np.sin(4.3 * np.ravel(y) / Wd * np.pi)
    hill = HILL_A * Wd * (1 - np.exp(-d / (HILL_L * Wd))) * rough
    return (bed + hill).reshape(np.shape(x))

# ---- oblique view: seen from the outlet side, tilted
SQ, EZ = float(os.environ.get('FIG3_SQ', '0.55')), float(os.environ.get('FIG3_EZ', '0.85'))
AZ = np.deg2rad(float(os.environ.get('FIG3_AZ', '45')))       # viewer at the lower left, looking up-right at AZ from the x axis
DV = np.array([np.cos(AZ), np.sin(AZ)])                       # view direction on the map
RV = np.array([np.sin(AZ), -np.cos(AZ)])                      # screen right on the map
def P(x, y, z):
    x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)
    return np.c_[x * RV[0] + y * RV[1], SQ * (x * DV[0] + y * DV[1]) + EZ * z]
ZB = -0.09 * Wd                                               # base of the slab
LIGHTV = np.array([-0.6, 0.5, 0.62]); LIGHTV /= np.linalg.norm(LIGHTV)

# terrain grid inside the basin
NG = 150
gx = np.linspace(bX.min(), bX.max(), NG); gy = np.linspace(bY.min(), bY.max(), int(NG * (bY.ptp() / bX.ptp())))
GX, GY = np.meshgrid(gx, gy)
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
GZ = gaussian_filter(zfun(GX, GY), 1.6)                     # smooth the creases between valleys
_zi = RegularGridInterpolator((gy, gx), GZ, bounds_error=False, fill_value=None)
def zs(x, y):                                                # elevation of the smoothed surface
    return _zi(np.c_[np.ravel(y), np.ravel(x)]).reshape(np.shape(x))
inside = MPath(np.c_[bX, bY]).contains_points(np.c_[GX.ravel(), GY.ravel()]).reshape(GX.shape)
dzdx = np.gradient(GZ, gx, axis=1); dzdy = np.gradient(GZ, gy, axis=0)
nrm = np.dstack([-dzdx, -dzdy, np.ones_like(GZ)]); nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
shade = np.clip(nrm @ LIGHTV, 0, 1)
zn = (GZ - GZ[inside].min()) / np.ptp(GZ[inside])
quads, cols, depth = [], [], []
for r in range(len(gy) - 1):
    for c in range(len(gx) - 1):
        if not (inside[r, c] and inside[r + 1, c] and inside[r, c + 1] and inside[r + 1, c + 1]):
            continue
        xs = [GX[r, c], GX[r, c + 1], GX[r + 1, c + 1], GX[r + 1, c]]
        ys = [GY[r, c], GY[r, c + 1], GY[r + 1, c + 1], GY[r + 1, c]]
        zq = [GZ[r, c], GZ[r, c + 1], GZ[r + 1, c + 1], GZ[r + 1, c]]
        quads.append(P(xs, ys, zq))
        s = shade[r, c]
        col = TOPF * (0.34 + 0.80 * s) + np.array([0.03, 0.02, -0.01]) * zn[r, c]
        col = 0.90 * col + 0.10 * np.ones(3)                   # light tones, so the rivers stay on top
        cols.append(np.clip(col, 0, 1)); depth.append(GX[r, c] * DV[0] + GY[r, c] * DV[1])
order = np.argsort(depth)[::-1]                               # back to front
quads = [quads[i] for i in order]; cols = [cols[i] for i in order]

# the basin edge: top line on the terrain and the walls below it that face the viewer
ring = np.c_[bX, bY]
ring = np.vstack([ring, ring[:1]])
dense = []
for a, b in zip(ring[:-1], ring[1:]):
    for t in np.linspace(0, 1, 4, endpoint=False):
        dense.append(a + (b - a) * t)
dense = np.array(dense + [ring[0]])
ez = zs(dense[:, 0], dense[:, 1])
area = 0.5 * np.sum(dense[:-1, 0] * dense[1:, 1] - dense[1:, 0] * dense[:-1, 1])   # >0: counter-clockwise
walls = []
for k in range(len(dense) - 1):
    a, b = dense[k], dense[k + 1]
    t = b - a; n = np.array([t[1], -t[0]]) * (1 if area > 0 else -1)   # outward normal
    n = n / (np.hypot(*n) + 1e-12)
    if n @ DV < 0.05:                                         # faces the viewer
        quad = np.vstack([P(a[0], a[1], ez[k]), P(b[0], b[1], ez[k + 1]), P(b[0], b[1], ZB), P(a[0], a[1], ZB)])
        f = 0.80 + 0.30 * max(0.0, -(n @ RV))                             # facets facing left are lit
        walls.append((quad, np.clip(WALL * f, 0, 1)))
top_edge = P(dense[:, 0], dense[:, 1], ez)

amax = accv[mb].max()
def width(i):
    return 0.8 + 1.7 * np.sqrt(accv[reaches[i][len(reaches[i]) // 2]] / amax)

def draw(ax, j):
    """State after iteration j (j = 0: start of the time step)."""
    for quad, col in walls:
        ax.add_patch(Polygon(quad, closed=True, fc=col, ec=col, lw=0.3, zorder=1))
    ax.add_collection(PolyCollection(quads, facecolors=cols, edgecolors='face', linewidths=0.35, zorder=2,
                                     rasterized=True))
    ax.plot(top_edge[:, 0], top_edge[:, 1], color=EDGE, lw=0.6, zorder=3)
    for i, l in enumerate(lines):                              # unaffected reaches first, the rain path on top
        if i in pos:
            continue
        q = P(l[:, 0], l[:, 1], zs(l[:, 0], l[:, 1]) + 0.002 * Wd)
        ax.plot(q[:, 0], q[:, 1], color=LIGHT, lw=width(i), solid_capstyle='round', zorder=4)
    for i in path:
        l = lines[i]
        q = P(l[:, 0], l[:, 1], zs(l[:, 0], l[:, 1]) + 0.002 * Wd)
        if pos[i] < j:
            col, a = BLUE, 1.0
        else:
            col, a = RED, 0.25 + 0.75 * 0.5 ** pos[i]   # a reach keeps its error until the correction reaches it
        ax.plot(q[:, 0], q[:, 1], color=col, alpha=a, lw=width(i), solid_capstyle='round', zorder=5)
        if i in down_of:                                       # dot at the confluence, coloured as the reach below
            nxt = path[pos[i] + 1]
            cn, an = (BLUE, 1.0) if pos[nxt] < j else (RED, 0.25 + 0.75 * 0.5 ** pos[nxt])
            e = q[-1]
            ax.plot([e[0]], [e[1]], 'o', ms=width(nxt) * 1.35, mfc=cn, mec='none', alpha=an, zorder=6)
    o = P(X[seine_root], Y[seine_root], zs(X[seine_root], Y[seine_root]))[0]
    ax.plot([o[0]], [o[1]], 'o', ms=6.2, mfc='white', mec=INK, mew=1.3, zorder=8)
    h = lines[path[0]]                                         # one curved arrow: water entering the rain reach at its upstream end
    q = P(h[:, 0], h[:, 1], zs(h[:, 0], h[:, 1]))
    e = q[0]                                                   # upstream end of the reach
    d = q[min(4, len(q) - 1)] - q[0]; d /= np.hypot(*d)       # flow direction there, on screen
    nrm = np.array([-d[1], d[0]])                              # across the flow, on screen
    NARR = int(os.environ.get('FIG3_NARROW', '3'))
    for k in range(NARR):                                      # parallel curved arrows into the upstream end
        off = (k - (NARR - 1) / 2) * 0.05 * Wd * nrm
        ek = e + off
        sk = ek - d * 0.11 * Wd + np.array([0.0, 0.10 * Wd])     # start above, upstream of the reach
        ax.annotate('', xy=ek, xytext=sk, zorder=9,
                    arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', color=RAIN, lw=0.9,
                                    shrinkA=0, shrinkB=1.5, connectionstyle='arc3,rad=0.45'))
        starts = starts + [sk] if k else [sk]
    if j == 0:                                                 # name the arrows once, left of them, in their colour
        st = np.array(starts)
        ax.text(st[:, 0].min() - 0.035 * Wd, st[:, 1].mean() - 0.02 * Wd, 'Rain', color=RAIN, fontsize=9.0,
                ha='right', va='center', zorder=9)
    ax.set_aspect('equal'); ax.axis('off')

PW = 472.31595 / 72.27
LAYOUT = os.environ.get('FIG3_LAYOUT', 'row')                # 'row' (1 x 4) or 'grid' (2 x 2)
if LAYOUT == 'row':
    fig, axs = plt.subplots(1, 4, figsize=(PW, 46 / 25.4), layout='constrained')
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))          # room above the panels for the titles
else:
    fig, axs = plt.subplots(2, 2, figsize=(PW, 112 / 25.4), layout='constrained')
axs = axs.ravel()
TITLES = ['Before iteration 1', 'After iteration 1', 'After iteration 2', 'After iteration 3']
allp = np.vstack([top_edge, P(dense[:, 0], dense[:, 1], np.full(len(dense), ZB))])
for j, ax in enumerate(axs):
    draw(ax, j)
    ax.set_xlim(allp[:, 0].min() - 0.02 * Wd, allp[:, 0].max() + 0.02 * Wd)
    ax.set_ylim(allp[:, 1].min() - 0.01 * Wd, allp[:, 1].max() + 0.03 * Wd)
    YB = 13.0                                                  # baseline of the letter and of the first title line, above the panel [pt]
    ax.annotate('abcd'[j], (0.0, 1.0), xycoords='axes fraction', xytext=(0, YB), textcoords='offset points',
                fontsize=11, fontweight='bold', ha='left', va='baseline')
    for n, line in enumerate(TITLES[j].split('\n')):           # line by line, so the letter sits on the first line
        ax.annotate(line, (0.5, 1.0), xycoords='axes fraction', xytext=(0, YB - n * 10.5 * 1.15),
                    textcoords='offset points', fontsize=10.5, color=INK, ha='center', va='baseline')   # centred, as in Figure 1
fig.legend(handles=[Line2D([], [], color=BLUE, lw=3.0, solid_capstyle='projecting', label='equals the serial solution'),
                    Line2D([], [], color=RED, lw=3.0, solid_capstyle='projecting', label='still differs by a large amount'),
                    Line2D([], [], color=RED, alpha=0.4, lw=3.0, solid_capstyle='projecting',
                           label='still differs by a small amount'),
                    Line2D([], [], color=LIGHT, lw=2.0, solid_capstyle='projecting', label='unaffected reach')],
           loc='outside lower center', ncol=4, frameon=False, fontsize=9.0, handlelength=1.1, handletextpad=0.4,
           columnspacing=0.9)

FIG = os.path.join(HERE, '..', 'figures')
OUT = os.environ.get('FIG_OUT', 'fig_isometric')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG, OUT + '.' + ext), dpi=600)
COPY = os.environ.get('SURGE_FIGURE_COPY')
if COPY:
    for ext in ('png', 'pdf'):
        shutil.copy(os.path.join(FIG, OUT + '.' + ext), os.path.join(COPY, OUT + '.' + ext))
print('rain path (Figure 1b numbers):', [serial[r] for r in path], '| quads', len(quads), '| saved', OUT)
