"""Helper of fig_iteration_gap.py, which reuses its solver e_numbers(); not a figure of the manuscript.
Run on its own it draws the change of the discharge along one river in one time step,
serial solution against the fully parallel scheme after each iteration. It was panel e of
Figure 1 (fig_method_seine.py) and is drawn here on its own, with the same numbers.
Run:  python fig_wave_attenuation.py
Output: paper/figures/fig_wave_attenuation_bars.{png,pdf}
"""
import os, sys, shutil
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import setup_paper1_style
setup_paper1_style()
PW = 166.0 / 25.4                                   # built at the 166 mm text width


def e_numbers(N=8, DT=4320.0, BETA=0.6, n=0.03, S0=1e-3, P=30.0, Q0=100.0, QIN=200.0, K=3):
    """Equation 7 on a chain of N cells, Courant number 1 at the base flow Q0; the inflow at the
    upstream end rises from Q0 to QIN in this time step. Returns the serial change per unit rise
    and, for iterations 1..K of the fully parallel scheme, the serial solution minus it."""
    ALPHA = (n * P ** (2 / 3) / np.sqrt(S0)) ** BETA
    DX = DT / (ALPHA * BETA * Q0 ** (BETA - 1))     # Cr = 1 at Q0
    def solve(qin, qold):
        rhs = DT / DX * qin + ALPHA * qold ** BETA
        Q = np.maximum(qold, 1e-6)
        for _ in range(30):
            f = DT / DX * Q + ALPHA * Q ** BETA - rhs
            Q = np.maximum(Q - f / (DT / DX + ALPHA * BETA * Q ** (BETA - 1)), 1e-6)
        return Q
    qold = np.full(N, Q0); serial = np.empty(N); up = QIN
    for c in range(N):
        serial[c] = solve(up, qold[c]); up = serial[c]
    q = qold.copy(); left = []
    for _ in range(K):
        q = solve(np.r_[QIN, q[:-1]], qold); left.append((serial - q) / (QIN - Q0))
    assert min(l.min() for l in left) > -1e-9, 'fully parallel scheme above the serial solution'
    return (serial - qold) / (QIN - Q0), left


change, left = e_numbers()
NE = len(change); xe = np.arange(NE)
BLUE, RED, GREY, INK = '#2e79b8', '#C0392B', '#D5D8DC', '#333333'

H = 52.0 / 25.4
XIN, YMAX = -1.6, 1.22                               # inflow bar left of cell 1; bars to scale with it
fig = plt.figure(figsize=(PW, H), layout='none')
top, bot = 1.0 - 1.5 / 25.4 / H, 7.0 / 25.4 / H     # room below for Upstream / Downstream
rows = [([('Serial scheme (Figure 1', 'normal'), ('b', 'bold'), (')', 'normal')], change, BLUE, None)] + \
       [([('Fully parallel scheme (Figure 1', 'normal'), ('d', 'bold'),
          (') after iteration $i=%d$' % (i + 1), 'normal')], left[i], RED, change) for i in range(len(left))]
hrow = (top - bot) / len(rows)
X0, X1 = 0.30, 0.995                                 # provisional; the bars are moved after the labels
axes = []
for r, (lab, v, col, ghost) in enumerate(rows):
    ax = fig.add_axes([X0, top - (r + 1) * hrow + 0.01, X1 - X0, hrow - 0.02])
    if ghost is not None:
        ax.bar(xe, ghost, width=0.56, color=GREY, lw=0, zorder=1)
    ax.bar(xe, np.where(v >= 0.005, v, 0), width=0.56, color=col, lw=0, zorder=2)
    for j in (range(4) if ghost is None else [int(np.argmax(v))]):   # the halving shows in the first row
        ax.text(j, v[j] + 0.04, '%.2f' % v[j], ha='center', va='bottom', fontsize=8, color=col,
                fontweight='bold')
    if r == 0:                                       # the unit rise of the inflow that the bars are scaled by
        ax.bar(XIN, 1.0, width=0.56, facecolor='white', edgecolor=INK, lw=0.8, zorder=2)
        ax.text(XIN, 1.04, '1', ha='center', va='bottom', fontsize=8, color=INK, fontweight='bold')
        ax.annotate('', xy=(-0.38, 0.5), xytext=(XIN + 0.36, 0.5),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.8, mutation_scale=8))
        ax.text(XIN - 0.36, 0.5, 'Rise of the\ninflow', fontsize=8, color=INK, ha='right', va='center', linespacing=1.1)
    ax.axhline(0, color=INK, lw=0.8, zorder=3)
    ax.set_xlim(XIN - 0.6, NE - 0.4); ax.set_ylim(0, YMAX); ax.axis('off')
    if r == 0:                                       # labels on the data instead of a key
        ax.text(NE - 0.4, 0.45, 'Rise of the outflow of each cell, serial\nsolution, per unit rise of the inflow', fontsize=8,
                color=BLUE, ha='right', va='bottom', linespacing=1.1)
    if r == 1:
        ax.text(NE - 0.4, 0.08, 'Serial solution minus the fully\nparallel scheme, in each cell',
                fontsize=8, color=RED, ha='right', va='bottom', linespacing=1.1)
    if r == 2:
        ax.text(NE - 0.4, 0.08, 'Cells where the fully parallel\nscheme equals the serial solution',
                fontsize=8, color='#7f8790', ha='right', va='bottom', linespacing=1.1)
    if r == len(rows) - 1:
        ax.text(-0.3, -0.08, 'Upstream', fontsize=8, color='#555', ha='left', va='top')
        ax.text(NE - 0.7, -0.08, 'Downstream', fontsize=8, color='#555', ha='right', va='top')
    axes.append((ax, lab))

fig.canvas.draw()
_r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
def runs(x, y, parts):
    # one label from several pieces, so the panel letters in it can be bold
    t = None
    for txt, wt in parts:
        t = fig.text(x, y, txt, ha='left', va='bottom', fontsize=8, color=INK, fontweight=wt)
        x = inv.transform((t.get_window_extent(_r).x1, 0))[0]
    return t
XL = 1.0 / 25.4 / PW
labs = [runs(XL, ax.get_position().y0, lab) for ax, lab in axes]
x_end = max(inv.transform((t.get_window_extent(_r).x1, 0))[0] for t in labs)
for ax, _ in axes:                                   # the bars start 3 mm after the longest label
    p = ax.get_position(); ax.set_position([x_end + 3.0 / 25.4 / PW, p.y0, X1 - x_end - 3.0 / 25.4 / PW, p.height])

# centre the drawing left to right, then trim the space above and below to 1 mm
fig.canvas.draw()
img = np.asarray(fig.canvas.buffer_rgba())[..., :3].min(axis=2)
cols = np.nonzero((img < 235).any(axis=0))[0]
dx = ((img.shape[1] - 1 - cols[-1]) - cols[0]) / 2.0 / img.shape[1]
for ax in fig.axes:
    p = ax.get_position(); ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
for t in fig.texts:
    x, y = t.get_position(); t.set_position((x + dx, y))
rws = np.nonzero((img < 235).any(axis=1))[0]
pad = 1.0 / 25.4
bb = Bbox([[0.0, max((img.shape[0] - 1 - rws[-1]) / fig.dpi - pad, 0.0)],
           [PW, min((img.shape[0] - rws[0]) / fig.dpi + pad, H)]])
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
OUT = os.environ.get('FIG3_OUT', 'fig_wave_attenuation_bars')
fig.savefig(os.path.join(FIG, OUT + '.png'), dpi=600, bbox_inches=bb)
fig.savefig(os.path.join(FIG, OUT + '.pdf'), bbox_inches=bb)
COPY = os.environ.get('SURGE_FIGURE_COPY')
if COPY:
    for f in ('fig_wave_attenuation.png', 'fig_wave_attenuation.pdf'):
        shutil.copy(os.path.join(FIG, f), os.path.join(COPY, f))
print('serial change', np.round(change[:4], 2), '| saved fig_wave_attenuation.png/pdf')
