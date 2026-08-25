"""Paper1 method schematic: the same drainage network solved three ways --
(a) serial C: one cell at a time, in topological order (CPU);
(b) exact wavefront: one topological level at a time (GPU, O(levels) steps);
(c) Jacobi fixed point: all cells at once, k~=3 sweeps (GPU, depth-independent).
Pure schematic; no data."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper_style
    setup_paper_style(base_size=8.0)
except Exception as e:
    print('paper_style fallback:', e)
    plt.rcParams.update({'font.size': 8.0, 'font.family': 'sans-serif'})
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(_HERE, '..', 'figure_data')

LVL_CMAP = plt.get_cmap('YlGnBu')

# illustrative drainage tree: id -> (x, y, level)   level 1=headwater ... 4=outlet
nodes = {
    0: (0.50, 0.10, 4),
    1: (0.33, 0.36, 3), 2: (0.67, 0.36, 3),
    3: (0.18, 0.60, 2), 4: (0.42, 0.60, 2), 5: (0.58, 0.60, 2), 6: (0.82, 0.60, 2),
    7: (0.10, 0.84, 1), 8: (0.25, 0.84, 1), 9: (0.37, 0.84, 1),
    10: (0.53, 0.84, 1), 11: (0.75, 0.84, 1), 12: (0.90, 0.84, 1),
}
edges = [(7, 3), (8, 3), (3, 1), (9, 4), (4, 1), (10, 5), (5, 2),
         (11, 6), (12, 6), (6, 2), (1, 0), (2, 0)]
maxlv = 4
N = len(nodes)


_HALO = [pe.withStroke(linewidth=1.4, foreground='#33333a')]   # keeps white text legible on any node colour


def draw(ax, colors, title, sub, keytext, labels=None):
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.06, 1.06); ax.axis('off')
    ax.set_title(title, pad=2)
    for a, b in edges:                                 # arrow points downstream (a -> b, toward the outlet)
        ax.annotate('', xy=(nodes[b][0], nodes[b][1]), xytext=(nodes[a][0], nodes[a][1]),
                    arrowprops=dict(arrowstyle='-|>', color='#9aa0a6', lw=1.0,
                                    shrinkA=6, shrinkB=7, mutation_scale=9), zorder=1)
    for nid, (x, y, lv) in nodes.items():
        ax.scatter(x, y, s=175, c=[colors[nid]], edgecolors='white', lw=0.7, zorder=3)
        if labels is not None:                     # number = the step at which this cell is computed
            ax.text(x, y, labels[nid], ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='white', zorder=4, path_effects=_HALO)
    ax.text(0.5, 0.99, keytext, ha='center', va='top', fontsize=6.8, color='#555', style='italic')
    ax.text(0.5, -0.04, sub, fontsize=7.0, ha='center', va='top', color='#333')


fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15), constrained_layout=False)  # hand-placed schematic

# (a) serial: number = processing (topological) order 1..N; colour reinforces the progression
order = sorted(nodes, key=lambda n: (nodes[n][2], n))
oidx = {n: i for i, n in enumerate(order)}
vir = plt.get_cmap('turbo')   # higher low-end contrast so (b)'s first few steps separate clearly
c_serial = {n: vir(oidx[n] / (N - 1)) for n in nodes}          # (a) colour = serial compute order 1..N
c_wave = {n: vir((nodes[n][2] - 1) / (N - 1)) for n in nodes}  # (b) colour = compute step (=level) on the SAME scale: level k -> serial order k's colour
c_jac = {n: vir(0.0) for n in nodes}                           # (c) all computed at once -> everyone gets the order-1 colour
lab_serial = {n: str(oidx[n] + 1) for n in nodes}
draw(axes[0], c_serial, '(a) Serial scheme',
     'One cell at a time\n$N$ sequential steps',
     'Number = compute order', labels=lab_serial)
# (all three panels share the same idea: the node number is the cell's compute order)

# (b) wavefront: number = level = the step; same-level cells share a step; colour banded by level
lab_wave = {n: str(nodes[n][2]) for n in nodes}   # level 1 (headwaters) first ... level L (outlet) last
draw(axes[1], c_wave, '(b) Wavefront scheme',
     'One level at a time\n$L$ level steps',
     'Number = compute order (by level)', labels=lab_wave)

# (c) jacobi: no ordering -- all cells updated every sweep (hence unnumbered)
draw(axes[2], c_jac, '(c) Jacobi fixed-point scheme',
     'All cells at once\nRepeated iterations',
     'No compute order: all at once', labels=None)

fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.15, wspace=0.06)
fig.savefig(os.path.join(FIG, 'fig_method.png'), dpi=600); fig.savefig(os.path.join(FIG, 'fig_method.pdf'))
print('saved fig_method.png (3-panel serial/wavefront/jacobi)')
