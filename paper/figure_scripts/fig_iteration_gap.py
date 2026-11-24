"""Figure 4 of the manuscript: how the parallel iteration closes the gap on a chain of cells. Six cells at Courant number 1, no lateral inflow, the
inflow of cell 1 steps from Q_init to 2 Q_init in one time step (Equation 3). (a) serial solution, (b) serial minus the fully parallel scheme after
iterations 1-3, (c) the largest gap per iteration with the reference g_c^j, g_c = 1/2. Panel c stops at
iteration 5: after iteration 6 all six cells equal the serial solution and the gap is zero.
Run:  python fig_iteration_gap.py
Output: paper/figures/fig_iteration_gap.{png,pdf}
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import setup_paper1_style, panel_label
setup_paper1_style()
# text tiers of Figure 1 (fig_method_seine.py): letters 11, titles 10.5, labels and annotations 9, ticks 8
FS_LETTER, FS_TITLE, FS_ANN = 11.0, 10.5, 9.0
plt.rcParams.update({'font.size': FS_ANN, 'axes.labelsize': FS_ANN, 'axes.titlesize': FS_TITLE,
                     'legend.fontsize': FS_ANN, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0})


def e_numbers(N=8, DT=4320.0, BETA=0.6, n=0.03, S0=1e-3, P=30.0, Q0=100.0, QIN=200.0, K=3):
    """Equation 3 on a chain of N cells, Courant number 1 at the base flow Q0; the inflow at the
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


N = 6
change, left = e_numbers(N=N, K=N)          # left[j-1] = serial minus iteration j, in Q_init
serial = 1.0 + change
cells = np.arange(1, N + 1)
INK, GREY, BLUE, RED, SHADE = '#333333', '#7f8790', '#2e79b8', '#C0392B', '#e3edf7'
QI = r'Q_{\mathrm{init}}'

PW = 472.31595 / 72.27                       # text width in inches
fig = plt.figure(figsize=(PW, 76 / 25.4), layout='constrained')
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.15, 0.08, 0.85])   # column 2 is an empty spacer, so the y label of c clears b

# (a) serial solution
ax = fig.add_subplot(gs[0])
ax.axhline(2.0, color=GREY, lw=0.8, ls=(0, (1.5, 1.5)))
ax.axhline(1.0, color=GREY, lw=0.8, ls=(0, (1.5, 1.5)))
ax.text(N + 0.3, 2.03, r'Inflow, $2%s$' % QI, color=GREY, ha='right', va='bottom')
ax.text(N + 0.3, 0.97, r'Initial, $%s$' % QI, color=GREY, ha='right', va='top')
ax.plot(cells, serial, color=INK, lw=1.4, marker='o', ms=4)
for c, v in zip(cells, serial):
    ax.text(c + 0.1, v + 0.03, '%.2f' % v, ha='left', va='bottom', color=INK)   # upper right, clear of the line
ax.set_xlim(0.5, N + 0.5); ax.set_ylim(0.8, 2.25)
ax.set_xticks(cells)
ax.set_xlabel('Cell, counted downstream')
ax.set_ylabel(r'Outflow after one time step ($%s$)' % QI)
ax.set_title('Serial solution', loc='center')
ax.spines[['top', 'right']].set_visible(False)
panel_label(ax, 'a').set_fontsize(FS_LETTER)

# (b) gap after iterations 1-3, one row each
sub = gs[1].subgridspec(3, 1, hspace=0.12)
ymax = left[0].max() * 1.6   # headroom for the 9 pt value labels under the row title
for j in range(3):
    axb = fig.add_subplot(sub[j])
    axb.axvspan(0.5, j + 1.5, color=SHADE, lw=0, zorder=0)            # cells equal to the serial solution
    g = left[j]
    show = g > 5e-3
    axb.bar(cells[show], g[show], width=0.6, color=RED, lw=0, zorder=2)
    axb.plot(cells[~show], np.zeros((~show).sum()), 'o', color=BLUE, ms=3.5, zorder=3, clip_on=False)
    for c, v in zip(cells[show], g[show]):
        axb.text(c, v + ymax * 0.04, '%.2f' % v, ha='center', va='bottom', color=RED)
    axb.text(N + 0.4, ymax * 0.95, 'After iteration %d' % (j + 1), ha='right', va='top', color=INK,
             fontweight='bold')
    axb.set_xlim(0.5, N + 0.5); axb.set_ylim(0, ymax)
    axb.set_xticks(cells); axb.set_yticks([0, 0.2])
    axb.spines[['top', 'right']].set_visible(False)
    if j < 2:
        axb.tick_params(labelbottom=False)
    if j == 0:
        axb.set_title('Gap to the serial solution', loc='center')
        panel_label(axb, 'b').set_fontsize(FS_LETTER)
    if j == 1:
        axb.set_ylabel(r'Gap ($%s$)' % QI)
    if j == 2:
        axb.set_xlabel('Cell, counted downstream')

# (c) largest gap per iteration
axc = fig.add_subplot(gs[3])
J = np.arange(1, N)                          # iteration N leaves no gap
gmax = np.array([left[j - 1].max() for j in J])
axc.plot(J, gmax[0] * 0.5 ** (J - 1), color=INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
axc.plot(J, gmax, 'o', color=RED, ms=5, zorder=2)
axc.set_yscale('log')
axc.set_xlim(0.6, N - 0.6); axc.set_xticks(J)
axc.set_ylim(gmax.min() * 0.5, gmax.max() * 2.2)
axc.set_xlabel('Iteration $j$')
axc.set_ylabel(r'Largest gap ($%s$)' % QI)
# the label sits under the dashed line midway between iterations 2 and 3, and a vertical
# leader rises from it to the line, clear of the dots
_tc = axc.text(2.5, gmax.min() * 1.3, 'slope $g_c = 1/2$', ha='center', va='center', color=INK)
fig.canvas.draw()
_bb = _tc.get_window_extent(fig.canvas.get_renderer()).transformed(axc.transData.inverted())
_xa = 2.5
axc.annotate('', xy=(_xa, gmax[0] * 0.5 ** (_xa - 1)), xytext=(_xa, _bb.y1),
             arrowprops=dict(arrowstyle='-|>', lw=0.6, color=INK, mutation_scale=6, shrinkA=1, shrinkB=1))
axc.set_title('The largest gap halves', loc='center')
axc.spines[['top', 'right']].set_visible(False)
panel_label(axc, 'c').set_fontsize(FS_LETTER)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIG, 'fig_iteration_gap.' + ext), dpi=600)
print('serial', np.round(serial, 2), '| largest gap', np.round(gmax, 3), '| saved fig_iteration_gap')
