"""River network depth figure (paper1 section 4.1), from the matched-Courant experiment.

Every one of the five global river networks (30 to 1 arc-minute, 116 to 4,696
topological levels) is driven to the same median Courant number, 0.15, 0.25 and 0.40,
and SURGE after k iterations is compared with GPU-ordered on the same network
(global_results/code/bench_depth_matched_courant.py). Matching the Courant number
separates river network depth from the Courant number, which the earlier fixed-runoff
design could not do (NOTES_2026-09-02_geometry_and_reference.md).
Panel (a): 99th-percentile relative difference against k at Cr = 0.25, one curve per network.
Panel (b): the same at k = 3 against the number of topological levels, one line per Courant number.
Data: figure_data/depth_independence_matched.json.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator


def plain_log_ticks(ax, vals):
    # decimal tick labels on a log axis, easier to read than powers of ten for percentages
    ax.yaxis.set_major_locator(FixedLocator(vals))
    ax.yaxis.set_major_formatter(FixedFormatter([('%g' % v) for v in vals]))
    ax.yaxis.set_minor_locator(NullLocator())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper1_style, panel_label; setup_paper1_style()
except Exception as e:
    print('paper_style unavailable:', e)
# built at the text width, so the sizes set in paper_style are the sizes that print
PW = 166.0 / 25.4
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(_HERE, '..', 'figure_data')
M = json.load(open(os.path.join(DATA, 'depth_independence_matched.json')))
NAMES = {'30min': '30 arcmin', '15min': '15 arcmin', '05min': '5 arcmin',
         '03min': '3 arcmin', '01min': '1 arcmin'}
nets = []
for key, v in M.items():
    lab = next(n for t, n in NAMES.items() if t in key)
    nets.append((v['levels'], lab, v['regimes']))
nets.sort()
KS = [1, 2, 3, 5, 8]
CRS = ['0.15', '0.25', '0.40']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PW, 2.38 * PW / 5.3))
norm = plt.Normalize(np.log10(nets[0][0]), np.log10(nets[-1][0]))

# (a) the difference against the number of iterations at one Courant number
for lv, lab, reg in nets:
    p99 = [reg['0.25']['k_error'][str(k)][1] * 100 for k in KS]
    ax1.plot(KS, p99, 'o-', color=cm.viridis(norm(np.log10(lv))), lw=1.2, ms=4.5, zorder=4,
             markeredgecolor='k', markeredgewidth=0.5, label='%s, %s levels' % (lab, format(lv, ',')))
ax1.set_yscale('log'); ax1.set_xticks(KS)
plain_log_ticks(ax1, [0.01, 0.1, 1, 10])
ax1.set_xlabel('Jacobi iterations $k$')
ax1.set_ylabel('99th percentile of relative\ndifference [%]')
panel_label(ax1, 'a'); ax1.set_title('Difference against iterations, median Cr of 0.25')
ax1.legend(loc='lower left', fontsize=7.5, labelspacing=0.3, borderpad=0.4,
           handlelength=1.6, handletextpad=0.5, framealpha=0.85, facecolor='white', edgecolor='none')
ax1.grid(False, which='both')

# (b) the difference at three iterations against the number of topological levels
lvs = [n[0] for n in nets]
for cr, c in zip(CRS, ['#9ecae1', '#4292c6', '#08306b']):
    y = [reg[cr]['k_error']['3'][1] * 100 for _, _, reg in nets]
    ax2.plot(lvs, y, 'o-', color=c, lw=1.3, ms=4.5, zorder=4, markeredgecolor='k', markeredgewidth=0.5)
    ax2.annotate('Cr = %s' % cr, (lvs[-1], y[-1]), textcoords='offset points', xytext=(7, 0),
                 ha='left', va='center', fontsize=8.0, color=c)
    print('Cr %s k=3 p99 %%:' % cr, [round(v, 2) for v in y])
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlim(lvs[0] / 1.4, lvs[-1] * 3.2)
plain_log_ticks(ax2, [0.1, 0.3, 1, 3, 10]); ax2.set_ylim(0.1, 10)
ax2.set_xlabel('Topological levels of the river network')
ax2.set_ylabel('99th percentile of relative\ndifference at three iterations [%]')
panel_label(ax2, 'b'); ax2.set_title('Difference at three iterations against depth')
ax2.grid(False, which='both')

out = os.path.join(FIG, 'fig_depth_independence.png')
fig.savefig(out, dpi=600); fig.savefig(out[:-4] + '.pdf')   # constrained_layout handles spacing; no bbox=tight (house style)
print('wrote', out)
