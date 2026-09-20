"""Figure: every compute configuration at every resolution in one bar chart.

Wall-clock per simulated day on a log axis, four bars per resolution, and the speedup
over CPU-serial printed above each bar. Replaces the former resolution-scaling and scheme-throughput figures.
Data: figure_data/resolution_scaling.json (unified timing protocol).
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import setup_paper1_style, panel_label
setup_paper1_style()
PW = 166.0 / 25.4
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'figure_data')
OUT = os.path.join(HERE, '..', 'figures', 'fig_timing_bars')

rows = json.load(open(os.path.join(DATA, 'resolution_scaling.json')))['rows']
CONF = [('serialC', 'CPU-serial (1 core)', '0.45'),
        ('cpu_parallel', 'CPU-parallel (24 cores)', '#9ECAE1'),
        ('gpu_ordered', 'GPU-ordered (1 GPU)', '#D55E00'),
        ('surge', 'SURGE (1 GPU)', '#0072B2')]
resname = {"30'": "30'", "15'": "15'", "5'": "5'", "3'": "3'", "1'": "1'"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(PW, 2.6 * PW / 5.3), layout='constrained',
                              gridspec_kw=dict(width_ratios=(1.7, 1.0)))
SPD = 20          # routing time steps in a simulated day (Table A1)
n = len(rows); w = 0.2
x = np.arange(n)
for j, (key, label, col) in enumerate(CONF):
    t = np.array([r[key] for r in rows]) * SPD / 1e3      # ms a time step -> s a simulated day
    xb = x + (j - 1.5) * w
    ax.bar(xb, t, width=w, color=col, edgecolor='k', linewidth=0.4, label=label, zorder=3)
ax.set_yscale('log')
ax.set_ylim(1e-3, 2e4)
ax.set_xticks(x)
def _cells(n):
    return '%.0fk' % (n / 1e3) if n < 1e6 else '%.1fM' % (n / 1e6)


ax.set_xticklabels(['%s arcmin\n%s cells\n%s levels\n$k$ = %d'
                    % (r['res'].rstrip("'"), _cells(r['N']), '{:,}'.format(r['levels']), r['k'])
                    for r in rows])
ax.set_ylabel('Wall-clock per simulated day [s]')
ax.legend(loc='upper left', ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)
ax.grid(False, which='both')
panel_label(ax, 'a'); ax.set_title('Wall-clock per simulated day')

# (b) the same runs as a speedup over CPU-serial, against the size of the river network
cells = np.array([r['N'] for r in rows], float)
for key, label, col in CONF[1:]:
    sp = np.array([r['serialC'] / r[key] for r in rows])
    ax2.plot(cells, sp, 'o-', color=col, lw=1.2, ms=5, markeredgecolor='k', markeredgewidth=0.4,
             label=label.split(' (')[0], zorder=3)
ax2.axhline(1.0, color='0.45', lw=0.7, ls=(0, (4, 2)), zorder=2)
ax2.annotate('CPU-serial, one core', xy=(5e4, 1.0), xytext=(0, 3), textcoords='offset points',
             ha='left', va='bottom', fontsize=7.0, color='0.45')
for c, r in zip(cells, rows):
    sp = r['serialC'] / r['surge']
    ax2.annotate('{:,.0f}'.format(sp), (c, sp), textcoords='offset points', xytext=(0, 7),
                 ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=CONF[3][2])
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlim(4e4, 1.4e8); ax2.set_ylim(0.15, 1500)
ax2.set_xlabel('River cells in the network')
ax2.set_ylabel('Speedup over CPU-serial')
ax2.legend(loc='center left', frameon=False, fontsize=7.0, handlelength=1.6,
           labelspacing=0.3, borderpad=0.2)   # the band between the two upper curves is free
ax2.grid(False, which='both')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.tick_params(top=False, right=False)
panel_label(ax2, 'b'); ax2.set_title('Speedup over one CPU core')

fig.savefig(OUT + '.png', dpi=600); fig.savefig(OUT + '.pdf')
print('wrote', OUT + '.{png,pdf}')
