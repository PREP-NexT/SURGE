"""Resolution-scaling figure (paper1): measured single-run speedup vs network size,
A100 eager and fused, 30' -> 1'. All numbers measured (no extrapolation)."""
import os
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# the measured values of Table 7, kept in one file that Figure 9 reads too
import json
RS = json.load(open(os.path.join(DATA, 'resolution_scaling.json')))['rows']
res   = [r['res'] for r in RS]
cells = np.array([r['N'] for r in RS], float)
serial= np.array([r['serialC'] for r in RS])   # ms, 1 CPU core
fused = np.array([r['surge'] for r in RS])     # ms, A100, SURGE
sp_f = serial/fused

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PW, 2.38 * PW / 5.3))

# (a) speedup vs network size
ax1.loglog(cells, sp_f, 'o-', color='#0072B2', lw=1.4, ms=5, label='SURGE')
for x, y, r in zip(cells, sp_f, res):
    ax1.annotate(r, (x, y), textcoords='offset points', xytext=(3, -9), fontsize=8.0)
ax1.annotate('{:,.0f}$\\times$'.format(sp_f[-1]), (cells[-1], sp_f[-1]), textcoords='offset points',
             xytext=(-6, -12), fontsize=8.0, fontweight='bold', color='#0072B2', ha='right')
ax1.set_xlabel('River cells in global network')
ax1.set_ylabel('Speedup over CPU-serial')
panel_label(ax1, 'a'); ax1.set_title('The speedup grows with resolution')
ax1.legend(loc='lower right')
ax1.grid(False, which='both')

# (b) per-substep wall-clock
ax2.loglog(cells, serial, '^-', color='0.35', lw=1.2, ms=4.5, label='CPU-serial')
ax2.loglog(cells, fused, 'o-', color='#0072B2', lw=1.4, ms=5, label='SURGE')
ax2.set_xlabel('River cells in global network')
ax2.set_ylabel('Wall-clock per substep [ms]')
panel_label(ax2, 'b'); ax2.set_title('The cost grows with network size')
ax2.legend(loc='upper left')
ax2.grid(False, which='both')

out = os.path.join(FIG, 'fig_resolution_scaling.png')
fig.savefig(out, dpi=600); fig.savefig(out[:-4] + '.pdf')   # constrained_layout handles spacing; no bbox=tight (house style)
print('wrote', out)
print('fused speedups:', dict(zip(res, np.round(sp_f).astype(int))))
