"""Resolution-scaling figure (paper1): measured single-run speedup vs network size,
A100 eager and fused, 30' -> 1'. All numbers measured (no extrapolation)."""
import os
import sys, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper_style; setup_paper_style(base_size=8.0)
except Exception as e:
    print('paper_style unavailable:', e)
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(_HERE, '..', 'figure_data')

res   = ["30'", "15'", "5'", "3'", "1'"]
cells = np.array([70648, 267950, 2307532, 6222566, 55812946], float)
levels= [116, 251, 782, 1436, 4696]
serial= np.array([10.1, 39.3, 387.0, 1432.7, 12931.0])   # ms, 1 CPU core
eager = np.array([1.738, 1.773, 3.913, 12.871, 107.2])   # ms, A100 eager
fused = np.array([0.137, 0.149, 0.550, 1.419, 12.367])   # ms, A100 fused
sp_e, sp_f = serial/eager, serial/fused

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.1))

# (a) speedup vs network size
ax1.loglog(cells, sp_f, 'o-', color='#0072B2', lw=1.4, ms=5, label='GPU-fused')
ax1.loglog(cells, sp_e, 's--', color='#D55E00', lw=1.2, ms=4.5, label='GPU-eager')
for x, y, r in zip(cells, sp_f, res):
    ax1.annotate(r, (x, y), textcoords='offset points', xytext=(3, -9), fontsize=7.0)
ax1.annotate('1,046$\\times$', (cells[-1], sp_f[-1]), textcoords='offset points',
             xytext=(-4, 7), fontsize=8.0, fontweight='bold', color='#0072B2', ha='right')
ax1.set_xlabel('River cells in global network')
ax1.set_ylabel('Speedup over CPU-serial')
ax1.set_title('(a) The speedup grows with resolution')
ax1.legend(loc='upper left')
ax1.grid(False, which='both')

# (b) per-substep wall-clock
ax2.loglog(cells, serial, '^-', color='0.35', lw=1.2, ms=4.5, label='CPU-serial')
ax2.loglog(cells, eager, 's--', color='#D55E00', lw=1.2, ms=4.5, label='GPU-eager')
ax2.loglog(cells, fused, 'o-', color='#0072B2', lw=1.4, ms=5, label='GPU-fused')
ax2.set_xlabel('River cells in global network')
ax2.set_ylabel('Wall-clock per substep [ms]')
ax2.set_title('(b) The per-substep cost grows with network size')
ax2.legend(loc='upper left')
ax2.grid(False, which='both')

out = os.path.join(FIG, 'fig_resolution_scaling.png')
fig.savefig(out, dpi=600); fig.savefig(out[:-4] + '.pdf')   # constrained_layout handles spacing; no bbox=tight (house style)
print('wrote', out)
print('fused speedups:', dict(zip(res, np.round(sp_f).astype(int))))
