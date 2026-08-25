"""Adding the full GRanD dam set (7,320 dams -> ~6.5-7.3k reservoir cells,
6,881 km3) to the three schemes. (a) per-substep cost WITH reservoirs across
resolutions -- the wavefront/Jacobi gap is unchanged and reservoirs add only a
few percent; (b) Jacobi still reproduces the EXACT wavefront with reservoirs
active (big-river relative error vs sweeps, with vs without dams). A100, fp32.
Data: figure_data/scheme_bench.json (bench_reservoir.py on ASPIRE A100)."""
import os, sys, json, numpy as np
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

J = json.load(open(os.path.join(DATA, 'scheme_bench.json')))
B = {r['res']: r for r in J['withdam']}
A = {r['res']: r for r in J['nodam']}
order = ['5\'', '3\'', '1\'']
cells = np.array([B[r]['N'] for r in order], float)
wfB = np.array([B[r]['wavefront'] for r in order]); jeB = np.array([B[r]['jacobi_eager'] for r in order])
jfB = np.array([B[r]['jacobi_fused'] for r in order])
wfA = np.array([A[r]['wavefront'] for r in order]); jeA = np.array([A[r]['jacobi_eager'] for r in order])
jfA = np.array([A[r]['jacobi_fused'] for r in order])
C_WF = '#D55E00'; C_JE = '#E69F00'; C_JF = '#0072B2'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.15))
# (a) per-substep cost WITH reservoirs (solid) vs without (faint) -- three schemes
ax1.loglog(cells, wfB, 's-', color=C_WF, lw=1.3, ms=5, label='GPU-wavefront + dams')
ax1.loglog(cells, jeB, 'o-', color=C_JE, lw=1.2, ms=4.5, label='GPU-eager + dams')
ax1.loglog(cells, jfB, 'o-', color=C_JF, lw=1.4, ms=5.5, label='GPU-fused + dams')
ax1.loglog(cells, wfA, 's:', color=C_WF, lw=1.0, ms=3, alpha=0.5)
ax1.loglog(cells, jeA, 'o:', color=C_JE, lw=1.0, ms=3, alpha=0.5)
ax1.loglog(cells, jfA, 'o:', color=C_JF, lw=1.0, ms=3, alpha=0.5, label='Same, no dams (faint)')
ax1.annotate('%.0f$\\times$' % (wfB[-1] / jfB[-1]), (cells[-1], (wfB[-1] * jfB[-1]) ** 0.5),
             fontsize=7.5, color=C_WF, ha='right', fontweight='bold')
for x, y, r in zip(cells, jfB, order):
    ax1.annotate(r, (x, y), textcoords='offset points', xytext=(3, -10), fontsize=7.0, color=C_JF)
ax1.set_xlabel('River cells in global network'); ax1.set_ylabel('Wall-clock per substep [ms]')
ax1.set_title('(a) Reservoirs add a few per cent to the cost')
ax1.legend(loc='center left'); ax1.grid(False, which='both')

# (b) accuracy: Jacobi vs exact wavefront, with vs without reservoirs (5-arcmin)
acc = J['accuracy_5min_bigriver_relerr']
ks = [1, 2, 3, 5]
ax2.plot(ks, [acc['with_dams'][str(k)] * 100 for k in ks], 'o-', color='#0072B2', lw=1.4, ms=5.5, label='With GRanD dams')
ax2.plot(ks, [acc['without_dams'][str(k)] * 100 for k in ks], 's--', color='0.5', lw=1.1, ms=4, label='Without dams')
ax2.set_xlabel('Jacobi iterations $k$'); ax2.set_ylabel('Median $E^{\mathrm{rel}}_{c,k}$ over mainstem cells [%]')
ax2.set_xticks(ks)
ax2.set_title('(b) Reservoirs do not change convergence')
ax2.legend(); ax2.grid(False, which='both')
ax2.text(0.5, 0.06, 'GPU-eager reproduces GPU-wavefront\nwith dams active',
         transform=ax2.transAxes, fontsize=7.0, ha='center', color='#444', style='italic')

fig.savefig(os.path.join(FIG, 'fig_reservoir.png'), dpi=600); fig.savefig(os.path.join(FIG, 'fig_reservoir.pdf'))
print('wrote fig_reservoir.png')
print('reservoir overhead (fused):', {r: round(B[r]['jacobi_fused'] / A[r]['jacobi_fused'], 3) for r in order})
