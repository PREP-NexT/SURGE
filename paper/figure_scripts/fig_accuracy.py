"""Global-version fig_courant_theory + poc_jacobi_validation for paper1 §3.1,
from bench_global_accuracy outputs (serial-C oracle on the 5-arcmin global net)."""
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
res = json.load(open(os.path.join(DATA, 'accuracy.json')))
pc = np.load(os.path.join(DATA, 'accuracy_bigcells.npz'))   # big-river cells only
th = np.array(res['theory'])                       # (Cr, k, err)
ks_all = [1, 2, 3, 5, 8]
kk = {int(k): res['k'][str(k)] for k in ks_all}

# ================= fig_courant_theory (theory collapse) =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.1))
cols = {1: '#D55E00', 3: '#0072B2', 8: '#009E73'}
for k in (1, 3, 8):
    m = th[:, 1] == k
    Cr = th[m, 0]; err = th[m, 2]; g = Cr / (1 + Cr); pred = g ** (k + 1) / (1 - g)
    ax1.loglog(pred, err, 'o', ms=1.4, color=cols[k], alpha=0.12)
    # binned median (clean trend through the scatter)
    pv = np.clip(pred, 1e-9, None)
    bins = np.logspace(np.log10(pv.min()), np.log10(pv.max()), 11)
    ib = np.digitize(pv, bins); bx, by = [], []
    for i in range(1, len(bins)):
        s = ib == i
        if s.sum() >= 5:
            bx.append(np.median(pv[s])); by.append(np.median(err[s]))
    ax1.loglog(bx, by, 'o-', color=cols[k], ms=4.5, lw=1.4, label=f'GPU-eager, $k$={k}')
lims = [3e-6, 1]
ax1.plot(lims, lims, 'k--', lw=0.8, label='1:1')
ax1.set_xlim(3e-6, 1); ax1.set_ylim(2e-3, 1)
ax1.set_xlabel('Predicted $g^{k+1}/(1-g)$'); ax1.set_ylabel('$\mathrm{rRMSE}_c$')
ax1.set_title('(a) Error scales with the Courant damping law')
ax1.legend(loc='upper left', markerscale=1.6)
ax1.grid(False, which='both')

p99 = [kk[k]['rmse_p99'] for k in ks_all]
med = [max(kk[k]['rmse_med'], 1e-9) for k in ks_all]
ax2.semilogy(ks_all, p99, 'o-', color='#0072B2', lw=1.3, ms=5, label='99th percentile')
ax2.semilogy(ks_all, med, 's--', color='0.5', lw=1.1, ms=4, label='Median')
ax2.set_xlabel('Jacobi iterations $k$'); ax2.set_ylabel('$\mathrm{rRMSE}_c$ over $\mathcal{C}$')
ax2.set_title('(b) The error falls monotonically with $k$')
ax2.legend(); ax2.grid(False, which='both')
fig.savefig(f'{FIG}/fig_courant_theory.png', dpi=600); fig.savefig(f'{FIG}/fig_courant_theory.pdf')   # constrained_layout; no bbox=tight
print('wrote fig_courant_theory.png')

# ================= poc_jacobi_validation (hydrograph + dist + level) =================
level = pc['level']   # topological level, big-river cells
fig = plt.figure(figsize=(6.9, 5.4), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.0])
axa = fig.add_subplot(gs[0, :])     # time series, full width
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])
# (a) hydrograph at largest-flow cell
days = np.arange(len(res['hydro']['serialC']))
axa.plot(days, res['hydro']['serialC'], 'k-', lw=1.3, label='CPU-serial')
for k, c in [(1, '#D55E00'), (2, '#E69F00'), (3, '#0072B2')]:
    axa.plot(days, res['hydro'][str(k)], '-', lw=0.9, color=c, alpha=0.9, label=f'GPU-eager, $k$={k}')
axa.set_xlabel('Day of 2010'); axa.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
axa.set_title('(a) Three iterations reproduce the Amazon-mouth discharge')
axa.legend(ncol=2)
# (b) per-cell RMSE distribution (big rivers), k=1,2,3
for k, c in [(1, '#D55E00'), (2, '#E69F00'), (3, '#0072B2')]:
    e = pc['err_k%d' % k]; e = e[np.isfinite(e) & (e > 0)]
    axb.hist(np.log10(e), bins=60, histtype='step', color=c, lw=1.2, label=f'GPU-eager, $k$={k}')
axb.set_xlabel('$\log_{10}(\mathrm{rRMSE}_c)$'); axb.set_ylabel('High-discharge cells')
axb.set_title('(b) The error tail contracts as $k$ grows')
axb.legend()
# (c) mean error vs topological level, k=1,2,3
lv = level
bins = np.linspace(0, lv.max(), 40)
idx = np.digitize(lv, bins)
for k, c in [(1, '#D55E00'), (2, '#E69F00'), (3, '#0072B2')]:
    ek = pc['err_k%d' % k]
    me = [np.mean(ek[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
    axc.semilogy(bins[1:], me, '-', color=c, lw=1.3, label=f'GPU-eager, $k$={k}')
axc.set_xlabel('Topological level'); axc.set_ylabel('Mean $\mathrm{rRMSE}_c$')
axc.set_title('(c) The error does not grow with depth')
axc.legend()
fig.savefig(f'{FIG}/poc_jacobi_validation.png', dpi=600); fig.savefig(f'{FIG}/poc_jacobi_validation.pdf')   # constrained_layout; no bbox=tight
print('wrote poc_jacobi_validation.png')
print('tab:accuracy rows:', {k: (round(kk[k]['rmse_p99']*100,2), round(kk[k]['nse_med'],4)) for k in ks_all})
print('wavefront vs serialC maxrel:', res['wavefront_vs_serialC_maxrel'])
print('focused/blocked:', res['k'].get('focused_3+5'), res['k'].get('blocked_gs_3'))
