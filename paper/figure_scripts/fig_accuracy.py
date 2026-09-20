"""Global-version fig_courant_theory + poc_jacobi_validation for paper1 section 4.1,
from bench_global_accuracy outputs (serial-C oracle on the 5-arcmin global net).

fig_courant_theory carries the four summaries of the truncation over the evaluation set,
the 106,661 cells whose annual-mean discharge exceeds 100 m3/s. All four panels describe
that same set, so they belong in one figure: (a) the error against the scale the theory
predicts, (b) how it falls with the iteration count, (c) the shape of its distribution,
(d) how it grows along the river. accuracy_bigcells.npz, accuracy_serialC_area.npz and
metric_maps_serialC.npz hold the same cells in the same order and agree on rRMSE to 6e-6,
so panels drawn from different files still compare.

poc_jacobi_validation is the single-cell counterpart, one hydrograph at the Amazon mouth.
"""
import os, sys, json, numpy as np
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
res = json.load(open(os.path.join(DATA, 'accuracy.json')))
pc = np.load(os.path.join(DATA, 'accuracy_bigcells.npz'))   # big-river cells only
th = np.array(res['theory'])                       # (Cr, k, err)
ks_all = [1, 2, 3, 5, 8]
kk = {int(k): res['k'][str(k)] for k in ks_all}
LEG = dict(fontsize=8.0, labelspacing=0.4, borderpad=0.45, handlelength=1.6,
           handletextpad=0.5, framealpha=0.85, facecolor='white', edgecolor='none')

# ================= fig_courant_theory: the truncation over the evaluation set ==========
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(PW, 4.76 * PW / 5.3))
# every cell of the year-long serial-C comparison, so k=20 can be shown too; the
# scatter is thinned for legibility while the binned medians use all of them
MM = np.load(os.path.join(DATA, 'metric_maps_serialC.npz'))
CR_ALL = MM['courant'].astype(np.float64)
rng = np.random.default_rng(0)
thin = rng.choice(CR_ALL.size, size=3000, replace=False)
cols = {1: '#D55E00', 3: '#0072B2', 8: '#009E73', 20: '#7A4FBF'}
for k in (1, 3, 8, 20):
    err = MM['rrmse_k%d' % k].astype(np.float64)
    g = CR_ALL / (1 + CR_ALL); pred = g ** (k + 1) / (1 - g)
    ax1.loglog(pred[thin], np.clip(err[thin], 1e-9, None), 'o', ms=1.4,
               color=cols[k], alpha=0.12, zorder=1)
    # binned median over equal-count bins: equal-width log bins leave the left
    # tail with a handful of cells per bin, whose medians are pure noise
    pv = np.clip(pred, 1e-12, None)
    q = np.quantile(pv, np.linspace(0, 1, 11)); q[-1] *= 1.0001
    ib = np.digitize(pv, q)
    bx = [np.median(pv[ib == i]) for i in range(1, 11)]
    by = [max(np.median(err[ib == i]), 1e-9) for i in range(1, 11)]
    ax1.loglog(bx, by, 'o-', color=cols[k], ms=4.5, lw=1.4, zorder=5,
               markeredgecolor='k', markeredgewidth=0.5,
               label=f'{k} Jacobi iteration' + ('' if k == 1 else 's'))
LIMS = (1e-9, 3)          # both axes share the range so 1:1 is the true diagonal
ax1.plot(LIMS, LIMS, 'k--', lw=0.8, zorder=3)
ax1.set_xlim(*LIMS); ax1.set_ylim(*LIMS)
ax1.set_xlabel('Predicted error scale $\\varepsilon_k$'); ax1.set_ylabel('Relative RMSE of cell discharge')
panel_label(ax1, 'a'); ax1.set_title('The difference follows the predicted scale')
# label the 1:1 line with a leader, so it does not take a legend row
ax1.annotate('1:1 line', xy=(3.0e-8, 3.0e-8), xytext=(2.0e-6, 2.0e-9),
             fontsize=8.0, ha='left', va='center', zorder=6,
             arrowprops=dict(arrowstyle='-', lw=0.6, color='k',
                             shrinkA=1.5, shrinkB=1.5))
# at 8 pt a lower-right legend covers the converged clouds; the upper left above the
# diagonal holds no cells
ax1.legend(loc='upper left', markerscale=1.0, title='SURGE', **LEG)
ax1.get_legend().get_title().set_fontsize(8.0)
ax1.grid(False, which='both')

SC = np.load(os.path.join(DATA, 'accuracy_serialC_area.npz'))
ks_all = [1, 2, 3, 5, 8, 20]
p99 = [np.percentile(SC['err_k%d' % k], 99) for k in ks_all]
med = [max(np.median(SC['err_k%d' % k]), 1e-9) for k in ks_all]
ax2.semilogy(ks_all, p99, 'o-', color='#0072B2', lw=1.3, ms=5,
             markeredgecolor='k', markeredgewidth=0.5,
             label='99th percentile,\ncells with $\\overline{Q}_c>100$ m$^3$ s$^{-1}$')
ax2.semilogy(ks_all, med, 's--', color='0.5', lw=1.1, ms=4,
             markeredgecolor='k', markeredgewidth=0.5,
             label='Median,\ncells with $\\overline{Q}_c>100$ m$^3$ s$^{-1}$')
ax2.set_xlabel('The number of Jacobi iterations $k$'); ax2.set_ylabel('Relative RMSE of cell discharge')
panel_label(ax2, 'b'); ax2.set_title('The difference falls monotonically with $k$')
ax2.legend(**LEG)
ax2.grid(False, which='both')

# (c) per-cell RMSE distribution over the same cells, at every iteration count the
# table reports. accuracy_bigcells.npz holds only k=1,2,3, so panels (c) and (d) read
# the full run, which carries the same cells in the same order and agrees on rRMSE to
# 6e-6. The left edge is the fp32 resolution of the comparison, 1.1e-7, where the
# converged runs pile up
FU = np.load(os.path.join(DATA, 'accuracy_full_serialC.npz'))
KCOL = [(1, '#D55E00'), (2, '#E69F00'), (3, '#0072B2'),
        (5, '#CC79A7'), (8, '#009E73'), (20, '#7A4FBF')]
_hb = np.logspace(-7.2, 1.0, 96)
for k, c in KCOL:
    e = FU['rrmse_k%d' % k].astype(np.float64); e = e[np.isfinite(e) & (e > 0)]
    ax3.hist(e, bins=_hb, histtype='step', color=c, lw=1.1,
             label=f'{k} Jacobi iteration' + ('' if k == 1 else 's'))
ax3.set_xscale('log'); ax3.set_xlim(1e-7, 10)
ax3.set_xlabel('Relative RMSE of cell discharge')
ax3.set_ylabel('Number of cells ($\\overline{Q}_c>100$ m$^3$ s$^{-1}$)')
panel_label(ax3, 'c'); ax3.set_title('The largest differences shrink with more iterations')
# moved right of the twenty-iteration peak at the left edge, which it covered at 8 pt
ax3.legend(title='SURGE', loc='upper left', bbox_to_anchor=(0.16, 1.0), **LEG)
ax3.get_legend().get_title().set_fontsize(8.0)
ax3.grid(False, which='both')

# (d) error against the length of the river upstream, in equal-count depth bins:
# equal-width bins leave the deep tail with a few dozen cells, whose medians are noise
lv = FU['level']
_q = np.quantile(lv, np.linspace(0, 1, 13)); _q[-1] += 1
idx = np.digitize(lv, _q)
_x = [np.median(lv[idx == i]) for i in range(1, 13)]
for k, c in KCOL:
    ek = FU['rrmse_k%d' % k].astype(np.float64)
    ax4.semilogy(_x, [np.median(ek[idx == i]) for i in range(1, 13)], 'o-', color=c,
                 lw=1.2, ms=3.2, markeredgecolor='k', markeredgewidth=0.4,
                 label=f'{k} Jacobi iteration' + ('' if k == 1 else 's'))
ax4.set_xlabel('Upstream river length (number of cells)')
ax4.set_ylabel('Relative RMSE of cell discharge')
panel_label(ax4, 'd'); ax4.set_title('Longer rivers carry larger differences')
ax4.legend(loc='lower right', title='SURGE', **LEG)
ax4.get_legend().get_title().set_fontsize(8.0)
ax4.grid(False, which='both')
fig.savefig(f'{FIG}/fig_courant_theory.png', dpi=600); fig.savefig(f'{FIG}/fig_courant_theory.pdf')   # constrained_layout; no bbox=tight
print('wrote fig_courant_theory.png (2x2)')

# ================= poc_jacobi_validation: one hydrograph at the Amazon mouth ==========
fig = plt.figure(figsize=(PW, 2.05 * PW / 5.3), constrained_layout=True)
axa = fig.add_subplot(111)
days = np.arange(len(res['hydro']['serialC']))
axa.plot(days, res['hydro']['serialC'], 'k-', lw=1.3, label='CPU-serial')
for k, c in [(1, '#D55E00'), (2, '#E69F00'), (3, '#0072B2')]:
    axa.plot(days, res['hydro'][str(k)], '-', lw=0.9, color=c, alpha=0.9,
             label=f'SURGE, {k} Jacobi iteration' + ('' if k == 1 else 's'))
axa.set_xlabel('Day of 2010'); axa.set_ylabel('Discharge [m$^3$ s$^{-1}$]')
axa.set_xlim(days[0], days[-1])
# headroom above the April peak for the legend, which covered the peak at 8 pt
_lo, _hi = axa.get_ylim(); axa.set_ylim(_lo, _lo + (_hi - _lo) * 1.30)
axa.legend(ncol=2, columnspacing=1.2, **LEG)
# the section 3.3 metrics of the three-iteration series at this cell, from the plotted data
_o = np.asarray(res['hydro']['serialC'], float); _s = np.asarray(res['hydro']['3'], float)
_r = np.corrcoef(_s, _o)[0, 1]
_kge = 1 - np.sqrt((_r - 1) ** 2 + (_s.std() / _o.std() - 1) ** 2 + (_s.mean() / _o.mean() - 1) ** 2)
_nse = 1 - ((_s - _o) ** 2).sum() / ((_o - _o.mean()) ** 2).sum()
_amp = abs(_s.max() - _o.max()) / _o.max() * 100
_tpk = abs(int(_s.argmax()) - int(_o.argmax()))
axa.text(0.012, 0.97, 'SURGE, 3 Jacobi iterations\n'
         + r'$\mathrm{KGE}_c$ = %.3f,  $\mathrm{NSE}_c$ = %.3f' % (_kge, _nse) + '\n'
         + r'$E^{\mathrm{amp}}_c$ = %.1f%%,  $E^{\mathrm{time}}_c$ = %d days' % (_amp, _tpk),
         transform=axa.transAxes, ha='left', va='top', fontsize=8.0, linespacing=1.35)
print('Amazon mouth k=3: KGE %.4f NSE %.4f Eamp %.2f%% Etime %d d' % (_kge, _nse, _amp, _tpk))
axa.grid(False, which='both')
fig.savefig(f'{FIG}/poc_jacobi_validation.png', dpi=600); fig.savefig(f'{FIG}/poc_jacobi_validation.pdf')   # constrained_layout; no bbox=tight
print('wrote poc_jacobi_validation.png (hydrograph only)')
print('tab:accuracy rows (serial-C run):',
      {k: round(float(np.percentile(SC['err_k%d' % k], 99)) * 100, 2) for k in ks_all})
print('wavefront vs serialC maxrel:', res['wavefront_vs_serialC_maxrel'])
print('focused/blocked:', res['k'].get('focused_3+5'), res['k'].get('blocked_gs_3'))
