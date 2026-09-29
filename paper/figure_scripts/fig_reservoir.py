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
    from paper_style import setup_paper1_style, panel_label; setup_paper1_style()
except Exception as e:
    print('paper_style unavailable:', e)
# built at the text width, so the sizes set in paper_style are the sizes that print
PW = 166.0 / 25.4
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PW, 2.42 * PW / 5.3))
# (a) what the dams actually cost: percentage added, per scheme and resolution
RT = json.load(open(os.path.join(DATA, 'reservoir_timing.json')))['rows']
schemes = [('gpu_ordered', C_WF, 'GPU-ordered'),
           ('surge', C_JF, 'SURGE')]
xpos = np.arange(len(order)); w = 0.34
for s, (key, col, lab) in enumerate(schemes):
    tot = [100.0 * (RT[r][key]['dams'] / RT[r][key]['none'] - 1.0) for r in order]
    xb = xpos + (s - 0.5) * w
    ax1.bar(xb, tot, width=w, color=col, edgecolor='k', linewidth=0.5, label=lab, zorder=3)
    for x, v in zip(xb, tot):
        ax1.annotate('%.1f' % v, (x, v), textcoords='offset points', xytext=(0, 3),
                     ha='center', va='bottom', fontsize=8.0, color='0.25')
ax1.axhline(0, color='k', lw=0.7, zorder=2)
ax1.set_xticks(xpos)
ax1.set_xticklabels(['5 arcmin\n$k$ = %d' % RT["5'"]['surge']['k'],
                     '3 arcmin\n$k$ = %d' % RT["3'"]['surge']['k'],
                     '1 arcmin\n$k$ = %d' % RT["1'"]['surge']['k']])
ax1.set_ylabel('Increase in wall-clock with the dams [%]')
ax1.set_ylim(0, 17)
panel_label(ax1, 'a'); ax1.set_title('Dams add one to eight per cent to SURGE')
ax1.legend(loc='upper center', ncol=2, fontsize=8.0, handlelength=1.4, handletextpad=0.5,
           columnspacing=1.0, borderpad=0.3, framealpha=0.9, facecolor='white', edgecolor='none')
ax1.grid(False, which='both')

# (b) the criterion of section 4.1 with and without the dams, over the 2010 year
CV = json.load(open(os.path.join(DATA, 'reservoir_convergence.json')))
THRESH = 0.01                                  # 1 - KGE at the 5th percentile, the threshold
for key, col, mk, lab, off, ha_ in (('without_dams', '0.5', 's--', 'SURGE without dams', (-9, -31), 'right'),
                                    ('with_dams', '#0072B2', 'o-', 'SURGE with dams',
                                     (-9, -17), 'right')):
    d = CV[key]
    ks_ = sorted((int(k) for k in d), key=int)
    ks_ = [k for k in ks_ if k <= 12]
    ax2.plot(ks_, [max(d[str(k)]['p5'], 1e-7) for k in ks_], mk, color=col, lw=1.2, ms=4,
             markeredgecolor='k', markeredgewidth=0.4, label=lab)
ax2.axhline(THRESH, color='#d1441f', lw=0.8, ls=(0, (4, 2)), zorder=1)
ax2.text(12.4, THRESH * 1.12, 'Threshold, $\\mathrm{KGE}_c \\geq 0.99$\n'
         'at 95% of cells', fontsize=7.5, color='#d1441f', ha='right', va='bottom',
         linespacing=1.25)      # above the line and right of where the curves cross it
ax2.set_yscale('log')
ax2.set_xlabel('The number of parallel iterations $k$')
ax2.set_ylabel('1 $-$ KGE$_c$ at the 5th percentile')
ax2.set_xticks([1, 3, 5, 7, 9, 11]); ax2.set_xlim(0.5, 12.5); ax2.set_ylim(8e-4, 1.0)
panel_label(ax2, 'b'); ax2.set_title('Reservoirs do not change convergence')
ax2.legend(loc='upper right', fontsize=8.0, labelspacing=0.35, borderpad=0.45, handlelength=1.6,
           handletextpad=0.5, framealpha=0.85, facecolor='white', edgecolor='none')
ax2.grid(False, which='both')

fig.savefig(os.path.join(FIG, 'fig_reservoir.png'), dpi=600); fig.savefig(os.path.join(FIG, 'fig_reservoir.pdf'))
print('wrote fig_reservoir.png')
print('reservoir overhead, SURGE:', {r: round(RT[r]['surge']['dams'] / RT[r]['surge']['none'], 3) for r in order})
