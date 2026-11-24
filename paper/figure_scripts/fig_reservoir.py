"""Figure 9 of the manuscript: the full GRanD dam set (7,320 dams, 6,501 to 7,270 reservoir cells).
(a) wall-clock added by the dams to GPU-ordered and SURGE at 5, 3 and 1 arcmin, with and without
the dams timed in the same job on one A100; (b) 1 - KGE at the 5th percentile against the iteration
count on the 5 arcmin network, with and without the dams.
Data: figure_data/reservoir_timing.json, reservoir_convergence.json, reservoir_k7_scores.json."""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper1_style, panel_label; setup_paper1_style()
except Exception as e:
    print('paper_style unavailable:', e)
# text tiers of Figure 1 (fig_method_seine.py): letters 11, titles 10.5, labels and annotations 9, ticks 8
FS_LETTER, FS_TITLE, FS_ANN = 11.0, 10.5, 9.0
plt.rcParams.update({'font.size': FS_ANN, 'axes.labelsize': FS_ANN, 'axes.titlesize': FS_TITLE,
                     'legend.fontsize': FS_ANN, 'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0})
# built at the text width, so the sizes set in paper_style are the sizes that print
PW = 166.0 / 25.4
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(_HERE, '..', 'figure_data')

order = ['5\'', '3\'', '1\'']
C_WF = '#858585';   # the grey of GPU-ordered in Figure 7
C_JE = '#E69F00'; C_JF = '#0072B2'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PW, 2.42 * PW / 5.3))
# (a) what the dams actually cost: percentage added, per scheme and resolution
RT = json.load(open(os.path.join(DATA, 'reservoir_timing.json')))['rows']
# with and without dams come from the same job (2026-11-24), and the runs without dams are
# also the ones of Table S5 (resolution_scaling.json)
schemes = [('gpu_ordered', C_WF, 'GPU-ordered'),
           ('surge', C_JF, 'SURGE')]
xpos = np.arange(len(order)); w = 0.34
for s, (key, col, lab) in enumerate(schemes):
    tot = [100.0 * (RT[r][key]['dams'] / RT[r][key]['none'] - 1.0) for r in order]
    xb = xpos + (s - 0.5) * w
    ax1.bar(xb, tot, width=w, color=col, edgecolor='k', linewidth=0.5, label=lab, zorder=3)
    for x, v in zip(xb, tot):
        ax1.annotate('%.1f' % v, (x, v), textcoords='offset points', xytext=(0, 3),
                     ha='center', va='bottom', fontsize=FS_ANN, color='0.25')
ax1.axhline(0, color='k', lw=0.7, zorder=2)
ax1.set_xticks(xpos)
ax1.set_xticklabels(['%s arcmin\n{:,}\nreservoir cells\n$k$ = %d'.format(RT[r]['reservoir_cells']) % (r.rstrip("'"), RT[r]['surge']['k'])
                     for r in ("5'", "3'", "1'")])   # the cell counts the text quotes
ax1.tick_params(axis='x', labelsize=FS_ANN)   # the group labels are words, not tick numbers
ax1.set_ylabel('Increase in wall-clock with reservoirs included [%]')
ax1.set_ylim(0, 19)
# both titles sit a little higher, so the one of b clears the 10^0 at the top of its axis
TPAD = plt.rcParams['axes.titlepad'] + 6.0
panel_label(ax1, 'a', pad_pt=TPAD).set_fontsize(FS_LETTER); ax1.set_title('Reservoirs add one to eight percent to SURGE', loc='center', pad=TPAD, x=0.55)   # shifted right, clear of the letter
ax1.legend(loc='upper center', ncol=2, fontsize=FS_ANN, handlelength=1.4, handletextpad=0.5,
           columnspacing=1.0, borderpad=0.3, framealpha=0.9, facecolor='white', edgecolor='none')
ax1.grid(False, which='both')

# (b) the criterion of section 4.1 with and without the dams, over the 2010 year
CV = json.load(open(os.path.join(DATA, 'reservoir_convergence.json')))
THRESH = 0.01                                  # 1 - KGE at the 5th percentile, the threshold
for key, col, mk, lab, off, ha_ in (('without_dams', '0.5', 's--', 'SURGE without reservoirs', (-9, -31), 'right'),
                                    ('with_dams', '#0072B2', 'o-', 'SURGE with reservoirs',
                                     (-9, -17), 'right')):
    d = CV[key]
    ks_ = sorted((int(k) for k in d), key=int)
    ks_ = [k for k in ks_ if k <= 12]
    ax2.plot(ks_, [max(d[str(k)]['p5'], 1e-7) for k in ks_], mk, color=col, lw=1.2, ms=4,
             markeredgecolor='k', markeredgewidth=0.4, label=lab)
ax2.axhline(THRESH, color='#d1441f', lw=0.8, ls=(0, (4, 2)), zorder=1)
import matplotlib.transforms as mtransforms
ax2.text(0.995, THRESH * 1.12,   # worded and placed as in Figure 6: right end, just above the line
         'Threshold: $\\mathrm{KGE} \\geq 0.99$\n(95% of cells)',   # two lines, clear of the curves
         transform=mtransforms.blended_transform_factory(ax2.transAxes, ax2.transData),
         ha='right', va='bottom', fontsize=FS_ANN, color='#d1441f')
# the scores at the seven iterations the network needs, in the empty corner below the curves,
# written as in Figure 5f; columns drawn separately so the numbers line up
S7 = json.load(open(os.path.join(DATA, 'reservoir_k7_scores.json')))
_rows = [('At $k=7$', 'Median $\\mathrm{KGE}$', 'Max $|\\mathrm{bias}|$')]
for key, lab in (('with_dams', 'With reservoirs'), ('without_dams', 'Without reservoirs')):
    _rows.append((lab, '%.4f' % S7[key]['kge_median'], '%.2f%%' % (100 * S7[key]['bias_absmax'])))
# at 9 pt the columns are placed from the widths of their widest entries, with a fixed gap
fig.canvas.draw(); _r0 = fig.canvas.get_renderer()
def _wfrac(txt):
    _t = ax2.text(0, 0, txt, transform=ax2.transAxes, fontsize=FS_ANN)
    _w = _t.get_window_extent(_r0).width / ax2.get_window_extent(_r0).width
    _t.remove(); return _w
_colw = [max(_wfrac(row[c]) for row in _rows) for c in range(3)]
_gap = 0.045
_x0 = 0.995 - sum(_colw) - 2 * _gap          # last column ends at the right frame
_xs = (_x0, _x0 + _colw[0] + _gap, _x0 + _colw[0] + _colw[1] + 2 * _gap)
for i, row in enumerate(_rows):
    for x, s in zip(_xs, row):
        ax2.text(x, 0.93 - i * 0.075, s, transform=ax2.transAxes, fontsize=FS_ANN, color='k',
                 ha='left', va='baseline')
print('table columns at', np.round(_xs, 3))
ax2.set_yscale('log')
ax2.set_xlabel('Parallel iterations $k$')
ax2.set_ylabel('$1 - \\mathrm{KGE}$ at the 5th percentile')
ax2.set_xticks([1, 3, 5, 7, 9, 11]); ax2.set_xlim(0.5, 12.5); ax2.set_ylim(2e-4, 1.0)
panel_label(ax2, 'b', pad_pt=TPAD).set_fontsize(FS_LETTER); ax2.set_title('Reservoirs do not change convergence', loc='center', pad=TPAD)
# each curve is named beside it by a leader, in place of a key
_wd, _nd = CV['with_dams'], CV['without_dams']
# vertical leaders: the run with reservoirs keeps its label right of its point at k = 4 and
# drops a leader to the curve at k = 6; the run without them is named under the curves at
# k = 10, with a leader up to its point there
fig.canvas.draw(); _r = fig.canvas.get_renderer()
_tw = ax2.text(4.55, _wd['4']['p5'], 'SURGE with reservoirs', fontsize=FS_ANN, color='#0072B2',
               ha='left', va='center')
_bb = _tw.get_window_extent(_r).transformed(ax2.transData.inverted())
ax2.plot([6, 6], [_bb.y0, _wd['6']['p5'] * 1.12], color='#1A1A1A', lw=0.6)
_tn = ax2.text(10, 6.5e-4, 'SURGE without reservoirs', fontsize=FS_ANN, color='0.35',
               ha='center', va='center')
_bb = _tn.get_window_extent(_r).transformed(ax2.transData.inverted())
ax2.plot([10, 10], [_bb.y1, _nd['10']['p5'] / 1.12], color='#1A1A1A', lw=0.6)
ax2.grid(False, which='both')

from paper_style import myriad_ticks; myriad_ticks(fig)   # tick numbers in Myriad Pro, symbols in STIX
fig.savefig(os.path.join(FIG, 'fig_reservoir.png'), dpi=600); fig.savefig(os.path.join(FIG, 'fig_reservoir.pdf'))
print('wrote fig_reservoir.png')
print('reservoir overhead, SURGE:', {r: round(RT[r]['surge']['dams'] / RT[r]['surge']['none'], 3) for r in order})
