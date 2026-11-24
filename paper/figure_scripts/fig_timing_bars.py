"""Figure: every compute configuration at every resolution in one bar chart.

Wall-clock per simulated day on a log axis, four bars per resolution, and the speedup
over CPU-serial printed above each bar. Replaces the former resolution-scaling and scheme-throughput figures.
Data: figure_data/resolution_scaling.json (unified timing protocol).
"""
import os, sys, json
import numpy as np
import matplotlib, matplotlib.ticker
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import setup_paper1_style, panel_label
setup_paper1_style()
# Fig 1 size tier: letter 11, title 10.5, labels and annotations 9, tick numbers 8
FS_LETTER, FS_TITLE, FS_ANN = 11.0, 10.5, 9.0
plt.rcParams.update({'axes.titlesize': FS_TITLE, 'axes.labelsize': FS_ANN,
                     'legend.fontsize': FS_ANN, 'font.size': FS_ANN})
PW = 166.0 / 25.4
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'figure_data')
OUT = os.path.join(HERE, '..', 'figures', 'fig_timing_bars')

rows = json.load(open(os.path.join(DATA, 'resolution_scaling.json')))['rows']
# only SURGE is in colour; the three it is compared with are greys of different depth,
# the same in both panels
CONF = [('serialC', 'CPU-serial (1 core)', '#3F3F3F'),
        ('cpu_parallel', 'CPU-parallel (24 cores)', '#C4C4C4'),
        ('gpu_ordered', 'GPU-ordered (1 GPU)', '#858585'),
        ('surge', 'SURGE (1 GPU)', '#0072B2')]
resname = {"30'": "30'", "15'": "15'", "5'": "5'", "3'": "3'", "1'": "1'"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(PW, 2.85 * PW / 5.3), layout='constrained',
                              gridspec_kw=dict(width_ratios=(1.9, 1.0)))
SPD = 20          # routing time steps in a simulated day (Table A1)
n = len(rows); w = 0.2
x = np.arange(n)
for j, (key, label, col) in enumerate(CONF):
    t = np.array([r[key] for r in rows]) * SPD / 1e3      # ms a time step -> s a simulated day
    xb = x + (j - 1.5) * w
    ax.bar(xb, t, width=w, color=col, edgecolor='k', linewidth=0.4, label=label, zorder=3)
ax.set_yscale('log')
# the wall-clock the text quotes for CPU-serial and SURGE at 1 arc-minute, written
# upright above their bars because the bars are narrower than the numbers
def _secs(v):
    return '%.0f ms' % (v * 1e3) if v < 1 else ('%.1f s' % v if v < 100 else '%.0f s' % v)
for i, r in enumerate(rows):
    if r['res'] not in ("1'", "5'"):
        continue
    for j, key in ((0, 'serialC'), (3, 'surge')):
        v = r[key] * SPD / 1e3
        ax.annotate(_secs(v), (x[i] + (j - 1.5) * w, v), xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', rotation=90, fontsize=FS_ANN, color='k')
ax.set_ylim(1e-3, 1e5)
ax.set_xticks(x)
def _cells(n):
    return '%.0fk' % (n / 1e3) if n < 1e6 else '%.1fM' % (n / 1e6)


ax.set_xticklabels(['%s arcmin\n%s cells\n%s levels\n$k$ = %d'
                    % (r['res'].rstrip("'"), _cells(r['N']), '{:,}'.format(r['levels']), r['k'])
                    for r in rows])
ax.set_ylabel('Wall-clock per simulated day [s]')
ax.legend(loc='upper left', ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)
ax.grid(False, which='both')
ax.set_title('Wall-clock per simulated day', loc='center')

# (b) the same runs as a speedup over CPU-serial, against the size of the river network
cells = np.array([r['N'] for r in rows], float)
for key, label, col in CONF[1:]:
    sp = np.array([r['serialC'] / r[key] for r in rows])
    ax2.plot(cells, sp, 'o-', color=col, lw=1.2, ms=5, markeredgecolor='k', markeredgewidth=0.4,
             label=label.split(' (')[0], zorder=3)
ax2.axhline(1.0, color=CONF[0][2],   # the grey of CPU-serial in a
            lw=0.7, ls=(0, (4, 2)), zorder=2)
ax2.annotate('One CPU core', xy=(1.4e8, 1.0), xytext=(3, 0), textcoords='offset points',
             ha='left', va='center', fontsize=FS_ANN, color=CONF[0][2], annotation_clip=False)
for c, r in zip(cells, rows):
    sp = r['serialC'] / r['surge']
    ax2.annotate('{:,.0f}'.format(sp), (c, sp), textcoords='offset points', xytext=(0, 7),
                 ha='center', va='bottom', fontsize=FS_ANN, fontweight='bold', color=CONF[3][2])
# the range of GPU-ordered that the text quotes, at its two ends
_go = [r['serialC'] / r['gpu_ordered'] for r in rows]
for c, sp, dy, va in ((cells[0], _go[0], 6, 'bottom'), (cells[-1], _go[-1], -7, 'top')):
    ax2.annotate(('%.2f' if sp < 1 else '%.1f') % sp, (c, sp), textcoords='offset points', xytext=(0, dy),
                 ha='center', va=va, fontsize=FS_ANN, fontweight='bold', color=CONF[2][2])
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlim(3e4, 1.4e8); ax2.set_ylim(0.15, 1500)
ax2.set_xlabel('River cells in the network')
ax2.set_ylabel('Speedup')
# each curve is named at its right end, in its own colour, in place of a key
for key, label, col in CONF[2:]:
    sp_end = rows[-1]['serialC'] / rows[-1][key]
    ax2.annotate(label.split(' (')[0], (cells[-1], sp_end), xytext=(7, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=FS_ANN, color=col if key == 'surge' else '0.25',
                 annotation_clip=False)
# CPU-parallel ends on the CPU-serial line, so it is named above its highest stretch instead
_cp = np.array([r['serialC'] / r['cpu_parallel'] for r in rows])
_i = int(np.argmax(_cp)); _i0 = _i - 1 if _cp[_i - 1] >= _cp[min(_i + 1, n - 1)] else _i + 1
_a, _b = sorted((_i0, _i))
ax2.annotate('CPU-parallel', (np.sqrt(cells[_a] * cells[_b]), _cp[_i]), xytext=(0, 6),
             textcoords='offset points', ha='center', va='bottom', fontsize=FS_ANN, color='#6E6E6E')   # darker than its pale curve so it reads
ax2.grid(False, which='both')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.tick_params(top=False, right=False)
# the resolution of each river network on a second axis along the top, above each point
_sec = ax2.secondary_xaxis('top')
_sec.set_xticks(cells, [r['res'].rstrip("'") for r in rows])
_sec.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
_sec.tick_params(pad=1.5)
ax2.xaxis.labelpad = 1.5
ax2.set_title('Speedup over CPU-serial', loc='center', pad=9)   # clear of the resolution numbers

# the title of b sits above the resolution axis; the title of a is lifted to the same line,
# and each letter sits on that line at the left of its frame. The layout is frozen first so
# the two stay level when the figure is saved
# the resolution row is moved, with the layout still live, until its numbers end level with
# the four-line labels under a; then the layout is frozen
fig.canvas.draw()
_r = fig.canvas.get_renderer()
fig.set_layout_engine('none')
fig.canvas.draw()
_last = max(_sec.get_xticklabels(), key=lambda t: t.get_window_extent(_r).x1).get_window_extent(_r)
_xy = ax2.transAxes.inverted().transform((_last.x1, 0.5 * (_last.y0 + _last.y1)))
ax2.annotate('arcmin', xy=_xy, xycoords='axes fraction', xytext=(4, 0), textcoords='offset points',
             ha='left', va='center', fontsize=FS_ANN, annotation_clip=False)
_dy = ax2.title.get_window_extent(_r).y0 - ax.title.get_window_extent(_r).y0
ax._autotitlepos = False                       # keep the y set here
ax.title.set_y(ax.title.get_position()[1] + _dy / ax.bbox.height)
fig.canvas.draw()
for _a, _l in ((ax, 'a'), (ax2, 'b')):
    _yb = _a.transAxes.inverted().transform((0, _a.title.get_window_extent(_r).y0))[1]
    _a.annotate(_l, xy=(0, _yb), xycoords='axes fraction', xytext=(-1.2 * 72 / 25.4, 0),
                textcoords='offset points', ha='right', va='bottom', fontsize=FS_LETTER,
                fontweight='bold', annotation_clip=False)
from paper_style import myriad_ticks; myriad_ticks(fig)   # tick numbers in Myriad Pro, symbols in STIX
fig.savefig(OUT + '.png', dpi=600); fig.savefig(OUT + '.pdf')
print('wrote', OUT + '.{png,pdf}')
