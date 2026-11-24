"""One row per river network, the shortfall in KGE against the number of parallel
iterations. Each iteration count carries a gradient bar (density along the bar), a thin
5-95 per cent interval and a median dot, after ggdist's gradientinterval. Rows share the
iteration axis, so the same k can be compared down the column, and each row picks out its
own crossing, the first k at which 95 per cent of its cells reach THRESH, drawing that
iteration count at full strength and the others faded.
"""
import os, sys, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper1_style, panel_label; setup_paper1_style()
except Exception as e:
    print('paper_style unavailable:', e)
# maths stays in STIX from paper_style, as in every other figure
# paper_style asks for a constrained layout, and the PDF writer reinstates it at save time even
# when the figure was built without one, which threw away the margins set here and moved every
# axes in the PDF alone. Turning it off in the parameters keeps the PDF and the PNG identical.
matplotlib.rcParams['figure.constrained_layout.use'] = False
# the text tiers of Figure 1 (fig_method_seine.py): letters 11, row titles
# 10.5, annotations and labels 9, tick numbers 8; set here, paper_style is shared
FS_LETTER, FS_TITLE, FS_ANN, FS_TICK = 11.0, 10.5, 9.0, 8.0
matplotlib.rcParams.update({'font.size': FS_ANN, 'axes.labelsize': FS_ANN,
                            'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK})
PW = 166.0 / 25.4
_HERE = os.path.dirname(os.path.abspath(__file__))
# the per-cell metrics of bench_year_resolution.py, one folder per network, if you have them;
# without them the script draws from the summary in figure_data/
SRC = os.environ.get('SURGE_YEAR_METRICS', os.path.join(_HERE, '..', 'figure_data', 'year_metrics'))
OUT = os.path.join(_HERE, '..', 'figures', 'fig_iterations_by_resolution.png')
NETS = [('30min_area', '30 arcmin', 116), ('15min_area', '15 arcmin', 251),
        ('05min_aspire', '5 arcmin', 782), ('03min_area', '3 arcmin', 1436),
        ('01min_area', '1 arcmin', 4696)]
THRESH, FLOOR = 0.99, 1e-7
YLO, YHI = np.log10(FLOOR), 0.6
YAX = YLO - 0.55, YHI + 0.15          # axis limits, clear of the clamped ends


def ramp(name, rgb):
    c = LinearSegmentedColormap.from_list(name, rgb)(np.linspace(0, 1, 256))
    c[:, 3] = np.linspace(0, 1, 256)                 # opacity carries the density
    return LinearSegmentedColormap.from_list(name + '_a', c)


COOL, WARM = ramp('cool', ['#7fb3c8', '#1f6f8b']), ramp('warm', ['#f0a882', '#d1441f'])
PLANNED = ()                    # every network has its results, so no column is held open
avail = NETS                    # a network without results yet keeps its row as a placeholder


# The per-cell year metrics are hundreds of megabytes, so the repository carries the
# summary they reduce to, the density and the percentiles this figure draws. The script
# reads the per-cell file where it is available and the summary otherwise, and both give
# the same figure.
SUMMARY = os.path.join(_HERE, '..', 'figure_data', 'iterations_by_resolution.npz')
_SUM = np.load(SUMMARY) if os.path.exists(SUMMARY) else None


def has(tag):
    return (os.path.exists(os.path.join(SRC, tag, 'metrics.npz'))
            or (_SUM is not None and tag + '_ks' in _SUM.files))


def _cells(tag):
    f = os.path.join(SRC, tag, 'metrics.npz')
    return np.load(f) if os.path.exists(f) else None


def p5_kge(tag, k):
    M = _cells(tag)
    if M is not None:
        return float(np.percentile(M['kge_k%d' % k].astype(float), 5))
    return float(_SUM['%s_p5kge%d' % (tag, k)])


def density_and_quartiles(tag, k, bins):
    """The density over the bins and the 5th, 50th and 95th percentiles of log10(1-KGE)"""
    M = _cells(tag)
    if M is not None:
        v = np.log10(np.maximum(1.0 - M['kge_k%d' % k].astype(float), FLOOR))
        return np.histogram(v, bins=bins, density=True)[0], np.percentile(v, [5, 50, 95])
    return _SUM['%s_d%d' % (tag, k)].astype(float), _SUM['%s_q%d' % (tag, k)].astype(float)


def ks_of(tag):
    """Iteration counts to draw for a network: up to one past its crossing, no further,
    since a network that already matches gains nothing from the iterations beyond."""
    M = _cells(tag)
    if M is not None:
        ks = sorted(int(m.group(1)) for m in (re.match(r'kge_k(\d+)$', k) for k in M.files) if m)
    else:
        ks = [int(k) for k in _SUM[tag + '_ks']]
    cr = crossing(tag, ks)
    return ks if cr is None else [k for k in ks if k <= ks[min(ks.index(cr) + 1, len(ks) - 1)]]


def crossing(tag, ks):
    return next((k for k in ks if p5_kge(tag, k) >= THRESH), None)


KS_ALL = sorted({k for t, _, _ in avail if has(t) for k in ks_of(t)} | set(PLANNED))
CHOSEN = {}                     # network -> (median Courant number, iterations it needs)
# the network name sits in a top corner inside the frame, on the side the chosen k is not on
fig, axs = plt.subplots(len(avail), 1, figsize=(PW, (28 * len(avail) + 22) / 25.4), sharex=True,
                        constrained_layout=False)   # paper_style turns it on, and it would undo the margins
fig.subplots_adjust(left=0.115, right=0.985, top=0.94, bottom=0.075, hspace=0.34)
grid = np.linspace(YLO, YHI, 400)
PAD = matplotlib.rcParams['axes.titlepad']      # the offset panel_label uses for the letters
for row, (ax, (tag, lab, lv)) in enumerate(zip(np.atleast_1d(axs), avail)):
    panel_label(ax, 'abcde'[row]).set_fontsize(FS_LETTER)
    if not has(tag):
        ax.set_ylim(*YAX); ax.set_xlim(-0.6, len(KS_ALL) - 0.4)
        ax.yaxis.set_major_locator(FixedLocator([-6, -4, -2, 0]))
        ax.yaxis.set_major_formatter(FixedFormatter(['$10^{%d}$' % t for t in [-6, -4, -2, 0]]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.axhline(np.log10(1 - THRESH), color='#d1441f', lw=0.7, ls=(0, (4, 2)), zorder=3)
        ax.annotate('%s, %s levels' % (lab, format(lv, ',')), xy=(0, 1), xycoords='axes fraction',
                    xytext=(0, PAD), textcoords='offset points', ha='left', va='baseline',
                    fontsize=FS_TITLE)      # same baseline as the panel letter
        ax.text(0.5, 0.45, 'results pending', transform=ax.transAxes, ha='center',
                va='center', fontsize=FS_ANN, color='0.45')
        ax.grid(False, which='both')
        print('%-13s placeholder' % tag)
        continue
    KS = ks_of(tag)
    cross = crossing(tag, KS)
    if cross is not None:
        ax.axvspan(KS_ALL.index(cross) - 0.5, KS_ALL.index(cross) + 0.5, color='#f2efe4', zorder=0)
    for k in KS:
        i = KS_ALL.index(k)
        near = k == cross          # only the chosen iteration count is at full strength
        hist, q = density_and_quartiles(tag, k, grid)
        edges = grid
        d = np.convolve(hist, np.ones(9) / 9, mode='same')
        d = d / max(d.max(), 1e-12) * (1.0 if near else 0.5)
        mid = 0.5 * (edges[:-1] + edges[1:])
        above = mid > np.log10(1 - THRESH)
        for mask, cmap in ((~above, COOL), (above, WARM)):
            ax.imshow(np.where(mask, d, np.nan)[:, None], extent=(i - 0.34, i + 0.34, YLO, YHI),
                      origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1,
                      interpolation='bilinear', zorder=2)
        col = '0.15' if near else '0.55'
        ax.plot([i] * 2, [q[0], q[2]], color=col, lw=0.7, zorder=4)
        ax.plot([i], [q[1]], 'o', color='white' if not near else col, ms=2.6 if not near else 3.2,
                markeredgecolor=col, markeredgewidth=0.7, zorder=5)
    ax.axhline(np.log10(1 - THRESH), color='#d1441f', lw=0.7, ls=(0, (4, 2)), zorder=3)
    ax.set_ylim(*YAX); ax.set_xlim(-0.6, len(KS_ALL) - 0.4)
    ticks = [-6, -4, -2, 0]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(['$10^{%d}$' % t for t in ticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.annotate('%s, %s levels' % (lab, format(lv, ',')), xy=(0, 1), xycoords='axes fraction',
                xytext=(0, PAD), textcoords='offset points', ha='left', va='baseline', fontsize=FS_TITLE)
    if cross is not None:
        ax.annotate('$k=%d$' % cross, xy=(KS_ALL.index(cross), YHI), xytext=(0, -1),
                    textcoords='offset points', ha='center', va='top', fontsize=FS_ANN,
                    fontweight='bold')
        # how many times fewer sequential steps than the level-by-level scheme, beside the k
        # label, on its left in the last row where k sits near the right end of the axis
        _kl = ax.texts[-1]
        _left = KS_ALL.index(cross) > 0.7 * len(KS_ALL)
        ax.annotate('(%d-fold fewer)' % round(lv / cross), xy=(0.0 if _left else 1.0, 0.5),
                    xycoords=_kl, xytext=(-3 if _left else 3, 0), textcoords='offset points',
                    ha='right' if _left else 'left', va='center', fontsize=FS_ANN,
                    color='k')   # words in the text face, as elsewhere
        print('   %s: %d levels / k=%d -> %d-fold fewer' % (tag, lv, cross, round(lv / cross)))
    else:
        ax.text(0.01, 0.93, 'not reached by $k=%d$' % KS[-1], transform=ax.transAxes,
                ha='left', va='top', fontsize=FS_ANN, color='0.35')
    _M = _cells(tag)
    CHOSEN[tag] = (float(np.median(_M['courant'].astype(float))) if _M is not None else float('nan'), cross)
    if row == 0 and 3 in KS:                    # one real bar carries the labels, so no separate key
        j = KS_ALL.index(3)
        _, _q3 = density_and_quartiles(tag, 3, grid)
        lo95, med95 = _q3[0], _q3[1]

        def label(text, y, colour, x_from):
            """Horizontal leader from the bar itself to the text on its right."""
            ax.annotate(text, xy=(x_from, y), xytext=(j + 0.95, y), ha='left', va='center',
                        fontsize=FS_ANN, color=colour,
                        arrowprops=dict(arrowstyle='-|>', lw=0.5, color=colour, shrinkA=1, shrinkB=2,
                                        mutation_scale=6, facecolor=colour, edgecolor=colour))
        label('spread across cells', med95 + 1.4, '#1f6f8b', j + 0.34)   # placed so the text clears the threshold line
        label('median cell', med95, '0.35', j + 0.06)
        label('5th to 95th percentile', lo95 + 0.3, '0.35', j)
    ax.grid(False, which='both')
    print('%-13s crossing k = %s' % (tag, cross))
print('crossings: %s' % {t: v[1] for t, v in CHOSEN.items()})

axs[-1].set_xticks(range(len(KS_ALL)))
axs[-1].set_xticklabels([str(k) for k in KS_ALL])
axs[-1].set_xlabel('Parallel iterations $k$')
# the threshold is named on the first row, sitting on its own line
import matplotlib.transforms as mtransforms
axs[0].text(0.995, np.log10(1 - THRESH) + 0.12,   # at the right end of the line, clear of the bars
            'Threshold: $\\mathrm{KGE} \\geq %g$ (95%% of cells)' % THRESH,
            transform=mtransforms.blended_transform_factory(axs[0].transAxes, axs[0].transData),
            ha='right', va='bottom', fontsize=FS_ANN, color='#d1441f')
fig.text(0.010, 0.5, '$1-\\mathrm{KGE}$ against the serial solution', fontsize=FS_ANN,
         rotation=90, ha='left', va='center')
from paper_style import myriad_ticks; myriad_ticks(fig)   # tick numbers in Myriad Pro, symbols in STIX
fig.savefig(OUT, dpi=600); fig.savefig(OUT[:-4] + '.pdf')
print('wrote', OUT)
