"""How the Kling-Gupta efficiency against the CWatM serial-C kernel improves with the number
of Jacobi iterations, the river network it is measured on, and one hydrograph.

(a-d) KGE after one, three, seven and twenty iterations, on one shared scale so the panels
compare; (e) annual-mean discharge of the serial solution, the Amazon mouth circled;
(f) daily discharge at the Amazon mouth for CPU-serial and SURGE with one to three
iterations, with the section 3.3 metrics of the three-iteration series. Only the evaluation
set is mapped, the 106,661 cells whose annual-mean discharge exceeds 100 m3/s, so the
summary in each map is over those cells and not over the whole network. Panel f was
Figure 5 before 2026-09-15.

Layout follows PREP-NexT/DOMINO-SEE (plot_global_degree_tele.py): ultraplot places the
panel grid and the colour bars; panel f alone is placed by hand after the first draw, its
y label on the left frame of the right-hand maps and its right end on the end of the KGE
bar label.
Data: figure_data/metric_maps_serialC.npz, figure_data/accuracy.json. Run in the geoplot conda env.
"""
import os, sys
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import ultraplot as pplt
import cartopy.crs as ccrs

pplt.rc['font.sans-serif'] = 'Myriad Pro'
# the other figures carry the panel letter inside a left-aligned, unbolded title
# (paper_style: axes.titlelocation left, axes.titleweight normal). ultraplot's own
# abc label is a separate bold letter beside a centred title, so it is switched off
# and the letter written into the title string instead. Those figures are drawn 5.3 in
# wide and scaled to the 6.54 in text width, so their 8.4 pt titles print at 10.4 pt;
# this figure is already 6.54 in, so it needs that size nominally to match on the page
# sizes from paper1_gpu_routing/画图规范与经验.md: 9 pt titles, 8 pt for everything else, and
# equations in STIX as in Figure 2
pplt.rc['title.size'] = 9.0
pplt.rc['font.size'] = 8.0
pplt.rc['label.size'] = 8.0
pplt.rc['tick.labelsize'] = 8.0
pplt.rc_matplotlib['mathtext.fontset'] = 'stix'
pplt.rc['title.weight'] = 'normal'
pplt.rc['title.loc'] = 'left'
pplt.rc['savefig.dpi'] = 600
pplt.rc_matplotlib['pdf.fonttype'] = 42
pplt.rc_matplotlib['ps.fonttype'] = 42

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from paper_style import panel_label
FIG = os.path.join(_HERE, '..', 'figures')
DATA = os.path.join(_HERE, '..', 'figure_data')
z = np.load(os.path.join(DATA, 'metric_maps_serialC.npz'))
lon, lat = z['lon'].astype(float), z['lat'].astype(float)
ann = z['ann'].astype(float)
KS = (1, 3, 7, 20)      # 7 is the iterations this network needs (Figure 3c, Table 6)
print('%d cells' % lon.size)

# one scale for the three efficiency panels, its lower end from the worst case
LO = float(np.floor(np.percentile(z['kge_k1'].astype(float), 2) * 20) / 20)
NORM = mcolors.Normalize(LO, 1.0)
TICKS = list(np.round(np.linspace(LO, 1.0, 5), 2))
# the pale end of either ramp vanishes against the land, and most cells sit at KGE = 1,
# so both are trimmed and the river network stays legible everywhere
CMAP_K = pplt.Colormap('dense_r', right=0.78)
CMAP_Q = pplt.Colormap('matter', left=0.12)
print('shared KGE scale %.2f .. 1.00' % LO)

# the four efficiency maps take the top two rows, the discharge map and the hydrograph
# the bottom row; every map is one column wide, so all five print at the same size
# the deepmind class gives 472.3 pt of text width, so the figure is drawn at that size
# and \includegraphics[width=\linewidth] leaves the type at its designed size
fig = pplt.figure(figwidth=6.54, dpi=200, share=False)
# subplot numbers follow the axes order below (1 discharge, 2 hydrograph, 3-6 KGE), and
# the array puts them in reading order a-f; the narrow column gap keeps the maps at their
# earlier size, and panel f is moved into place by hand, so it needs no gap of its own
axs = fig.subplots([[3, 4], [5, 6], [1, 2]], proj={1: 'robin', 3: 'robin', 4: 'robin', 5: 'robin', 6: 'robin'},
                   hspace=(2.5, 2.5), wspace=0.4)
MAPS = [axs[i] for i in (0, 2, 3, 4, 5)]
for _ax in MAPS:
    _ax.format(abc=False, coast=True, coastlinewidth=0.3, coastcolor='gray6',
               land=True, landcolor='gray2', labels=False, grid=False)
axs[1].format(abc=False)
# Antarctica carries no routed cells, so the view is cut at 60 S and the space it
# would take goes to the rivers instead. set_extent would clip to a rectangle in
# projected coordinates and flatten the curved sides of the ellipse, so only the
# vertical range is set and the horizontal one is left global
_PROJ = ccrs.Robinson()
_YLO = _PROJ.transform_point(0, -60, ccrs.PlateCarree())[1]
_YHI = _PROJ.transform_point(0, 85, ccrs.PlateCarree())[1]
for _ax in MAPS:
    _ax.set_global()
    _ax.set_ylim(_YLO, _YHI)

o = np.argsort(ann)                                   # large rivers drawn last
sc_q = axs[0].scatter(lon[o], lat[o], c=ann[o], s=0.05, cmap=CMAP_Q,
                      norm=mcolors.LogNorm(100, 2e5), linewidths=0,
                      transform=ccrs.PlateCarree(), rasterized=True, zorder=2)
axs[0].format(title='Annual-mean discharge from CPU-serial')
axs[0].scatter([-50.708], [-0.458], s=22, facecolors='none', edgecolors='k', linewidths=0.8,
               transform=ccrs.PlateCarree(), zorder=7)
# the circled cell is the Amazon mouth, whose hydrograph is panel f; the label sits in
# the open South Pacific, the only sea near the circle wide enough for it, with a leader
_px, _py = _PROJ.transform_point(-50.708, -0.458, ccrs.PlateCarree())
_tx, _ty = _PROJ.transform_point(-50.708 - 37.0, -0.458, ccrs.PlateCarree())   # the mouth's own
# latitude, which this projection draws as a horizontal line, so the leader is horizontal
axs[0].annotate('Amazon mouth\n(panel f)', xy=(_px, _py), xycoords='data', xytext=(_tx, _ty),
                textcoords='data', ha='right', va='center', fontsize=8.0, linespacing=1.1,
                arrowprops=dict(arrowstyle='-', lw=0.6, color='k', shrinkA=2.0, shrinkB=3.8),
                zorder=8, annotation_clip=False)
axs[0].colorbar(sc_q, loc='b', width=0.07, length=0.8, space=0.25,
                label='Annual-mean discharge [m$^3$ s$^{-1}$]',
                ticks=[100, 1000, 10000, 100000],
                ticklabels=['$10^{2}$', '$10^{3}$', '$10^{4}$', '$10^{5}$'],
                minorticks=[k * 10 ** e for e in (2, 3, 4) for k in range(2, 10)],
                extend='max')

# (f) the hydrograph at the Amazon mouth, the circled cell of panel e
import json
res = json.load(open(os.path.join(DATA, 'accuracy.json')))
H = res['hydro']; days = np.arange(len(H['serialC']))
ax = axs[1]
ax.plot(days, np.asarray(H['serialC']) / 1e3, color='k', lw=1.1, label='CPU-serial')
for kk, c in [(1, '#D55E00'), (3, '#E69F00'), (7, '#0072B2')]:
    ax.plot(days, np.asarray(H[str(kk)]) / 1e3, color=c, lw=0.7,
            label='SURGE, %d iteration%s' % (kk, '' if kk == 1 else 's'))
_o = np.asarray(H['serialC'], float); _s = np.asarray(H['7'], float)
_r = np.corrcoef(_s, _o)[0, 1]
_kge = 1 - np.sqrt((_r - 1) ** 2 + (_s.std() / _o.std() - 1) ** 2 + (_s.mean() / _o.mean() - 1) ** 2)
_nse = 1 - ((_s - _o) ** 2).sum() / ((_o - _o.mean()) ** 2).sum()
_bias = (_s.mean() / _o.mean() - 1) * 100
TITLE_F = 'Daily discharge at the Amazon mouth in 2010'
ax.format(xlabel='Day of 2010',
          ylabel='Discharge [10$^3$ m$^3$ s$^{-1}$]', xlim=(0, days[-1]), ylim=(50, 480),
          grid=False, xminorlocator='null', yminorlocator='null', ylabelpad=4)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)
ax.legend(loc='ur', ncols=2, frame=False, fontsize=7.0, handlelength=1.4, labelspacing=0.25,
          columnspacing=1.1)   # two by two, so it clears both the metrics and the inset
# a zoom on the April peak, where the iteration count shows itself: one iteration arrives
# nine days late, three two days, seven on the day
_zx, _zy = (85, 125), (225, 325)   # a round window, easy to read off the axes
axz = ax.inset_axes([0.545, 0.295, 0.430, 0.455], transform=ax.transAxes, zoom=False)
axz.set_facecolor('white'); axz.patch.set_alpha(1.0); axz.set_zorder(5)
axz.plot(days, _o / 1e3, color='k', lw=1.0)
for kk, c in [(1, '#D55E00'), (3, '#E69F00'), (7, '#0072B2')]:
    axz.plot(days, np.asarray(H[str(kk)], float) / 1e3, color=c, lw=0.8)
axz.set_xlim(*_zx); axz.set_ylim(*_zy)
axz.set_xticks([90, 100, 110, 120]); axz.set_yticks([250, 300])
axz.tick_params(labelsize=6.5, pad=1.5, length=2.0, top=False, right=False)
axz.grid(False, which='both')
for _sp in ('top', 'right'):
    axz.spines[_sp].set_visible(False)
# the window and the two leaders are drawn by hand, so the lines run from the right corners
# of the window to the left corners of the inset rather than corner to matching corner
from matplotlib.patches import ConnectionPatch
ax.add_patch(plt.Rectangle((_zx[0], _zy[0]), _zx[1] - _zx[0], _zy[1] - _zy[0], facecolor='none',
                           edgecolor='0.55', lw=0.5, zorder=4))
for _yr, _yi in ((_zy[1], 1.0), (_zy[0], 0.0)):
    ax.add_artist(ConnectionPatch(xyA=(_zx[1], _yr), coordsA=ax.transData,
                                  xyB=(0.0, _yi), coordsB=axz.transAxes,
                                  color='0.55', lw=0.5, zorder=4))

ax.text(0.03, 0.97, 'SURGE vs CPU-serial\n7 iterations\n' + r'NSE$_c$ %.3f' % _nse + '\n'
        + r'KGE$_c$ %.3f' % _kge + '\n' + r'bias$_c$ %+.2f%%' % _bias,
        transform=ax.transAxes, ha='left', va='top', fontsize=7.0, linespacing=1.3)

for i, k in enumerate(KS, start=2):
    v = z['kge_k%d' % k].astype(float)
    o = np.argsort(-v)                                # worst cells drawn last
    sc = axs[i].scatter(lon[o], lat[o], c=v[o], s=0.05, cmap=CMAP_K, norm=NORM,
                        linewidths=0, transform=ccrs.PlateCarree(),
                        rasterized=True, zorder=2)
    axs[i].format(title='SURGE with %d Jacobi iteration%s' % (k, '' if k == 1 else 's'))
    # at 8 pt the longer wording ran past the map frame; the caption says the statistics
    # are over the mapped cells
    # the summary sits in the open Atlantic, clear of both coasts. The explicit wording
    # runs to 96 per cent of the panel width and collides with the frame, so the second
    # line drops the qualifier that the first line and the caption already carry
    axs[i].text(-42, -35, 'Median KGE %.4f\n5th percentile KGE %.3f'
                % (np.median(v), np.percentile(v, 5)),
                transform=ccrs.PlateCarree(), fontsize=8.0, ha='left', va='top',
                zorder=6)

# both bars hang off the right edge, so every row loses the same width and the
# left frames line up
cb_k = fig.colorbar(sc, loc='r', rows=(1, 2), width=0.09, length=1.0, space=0.6, ticks=TICKS,
             extend='min', label='Kling-Gupta efficiency against CPU-serial')
for i_, ax_ in enumerate(axs):
    if i_ == 1:
        continue   # f is lettered after it is moved, in line with b
    panel_label(ax_, 'efabcd'[i_], pad_pt=pplt.rc['title.pad'])
# panel f spans from the left frame of the right-hand maps to the end of the KGE bar label
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_inv = fig.transFigure.inverted()
_pd = axs[3].get_position()
_lab = cb_k.ax.yaxis.label.get_window_extent(_r).transformed(_inv)
_pb = axs[1].get_position()
print('map b left %.4f, bar label right %.4f, f before %.4f..%.4f' % (_pd.x0, _lab.x1, _pb.x0, _pb.x1))
# the y label of f, not its axis, starts at the left frame of the right-hand maps
_yl = axs[1].yaxis.label.get_window_extent(_r).transformed(_inv)
_off = _pb.x0 - _yl.x0
_x0 = _pd.x0 + _off
axs[1].set_position([_x0, _pb.y0, _lab.x1 - _x0, _pb.height])
fig.canvas.draw()
_yl2 = axs[1].yaxis.label.get_window_extent(fig.canvas.get_renderer()).transformed(_inv)
# letter and title of f sit over the left frame of b, as the letter and title of b do
import matplotlib.transforms as mtransforms
_bt = mtransforms.blended_transform_factory(fig.transFigure, axs[1].transAxes)
_pad = pplt.rc['title.pad']
axs[1].annotate('f', xy=(_pd.x0, 1), xycoords=_bt, xytext=(-1.2 * 72.0 / 25.4, _pad),
                textcoords='offset points', ha='right', va='baseline', fontsize=10.0,
                fontweight='bold', annotation_clip=False)
axs[1].annotate(TITLE_F, xy=(_pd.x0, 1), xycoords=_bt, xytext=(0, _pad),
                textcoords='offset points', ha='left', va='baseline',
                fontsize=pplt.rc['title.size'], annotation_clip=False)
print('map widths [in]: a %.3f  c %.3f  d %.3f' % tuple(axs[i].get_position().width * fig.get_figwidth() for i in (0, 2, 3)))
print('f y label left %.4f vs map b left %.4f; f axis right %.4f vs bar label right %.4f' % (_yl2.x0, _pd.x0, axs[1].get_position().x1, _lab.x1))
fig.savefig(os.path.join(FIG, 'fig_kge_convergence.png'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_kge_convergence.pdf'), bbox_inches='tight')
print('wrote fig_kge_convergence.png/pdf')
