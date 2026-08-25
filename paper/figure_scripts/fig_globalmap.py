"""Global validation figure (last figure): left = discharge map with the 12
validation river mouths lettered a-l; right = routed vs CWatM at those mouths,
same letters. Panels equal height, side by side. Run in geoplot conda env."""
import os
import sys, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from paper_style import setup_paper_style; setup_paper_style(base_size=8.0)
except Exception as e:
    print('paper_style unavailable:', e)
_HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(_HERE, '..', 'figures'); os.makedirs(FIG, exist_ok=True)
DATA = os.path.join(_HERE, '..', 'figure_data')

OUT = os.path.join(FIG, 'fig_global_validation.png')
# compact plotting data: discharge cells >50 m3/s (subsampled) + the 12 mouths
m = np.load(os.path.join(DATA, 'global_map.npz'))
lon, lat, logq = m['lon'], m['lat'], m['logq']
rows = json.load(open(os.path.join(DATA, 'rivermouths.json')))   # top-12 mouths, precomputed
LAB = [str(i + 1) for i in range(len(rows))]          # rivers numbered 1-12
names = [x['name'] for x in rows]
ours = np.array([x['routed'] for x in rows]); cwat = np.array([x['cwat'] for x in rows])
mlons = np.array([x['lon'] for x in rows]); mlats = np.array([x['lat'] for x in rows])
medabs = float(np.median(np.abs(ours / cwat - 1)))
halo = [pe.withStroke(linewidth=1.6, foreground='white')]

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(7.8, 2.4))

# ---- (a) map: TRUE aspect (no stretch); fills width, letterbox cropped by tight bbox ----
axm = fig.add_axes([0.04, 0.05, 0.52, 0.86], projection=proj)
axm.set_extent([-180, 180, -56, 84], crs=proj)
axm.add_feature(cfeature.LAND, facecolor='#efefec', edgecolor='none', zorder=0)
axm.add_feature(cfeature.COASTLINE, linewidth=0.22, edgecolor='#9a9a9a', zorder=1)
sc = axm.scatter(lon, lat, c=logq, s=0.05, cmap='viridis',
                 vmin=1.7, vmax=5.2, transform=proj, rasterized=True, zorder=2)
axm.scatter(mlons, mlats, s=15, facecolors='none', edgecolors='#d62728', linewidths=0.9,
            transform=proj, zorder=4)
for i, (mlo, mla) in enumerate(zip(mlons, mlats)):
    axm.text(mlo + 3, mla + 3, LAB[i], fontsize=6.6, fontweight='bold', color='#b81414',
             transform=proj, zorder=5, path_effects=halo)
axm.set_title('(a) Annual-mean discharge routed from the 2010 global runoff')
cax = inset_axes(axm, width='2.2%', height='100%', loc='lower left',
                 bbox_to_anchor=(1.015, 0.0, 1, 1), bbox_transform=axm.transAxes, borderpad=0)
cb = fig.colorbar(sc, cax=cax, orientation='vertical')
cb.set_label('$\log_{10} Q$ [m$^3$ s$^{-1}$]'); cb.ax.tick_params(labelsize=7.0)

# ---- (b) scatter: square, same drawn height as the map, just right of it ----
fig.canvas.draw()
mp = axm.get_position()
sqw = mp.height * fig.get_figheight() / fig.get_figwidth()   # width fraction for a square axes
axs = fig.add_axes([mp.x1 + 0.155, mp.y0, sqw, mp.height])
axs.loglog(cwat, ours, 'o', ms=4.5, color='#0072B2', markeredgecolor='black',
           markeredgewidth=0.6, alpha=0.9, zorder=3)
axs.plot(cwat, ours, 'o', ms=1.5, color='black', zorder=6)   # centre dot the leaders point to
lo, hi = 1e4, 2e5
axs.plot([lo, hi], [lo, hi], 'k--', lw=0.8, zorder=1)
for f in (1.2, 0.8):
    axs.plot([lo, hi], [lo*f, hi*f], ':', color='grey', lw=0.6, zorder=1)
# name the reference lines at their upper ends (45-degree slope on the square
# log-log axes), with a white halo so they stay legible over the guides
_ref_halo = [pe.withStroke(linewidth=1.8, foreground='white')]
axs.text(1.00e5, 1.00e5 * 1.06, '1:1', fontsize=6.5, color='black',
         rotation=45, ha='center', va='bottom', path_effects=_ref_halo, zorder=5)
axs.text(0.80e5, 0.80e5 * 1.2 * 1.06, '+20%', fontsize=6.2, color='grey',
         rotation=45, ha='center', va='bottom', path_effects=_ref_halo, zorder=5)
axs.text(1.30e5, 1.30e5 * 0.8 * 0.94, '−20%', fontsize=6.2, color='grey',
         rotation=45, ha='center', va='top', path_effects=_ref_halo, zorder=5)
# labels sit off the crowded diagonal: even indices to the upper-left triangle,
# odd to the lower-right, each with a thin leader and a white halo so they never
# overlap a marker or each other (points are monotonic along the diagonal).
_txt_halo = [pe.withStroke(linewidth=1.6, foreground='white')]
for i in range(len(rows)):
    if i % 2 == 0:
        dx, dy, ha, va = -12, 10, 'right', 'bottom'
    else:
        dx, dy, ha, va = 12, -10, 'left', 'top'
    axs.annotate(LAB[i], (cwat[i], ours[i]), textcoords='offset points', xytext=(dx, dy),
                 fontsize=6.6, fontweight='bold', color='#222', ha=ha, va=va, zorder=4,
                 path_effects=_txt_halo,
                 arrowprops=dict(arrowstyle='-', lw=0.5, color='#9aa0a6',
                                 shrinkA=0.0, shrinkB=0.0))   # leader reaches the marker centre
axs.set_xlim(8.5e3, 2.3e5); axs.set_ylim(8.5e3, 2.3e5)
axs.set_xlabel('CWatM discharge [m$^3$ s$^{-1}$]')
axs.set_ylabel('GPU-eager discharge [m$^3$ s$^{-1}$]')
axs.set_title('(b) The median $E^{\mathrm{mouth}}_r$ is %.0f%%' % (100*medabs))
axs.tick_params(labelsize=7.0)

# PNG only: dense 5e4-point global scatter is inherently raster; vector PDF
# misplaces the rasterized layer under cartopy + tight bbox.
fig.savefig(OUT, dpi=600, bbox_inches='tight'); plt.close(fig)
print('wrote', OUT, '| median|ratio-1|=%.1f%%' % (100*medabs))
print('numbers:', ', '.join('%s=%s' % (LAB[i], names[i]) for i in range(len(rows))))
