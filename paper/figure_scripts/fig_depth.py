"""Depth-independence figure (paper1), measured across five global resolutions.
Each network's Jacobi k-sweep truncation is compared to the EXACT wavefront on the
same network, at three runoff levels bracketing the operational Courant regime.
Panel (a): k=3 p99 error vs median Courant, coloured by #levels -> all resolutions
collapse onto one error(Cr) curve, so error is set by Courant, not by depth.
Panel (b): sweeps k needed for p99<5% vs #levels -> flat (k~3) across 116->4696."""
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

# measured: {label: (levels, [(Cr_med, p99_k1,p99_k2,p99_k3,p99_k5,p99_k8), ...3 runoff])}
DATA = {
 "30'": (116,  [(0.23,0.131,0.0376,0.0121,0.00137,7.71e-5),
                (0.43,0.249,0.117,0.0568,0.0143,0.00192),
                (0.83,0.398,0.261,0.174,0.0783,0.0227)]),
 "15'": (251,  [(0.27,0.159,0.0536,0.0195,0.00296,2.13e-4),
                (0.51,0.291,0.151,0.0816,0.0247,0.00457),
                (0.98,0.439,0.306,0.217,0.111,0.0421)]),
 "5'":  (782,  [(0.17,0.118,0.0332,0.00997,9.91e-4,3.75e-5),
                (0.33,0.234,0.105,0.0485,0.0105,0.00116),
                (0.62,0.381,0.237,0.150,0.0592,0.0152)]),
 "3'":  (1436, [(0.42,0.235,0.107,0.0511,0.0128,0.00193),
                (0.79,0.389,0.250,0.164,0.0728,0.0230),
                (1.51,0.530,0.419,0.337,0.219,0.117)]),
 "1'":  (4696, [(0.45,0.205,0.0831,0.0359,0.00788,0.00114),
                (0.86,0.364,0.220,0.137,0.0563,0.0168),
                (1.64,0.514,0.399,0.314,0.199,0.104)]),
}

kidx = {1:1,2:2,3:3,5:4,8:5}
levels = [v[0] for v in DATA.values()]
import matplotlib.cm as cm
norm = plt.Normalize(np.log10(min(levels)), np.log10(max(levels)))
cmap = cm.viridis

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.1))

# (a) k=3 p99 vs Courant, coloured by levels
for lab, (lv, rows) in DATA.items():
    cr = [r[0] for r in rows]; p99 = [r[kidx[3]] for r in rows]
    c = cmap(norm(np.log10(lv)))
    ax1.plot(cr, p99, 'o-', color=c, lw=1.2, ms=5, label=f'{lab} ({lv:,} levels)')
# theory envelope g^(k+1)/(1-g), k=3
crx = np.linspace(0.1, 1.6, 50); g = crx/(1+crx)
ax1.plot(crx, g**4/(1-g), 'k--', lw=0.9, alpha=0.7, label='$g^{4}/(1-g)$ theory')
ax1.set_xlabel('Median Courant number'); ax1.set_ylabel('99th percentile of $E^{\mathrm{rel}}_{c,k}$ at $k{=}3$')
ax1.set_yscale('log'); ax1.set_title('(a) The error is set by the Courant number')
ax1.legend(loc='lower right')
ax1.grid(False, which='both')

# (b) k=3 p99 error vs #levels, at MATCHED Courant (~0.45): isolates depth from Cr
TARGET_CR = 0.45
lv_sorted = sorted(DATA.items(), key=lambda kv: kv[1][0])
xs, ys, crs = [], [], []
for lab, (lv, rows) in lv_sorted:
    row = min(rows, key=lambda r: abs(r[0] - TARGET_CR))   # regime closest to target Cr
    xs.append(lv); ys.append(row[kidx[3]]); crs.append(row[0])
ax2.plot(xs, ys, 'o-', color='#0072B2', lw=1.4, ms=6)
ax2.set_xscale('log'); ax2.set_ylim(0, max(ys) * 1.35)
ax2.set_xlabel('Topological levels in network')
ax2.set_ylabel('$E^{\mathrm{rel}}_{c,k}$ at $k{=}3$ and $\\mathrm{Cr}\\approx0.45$')
ax2.set_title('(b) The error is independent of network depth')
ax2.grid(False, which='both')
below = {0, 2}   # first and third points: label below so text clears the curve
for i, (x, y, cr) in enumerate(zip(xs, ys, crs)):
    dy, va = (-9, 'top') if i in below else (7, 'bottom')
    ha, dx = ('left', 2) if i == 0 else ('center', 0)   # first point: left-align so it clears the y-axis
    ax2.annotate('$\\mathrm{Cr}=%.2f$' % cr, (x, y), textcoords='offset points', xytext=(dx, dy),
                 fontsize=7.0, ha=ha, va=va, color='0.4')
print('panel b (levels, k3p99, Cr):', list(zip(xs, [round(v,3) for v in ys], crs)))

out = os.path.join(FIG, 'fig_depth_independence.png')
fig.savefig(out, dpi=600); fig.savefig(out[:-4] + '.pdf')   # constrained_layout handles spacing; no bbox=tight (house style)
print('wrote', out, '| resolutions:', list(DATA.keys()))
