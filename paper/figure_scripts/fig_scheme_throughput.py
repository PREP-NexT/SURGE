"""Throughput & synchronisation cost of the three schemes Fig.1 sketches
(serial / wavefront / Jacobi). (a) measured per-substep wall-clock across global
resolutions; (b) synchronisation barriers per substep -- the mechanism: serial = N
(one per cell), wavefront = L (one per level), Jacobi = 3 (constant). A100, fp32.
Data: figure_data/scheme_bench.json (bench_wavefront.py on ASPIRE A100)."""
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

A = json.load(open(os.path.join(DATA, 'scheme_bench.json')))['nodam']
cells = np.array([r['N'] for r in A], float)
lv = np.array([r['levels'] for r in A], float)
serial = np.array([r['serialC'] for r in A]); wf = np.array([r['wavefront'] for r in A])
je = np.array([r['jacobi_eager'] for r in A]); jf = np.array([r['jacobi_fused'] for r in A])
res = [r['res'] for r in A]

C_SER = '0.35'; C_WF = '#D55E00'; C_JE = '#E69F00'; C_JF = '#0072B2'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 3.2))

# (a) per-substep wall-clock
ax1.loglog(cells, serial, '^-', color=C_SER, lw=1.2, ms=4.5, label='CPU-serial')
ax1.loglog(cells, wf, 's--', color=C_WF, lw=1.3, ms=4.5, label='GPU-wavefront')
ax1.loglog(cells, je, 'o-', color=C_JE, lw=1.2, ms=4, label='GPU-eager')
ax1.loglog(cells, jf, 'o-', color=C_JF, lw=1.4, ms=5, label='GPU-fused')
ax1.annotate('%.0f$\\times$' % (wf[-1] / jf[-1]), (cells[-1], (wf[-1] * jf[-1]) ** 0.5),
             fontsize=7.5, color=C_WF, ha='right', fontweight='bold')
ax1.set_xlabel('River cells in global network'); ax1.set_ylabel('Wall-clock per substep [ms]')
ax1.set_title('(a) GPU-wavefront gains little over CPU-serial')
ax1.legend(loc='upper left'); ax1.grid(False, which='both')

# (b) synchronisation barriers per substep (the mechanism)
ax2.loglog(cells, cells, '^-', color=C_SER, lw=1.2, ms=4.5, label='CPU-serial')
ax2.loglog(cells, lv, 's--', color=C_WF, lw=1.3, ms=4.5, label='GPU-wavefront')
ax2.loglog(cells, np.full_like(cells, 3), 'o-', color=C_JF, lw=1.4, ms=5, label='GPU-eager')
for x, l, rr in zip(cells, lv, res):
    ax2.annotate(rr, (x, l), textcoords='offset points', xytext=(2, -9), fontsize=7.0, color=C_WF)
ax2.set_xlabel('River cells in global network'); ax2.set_ylabel('Synchronisation barriers per substep')
ax2.set_title('(b) The barrier count sets the cost')
ax2.legend(loc='upper left'); ax2.grid(False, which='both')

fig.savefig(os.path.join(FIG, 'fig_scheme_throughput.png'), dpi=600); fig.savefig(os.path.join(FIG, 'fig_scheme_throughput.pdf'))
print('wrote fig_scheme_throughput.png')
print('wavefront vs fused Jacobi:', dict(zip(res, np.round(wf / jf).astype(int))))
print('wavefront vs serial-C:', dict(zip(res, np.round(serial / wf, 2))))
