"""Figure 2, the Keynote drawing of the implicit difference, redrawn in matplotlib.

Every position, size and colour is read from figure_data/river_reach_text.json and
figure_data/river_reach_shapes.json, which measure_river_reach_text.py and
measure_river_reach_shapes.py extract from the Keynote export
paper1_gpu_routing/figures/river_reach.png. Nothing here is placed by eye.

The drawing works in the pixel coordinates of that export (2424 x 620, y downwards) and
the figure is built at its printed size, the 166 mm text width, so it goes in at
\\linewidth without scaling. Words are Myriad Pro and the equations STIX, as
in the Keynote original.

Writes paper1_gpu_routing/figures/fig_river_reach.{pdf,svg,png}, plus a PNG
at exactly the export's pixel grid for a pixel-by-pixel comparison. The original is left
untouched.
"""
import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'figure_data')
OUTDIR = os.environ.get('SURGE_FIGURE_DIR', os.path.join(HERE, '..', 'figures'))

for p in fm.findSystemFonts(fontpaths='/usr/share/fonts/opentype/MyriadPro'):
    try:
        fm.fontManager.addfont(p)
    except Exception:
        pass
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Myriad Pro', 'DejaVu Sans', 'Arial'],
    'mathtext.fontset': 'stix',
    # the label positions were measured on unhinted text, so the output is unhinted too
    'text.hinting': 'no_hinting', 'text.hinting_factor': 1,
})

T = json.load(open(os.path.join(DATA, 'river_reach_text.json')))
SH = json.load(open(os.path.join(DATA, 'river_reach_shapes.json')))
WPX, HPX = SH['width_px'], SH['height_px']
WMM, HMM = T['width_mm'], T['height_mm']
PX2PT = T['mm_per_px'] * 72.0 / 25.4          # line widths are given in points
MM = 1 / 25.4

fig = plt.figure(figsize=(WMM * MM, HMM * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WPX)
ax.set_ylim(HPX, 0)
ax.axis('off')


def z(n):
    return dict(zorder=n)


# --relabel moves the drawing of panel a and the labels that sit on it left by SHIFT_A_MM, away
# from panel b. The title, the inflow labels and the channel-length label stay where they are
import sys
SHIFT_A_MM = 5.0 if '--relabel' in sys.argv else 0.0
SHIFT_A_LABELS = ('lateral', 'xsec', 'xsec_area', 'outflow')
_d = SHIFT_A_MM / T['mm_per_px']        # px
for f_ in SH['faces']:
    f_['polygon'] = [[x_ - _d, y_] for x_, y_ in f_['polygon']]
    # the fitted gradients are functions of the pixel position, so they move with the face
    f_['rgb_coef'] = [[c_[0] + c_[1] * _d, c_[1], c_[2]] for c_ in f_['rgb_coef']]
for e_ in SH['edges']:
    e_['p'] = [e_['p'][0] - _d, e_['p'][1]]
    e_['q'] = [e_['q'][0] - _d, e_['q'][1]]
SH['block_arrows']['polygons'] = [[[x_ - _d, y_] for x_, y_ in pl] for pl in SH['block_arrows']['polygons']]
for a_ in SH['lateral_arrows']:
    a_['x'] -= _d
SH['dimension']['p'][0] -= _d
SH['dimension']['q'][0] -= _d
for tk_ in SH['dimension']['ticks']:
    tk_['x0'] -= _d
    tk_['x1'] -= _d
SH['xsec_arrow']['base_x'] -= _d
SH['xsec_arrow']['tip_x'] -= _d
SH['xsec_arrow']['dots'] = [[x_ - _d, y_, sz_] for x_, y_, sz_ in SH['xsec_arrow']['dots']]


# ---- (a) the prism: faces with their fitted gradients, then the edges over them -----
for f in SH['faces']:
    P = np.array(f['polygon'])
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    nx, ny = int(np.ceil(x1 - x0)) + 2, int(np.ceil(y1 - y0)) + 2
    XX, YY = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    img = np.stack([c[0] + c[1] * XX + c[2] * YY for c in f['rgb_coef']], -1) / 255.0
    shade = ax.imshow(np.clip(img, 0, 1), extent=(x0, x1, y1, y0), origin='upper',
                      interpolation='bilinear', **z(1))
    clip = Polygon(P, closed=True, facecolor='none', edgecolor='none')
    ax.add_patch(clip)
    shade.set_clip_path(clip)

# each edge is drawn in short pieces, each in the colour measured over that stretch, so an
# edge keeps the tint of the translucent face that covers part of it. Projecting caps let
# neighbouring pieces overlap and leave no seam; the two corners get round ends
for e in SH['edges']:
    p0, q0 = np.array(e['p']), np.array(e['q'])
    fr = np.array(e['colour_fr'])
    cut = np.r_[0.0, 0.5 * (fr[1:] + fr[:-1]), 1.0]
    pts = p0 + cut[:, None] * (q0 - p0)
    lw = e['width_px'] * PX2PT
    ax.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], 1), colors=e['colours'],
                                     linewidths=lw, capstyle='projecting', zorder=3))
    for end, colr in ((p0, e['colours'][0]), (q0, e['colours'][-1])):
        ax.plot([end[0]], [end[1]], marker='o', ms=lw, mfc=colr, mec='none', zorder=3)

ba = SH['block_arrows']
for poly in ba['polygons']:
    ax.add_patch(Polygon(poly, closed=True, facecolor=ba['colour'], edgecolor='none', **z(4)))

# the five lateral-flow arrows are copies of one Keynote arrow; the copies whose head narrows
# at one pixel per row, which no edge crosses, give its shape, and each copy keeps its own top
LA = SH['lateral_arrows']
clean = [a for a in LA if abs(a['slope'] + 1.0) < 0.1]
LEN = float(np.median([a['tip'] - a['top'] for a in clean]))
HL = float(np.median([a['tip'] - a['head_base'] for a in clean]))
SW = float(np.median([a['shaft_px'] for a in clean]))
HW = float(np.median([a['head_px'] for a in clean]))
for a in LA:
    tip = a['top'] + LEN
    base = tip - HL
    ax.add_patch(Rectangle((a['x'] - SW / 2, a['top']), SW, base - a['top'] + 0.5,
                           facecolor='k', edgecolor='none', **z(5)))
    ax.add_patch(Polygon([(a['x'] - HW / 2, base), (a['x'] + HW / 2, base), (a['x'], tip)],
                         closed=True, facecolor='k', edgecolor='none', **z(5)))

dm = SH['dimension']
p, q = np.array(dm['p']), np.array(dm['q'])
u = (q - p) / np.linalg.norm(q - p)
nv = np.array([-u[1], u[0]])
pa, qa = p + dm['head_len_px'] * u, q - dm['head_len_px'] * u
ax.plot([pa[0], qa[0]], [pa[1], qa[1]], color='k', lw=dm['line_px'] * PX2PT,
        solid_capstyle='butt', **z(5))
for tip, b in ((p, pa), (q, qa)):
    ax.add_patch(Polygon([b + nv * dm['head_px'] / 2, b - nv * dm['head_px'] / 2, tip],
                         closed=True, facecolor='k', edgecolor='none', **z(5)))
for tk in dm['ticks']:
    ax.add_patch(Rectangle((tk['x0'], tk['y0']), tk['x1'] - tk['x0'], tk['y1'] - tk['y0'],
                           facecolor='k', edgecolor='none', **z(5)))

xa = SH['xsec_arrow']
ax.add_patch(Polygon([(xa['base_x'], xa['y'] - xa['half_h']), (xa['base_x'], xa['y'] + xa['half_h']),
                      (xa['tip_x'], xa['y'])], closed=True, facecolor='k', edgecolor='none', **z(5)))
for dx, dy, size in xa['dots']:
    ax.add_patch(Rectangle((dx - size / 2, dy - size / 2), size, size,
                           facecolor='k', edgecolor='none', **z(5)))

# ---- (b) the stencil -------------------------------------------------------------------
st = SH['stencil']
for k, ln in st['lines'].items():
    for s in np.arange(ln['first'], ln['last'] - ln['dash'] / 2, ln['period']):
        if k.startswith('row'):
            r = Rectangle((s, ln['centre'] - ln['thick'] / 2), ln['dash'], ln['thick'])
        else:
            r = Rectangle((ln['centre'] - ln['thick'] / 2, s), ln['thick'], ln['dash'])
        r.set(facecolor='k', edgecolor='none', zorder=2)
        ax.add_patch(r)

for key, c in st['circles'].items():
    col = st['blue'] if key.startswith('A_') else st['orange']
    if c['kind'] == 'open':
        ax.add_patch(Circle((c['x'], c['y']), 0.5 * (c['r_out'] + c['r_in']), facecolor='white',
                            edgecolor=col, lw=(c['r_out'] - c['r_in']) * PX2PT, **z(6)))
    else:
        ax.add_patch(Circle((c['x'], c['y']), c['r_out'], facecolor=col, edgecolor='none', **z(6)))

sa, ta = st['arrows']['space'], st['arrows']['time']
ax.add_patch(Rectangle((sa['tail'], sa['centre'] - sa['thick'] / 2), sa['base'] - sa['tail'] + 0.5,
                       sa['thick'], facecolor=st['orange'], edgecolor='none', **z(5)))
ax.add_patch(Polygon([(sa['base'], sa['centre'] - sa['head'] / 2), (sa['base'], sa['centre'] + sa['head'] / 2),
                      (sa['tip'], sa['centre'])], closed=True, facecolor=st['orange'],
                     edgecolor='none', **z(5)))
ax.add_patch(Rectangle((ta['centre'] - ta['thick'] / 2, ta['base'] - 0.5), ta['thick'],
                       ta['tail'] - ta['base'] + 0.5, facecolor=st['blue'], edgecolor='none', **z(5)))
ax.add_patch(Polygon([(ta['centre'] - ta['head'] / 2, ta['base']), (ta['centre'] + ta['head'] / 2, ta['base']),
                      (ta['centre'], ta['tip'])], closed=True, facecolor=st['blue'],
                     edgecolor='none', **z(5)))

# ---- every label, token by token, at its measured baseline -----------------------------
# The two panel titles follow paper1_gpu_routing/画图规范与经验.md: the panel letter is a bold
# 10 pt lower-case letter where "(a)" and "(b)" stood, and the title that follows is one 9 pt
# line on the measured baseline. Every other label keeps its measured size and place.
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import sys
LETTER_PT, TITLE_PT, LETTER_GAP_MM = 10.0, 9.0, 1.2

# --relabel writes a trial whose symbols follow the text: panel a carries the continuous
# quantities of equation (1), panel b the corner values of equation (3). Each new label keeps
# the anchor of the label it replaces: the start of the word it follows, the right edge of the
# labels left of the upstream line, the left edge of those right of the downstream line, and
# the centre of the two area labels
RELABEL = '--relabel' in sys.argv
_R = fig.canvas.get_renderer()


def width_mm(s, size):
    tx = fig.text(0, 0, s, fontsize=size)
    w = tx.get_window_extent(_R).width / fig.dpi * 25.4
    tx.remove()
    return w


def put(x_mm, base_mm, s, size, colour, ha='left'):
    fig.text(x_mm / WMM, 1.0 - base_mm / HMM, s, fontsize=size, color=colour, ha=ha,
             va='baseline', zorder=10)


def find_tok(lab, name, role=None):
    for t_ in lab['tokens']:
        if t_['token'] == name and (role is None or t_.get('role') == role):
            return t_
    raise KeyError((lab['name'], name, role))


def run(lab, start_tok, parts):
    # parts: (string, size) drawn one after another from the start token, word spacing kept
    x = start_tok['x_mm']
    base = start_tok['baseline_mm']
    col = start_tok['colour']
    for i_, (s, size) in enumerate(parts):
        put(x, base, s, size, col)
        x += width_mm(s, size) + (1.0 if i_ < len(parts) - 1 else 0.0)


PX = T['mm_per_px']
MATH_PT = 10.9
UNIT_PT = 10.0
INFLOW_DOWN_MM = 2.0
# panel a names the quantities without units: the text gives them right after equation (1), and
# without them the labels end well clear of panel b
RELABELS = {
    'lateral': lambda lab: run(lab, find_tok(lab, 'Lateral'), [('Lateral inflow', find_tok(lab, 'Lateral')['size_pt']),
                                                        (r'$q(\xi,\tau)$', MATH_PT)]),
    # "Inflow Q(x,tau)" on one line reaches the corner of the box, so the function goes on the
    # line where the unit stood
    # with panel a moved left the inflow arrow reaches Q(x,tau), so both inflow lines drop by
    # INFLOW_DOWN_MM, below the tail of the arrow
    'inflow': lambda lab: run(lab, dict(find_tok(lab, 'Inflow'),
                                        baseline_mm=find_tok(lab, 'Inflow')['baseline_mm'] + INFLOW_DOWN_MM),
                              [('Inflow', find_tok(lab, 'Inflow')['size_pt'])]),
    'inflow_unit': lambda lab: put(lab['tokens'][0]['x_mm'], lab['tokens'][0]['baseline_mm'] + INFLOW_DOWN_MM,
                                   r'$Q(x,\tau)$', MATH_PT, lab['tokens'][0]['colour']),
    'xsec_area': lambda lab: run(lab, find_tok(lab, 'area'), [('area', find_tok(lab, 'area')['size_pt']),
                                                       (r'$A(\xi,\tau)$', MATH_PT)]),
    'outflow': lambda lab: run(lab, find_tok(lab, 'Outflow'), [('Outflow', find_tok(lab, 'Outflow')['size_pt']),
                                                        (r'$Q(x+\Delta x,\tau)$', MATH_PT)]),
    'length': lambda lab: [put(t_['x_mm'], t_['baseline_mm'], t_['token'], t_['size_pt'], t_['colour'])
                           for t_ in lab['tokens'] if 'role' not in t_],
}


def corner(lab, s, how):
    xs0 = min(t_['box_px'][0] for t_ in lab['tokens']) * PX
    xs1 = max(t_['box_px'][2] for t_ in lab['tokens']) * PX
    base = [t_ for t_ in lab['tokens'] if t_.get('role') == 'base'][0]
    col = base['colour']
    if how == 'right':
        put(xs1, base['baseline_mm'], s, MATH_PT, col, ha='right')
    elif how == 'left':
        put(xs0, base['baseline_mm'], s, MATH_PT, col, ha='left')
    else:
        put(0.5 * (xs0 + xs1), base['baseline_mm'], s, MATH_PT, col, ha='center')


RELABELS.update({
    'Qin_tdt': lambda lab: corner(lab, r'$Q(x,t+\Delta t)$', 'right'),
    'Qin_t': lambda lab: corner(lab, r'$Q(x,t)$', 'right'),
    'Qout_tdt': lambda lab: corner(lab, r'$Q(x+\Delta x,t+\Delta t)$', 'left'),
    'Qout_t': lambda lab: corner(lab, r'$Q(x+\Delta x,t)$', 'left'),
    'A_tdt': lambda lab: corner(lab, r'$A(t+\Delta t)$', 'center'),
    'A_t': lambda lab: corner(lab, r'$A(t)$', 'center'),
})
for lab in T['labels']:
    if lab['name'] in SHIFT_A_LABELS:
        for t_ in lab['tokens']:
            t_['x_mm'] -= SHIFT_A_MM
            t_['box_px'] = [t_['box_px'][0] - SHIFT_A_MM / PX, t_['box_px'][1],
                            t_['box_px'][2] - SHIFT_A_MM / PX, t_['box_px'][3]]
for lab in T['labels']:
    if RELABEL and lab['name'] in RELABELS:
        RELABELS[lab['name']](lab)
        continue
    if lab['name'] in ('title_a', 'title_b'):
        head, words = lab['tokens'][0], lab['tokens'][1:]
        letter = head['token'].strip('()')
        base = float(np.mean([w_['baseline_mm'] for w_ in words]))
        fig.text(head['x_mm'] / WMM, 1.0 - base / HMM, letter, fontsize=LETTER_PT,
                 fontweight='bold', color='#000000', ha='left', va='baseline', zorder=10)
        adv = TextPath((0, 0), letter, size=LETTER_PT,
                       prop=FontProperties(family='Myriad Pro', weight='bold')).get_extents().x1 * 25.4 / 72
        title = ' '.join(w_['token'] for w_ in words).replace('$t$ $+$ $\\Delta t$', '$t + \\Delta t$')
        fig.text((head['x_mm'] + adv + LETTER_GAP_MM) / WMM, 1.0 - base / HMM, title,
                 fontsize=TITLE_PT, color=words[0]['colour'], ha='left', va='baseline', zorder=10)
        continue
    for tok in lab['tokens']:
        fig.text(tok['x_mm'] / WMM, 1.0 - tok['baseline_mm'] / HMM, tok['token'],
                 fontsize=tok['size_pt'], color=tok['colour'], ha='left', va='baseline', zorder=10)

# not river_reach: the PNG written below would overwrite the Keynote export the measurements read
base = os.path.join(OUTDIR, '_trial_river_reach_relabel' if '--relabel' in sys.argv else 'fig_river_reach')
fig.savefig(base + '.pdf')
fig.savefig(base + '.svg')
# the PNGs are rasterised from the PDF by Cairo rather than drawn by Agg. Agg snaps text to
# whole output pixels, so only a raster of the PDF shows the figure the paper prints; the
# second one lands on the Keynote export's own 2424 x 620 grid for a pixel comparison
subprocess.run(['pdftocairo', '-r', '600', '-png', '-singlefile', base + '.pdf', base],
               check=True, capture_output=True)
subprocess.run(['pdftocairo', '-scale-to-x', str(WPX), '-scale-to-y', str(HPX), '-png',
                '-singlefile', base + '.pdf', base + '_grid'], check=True, capture_output=True)
print('wrote %s.{pdf,svg,png} and %s_grid.png' % (base, base))
print('lateral-arrow template: length %.0f px, head %.0f x %.0f px, shaft %.0f px (from %d clean copies)'
      % (LEN, HW, HL, SW, len(clean)))
