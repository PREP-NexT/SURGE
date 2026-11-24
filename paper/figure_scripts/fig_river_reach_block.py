"""Figure 2. Panel (a) is a reach of river cut through two hillslopes, seen level with the
water, with the lateral inflow arriving over the banks. Panel (b) is the cell in distance and
time, drawn as a parallelogram in the attitude of the reach in (a): the upstream end is the
upper edge and the downstream end the lower edge, the sides run parallel to the channel, and
time runs to the right. Its upper and lower edges are level with the top and foot of the block.

Each panel title is centred over its panel, and each letter stays at the left of its panel,
as in Figure 1. Every label is 9 pt.

Writes paper/figures/fig_river_reach.{pdf,png}."""
import os, json, subprocess
import matplotlib.patheffects
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'figure_data')
OUT  = os.path.join(HERE, '..', 'figures', 'fig_river_reach')
for p in fm.findSystemFonts(fontpaths='/usr/share/fonts/opentype/MyriadPro'):
    try: fm.fontManager.addfont(p)
    except Exception: pass
plt.rcParams.update({'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                     'font.family':'sans-serif',
                     'font.sans-serif':['Myriad Pro','DejaVu Sans','Arial'],
                     'mathtext.fontset':'stix',
                     'text.hinting':'no_hinting', 'text.hinting_factor':1})

T = json.load(open(os.path.join(DATA, 'river_reach_text.json')))
SH = json.load(open(os.path.join(DATA, 'river_reach_shapes.json')))
WPX, HPX = SH['width_px'], SH['height_px']
MM = 1/25.4
DROP = 26                      # both panels sit this much below the titles
PAD_B = 44 + DROP              # room under the block for the label of the section
fig = plt.figure(figsize=(T['width_mm']*MM, T['height_mm']*MM*(HPX+PAD_B)/HPX))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0, WPX); ax.set_ylim(HPX+PAD_B, 0); ax.axis('off')

# ---------------- the reach, seen level with the water --------------------------------
# The eye is level with the ground, so the cross-section at the near end is drawn true and
# the reach recedes to the upper right. v runs across the valley, w upwards, u downstream.
X0, Y0 = 622.0, 226.0 + DROP
RU, RV = 178.0, 200.0          # the reach recedes up and to the right, as the section does
SV, SW = 320.0,  56.0          # half width across, and the freeboard above the water
SOILW  =   1.15                # soil below the water line, in freeboards

VW   = 0.66                    # the valley reaches the flat ground at this half width
DEEP  = 1.05                   # the bed lies this far below the water line, in freeboards

def terrain(v):
    """One curve for the whole section, from the bed up to the flat ground. It leaves the
    bed and meets the ground with zero slope, so the wetted perimeter carries no corner."""
    t = np.clip(abs(v)/VW, 0, 1)
    return -DEEP + (DEEP + 1.0)*(0.5 - 0.5*np.cos(np.pi*t))

WCH = VW*np.arccos(1 - 2*DEEP/(DEEP + 1.0))/np.pi     # where that curve cuts the water line

def ground(v):                 # what the eye sees from above: water where the bed is wet
    return max(terrain(v), 0.0)

def bed(v):
    return terrain(v)

def proj(u, v, w):
    return (X0 - RU*u + SV*v, Y0 + RV*u - SW*w)

GREEN_HI, GREEN_LO = np.array([0.60,0.82,0.46]), np.array([0.42,0.66,0.34])
BLUE_HI,  BLUE_LO  = np.array([0.62,0.82,0.95]), np.array([0.32,0.62,0.86])
SOILPX_HI,  SOILPX_LO  = np.array([0.96,0.72,0.56]), np.array([0.84,0.58,0.44])
INK = '#1A1A1A'

VS = np.linspace(-1, 1, 400)

# the right face of the block, the one the recession leaves in view
right = ([proj(0.0, 1, ground(1)), proj(1.0, 1, ground(1))]
         + [proj(1.0, 1, -SOILW), proj(0.0, 1, -SOILW)])
ax.add_patch(Polygon(right, closed=True, facecolor=tuple(SOILPX_LO), edgecolor='none', zorder=1))

# the ground and the water, each a strip running from the near end to the far end
def strip(v0, v1, colour, z=2):
    P = [proj(0.0, v0, ground(v0)), proj(0.0, v1, ground(v1)),
         proj(1.0, v1, ground(v1)), proj(1.0, v0, ground(v0))]
    ax.add_patch(Polygon(P, closed=True, facecolor=colour, edgecolor='none', zorder=z))

NS = 90
for a_, b_, base, top in ((-1, -VW, GREEN_HI, GREEN_HI),          # flat ground, left
                          (-VW, -WCH, GREEN_HI, GREEN_LO),                # the bank
                          (-WCH, WCH, BLUE_HI, BLUE_HI),                  # the water surface
                          (WCH, VW, GREEN_LO, GREEN_HI),                  # the bank
                          (VW, 1, GREEN_HI, GREEN_HI)):                   # flat ground, right
    vs = np.linspace(a_, b_, NS+1)
    for k in range(NS):
        t = (k + 0.5)/NS
        strip(vs[k], vs[k+1], tuple(base + (top-base)*t))

# the cross-section at the near end: soil, then the water in the cut
prof = [proj(1.0, v, terrain(v)) for v in VS]      # the true section, bed included
face = prof + [proj(1.0, 1, -SOILW), proj(1.0, -1, -SOILW)]
ax.add_patch(Polygon(face, closed=True, facecolor=tuple(SOILPX_HI), edgecolor='none', zorder=3))
wat  = ([proj(1.0, v, 0.0) for v in VS if abs(v) <= WCH]
        + [proj(1.0, v, terrain(v)) for v in VS[::-1] if abs(v) <= WCH])
ax.add_patch(Polygon(wat, closed=True, facecolor=tuple(BLUE_LO), edgecolor='none', zorder=4))

# the lines: the near profile with its bed, the far profile, the two long edges, the foot
ax.plot([p[0] for p in prof], [p[1] for p in prof], color=INK, lw=1.0, zorder=5)
# the section outline already carries the bed, so nothing extra is drawn for it
farp = [proj(0.0, v, ground(v)) for v in VS]   # at the far end only the skyline shows
ax.plot([p[0] for p in farp], [p[1] for p in farp], color=INK, lw=0.9, zorder=5)
for v_ in (-1, 1):             # the long edges of the block, nothing across the ground
    p0, p1 = proj(0.0, v_, ground(v_)), proj(1.0, v_, ground(v_))
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=0.9, zorder=5)
for u_ in (0.0, 1.0):          # the water line of each section, dashed as a water level is
    p0, p1 = proj(u_, -WCH, 0.0), proj(u_, WCH, 0.0)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=0.9, ls=(0, (5, 3)), zorder=5)
foot = [proj(1.0, -1, -SOILW), proj(1.0, 1, -SOILW)]
ax.plot([foot[0][0], foot[1][0]], [foot[0][1], foot[1][1]], color=INK, lw=1.0, zorder=5)
for v_ in (-1, 1):
    p0, p1 = proj(1.0, v_, ground(v_)), proj(1.0, v_, -SOILW)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=1.0, zorder=5)
p0, p1 = proj(0.0, 1, -SOILW), proj(1.0, 1, -SOILW)          # the foot of the right face
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=0.9, zorder=5)
p0, p1 = proj(0.0, 1, ground(1)), proj(0.0, 1, -SOILW)        # its far corner
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=0.9, zorder=5)

# ---------------- lateral inflow over the banks ----------------------------------------
HALO = [matplotlib.patheffects.withStroke(linewidth=3.0, foreground='white')]
HALO_A = [matplotlib.patheffects.withStroke(linewidth=1.7, foreground='white')]
def surf_arrow(u, v0, v1, lw=1.0, hw=15.0, z=6, length=None):
    p = np.array(proj(u, v0, ground(v0)))
    q = np.array(proj(u, v1, ground(v1)))
    d = q - p; d = d/np.linalg.norm(d); n = np.array([-d[1], d[0]])
    if length is not None:         # keep the heading, set the length
        p = q - d*length
    ax.plot(*zip(p, q - d*hw*0.45), color=INK, lw=lw, solid_capstyle='butt', zorder=z,
            path_effects=HALO_A)
    ax.add_patch(Polygon([q, q - d*hw + n*hw*0.34, q - d*hw - n*hw*0.34], closed=True,
                         facecolor=INK, edgecolor='none', zorder=z, path_effects=HALO_A))

LARR = 0.64*np.linalg.norm(np.array(proj(0.5, -(VW-0.06), ground(VW-0.06)))
                          - np.array(proj(0.5, -(WCH+0.04), 0.0)))
for u_ in (0.18, 0.50, 0.82):
    surf_arrow(u_, -(VW-0.06), -(WCH+0.04), length=LARR)
for u_ in (0.06, 0.38, 0.70):      # the far bank, kept flatter and a little upstream
    surf_arrow(u_, 0.88, WCH+0.04, length=LARR)

# ---------------- the flow in and the flow out -----------------------------------------
ARROW_FILL, ARROW_EDGE = '#8F93C6', '#3B3F6E'
def water_arrow(u0, u1, vc=0.0, sv=0.072, hv=0.150, hu=0.20):
    """A flow arrow lying in the plane of the water, so it foreshortens with the reach"""
    pts = [(u0, vc+sv), (u1-hu, vc+sv), (u1-hu, vc+hv), (u1, vc),
           (u1-hu, vc-hv), (u1-hu, vc-sv), (u0, vc-sv)]
    ax.add_patch(Polygon([proj(u, v, 0.0) for u, v in pts], closed=True,
                         facecolor=ARROW_FILL, edgecolor=ARROW_EDGE, lw=0.9,
                         joinstyle='miter', zorder=6))

water_arrow(-0.44, -0.04)          # into the reach, on the water coming from upstream
water_arrow( 1.02,  1.42)          # out of the reach, on the water going downstream

# ---------------- reach length, in the style of the published figure -------------------
# the dimension sits square to the left bank, carried out from its two corners
qa = np.array(proj(0.0, -1.0, ground(-1.0)))
qb = np.array(proj(1.0, -1.0, ground(-1.0)))
d = (qb-qa)/np.linalg.norm(qb-qa); n = np.array([-d[1], d[0]])   # n points away from the block
OFFD = 52.0
pa, pb = qa + n*OFFD, qb + n*OFFD
for c_, e_ in ((qa, pa), (qb, pb)):                # the extension line out of each corner
    ax.plot(*zip(e_ - n*26, e_ + n*13), color=INK, lw=0.9, zorder=6)
ax.plot(*zip(pa+d*10, pb-d*10), color=INK, lw=1.1, zorder=6)
for tip, base in ((pa, pa+d*10), (pb, pb-d*10)):
    ax.add_patch(Polygon([tip, base+n*4.2, base-n*4.2], closed=True, facecolor=INK,
                         edgecolor='none', zorder=6))

# ---------------- the labels of panel (a) ----------------------------------------------
PT = 9.0
def lab(x, y, t, ha='center', va='center', pt=PT, rot=0, halo=False, c='k'):
    ax.text(x, y, t, fontsize=pt, ha=ha, va=va, color=c, zorder=8, rotation=rot,
            rotation_mode='anchor', path_effects=HALO if halo else None)

def leader(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color='k', lw=0.8, zorder=7)

_ang = np.degrees(np.arctan2(-d[1], d[0]))
_ang = _ang + 180 if (_ang < -90 or _ang > 90) else _ang
_m = 0.5*(pa+pb) + n*42                # the words sit outside the line, clear of it
lab(_m[0], _m[1], 'Channel length $\\Delta x$', rot=_ang)

_t = np.array(proj(0.50, -0.755, ground(-0.755)))   # written on the ground beside the bank
lab(_t[0], _t[1], 'Lateral inflow $q$', rot=_ang, pt=PT, halo=True)

_i = np.array(proj(-0.44, 0.0, 0.0))      # beside the tail of the arrow, on its left
lab(_i[0]-80, _i[1]-8, 'Inflow\n$Q(x)$', ha='right', c=ARROW_EDGE)
_o = np.array(proj(1.42, 0.0, 0.0))
lab(_o[0]+2, _o[1]+70, 'Outflow\n$Q(x+\\Delta x)$', ha='right', c=ARROW_EDGE)

_c = np.array(proj(1.0, 0.26, -DEEP*0.06))           # straight down out of the wet section
_cy = _o[1] + 70                                     # level with the outflow label
leader((_c[0]-26, _c[1]+10), (_c[0]-26, _cy-70))
lab(_c[0]+150, _cy, 'Wetted cross-section\narea $A$', ha='center', c=tuple(BLUE_LO*0.80))

# ---------------- the labels of panel (b) and the two panel titles ----------------------
# token positions were measured in millimetres from the top left of the Keynote export, so
# they are placed as fractions of this figure, whose width is the same and whose height is
# the export plus the room left under panel (a)
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
WMM = T['width_mm']; HMM = T['height_mm']*(HPX + PAD_B)/HPX
MATH_PT, LETTER_PT, TITLE_PT, LETTER_GAP_MM = 9.0, 11.0, 10.5, 1.2

def put(x_mm, base_mm, t, size, colour, ha='left'):
    fig.text(x_mm/WMM, 1.0 - base_mm/HMM, t, fontsize=size, color=colour, ha=ha,
             va='baseline', zorder=10)

TITLES = []
for lab in T['labels']:
    name = lab['name']
    if name in ('title_a', 'title_b'):
        head, words = lab['tokens'][0], lab['tokens'][1:]
        letter = head['token'].strip('()')
        base = float(np.mean([w['baseline_mm'] for w in words]))
        put(head['x_mm'], base, letter, LETTER_PT, '#000000')
        fig.texts[-1].set_fontweight('bold')
        adv = TextPath((0, 0), letter, size=LETTER_PT,
                       prop=FontProperties(family='Myriad Pro', weight='bold')).get_extents().x1*25.4/72
        title = ' '.join(w['token'] for w in words).replace('$t$ $+$ $\\Delta t$', '$t + \\Delta t$').replace('one grid cell', 'one cell')
        put(head['x_mm'] + adv + LETTER_GAP_MM, base, title, TITLE_PT, words[0]['colour'])
        TITLES.append((name, fig.texts[-2], fig.texts[-1]))   # letter, title

# ---------------- panel (b) as a parallelogram, in the attitude of the reach in (a) -------
# the upstream end is the upper edge and the downstream end the lower edge, both level; the
# sides run parallel to the channel of (a), down and to the left in the downstream direction,
# and time runs to the right along each end
ORANGE, BLUEB = '#D55D00', '#1972B2'
dd = np.array([-RU, RV]); dd = dd/np.hypot(*dd)          # downstream, as in (a)
# the upper edge is level with the top of the block in (a) and the lower edge with its foot,
# so the two panels line up across the row
BLOCK_TOP = proj(0.0, 1, ground(1))[1]
BLOCK_FOOT = proj(1.0, -1, -SOILW)[1]
LT = 360.0                                               # time step, in pixels
LS = (BLOCK_FOOT - BLOCK_TOP) / dd[1]                    # cell length, in pixels
P_ut = np.array([1600.0, BLOCK_TOP])                     # upstream end at t
P_utt = P_ut + [LT, 0.0]
P_dt = P_ut + dd*LS; P_dtt = P_dt + [LT, 0.0]
EXT = 55.0
def dotted(p, q, col='k'):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.9, ls=(0, (1.6, 1.8)), zorder=2)
# the level lines run in time (blue, as the time difference), the sloping lines in space
# (orange, as the space difference)
dotted(P_ut - [EXT, 0], P_utt + [EXT, 0], BLUEB); dotted(P_dt - [EXT, 0], P_dtt + [EXT, 0], BLUEB)
dotted(P_ut - dd*EXT, P_dt, ORANGE); dotted(P_utt - dd*EXT, P_dtt, ORANGE)   # carried on above, for the time labels
def node(p, col, filled):
    ax.add_patch(Circle(p, 12.5, facecolor=col if filled else 'white', edgecolor=col,
                        lw=1.0 if not filled else 0, zorder=6))
for p in (P_ut, P_utt, P_dt): node(p, ORANGE, False)
node(P_dtt, ORANGE, True)
A_t, A_tt = 0.5*(P_ut + P_dt), 0.5*(P_utt + P_dtt)
node(A_t, BLUEB, True); node(A_tt, BLUEB, True)
def txt(p, t, ha='center', va='baseline', c='k'):
    ax.text(p[0], p[1], t, fontsize=MATH_PT, ha=ha, va=va, color=c, zorder=8)
txt(P_ut + [-14, -36], r'$Q(x,t)$', ha='right')
txt(P_utt + [-14, -36], r'$Q(x,t+\Delta t)$', ha='right')
txt(P_dt + [0, 64], r'$Q(x+\Delta x,t)$')
txt(P_dtt + [0, 64], r'$Q(x+\Delta x,t+\Delta t)$')
txt(A_t + [-22, 0], r'$A(t)$', ha='right', va='center')
txt(A_tt + [22, 0], r'$A(t+\Delta t)$', ha='left', va='center')
txt(P_ut - dd*EXT + [6, -8], r'$t$', ha='left')
txt(P_utt - dd*EXT + [6, -8], r'$t+\Delta t$', ha='left')
# the two ends, as the two times are marked above: x and x + dx at the left end of each line
txt(P_ut - [EXT + 16, 0], r'$x$', ha='right', va='center')
txt(P_dt - [EXT + 16, 0], r'$x+\Delta x$', ha='right', va='center')
txt(P_utt + [EXT + 14, 0], 'Upstream end', ha='left', va='center', c=ARROW_EDGE)
txt(P_dtt + [EXT + 14, 0], 'Downstream end', ha='left', va='center', c=ARROW_EDGE)
# difference in time along the upstream end, difference in space along the side at t + dt
def arrow(p, q, col):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>,head_length=4.6,head_width=2.2',
                                 mutation_scale=1, color=col, lw=1.0, shrinkA=0, shrinkB=0,
                                 zorder=5))
ta0 = P_ut + dd*30 + [0.5*LT - 85, 0]          # centred across the cell at this height
arrow(ta0, ta0 + [170, 0], BLUEB)
txt(ta0 + [85, 18], 'Difference\nin time', ha='center', va='top', c=BLUEB)
sa0 = P_utt + dd*225 + [-40, 0]
arrow(sa0, sa0 + dd*135, ORANGE)
txt(sa0 + dd*67 + [-45, 0], 'Difference\nin space', ha='right', va='center', c=ORANGE)   # level with the middle of the arrow
# key
kx, ky = P_utt[0] + 120.0, 0.5*(P_ut[1] + P_dt[1]) - 47.0   # a column on the right, between the two ends
node((kx, ky), ORANGE, False); txt((kx + 26, ky), 'Known $Q$', ha='left', va='center')
node((kx, ky + 47), ORANGE, True); txt((kx + 26, ky + 47), 'Unknown $Q$', ha='left', va='center')
# as in Figure 1, each title is centred over its panel and each letter stays at the left
# edge of the panel; a title too wide to centre starts just after its letter
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_px = lambda xd: ax.transData.transform((xd, 0))[0]          # data x -> display x
_bx = [t.get_window_extent(_r) for t in ax.texts if t.get_position()[0] > 1100]
_bx += [p.get_window_extent(_r) for p in ax.patches if p.get_window_extent(_r).x0 > _px(1100)]
SPAN = {'title_a': (None, _px(proj(0.0, 1, ground(1))[0])),     # to the right face of the block
        'title_b': (min(b.x0 for b in _bx), max(b.x1 for b in _bx))}
_gap = LETTER_GAP_MM / 25.4 * fig.dpi
for name, lt, tt in TITLES:
    lb = lt.get_window_extent(_r)
    x0 = SPAN[name][0] if SPAN[name][0] is not None else lb.x0
    x0 = min(x0, lb.x0)
    w = tt.get_window_extent(_r).width
    left = max(0.5 * (x0 + SPAN[name][1]) - 0.5 * w, lb.x1 + _gap)
    tt.set_x(fig.transFigure.inverted().transform((left, 0))[0])
fig.savefig(OUT + '.pdf')
subprocess.run(['pdftocairo', '-r', '600', '-png', '-singlefile', OUT + '.pdf', OUT],
               check=True, capture_output=True)
print('wrote', OUT + '.{pdf,png}', '%.2f x %.2f mm' % (WMM, HMM))
