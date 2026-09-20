"""Paper1 Figure 2: a river network (derived from the Seine at 5 arcmin, simplified)
and the three schemes at reach level.
Run:  ~/anaconda3/envs/rtho/bin/python fig_method_seine.py
Output: paper/figures/fig_method.{png,pdf}, copied into paper1_gpu_routing/figures/

调标签位置只需改 LABEL_ADJUST 字典 (搜索这个名字), 按 (b) 图编号写调整,
写法见字典上方注释; 改完重跑本脚本即可预览。
"""

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# paper1 style: Myriad Pro words, STIX equations, and sizes from
# paper1_gpu_routing/画图规范与经验.md (8 pt text, 9 pt titles, 10 pt bold panel letters)
from paper_style import setup_paper1_style, panel_label
setup_paper1_style()
# built at the 166 mm text width, so the sizes set here are the sizes that print
PW = 166.0 / 25.4
from scipy.ndimage import gaussian_filter

_ROOT = os.environ.get('SURGE_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figure_data'))
d = np.load(os.path.join(_ROOT, 'routing_static_05min.npz'))   # set SURGE_DATA to the folder holding the network file
valid = ~d['mask2d']; lvl = d['level']
src, dst = d['edge_src'], d['edge_dst']
rows = np.argwhere(valid)
res = 1/12
lat = 90 - (rows[:,0]+0.5)*res; lon = -180 + (rows[:,1]+0.5)*res
n = len(lat)

box = (lat>46.9)&(lat<50.6)&(lon>-1.2)&(lon<6.2)
down = np.full(n, -1, np.int64); down[src] = dst
out = np.where(box)[0]
sub = np.full(n, -1, np.int64); sub[out] = np.arange(len(out))
dn_l = np.full(len(out), -1, np.int64)
has = (down[out]>=0) & box[np.clip(down[out],0,n-1)]
dn_l[np.where(has)[0]] = sub[down[out][has]]
m_ = len(out)
root = np.arange(m_)
for _ in range(2000):
    nxt = np.where(dn_l[root]>=0, dn_l[root], root)
    if np.all(nxt==root): break
    root = nxt
la, lo = lat[out], lon[out]
roots, counts = np.unique(root, return_counts=True)
cands = [(c, r) for r, c in zip(roots, counts) if lo[r] < 1.8 and la[r] > 48.8]
cands.sort(reverse=True)
seine_root = cands[0][1]
mb = root == seine_root
ids = np.where(mb)[0]
N = len(ids)

lv_l = lvl[out]
accv = np.ones(m_)
for L_ in np.sort(np.unique(lv_l[mb])):
    sel = mb & (lv_l == L_) & (dn_l >= 0)
    np.add.at(accv, dn_l[sel], accv[sel])

# ---- rotated metric coordinates: outlet points down
lam = np.cos(np.deg2rad(la[ids].mean()))
X0 = (lo - lo[ids].mean())*lam; Y0 = la - la[ids].mean()
cx, cy = X0[ids].mean(), Y0[ids].mean()
vx, vy = X0[seine_root]-cx, Y0[seine_root]-cy
dth = -np.pi/2 - np.arctan2(vy, vx)
X = (X0-cx)*np.cos(dth) - (Y0-cy)*np.sin(dth)
Y = (X0-cx)*np.sin(dth) + (Y0-cy)*np.cos(dth)

def simplify(p, step=3):
    p = np.asarray(p, float)
    if len(p) <= 4: return p
    q = np.vstack([p[0], p[1:-1:step], p[-1]])
    return q

def chaikin(p, it=2):
    p = np.asarray(p, float)
    for _ in range(it):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q.append(0.75*a+0.25*b); q.append(0.25*a+0.75*b)
        q.append(p[-1]); p = np.array(q)
    return p

def contract(T):
    keep = mb & (accv >= T)
    kidn = np.zeros(m_, int)
    sel = keep & (dn_l>=0)
    np.add.at(kidn, dn_l[sel & keep[np.clip(dn_l,0,m_-1)]], 1)
    node = keep & ((kidn==0)|(kidn>=2)|(dn_l<0)|~keep[np.clip(dn_l,0,m_-1)])
    reaches = []
    for c in np.where(node)[0]:
        if dn_l[c] < 0 or not keep[dn_l[c]]: continue
        ch = [c]; cur = dn_l[c]
        while cur >= 0 and keep[cur] and not node[cur]:
            ch.append(cur); cur = dn_l[cur]
        if cur >= 0 and keep[cur]: ch.append(cur)
        if len(ch) > 1: reaches.append(ch)
    return keep, reaches

# (b-d): major reaches
for T in [60,80,100,120,150,200,250,300,400]:
    keep, reaches = contract(T)
    if 13 <= len(reaches) <= 20: break
R = len(reaches)
ends_at = {}
for i,ch in enumerate(reaches): ends_at.setdefault(ch[-1], []).append(i)
ups_of = [ends_at.get(ch[0], []) for ch in reaches]
rlvl = [None]*R
def lv(i):
    if rlvl[i] is None:
        rlvl[i] = 0 if not ups_of[i] else 1+max(lv(j) for j in ups_of[i])
    return rlvl[i]
for i in range(R): lv(i)
Lr = max(rlvl)+1
rootr = [i for i,ch in enumerate(reaches) if dn_l[ch[-1]]<0 or not keep[dn_l[ch[-1]]]]
order = []
def dfs(i):
    for j in ups_of[i]: dfs(j)
    order.append(i)
for i in rootr: dfs(i)
serial = {i:k+1 for k,i in enumerate(order)}

# ==================== (b) 图河网编号位置手动调整 ====================
# key = (b) 图的串行编号 1..17。三种写法:
#   不写                        自动放置
#   ('beside', 参照编号, dx, dy)  钉在参照编号标签旁边: 参照标签位置 + (dx, dy)
#   ('nudge', dx, dy)           先自动放置, 再平移 (dx, dy)
# dx>0 右移, dy>0 上移; 单位为图内坐标, 一个标签宽约 0.34。
# (c)(d) 面板中同一条河段的标签自动继承相同调整。改完直接重跑本脚本预览。
LABEL_ADJUST = {
    8:  ('beside', 6, 0.22, 0.0),
    7:  ('beside', 4, 0.30, 0.0),
    12: ('nudge', 0.15, 0.14),
    9:  ('nudge', 0.05, -0.10),
    10: ('nudge', -0.2, -0.3),
    11: ('nudge', 0.0, -0.05),
    14: ('nudge', 0.05, 0.0),
    15: ('nudge', -0.05, 0.05),
    16: ('nudge', 0.05, -0.05),
    17: ('nudge', 0.05, 0.0),
}
# (a)图方向箭头位置, 相对河口: x = 河口x + dx; 箭头从 河口y+y1 画到 河口y+y0 (向下);
# 文字在箭头左侧 text_dx 处
FLOW_ARROW = dict(dx=0.9, y0=0.55, y1=1.55, text_dx=0.17)
LEADER_DIST = 0.22   # 标签中心离河段锚点超过这个距离就画引线 (原 0.30)
BY_NUM = {v: k for k, v in serial.items()}   # (b)图编号 -> 河段索引
chainsA = reaches
print(f'N={N} reaches={R} reach-levels={Lr}')

# ---- smooth basin boundary from the cell-footprint mask
r0, r1 = rows[out[ids],0].min()-3, rows[out[ids],0].max()+4
c0, c1 = rows[out[ids],1].min()-3, rows[out[ids],1].max()+4
g = np.zeros((r1-r0, c1-c0))
rc = np.unique(np.concatenate([np.asarray(ch) for ch in reaches]))
g[rows[out[rc],0]-r0, rows[out[rc],1]-c0] = 1.0
gs_ = gaussian_filter(g, 3.0)
cs = plt.contour(gs_, levels=[0.055])
plt.close()
paths = max(cs.allsegs[0], key=len)
if np.allclose(paths[0], paths[-1]): paths = paths[:-1]
# resample the closed curve evenly, then circular moving-average smoothing
seg = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[paths, paths[:1]], axis=0).T))]
u = np.linspace(0, seg[-1], 240, endpoint=False)
paths = np.c_[np.interp(u, seg, np.r_[paths[:,0], paths[:1,0]]),
              np.interp(u, seg, np.r_[paths[:,1], paths[:1,1]])]
w = 31; k_ = np.ones(w)/w
for _ in range(5):
    for j in (0,1):
        ext = np.r_[paths[-w:,j], paths[:,j], paths[:w,j]]
        paths[:,j] = np.convolve(ext, k_, 'same')[w:-w]
paths = np.vstack([paths, paths[:1]])
blat = 90 - (paths[:,1]+r0+0.5)*res; blon = -180 + (paths[:,0]+c0+0.5)*res
bX0 = (blon - lo[ids].mean())*lam; bY0 = blat - la[ids].mean()
bX = (bX0-cx)*np.cos(dth) - (bY0-cy)*np.sin(dth)
bY = (bX0-cx)*np.sin(dth) + (bY0-cy)*np.cos(dth)
bcx, bcy = bX.mean(), bY.mean()
bX = bcx + (bX-bcx)*1.04; bY = bcy + (bY-bcy)*1.04
# radial envelope: no river point may stick out
rX, rY = X[rc], Y[rc]
ang_p = np.arctan2(rY-bcy, rX-bcx); r_p = np.hypot(rX-bcx, rY-bcy)
ang_b = np.arctan2(bY-bcy, bX-bcx); r_b = np.hypot(bX-bcx, bY-bcy)
env = np.zeros_like(r_b)
for i in range(len(bX)):
    dd = np.angle(np.exp(1j*(ang_p - ang_b[i])))
    sel = np.abs(dd) < 0.14
    if sel.any():
        env[i] = r_p[sel].max() + 0.20
w2 = 23; k2 = np.ones(w2)/w2
r_b = np.maximum(r_b, env)
for _ in range(6):
    r_b = np.convolve(np.r_[r_b[-w2:], r_b, r_b[:w2]], k2, 'same')[w2:-w2]
    r_b = np.maximum(r_b, env)
w3 = 9; k3 = np.ones(w3)/w3
r_b = np.convolve(np.r_[r_b[-w3:], r_b, r_b[:w3]], k3, 'same')[w3:-w3]
bX = bcx + r_b*np.cos(ang_b); bY = bcy + r_b*np.sin(ang_b)
# pointed tip at the outlet: cut the bottom cap and taper into the outlet with beziers
if np.allclose([bX[0], bY[0]], [bX[-1], bY[-1]]): bX, bY = bX[:-1], bY[:-1]
Xo, Yo = X[seine_root], Y[seine_root]
dtip = np.hypot(bX-Xo, bY-Yo)
i0 = int(np.argmin(dtip))
bX = np.roll(bX, -i0); bY = np.roll(bY, -i0)
dtip = np.hypot(bX-Xo, bY-Yo)
keep_ = dtip >= 0.62
ii = np.where(keep_)[0]
kx, ky = bX[ii], bY[ii]
O = np.array([Xo, Yo - 0.03])
def bez(P0, C1, C2, P3, m=40):
    s = np.linspace(0, 1, m)[:, None]
    return ((1-s)**3*P0 + 3*(1-s)**2*s*C1 + 3*(1-s)*s**2*C2 + s**3*P3)
P1 = np.array([kx[0], ky[0]])
t1 = P1 - np.array([kx[3], ky[3]]); t1 /= np.hypot(*t1)
d1 = np.hypot(*(P1 - O))
B1 = bez(P1, P1 + t1*0.45*d1, O + (P1-O)/d1*0.4*d1, O)          # side A into the tip
P2 = np.array([kx[-1], ky[-1]])
t2 = P2 - np.array([kx[-4], ky[-4]]); t2 /= np.hypot(*t2)
d2 = np.hypot(*(P2 - O))
B2 = bez(P2, P2 + t2*0.45*d2, O + (P2-O)/d2*0.4*d2, O)          # side B into the tip
ring = np.vstack([np.c_[kx, ky], B2, B1[::-1]])
bX, bY = ring[:, 0], ring[:, 1]

cmap = plt.get_cmap('turbo')
# the layout was designed on a 5.3 in canvas; it is scaled to the print width as a whole,
# and the hand-placed schematic keeps its own margins, so the constrained layout is off
fig = plt.figure(figsize=(PW, 2.11 * PW / 5.3), layout='none')
gsp = fig.add_gridspec(1, 4, width_ratios=[1.12, 1, 1, 1], wspace=0.03,
                       left=0.022, right=0.995, top=0.875, bottom=0.095)

axa = fig.add_subplot(gsp[0])
amax = accv[mb].max()
for ch in chainsA:
    pts = chaikin(simplify(np.c_[X[ch], Y[ch]]), 3)
    w = 0.4 + 3.0*np.sqrt(accv[ch[len(ch)//2]]/amax)
    axa.plot(pts[:,0], pts[:,1], color='#2e79b8', lw=w, solid_capstyle='round', zorder=2)
axa.plot(bX, bY, color='black', lw=1.2, zorder=1)
axa.scatter([X[seine_root]], [Y[seine_root]], s=48, facecolors='white',
            edgecolors='#333', lw=1.2, zorder=4)
from matplotlib.lines import Line2D
axa.legend(handles=[
    Line2D([], [], color='#2e79b8', lw=2.2, label='River'),
    Line2D([], [], color='black', lw=1.2, label='Basin'),
    Line2D([], [], marker='o', mfc='white', mec='#333', ls='none', ms=5.5, label='Outlet'),
], loc='lower left', fontsize=8.0, frameon=False, handlelength=1.1,
    handletextpad=0.5, borderaxespad=0.0)
xo_, yo_ = X[seine_root], Y[seine_root]
_ar = FLOW_ARROW
axa.annotate('', xy=(xo_+_ar['dx'], yo_+_ar['y0']), xytext=(xo_+_ar['dx'], yo_+_ar['y1']),
             arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.0, mutation_scale=9))
axa.text(xo_+_ar['dx']+_ar['text_dx'], yo_+(_ar['y0']+_ar['y1'])/2, 'Flow direction',
         fontsize=8.0, color='#333', rotation=90, ha='center', va='center')
TITLES = [(axa, 'a', 'A river network')]   # letter and title go on once the panels are placed
axa.text(0.5, -0.01, 'The same network is solved\nin three ways', transform=axa.transAxes,
         fontsize=8.0, ha='center', va='top', color='#333')

def reach_panel(ax, letter, title, sub_t, num_of, vmax):
    allpts = [chaikin(simplify(np.c_[X[ch], Y[ch]]), 3) for ch in reaches]
    for i, pts in enumerate(allpts):
        col = cmap(0.08 + 0.84*num_of(i)/max(vmax,1))
        ax.plot(pts[:,0], pts[:,1], color=col, lw=1.9, solid_capstyle='round', zorder=2)
    lines = np.vstack(allpts)

    # LABEL_ADJUST (按b图编号) 翻译成河段索引
    beside, nudge, head = {}, {}, []
    for num, adj in LABEL_ADJUST.items():
        i = BY_NUM[num]
        if adj[0] == 'beside':
            base = BY_NUM[adj[1]]
            beside[i] = (base, np.array([adj[2], adj[3]]))
            head += [base, i]                       # 参照标签先放, 被钉标签跟上
        elif adj[0] == 'nudge':
            nudge[i] = np.array([adj[1], adj[2]])
    head = list(dict.fromkeys(head))
    order_idx = head + [j for j in range(len(allpts)) if j not in head]

    placed = []
    spot = {}

    def draw_label(i, q, col, pts):
        anchor = pts[len(pts)//2]
        if np.hypot(q[0]-anchor[0], q[1]-anchor[1]) > LEADER_DIST:   # 离河段远时画引线
            ax.plot([q[0], anchor[0]], [q[1], anchor[1]], color='#9aa0a6',
                    lw=0.55, zorder=3)
        ax.text(q[0], q[1], str(num_of(i)), fontsize=8.0,
                ha='center', va='center', zorder=4, color='#222',
                bbox=dict(boxstyle='square,pad=0.16', fc='white', ec=col, lw=0.8))

    def auto_place(i, pts):
        # 候选: 沿河段位置分数 frac 处, 法向偏移 0.17*sgn*s; 依次找第一个
        # 既不与已放标签相撞 (间距阈值 0.40/0.28), 又不压河道线 (距离阈值 0.14) 的位置
        sgn = 1 if i % 2 else -1
        q = None
        cands = [(f, s) for s in (1.0, -1.0, 1.7, -1.7, 2.4, -2.4, 3.2, -3.2)
                 for f in (0.5, 0.35, 0.65, 0.2, 0.8)]
        for frac, s in cands:
            k = max(1, min(len(pts)-2, int(frac*len(pts))))
            tv = pts[k+1] - pts[k-1]
            nm = np.hypot(*tv) + 1e-9
            nrm = np.array([-tv[1]/nm, tv[0]/nm])
            q = pts[k] + nrm*0.17*sgn*s
            if not all(abs(q[0]-r[0]) > 0.40 or abs(q[1]-r[1]) > 0.28 for r in placed):
                continue
            if np.hypot(lines[:,0]-q[0], lines[:,1]-q[1]).min() < 0.14:
                continue
            return q
        return q

    for i in order_idx:
        pts = allpts[i]
        col = cmap(0.08 + 0.84*num_of(i)/max(vmax,1))
        if i in beside:
            base, off = beside[i]
            q = spot[base] + off
        else:
            q = auto_place(i, pts)
            if i in nudge:
                q = q + nudge[i]
        placed.append(q)
        spot[i] = q
        draw_label(i, q, col, pts)
    ax.scatter([X[seine_root]], [Y[seine_root]], s=36, facecolors='white',
               edgecolors='#333', lw=1.0, zorder=3)
    TITLES.append((ax, letter, title))
    ax.text(0.5, -0.01, sub_t, transform=ax.transAxes, fontsize=8.0,
            ha='center', va='top', color='#333')

reach_panel(fig.add_subplot(gsp[1]), 'b', 'Serial scheme',
            f'One reach after another\n{R} sequential steps', lambda i: serial[i], R)
reach_panel(fig.add_subplot(gsp[2]), 'c', 'Partly parallel scheme',
            f'One level at a time\n{Lr} sequential steps', lambda i: rlvl[i]+1, Lr)
reach_panel(fig.add_subplot(gsp[3]), 'd', 'Fully parallel scheme',
            'All reaches at once\nA few iterations', lambda i: 1, 1)

allX = np.r_[X[rc], bX]; allY = np.r_[Y[rc], bY]   # 只按画出的河网+边界取画幅
# Every panel draws the network at one scale and over one vertical range, and every axes
# box has the same top and the same height, so the outlets and the lines of text under the
# networks line up across the row. With set_aspect('equal') each box would instead shrink
# to its own column, and panel a, whose column is wider, would come out taller than b-d.
# Panel a keeps its wider column and simply shows more ground to either side.
xlo, xhi = allX.min() - 0.12, allX.max() + 0.12
ylo, yhi = allY.min() - 0.12, allY.max() + 0.12
W_in, H_in = fig.get_size_inches()
boxes = [ax.get_position() for ax in fig.axes]          # the gridspec cells
row_top, row_h = boxes[0].y1, boxes[0].height * H_in
scale = min(min(b.width * W_in for b in boxes) / (xhi - xlo), row_h / (yhi - ylo))   # in per unit
box_h = (yhi - ylo) * scale / H_in
xc = 0.5 * (xlo + xhi)
for ax, b in zip(fig.axes, boxes):
    ax.set_position([b.x0, row_top - box_h, b.width, box_h])
    half = b.width * W_in / scale / 2.0
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('auto')
    ax.axis('off')

# Each panel's letter and title start where its drawing starts, the basin outline in a and
# the reaches with their number boxes in b-d, so title and drawing share one left edge.
# The drawing is centred in its column, so aligning to the column edge put the titles left
# of the networks
fig.canvas.draw()
_r = fig.canvas.get_renderer()
TITLE_PAD_PT, LETTER_GAP_PT = 3.0, 1.2 * 72 / 25.4
for ax, letter, title in TITLES:
    xs = [ln.get_window_extent(_r).x0 for ln in ax.lines if len(ln.get_xdata()) > 0]
    xs += [tx.get_bbox_patch().get_window_extent(_r).x0 for tx in ax.texts
           if tx.get_bbox_patch() is not None]
    left = ax.transAxes.inverted().transform((min(xs), 0.0))[0]
    lt = ax.annotate(letter, xy=(left, 1.0), xycoords='axes fraction', xytext=(0.0, TITLE_PAD_PT),
                     textcoords='offset points', ha='left', va='baseline', fontsize=10.0,
                     fontweight='bold', annotation_clip=False)
    w_pt = lt.get_window_extent(_r).width * 72.0 / fig.dpi
    ax.annotate(title, xy=(left, 1.0), xycoords='axes fraction',
                xytext=(w_pt + LETTER_GAP_PT, TITLE_PAD_PT), textcoords='offset points',
                ha='left', va='baseline', fontsize=9.0, annotation_clip=False)

import shutil
_FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
fig.savefig(os.path.join(_FIG, 'fig_method.png'), dpi=600)
fig.savefig(os.path.join(_FIG, 'fig_method.pdf'))
_COPY_TO = os.environ.get('SURGE_FIGURE_COPY')      # optional second destination, e.g. a manuscript tree
if _COPY_TO:
    for _f in ('fig_method.png', 'fig_method.pdf'):
        shutil.copy(os.path.join(_FIG, _f), os.path.join(_COPY_TO, _f))
print('saved fig_method.png/pdf')
