"""Fig 6: two ways an SCN stores a choice, side by side.

(a) Discrete multistability: at symmetric parameters the two chirality orbits sigma=(1,2,3)
    and sigma=(1,3,2) coexist, and the initial state picks one of the two.
(b) A continuous family: at mu_1 = mu_2 the isolated orbit is replaced by a one-parameter
    family sharing one firing word. The black curve is the set of orbit starts on the first
    bound, which is a single point for every other parameter setting in the paper.

Both panels are top-down projections along the mean decoder direction, so a reader compares
them directly. One legend, folded into the lower right of (b).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scn import step
import style

style.setup()

NEURON_COL = {1: '#B03A2E', 2: '#1F4E79', 3: '#1E7A48'}
ORBIT_COL = {'A': '#F5F5F5', 'B': '#2C3E50'}
DECAY_COL = '#7F8C8D'
FAMILY_COL = '#111111'

FS_BOUND, FS_NUM, FS_LEGEND, FS_PANEL = 30, 24, 25, 30
MS_DOT, MS_LEG = 34, 26
LW_SPIKE, LW_DECAY, LW_BOUND = 4.0, 2.5, 3.0
MUT_SPIKE, MUT_DECAY = 30, 26


def frame_of(d):
    z = np.array([0., 0, 1.]) if abs(d[2]) < 0.9 else np.array([0., 1., 0])
    u1 = np.cross(d, z); u1 /= np.linalg.norm(u1)
    u2 = np.cross(d, u1); u2 /= np.linalg.norm(u2)
    return np.column_stack([u1, u2])


def projector(axis, ref):
    """2-D chart on the plane normal to `axis`, with `ref` fixing the first direction."""
    axis = axis / np.linalg.norm(axis)
    e1 = ref - (ref @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return lambda y: np.array([y @ e1, y @ e2])


# ─────────────────────────── panel (a): chirality pair ───────────────────────────
def make_symmetric(alpha, cos_psi=0.507):
    axis = np.array([1, 1, 1]) / np.sqrt(3)
    ct = np.sqrt((1 - cos_psi) / 1.5)
    st = np.sqrt(1 - ct ** 2)
    e = [np.eye(3)[i] for i in range(3)]
    ep = [ei - (ei @ axis) * axis for ei in e]
    ep = [ei / np.linalg.norm(ei) for ei in ep]
    DS = [ct * ep[i] + st * axis for i in range(3)]
    DS = [d / np.linalg.norm(d) for d in DS]
    return {'DS': DS, 'ES': [frame_of(d) for d in DS], 'alphas': [alpha] * 3, 'K': 3, 'N': 3}


SYS_A = make_symmetric(0.7)


def orbit_from(bi0, xi0, n_warm=2000, n_keep=9):
    xi, bi = np.array(xi0), bi0
    for _ in range(n_warm):
        xi, bi = step(xi, bi, SYS_A)
        if bi is None or bi < 0:
            return None
    ys, bs = [], []
    for _ in range(n_keep):
        ys.append(SYS_A['alphas'][bi] * SYS_A['DS'][bi] + SYS_A['ES'][bi] @ xi)
        bs.append(bi)
        xi, bi = step(xi, bi, SYS_A)
        if bi is None or bi < 0:
            break
    return np.array(ys), np.array(bs)


def chirality_pair():
    found = {}
    rng = np.random.RandomState(0)
    for _ in range(300):
        out = orbit_from(0, rng.randn(2) * 0.3)
        if out is None:
            continue
        ys, bs = out
        key = tuple(bs[:3])
        rots = [tuple(np.roll(bs[:3], r)) for r in range(3)]
        for k in list(found):
            if k in rots:
                key = k
                break
        found.setdefault(key, (ys, bs))
        if len(found) >= 2:
            break
    (kA, (ysA, bsA)), (kB, (ysB, bsB)) = list(found.items())[:2]
    return (ysA, bsA), (ysB, bsB)


# ───────────────────── panel (b): the mu_1 = mu_2 orbit family ─────────────────────
BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', 'scn_bundle', 'extras', '_period3_ca_params.json')
P = json.load(open(os.path.normpath(BUNDLE), encoding='utf-8'))
ANG, AL_B, WORD_B = P['angles'], np.array(P['alphas']), P['word']
D_B = np.array([[np.sin(ANG[2 * i + 1]) * np.cos(ANG[2 * i]),
                 np.sin(ANG[2 * i + 1]) * np.sin(ANG[2 * i]),
                 np.cos(ANG[2 * i + 1])] for i in range(3)])
D_B = D_B / np.linalg.norm(D_B, axis=1, keepdims=True)
U_B = [frame_of(d) for d in D_B]


def transition(i, j):
    top = np.hstack([AL_B[j] * (U_B[j].T @ U_B[i]),
                     (AL_B[j] * (AL_B[i] + 1.0) * (U_B[j].T @ D_B[i])).reshape(-1, 1)])
    bot = np.hstack([(D_B[j] @ U_B[i]).reshape(1, -1),
                     np.array([[(AL_B[i] + 1.0) * float(D_B[j] @ D_B[i])]])])
    return np.vstack([top, bot])


def next_spike_B(y):
    """Only c = alpha/(D.y) <= 1 can fire: a bound is reached by decaying onto it."""
    dn = D_B @ y
    bj, bc, past = -1, -1.0, False
    for j in range(3):
        if dn[j] < 1e-12:
            continue
        c = AL_B[j] / dn[j]
        if c > 1.0 + 1e-9:
            past = True
        elif c > bc:
            bc, bj = c, j
    return bj, bc, past


def family_cycle(xi0, n=3):
    y, cur, segs, word = AL_B[WORD_B[0]] * D_B[WORD_B[0]] + U_B[WORD_B[0]] @ xi0, WORD_B[0], [], []
    for _ in range(n):
        yp = y + D_B[cur]
        j, c, past = next_spike_B(yp)
        if j < 0 or past:
            return None, None
        segs.append((y.copy(), yp.copy(), c * yp, cur))
        word.append(cur)
        y, cur = c * yp, j
    return segs, word


M = np.eye(3)
for k in range(len(WORD_B)):
    M = transition(WORD_B[k], WORD_B[(k + 1) % len(WORD_B)]) @ M
_ev = np.linalg.eigvals(M)
mu1 = _ev[np.argmax(np.abs(_ev))].real
_, _, Vt = np.linalg.svd(M - mu1 * np.eye(3))
null = Vt[-2:].T
rots_B = [tuple(WORD_B[r:] + WORD_B[:r]) for r in range(len(WORD_B))]

arc = []
for th in np.linspace(0, np.pi, 2001, endpoint=False):
    v = np.cos(th) * null[:, 0] + np.sin(th) * null[:, 1]
    if abs(v[-1]) < 1e-10:
        continue
    xi = (v[:-1] / v[-1]).real
    segs, w = family_cycle(xi)
    if w is not None and tuple(w) in rots_B:
        arc.append((th, xi.copy(),
                    AL_B[WORD_B[0]] * D_B[WORD_B[0]] + U_B[WORD_B[0]] @ xi, segs))
print('[family] admissible arc: %d samples, theta %.3f..%.3f' % (len(arc), arc[0][0], arc[-1][0]))


# ──────────────────────────────── draw ────────────────────────────────
def tint(color, t):
    """Blend toward white: t=1 keeps the colour, t=0 is white.

    Fading an arrow with alpha exposes the seam where the shaft meets the head, because a
    FancyArrowPatch is one patch whose parts overlap. Drawing a lighter colour at full
    opacity gives the same recession with no internal edges.
    """
    r, g, b = mcolors.to_rgb(color)
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


def draw_orbit(ax, proj, segs_colored, dot_fc, dot_ec, txt_col, strength=1.0, numbers=True):
    for y_pre, y_post, y_next, bi in segs_colored:
        p, q, r = proj(y_pre), proj(y_post), proj(y_next)
        ax.annotate('', xy=q, xytext=p, zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=tint(NEURON_COL[bi + 1], strength),
                                    lw=LW_SPIKE, mutation_scale=MUT_SPIKE))
        ax.annotate('', xy=r, xytext=q, zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color=tint(DECAY_COL, 0.85 * strength),
                                    lw=LW_DECAY, mutation_scale=MUT_DECAY))
    if dot_fc is None:
        return
    for y_pre, _, _, bi in segs_colored:
        p = proj(y_pre)
        ax.plot(p[0], p[1], 'o', color=dot_fc, markersize=MS_DOT, mec=dot_ec, mew=2.0, zorder=10)
        if numbers:
            ax.text(p[0], p[1], str(bi + 1), color=txt_col, ha='center', va='center_baseline',
                    fontsize=FS_NUM, weight='bold', zorder=11)


def set_view(ax, pts, pad=1.08):
    pts = np.asarray(pts)
    cx, cy = 0.5 * (pts[:, 0].min() + pts[:, 0].max()), 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    half = pad * max(np.abs(pts[:, 0] - cx).max(), np.abs(pts[:, 1] - cy).max())
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')
    ax.set_axis_off()


fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.6))

# (a)
projA = projector(np.array([1, 1, 1]), np.array(SYS_A['DS'][0]))
(ysA, bsA), (ysB, bsB) = chirality_pair()
inkA = []
for ys, bs, key in [(ysA, bsA, 'A'), (ysB, bsB, 'B')]:
    segs = []
    for k in range(3):
        y_pre = ys[k]
        y_post = y_pre + SYS_A['DS'][bs[k]]
        segs.append((y_pre, y_post, ys[k + 1], bs[k]))
    draw_orbit(axes[0], projA, segs, ORBIT_COL[key], '#111',
               '#111' if key == 'A' else '#F5F5F5')
    for s in segs:
        inkA += [projA(s[0]), projA(s[1]), projA(s[2])]
# Bound slices: the line through the two spike points that sit on bound i, extended a little
# past them, which is where the bound actually cuts this projection.
for i in range(3):
    on_i = [ys[k] for ys, bs in [(ysA, bsA), (ysB, bsB)] for k in range(3) if bs[k] == i]
    if len(on_i) < 2:
        continue
    q1, q2 = projA(on_i[0]), projA(on_i[1])
    mid, dirn = 0.5 * (q1 + q2), (q2 - q1)
    dirn = dirn / np.linalg.norm(dirn)
    ext = 1.5 * np.linalg.norm(q2 - q1)
    p1, p2 = mid - ext * dirn, mid + ext * dirn
    axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], ls=(0, (6, 4)),
                 color=NEURON_COL[i + 1], lw=LW_BOUND, alpha=0.85, zorder=2)
set_view(axes[0], inkA)

# (b)
projB = projector(D_B.sum(axis=0), D_B[0])
# Four members, fading in toward the last one. Seven overlapping copies read as a smear and
# hide the one thing the panel is for, the segment of starts.
sel = np.linspace(0, len(arc) - 1, 4).astype(int)
inkB = []
for n, k in enumerate(sel):
    _, _, _, segs = arc[k]
    last = (n == len(sel) - 1)
    draw_orbit(axes[1], projB, segs,
               ORBIT_COL['A'] if last else None, '#111', '#111',
               strength=1.0 if last else 0.26 + 0.17 * n, numbers=last)
    for s in segs:
        inkB += [projB(s[0]), projB(s[1]), projB(s[2])]
starts = np.array([projB(a[2]) for a in arc])
# Below the orbit dots (zorder 10) and their numbers (11), above the arrows, so the segment
# passes behind the numbered circle it starts from instead of cutting across it.
axes[1].plot(starts[:, 0], starts[:, 1], color=FAMILY_COL, lw=6, solid_capstyle='round',
             zorder=8)
set_view(axes[1], inkB)

# The two orbits are told apart by the light and dark dots, which the caption explains; the
# legend only names the three kinds of stroke.
legend_elems = [
    Line2D([0], [0], color=NEURON_COL[1], lw=LW_SPIKE, label='spike'),
    Line2D([0], [0], color=DECAY_COL, lw=LW_DECAY, label='decay'),
    Line2D([0], [0], color=FAMILY_COL, lw=6, label='orbit starts'),
    # Neutral grey: in the panels each bound carries its own neuron colour, so the entry is
    # there to name the dashed stroke, not to claim one particular neuron.
    Line2D([0], [0], color='#555555', lw=LW_BOUND, ls=(0, (6, 4)), label='bound'),
]
# One legend for both panels, in the lower right of the figure. Inside either panel it would
# sit on a trajectory, so it goes in reserved margin instead of over the drawing.
fig.legend(handles=legend_elems, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4,
           fontsize=FS_LEGEND, frameon=False, handletextpad=0.7, columnspacing=2.0)

style.panel_label(axes[0], 'a', subtitle='Two coexisting orbits', fontsize=FS_PANEL, y=1.0)
style.panel_label(axes[1], 'b', subtitle=r'A line attractor at $\mu_1=\mu_2$',
                  fontsize=FS_PANEL, y=1.0)

fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.13, wspace=0.05)
_here = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    out = os.path.normpath(os.path.join(_here, '..', 'figures', 'multistability_pair.' + ext))
    fig.savefig(out, dpi=200)
    print('saved', out)
