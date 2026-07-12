"""
Simon figure (K=4, N=3, sigma=RBYGYB, period-6): single paper-quality 3D panel.

Uses the numerically optimized decoder / drive configuration whose greedy
dynamics match the target word RBYGYB with admissibility margin 0.056 and a
strong contraction ratio |mu_2/mu_1| = 0.30 (perturbations halve every ~0.6
cycles). Prefer this stability profile over the higher-margin, weaker-stability
alternative (m=0.069, |mu_2/mu_1|=0.71) which also had a near-degenerate cone.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scn

try:
    import style
    style.setup()
except Exception:
    pass


# ─── decoders (paper-quality solution, pasted literally) ────────────────────
# Joint (D, alpha) inverse-solve result — see rbygyb-joint-inverse workflow.
# Warm-started from a per-neuron cone parametrisation with Y placed between
# B and G in azimuth (matching the sigma = R B Y G Y B transition graph),
# then all 12 params refined by Nelder-Mead.
DECODERS = np.array([
    [ 0.917321, -0.120376,  0.379515],   # R
    [-0.011784,  0.872237,  0.488941],   # B
    [ 0.321296, -0.241496,  0.915668],   # Y
    [ 0.783224,  0.517016,  0.345332],   # G
])
# renormalize just in case of tiny rounding
DECODERS = np.array([d / np.linalg.norm(d) for d in DECODERS])

ALPHAS = np.array([0.826021, 0.790467, 0.968666, 1.060437])

# firing-neuron palette: R B Y G
HEX = ['#e74c3c', '#2980b9', '#f1c40f', '#27ae60']
NAMES = ['R', 'B', 'Y', 'G']

WORD = [0, 1, 2, 3, 2, 1]  # RBYGYB

# Build tangent frames and an scn-style system dict.
FRAMES = [scn.tangent_frame(d) for d in DECODERS]
SYS = {
    'DS': [DECODERS[i] for i in range(4)],
    'ES': FRAMES,
    'alphas': list(ALPHAS),
    'K': 4, 'N': 3,
}


# ─── orbit reconstruction ────────────────────────────────────────────────────
orbit, eigs, xi_star = scn.eigenvector_orbit(WORD, SYS)


def draw_scene(ax, mpad=0.35):
    # (1) translucent bound quadrilaterals ---------------------------------
    label_anchors = []
    for i in range(4):
        d = DECODERS[i]; a = ALPHAS[i]
        foot = a * d
        t1 = FRAMES[i][:, 0]; t2 = FRAMES[i][:, 1]
        pts_on = [y for (y, bi, _) in orbit if bi == i]
        if pts_on:
            xis_on = [FRAMES[i].T @ (p - foot) for p in pts_on]
            xis_arr = np.array(xis_on + [np.zeros(2)])
        else:
            xis_arr = np.array([np.zeros(2)])
        u_min, v_min = xis_arr[:, 0].min() - mpad, xis_arr[:, 1].min() - mpad
        u_max, v_max = xis_arr[:, 0].max() + mpad, xis_arr[:, 1].max() + mpad
        corners = np.array([
            foot + u_min * t1 + v_min * t2,
            foot + u_max * t1 + v_min * t2,
            foot + u_max * t1 + v_max * t2,
            foot + u_min * t1 + v_max * t2,
        ])
        poly = Poly3DCollection([corners.tolist()], alpha=0.22,
                                facecolor=HEX[i], edgecolor=HEX[i],
                                linewidth=1.2)
        ax.add_collection3d(poly)
        # Anchor the R/B/Y/G label at a distinct corner of each polygon so
        # the labels don't overlap (all bound feet cluster near a common
        # direction, so polygon centers would pile up).
        corner_idx = i  # 0..3 → the four polygon corners in the loop above
        label_anchors.append(corners[corner_idx])

    # (2) spike arrows & (3) gray dashed decay curves ----------------------
    y_pres = [y.copy() for (y, _, _) in orbit]
    firers = [bi for (_, bi, _) in orbit]
    L = len(WORD)
    y_posts = [y_pres[k] + DECODERS[firers[k]] for k in range(L)]
    y_next_pres = [y_pres[(k + 1) % L] for k in range(L)]

    for k in range(L):
        col = HEX[firers[k]]
        yp = y_pres[k]; yq = y_posts[k]
        # solid spike as a plot segment
        ax.plot([yp[0], yq[0]], [yp[1], yq[1]], [yp[2], yq[2]],
                color=col, lw=2.2, alpha=0.95, zorder=5)
        # manual arrowhead via quiver (thin, colored)
        vec = yq - yp
        head_frac = 0.32
        base = yp + (1 - head_frac) * vec
        ax.quiver(base[0], base[1], base[2],
                  vec[0] * head_frac, vec[1] * head_frac, vec[2] * head_frac,
                  color=col, arrow_length_ratio=0.9, linewidth=2.2,
                  zorder=6)

    for k in range(L):
        yq = y_posts[k]; yn = y_next_pres[k]
        ax.plot([yq[0], yn[0]], [yq[1], yn[1]], [yq[2], yn[2]],
                color='0.55', lw=1.2, ls='--', alpha=0.85, zorder=4)

    # (4) orbit dots with step numbers 1..6 -------------------------------
    for k in range(L):
        y = y_pres[k]
        ax.scatter([y[0]], [y[1]], [y[2]],
                   color=HEX[firers[k]], s=180,
                   edgecolors='k', linewidths=0.9,
                   depthshade=False, zorder=10)
        # In the text near each dot show its step number so the reader can
        # trace the R -> B -> Y -> G -> Y -> B sequence.
        ax.text(y[0], y[1], y[2], str(k + 1),
                color='white', fontsize=9, fontweight='bold',
                ha='center', va='center', zorder=11)

    # (5) origin cross -----------------------------------------------------
    ax.scatter([0], [0], [0], marker='+', color='k', s=90, linewidths=1.6,
               depthshade=False, zorder=6)

    # (6) subtle R/B/Y/G labels at polygon corners -------------------------
    for i, c in enumerate(label_anchors):
        ax.text(c[0], c[1], c[2], NAMES[i], color=HEX[i],
                fontsize=14, fontweight='bold',
                ha='center', va='center', zorder=11,
                bbox=dict(facecolor='white', alpha=0.8,
                          edgecolor='none', pad=1.5))

    # (7) hide 3D box ------------------------------------------------------
    ax._axis3don = False
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0, 0, 0, 0))
        pane.fill = False
    ax.grid(False)


# ─── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(1, 1, 1, projection='3d')
draw_scene(ax, mpad=0.20)
ax.view_init(elev=18, azim=35)

# Snug view: include the four bound feet (a_i * D_i), all orbit points, and
# the origin. Center on the orbit/bounds cloud (origin lives at the corner).
foot_pts = np.array([ALPHAS[i] * DECODERS[i] for i in range(4)])
orb_pts = np.array([y for (y, _, _) in orbit])
all_pts = np.vstack([foot_pts, orb_pts, np.zeros((1, 3))])
lo = all_pts.min(axis=0) - 0.15
hi = all_pts.max(axis=0) + 0.15
mid = 0.5 * (lo + hi)
half = 0.5 * (hi - lo).max()
ax.set_xlim(mid[0] - half, mid[0] + half)
ax.set_ylim(mid[1] - half, mid[1] + half)
ax.set_zlim(mid[2] - half, mid[2] + half)
try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'simon_rbygyb.pdf'))
os.makedirs(os.path.dirname(_out), exist_ok=True)
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out}")


# ─── diagnostics ────────────────────────────────────────────────────────────
# (a) re-verify greedy dynamics matches RBYGYB (up to cyclic shift).
xi_k = xi_star.copy()
bi_k = WORD[0]
greedy_seq = []
for _ in range(2 * len(WORD)):
    greedy_seq.append(bi_k)
    xi_k, bi_k = scn.step(xi_k, bi_k, SYS)


def is_cyclic_shift(a, b):
    if len(a) != len(b):
        return False
    a2 = list(a) + list(a)
    return any(a2[i:i + len(b)] == list(b) for i in range(len(a)))


greedy_period = greedy_seq[:len(WORD)]
matches = is_cyclic_shift(greedy_period, WORD)

# (b) margin and stability ratio.
margin = scn.admissibility_margin(WORD, SYS)
abs_eigs = np.sort(np.abs(eigs))[::-1]
stability = abs_eigs[1] / abs_eigs[0]

print("--- diagnostics ---")
print(f"target word RBYGYB       : {WORD}")
print(f"greedy first-period seq  : {greedy_period}")
print(f"cyclic-shift match       : {matches}")
print(f"admissibility margin     : {margin:.6f}")
print(f"|mu_2/mu_1| stability    : {stability:.6f}")
print(f"return-map |eig| sorted  : {abs_eigs.tolist()}")
