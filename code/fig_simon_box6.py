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
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox


class Arrow3D(FancyArrowPatch):
    """A FancyArrowPatch placed by 3-D endpoints.

    Gives the same stroke as the 2-D arrows in the companion figure: one patch with a single
    '-|>' head whose size is set by mutation_scale. The 3-D quiver used before drew the head
    as a separate cone sized by a fraction of the shaft, which at these lengths produced a
    barb almost as long as the arrow.
    """

    def __init__(self, xs, ys, zs, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._xyz
        xp, yp, zp = proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xp[0], yp[0]), (xp[1], yp[1]))
        return float(np.min(zp))

import scn
import style

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

# Painting order of the bound planes, back to front: green, blue, red, yellow. The axes is
# built with computed_zorder=False, so this is what decides which translucent plane shows
# through which; depth sorting would otherwise pick an order from the viewpoint.
BOUND_ORDER = [3, 1, 0, 2]
VIEW_ELEV, VIEW_AZIM = 18, 35

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


def draw_scene(ax, mpad=0.35, mpad_by_neuron=None):
    """Draw the scene; return each bound's corners and every piece of ink to avoid.

    Explicit z-orders (the axes is built with computed_zorder=False) keep the bound
    quadrilaterals strictly behind the trajectory, so no plane is ever drawn over an arrow
    or a numbered dot. Matplotlib's own 3D depth sort cannot guarantee that: it orders whole
    artists by mean depth, which lets a large translucent polygon win against a short segment
    that is in front of it.
    """
    corners_per_bound, rects = [], []
    for i in range(4):
        d = DECODERS[i]; a = ALPHAS[i]
        foot = a * d
        t1 = FRAMES[i][:, 0]; t2 = FRAMES[i][:, 1]
        # Size each quadrilateral around the spikes that actually land on it. Including the
        # foot of the bound, the point of the plane nearest the origin, stretched the patches
        # far past the orbit and was what sent them hanging below the figure.
        pts_on = [y for (y, bi, _) in orbit if bi == i]
        xis_arr = (np.array([FRAMES[i].T @ (p - foot) for p in pts_on])
                   if pts_on else np.zeros((1, 2)))
        u_mid, v_mid = xis_arr[:, 0].mean(), xis_arr[:, 1].mean()
        pad_i = (mpad_by_neuron or {}).get(i, mpad)
        u_half = 0.5 * float(np.ptp(xis_arr[:, 0])) + pad_i
        v_half = 0.5 * float(np.ptp(xis_arr[:, 1])) + pad_i
        u_min, u_max = u_mid - u_half, u_mid + u_half
        v_min, v_max = v_mid - v_half, v_mid + v_half
        corners = np.array([
            foot + u_min * t1 + v_min * t2,
            foot + u_max * t1 + v_min * t2,
            foot + u_max * t1 + v_max * t2,
            foot + u_min * t1 + v_max * t2,
        ])
        corners_per_bound.append(corners)
        rects.append((u_min, u_max, v_min, v_max))

    # (2) spike arrows & (3) gray dashed decay curves ----------------------
    y_pres = [y.copy() for (y, _, _) in orbit]
    firers = [bi for (_, bi, _) in orbit]
    L = len(WORD)
    y_posts = [y_pres[k] + DECODERS[firers[k]] for k in range(L)]
    y_next_pres = [y_pres[(k + 1) % L] for k in range(L)]

    # An arrow pointing nearly along the line of sight projects to almost nothing and
    # disappears under the dot it starts from. Those few are lifted above the dot and drawn
    # with some transparency, so the arrow reads while its numbered circle still shows
    # through. D_G is the case here: it projects about twenty times shorter than the rest.
    el, az = np.radians(VIEW_ELEV), np.radians(VIEW_AZIM)
    w = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    proj_len = [float(np.linalg.norm(DECODERS[b] - (DECODERS[b] @ w) * w)) for b in firers]
    short = 0.25 * float(np.median(proj_len))

    for k in range(L):
        col = HEX[firers[k]]
        yp = y_pres[k]; yq = y_posts[k]
        tiny = proj_len[k] < short
        ax.add_artist(Arrow3D([yp[0], yq[0]], [yp[1], yq[1]], [yp[2], yq[2]],
                              arrowstyle='-|>', mutation_scale=22, lw=2.8, color=col,
                              alpha=0.80 if tiny else 1.0,
                              zorder=10.5 if tiny else 6))

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
                ha='center', va='center_baseline', zorder=11)

    # (5) collect every piece of ink a label must stay clear of ------------
    ink = []
    for k in range(L):
        ink.append(y_pres[k])
        ink.append(y_posts[k])
        for f in (0.25, 0.5, 0.75):                      # along the spike arrow
            ink.append(y_pres[k] + f * (y_posts[k] - y_pres[k]))
        for f in (0.25, 0.5, 0.75):                      # along the decay segment
            ink.append(y_posts[k] + f * (y_next_pres[k] - y_posts[k]))

    # (6) hide 3D box ------------------------------------------------------
    ax._axis3don = False
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_edgecolor((0, 0, 0, 0))
        pane.fill = False
    ax.grid(False)
    return corners_per_bound, np.array(ink), rects


def place_corner_labels(ax, corners_per_bound, ink):
    """Put each R/B/Y/G label on the corner of its own bound with the most free space.

    Distances are measured after projection, because what matters is overlap on the page and
    not in 3-D. Each label also avoids the labels already placed, so the four never collide.
    """
    from mpl_toolkits.mplot3d import proj3d

    def to2d(p):
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        return np.array([x, y])

    from matplotlib.path import Path

    ink2d = np.array([to2d(p) for p in ink])
    quads = [Path(np.array([to2d(c) for c in corners])) for corners in corners_per_bound]

    ink_mid = ink2d.mean(axis=0)
    clear_min = 0.055 * np.ptp(ink2d, axis=0).max()

    placed = []
    for i, corners in enumerate(corners_per_bound):
        # Sit the letter just outside its own corner, not on it, so neighbouring labels
        # separate and none of them lands on the quadrilateral's edge.
        centre = corners.mean(axis=0)
        cands = [c + 0.16 * (c - centre) / max(np.linalg.norm(c - centre), 1e-9)
                 for c in corners]
        scored = []
        for c in cands:
            c2 = to2d(c)
            # how many OTHER bound faces cover this corner: a label there would sit on them
            covered = sum(q.contains_point(c2) for j, q in enumerate(quads) if j != i)
            clear = np.min(np.linalg.norm(ink2d - c2, axis=1))
            if placed:
                clear = min(clear, np.min(np.linalg.norm(np.array(placed) - c2, axis=1)))
            scored.append((covered, clear, float(np.linalg.norm(c2 - ink_mid)), c, c2))
        # Free of other faces and clear of the drawing, then as close in as possible: a far
        # corner pushes the saved bounding box out and costs space on the page.
        free = [s for s in scored if s[0] == 0]
        pool = [s for s in free if s[1] >= clear_min] or free or scored
        _, _, _, best, best2 = min(pool, key=lambda s: s[2])
        placed.append(best2)
        ax.text(best[0], best[1], best[2], NAMES[i], color=HEX[i],
                fontsize=14, fontweight='bold', ha='center', va='center', zorder=12,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))


def _area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.array(poly)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


def _ccw(poly):
    a = sum((poly[i][0] * poly[(i + 1) % len(poly)][1]
             - poly[(i + 1) % len(poly)][0] * poly[i][1]) for i in range(len(poly)))
    return poly if a >= 0 else poly[::-1]


def clip_convex(subject, clipper):
    """Sutherland-Hodgman: intersect a polygon with a convex one. Both are convex here."""
    clipper = _ccw(list(clipper))
    out = list(subject)
    for k in range(len(clipper)):
        a, b = clipper[k], clipper[(k + 1) % len(clipper)]

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        src, out = out, []
        if not src:
            break
        s = src[-1]
        for e in src:
            se, ss = side(e), side(s)
            if se >= -1e-12:
                if ss < -1e-12:
                    t = ss / (ss - se)
                    out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
                out.append(e)
            elif ss >= -1e-12:
                t = ss / (ss - se)
                out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
            s = e
    return out


def clip_halfplane(poly, f):
    """Keep the part of a polygon where the affine functional f is non-negative."""
    out, n = [], len(poly)
    if n == 0:
        return out
    for k in range(n):
        s, e = poly[k - 1], poly[k]
        fs, fe = f(*s), f(*e)
        if fe >= 0:
            if fs < 0:
                t = fs / (fs - fe)
                out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
            out.append(e)
        elif fs >= 0:
            t = fs / (fs - fe)
            out.append((s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])))
    return out


def convex_difference(poly, cut):
    """poly minus a convex region, as a list of convex pieces.

    For each edge of `cut`, take the part of `poly` outside that edge but inside the edges
    already passed. The pieces are disjoint, convex, and their union is exactly the
    difference, so no general polygon-boolean machinery is needed.
    """
    cut = _ccw(list(cut))
    pieces, inside_so_far = [], []
    for k in range(len(cut)):
        a, b = cut[k], cut[(k + 1) % len(cut)]

        def outside(u, v, a=a, b=b):
            return -((b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0]))

        piece = clip_halfplane(poly, outside)
        for (a2, b2) in inside_so_far:
            def inside(u, v, a=a2, b=b2):
                return (b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0])
            piece = clip_halfplane(piece, inside)
        if len(piece) >= 3:
            pieces.append(piece)
        inside_so_far.append((a, b))
    return pieces


def draw_visible_planes(ax, rects, alpha=0.42):
    """Draw each bound only where it is in front; where it is hidden, draw nothing.

    For planes i and j, the hidden part of i is where i projects inside j's patch AND lies
    behind j along the view direction: a convex polygon intersected with a half-plane, so it
    stays convex. Subtracting those regions from i's patch leaves the part actually seen.

    The patches interpenetrate, so some cuts follow the line where two planes cross and are
    parallel to neither patch. That line is where the planes really do change places, so it
    belongs in the figure. Settling a pair by a painting order instead would be parallel to
    the front patch but would hide parts of the other one that are in view.
    """
    M = ax.get_proj()          # ax.M is only filled in during a draw
    el, az = np.radians(ax.elev), np.radians(ax.azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def to3d(i, u, v):
        return ALPHAS[i] * DECODERS[i] + u * FRAMES[i][:, 0] + v * FRAMES[i][:, 1]

    def screen(p):
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], M)
        return np.array([x, y])

    maps = []
    for i in range(4):
        o = screen(to3d(i, 0, 0))
        maps.append((o, np.column_stack([screen(to3d(i, 1, 0)) - o,
                                         screen(to3d(i, 0, 1)) - o])))

    def rect(i):
        u0, u1, v0, v1 = rects[i]
        return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]

    def overlap(i, j):
        """Where j's patch projects over i's, in i's chart."""
        a0, a1, b0, b1 = rects[j]
        quad_j3 = [to3d(j, a0, b0), to3d(j, a1, b0), to3d(j, a1, b1), to3d(j, a0, b1)]
        o_i, A_i = maps[i]
        Ainv = np.linalg.inv(A_i)
        return clip_convex(rect(i), [tuple(Ainv @ (screen(p) - o_i)) for p in quad_j3])

    def hidden_exact(i, j):
        """The part of that overlap where i really is the farther plane."""
        dn = float(DECODERS[j] @ view)

        def behind(u, v):
            s = float(ALPHAS[j] - DECODERS[j] @ to3d(i, u, v))
            return s / dn if abs(dn) > 1e-12 else -1.0

        return clip_halfplane(overlap(i, j), behind)

    for i in range(4):
        pieces = [rect(i)]
        for j in range(4):
            if j == i:
                continue
            hidden = hidden_exact(i, j)
            if len(hidden) < 3:
                continue
            pieces = [q for p in pieces for q in convex_difference(p, hidden)]

        # Cutting one quad out of another leaves slivers of no width along the shared edges.
        # They carry no ink but each one costs a Poly3DCollection, so drop them. Only
        # degenerate pieces go: a plane whose visible part falls into separate islands keeps
        # every island, because each is a part of the plane that really is in view.
        pieces = [p for p in pieces if _area(p) > 1e-6 * _area(rect(i))]

        for p in pieces:
            pts = [to3d(i, u, v) for (u, v) in p]
            ax.add_collection3d(Poly3DCollection(
                # No edge: the pieces are a subdivision of one plane, so outlining each one
                # would draw seams across its interior.
                [np.array(pts).tolist()], facecolor=HEX[i], edgecolor='none',
                alpha=alpha, zorder=1 + 0.2 * BOUND_ORDER.index(i)))


# ─── figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 6.5))
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
ax.set_proj_type('ortho')   # chart-to-screen must be affine to invert it
corners_per_bound, ink, rects = draw_scene(ax, mpad=0.28, mpad_by_neuron={2: 0.10})
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

# Snug view on the bound feet and the orbit. The origin is no longer drawn, so including it
# would only pad the frame with empty space.
foot_pts = np.array([ALPHAS[i] * DECODERS[i] for i in range(4)])
orb_pts = np.array([y for (y, _, _) in orbit])
all_pts = np.vstack([foot_pts, orb_pts])
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

# Occlusion shading and labels both read the projection, so they come after the view and the
# limits are final.
draw_visible_planes(ax, rects)
place_corner_labels(ax, corners_per_bound, ink)

def ink_bbox(fig, pad=0.03):
    """Bounding box of what is actually drawn, in inches.

    bbox_inches='tight' crops to the artists' reported extents, and a 3-D axes reports its
    whole box however little of it the projection fills. Here the canvas is rendered and the
    non-white pixels are measured instead, which for this figure removes about a quarter of
    the width and a fifth of the height.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    ink = (buf[:, :, :3] < 250).any(axis=2)
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    dpi, H = fig.dpi, buf.shape[0]
    return Bbox([[cols.min() / dpi - pad, (H - 1 - rows.max()) / dpi - pad],
                 [(cols.max() + 1) / dpi + pad, (H - rows.min()) / dpi + pad]])


_here = os.path.dirname(os.path.abspath(__file__))

# TWO output variants written from the same axes:
#   simon_rbygyb.pdf        : POSTER version with "(c) RBYGYB orbit" panel label
#                             at fontsize=38 (matches Box 6 co-panels).
#   simon_rbygyb_paper.pdf  : PAPER version with NO panel label — the paper
#                             refers to it as a whole (fig:simon), not as
#                             panel (c) of anything.

# ─── save paper version FIRST (no label) ────────────────────────────────
_out_paper = os.path.normpath(os.path.join(_here, '..', 'figures',
                                            'simon_rbygyb_paper.pdf'))
os.makedirs(os.path.dirname(_out_paper), exist_ok=True)
fig.savefig(_out_paper, dpi=300, bbox_inches=ink_bbox(fig))
print(f"Saved {_out_paper}")
_png = _out_paper.replace('.pdf', '.png')
fig.savefig(_png, dpi=200, bbox_inches=ink_bbox(fig))
print(f"Saved {_png}")

# ─── add poster label and save poster version ────────────────────────────
style.panel_label(ax, 'c', subtitle='RBYGYB orbit', fontsize=38, y=1.0)

_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'simon_rbygyb.pdf'))
os.makedirs(os.path.dirname(_out), exist_ok=True)
fig.savefig(_out, dpi=300, bbox_inches=ink_bbox(fig))
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
