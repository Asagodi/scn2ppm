"""
3-neuron WM example for poster Box 5 (Inverse Problem + WM).

Panel (a): Margin landscape m(alpha_1, alpha_2) at fixed alpha_3=1.0 for word RGB.
Panel (b): 3D geometry showing the period-3 orbit, same style as fig_3d_example panel (a).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import style

style.setup()

# ── 3 decoders on a wide cone around (1,1,1)/sqrt(3) for visual clarity ──
c_axis = np.array([1, 1, 1]) / np.sqrt(3)
u1 = np.array([1, -1, 0]) / np.sqrt(2)
u2 = np.cross(c_axis, u1); u2 /= np.linalg.norm(u2)

SPREAD = 0.7
DECODERS = []
for a in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
    d = c_axis + SPREAD * (np.cos(a) * u1 + np.sin(a) * u2)
    d /= np.linalg.norm(d)
    DECODERS.append(d)


def tangent_frame(d):
    N = len(d); cols = []
    for i in range(N):
        v = np.zeros(N); v[i] = 1.0; v -= (d @ v) * d
        for c in cols: v -= (c @ v) * c
        n = np.linalg.norm(v)
        if n > 1e-10: cols.append(v / n)
        if len(cols) == N - 1: break
    return np.column_stack(cols)


FRAMES = [tangent_frame(d) for d in DECODERS]
HEX = ['#e74c3c', '#27ae60', '#2980b9']   # R, G, B
NAMES = ['R', 'G', 'B']


def step(xi, bi, alphas):
    yp = (alphas[bi] + 1) * DECODERS[bi] + FRAMES[bi] @ xi
    best_j, best_c = -1, -1.0
    for j in range(3):
        if alphas[j] <= 0: continue
        dn = DECODERS[j] @ yp
        if dn < 1e-12: continue
        c = alphas[j] / dn
        if 0 < c <= 1.0 + 1e-10 and c > best_c:
            best_c, best_j = c, j
    if best_j < 0: return None, -1
    return FRAMES[best_j].T @ (best_c * yp), best_j


def transition_matrix(bi, bj, alphas):
    Di, Dj = DECODERS[bi], DECODERS[bj]
    Ei, Ej = FRAMES[bi], FRAMES[bj]
    ai, aj = alphas[bi], alphas[bj]
    M = np.zeros((3, 3))
    M[:2, :2] = aj * (Ej.T @ Ei)
    M[:2, 2] = aj * (ai + 1) * (Ej.T @ Di)
    M[2, :2] = (Ei.T @ Dj).ravel()
    M[2, 2] = (ai + 1) * (Dj @ Di)
    return M


def eigenvector_orbit(word, alphas):
    M = np.eye(3)
    for k in range(len(word)):
        bi, bj = word[k], word[(k + 1) % len(word)]
        M = transition_matrix(bi, bj, alphas) @ M
    ev, evc = np.linalg.eig(M)
    idx = np.argsort(np.abs(ev))[::-1]
    ev = ev[idx]; evc = evc[:, idx]
    xi_star = (evc[:2, 0] / evc[2, 0]).real
    orbit = []
    xi_k, bi_k = xi_star.copy(), word[0]
    for k in range(len(word)):
        y_k = alphas[bi_k] * DECODERS[bi_k] + FRAMES[bi_k] @ xi_k
        orbit.append((y_k.copy(), bi_k))
        yp = (alphas[bi_k] + 1) * DECODERS[bi_k] + FRAMES[bi_k] @ xi_k
        bj_n = word[(k + 1) % len(word)]
        c_n = alphas[bj_n] / (DECODERS[bj_n] @ yp)
        xi_k = FRAMES[bj_n].T @ (c_n * yp)
        bi_k = bj_n
    return orbit, ev, xi_star


def admissibility_margin(word, alphas):
    if any(a <= 0 for a in alphas):
        return -1.0
    try:
        orbit, _, xi_star = eigenvector_orbit(word, alphas)
    except Exception:
        return -1.0
    L = len(word)
    margins = []
    xi_k, bi_k = xi_star.copy(), word[0]
    for k in range(L):
        yp = (alphas[bi_k] + 1) * DECODERS[bi_k] + FRAMES[bi_k] @ xi_k
        cs = {}
        for j in range(3):
            if alphas[j] <= 0: continue
            dn = DECODERS[j] @ yp
            if dn < 1e-12: continue
            cs[j] = alphas[j] / dn
        target = word[(k + 1) % L]
        if target not in cs:
            return -1.0
        c_target = cs[target]
        for j, c_j in cs.items():
            if j != target:
                margins.append(c_target - c_j)
        if c_target <= 0:
            return -1.0
        xi_k = FRAMES[target].T @ (c_target * yp)
        bi_k = target
    return min(margins) if margins else -1.0


# ── Period-3 word RGB ──
WORD = [0, 1, 2]
ALPHAS = [1.0, 1.0, 1.0]

orbit, ev, xi_star = eigenvector_orbit(WORD, ALPHAS)
m = admissibility_margin(WORD, ALPHAS)
print(f"Word RGB: margin={m:.4f}, |mu2/mu1|={abs(ev[1]/ev[0]):.4f}")

# =====================================================================
# Combined 2-panel figure
# =====================================================================
fig = plt.figure(figsize=(15, 5.0))

# ── Panel (a): Margin landscape — 1/5 width ──
ax1 = fig.add_subplot(1, 5, 1)

n_grid = 100
a1_range = np.linspace(0.80, 1.20, n_grid)
a2_range = np.linspace(0.85, 1.25, n_grid)
margin_grid = np.full((n_grid, n_grid), np.nan)
a3_fixed = 1.0

for i, a1 in enumerate(a1_range):
    for j, a2 in enumerate(a2_range):
        margin_grid[j, i] = admissibility_margin(WORD, [a1, a2, a3_fixed])

vmax = np.nanmax(margin_grid[margin_grid > 0]) if np.any(margin_grid > 0) else 0.1
im = ax1.imshow(margin_grid, origin='lower', aspect='equal',
                 extent=[a1_range[0], a1_range[-1], a2_range[0], a2_range[-1]],
                 cmap='RdYlGn', vmin=-vmax * 0.5, vmax=vmax,
                 interpolation='bilinear')

try:
    ax1.contour(a1_range, a2_range, margin_grid, levels=[0], colors='k', linewidths=1.5)
except Exception:
    pass

ax1.plot(1.0, 1.0, '*', color='k', ms=14,
          markeredgecolor='white', markeredgewidth=0.6, zorder=10)

ax1.set_xlabel('$\\alpha_1$ (R)', fontsize=14)
ax1.set_ylabel('$\\alpha_2$ (G)', fontsize=14)
ax1.set_xticks([0.8, 1.0, 1.2])
ax1.set_yticks([0.9, 1.1])
ax1.text(0.04, 0.96, '$\\sigma=$RGB', transform=ax1.transAxes,
          fontsize=12, va='top', ha='left',
          bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=4))
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Margin $m$', fontsize=11)

# ── Panel (b): 3D geometry — original side view ──
ax2 = fig.add_subplot(1, 5, (2, 3), projection='3d')

def draw_3d(ax, view_elev, view_azim, zoom=1.0, crop_top=False):
    """Draw the orbit, bound patches, spike+decay segments, origin into ax."""
    # Spike bound patches
    for i in range(3):
        d = DECODERS[i]
        a = ALPHAS[i]
        foot = a * d
        t1 = FRAMES[i][:, 0]
        t2 = FRAMES[i][:, 1]
        pts_on = [orbit[k][0] for k in range(len(WORD)) if orbit[k][1] == i]
        xis_on = [FRAMES[i].T @ (p_ - foot) for p_ in pts_on]
        if xis_on:
            xis_arr = np.array(xis_on + [np.zeros(2)])
        else:
            xis_arr = np.array([np.zeros(2)])
        mpad = 0.3
        corners = np.array([
            foot + (xis_arr[:, 0].min() - mpad) * t1 + (xis_arr[:, 1].min() - mpad) * t2,
            foot + (xis_arr[:, 0].max() + mpad) * t1 + (xis_arr[:, 1].min() - mpad) * t2,
            foot + (xis_arr[:, 0].max() + mpad) * t1 + (xis_arr[:, 1].max() + mpad) * t2,
            foot + (xis_arr[:, 0].min() - mpad) * t1 + (xis_arr[:, 1].max() + mpad) * t2,
        ])
        poly = Poly3DCollection([corners.tolist()], alpha=0.18,
                                 facecolor=HEX[i], edgecolor=HEX[i], linewidth=1.1)
        ax.add_collection3d(poly)

    # Orbit points
    for k, (y, bi_) in enumerate(orbit):
        ax.scatter([y[0]], [y[1]], [y[2]], color=HEX[bi_], s=110,
                    edgecolors='k', linewidths=0.7, depthshade=False, zorder=10)

    # Spike + decay segments
    xi_k, bi_k = xi_star.copy(), WORD[0]
    for k in range(len(WORD)):
        y = ALPHAS[bi_k] * DECODERS[bi_k] + FRAMES[bi_k] @ xi_k
        yp = y + DECODERS[bi_k]
        ax.plot([y[0], yp[0]], [y[1], yp[1]], [y[2], yp[2]],
                 color=HEX[bi_k], lw=1.3, alpha=0.75)
        xi_n, bi_n = step(xi_k, bi_k, ALPHAS)
        y_next = ALPHAS[bi_n] * DECODERS[bi_n] + FRAMES[bi_n] @ xi_n
        ax.plot([yp[0], y_next[0]], [yp[1], y_next[1]], [yp[2], y_next[2]],
                 color='gray', lw=0.7, alpha=0.5, ls='--')
        xi_k, bi_k = xi_n, bi_n

    ax.scatter([0], [0], [0], color='k', s=35, depthshade=True, zorder=0, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    ax.view_init(elev=view_elev, azim=view_azim)
    if crop_top:
        # Tight bounds — eliminates 3D internal padding
        ax.set_xlim(-0.1, 1.05); ax.set_ylim(-0.1, 1.05); ax.set_zlim(-0.1, 1.05)
    if zoom != 1.0:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)


# Original side view
draw_3d(ax2, view_elev=70, view_azim=180)

# Panel (c): view along (1,1,1)
ax3 = fig.add_subplot(1, 5, (4, 5), projection='3d')
draw_3d(ax3, view_elev=35.264, view_azim=45, zoom=1.3, crop_top=True)

fig.tight_layout(w_pad=0.5)
# Move ax3 closer to ax2 by ~half its width-of-gap
pos2 = ax2.get_position()
pos3 = ax3.get_position()
shift = (pos3.x0 - (pos2.x0 + pos2.width)) * 0.8  # close 80% of the visible gap
ax3.set_position([pos3.x0 - shift, pos3.y0, pos3.width, pos3.height])
# style.panel_labels(fig, [ax1, ax2])

import os
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'wm_figure.pdf'))
os.makedirs(os.path.dirname(_out), exist_ok=True)
fig.savefig(_out, dpi=300, bbox_inches='tight')
print(f"Saved {_out}")
