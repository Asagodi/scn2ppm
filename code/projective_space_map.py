"""Figure: Projective space and a projective map, 2 panels on 1 row.

Panels:
  (a) Projective space RP^1: lines through the origin in R^2, each
      identified with a single point of RP^1 (antipodal pair, same color).
  (b) A projective map: a linear map A on R^2 descends to a well-defined
      map on RP^1, i.e. it permutes the same set of lines (colors are
      preserved as labels) but redistributes them non-uniformly around
      the circle of directions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import style

style.setup()

rng = np.random.default_rng(0)

# ── Shared setup ──
# N directions, evenly spaced over [0, pi) -- each theta_i together with
# theta_i + pi is a single point of RP^1 (a line through the origin).
N = 7
thetas = np.linspace(0, np.pi, N, endpoint=False) + 0.08
COLORS = ['#4C72B0', '#DD8452', '#55A467', '#C44E52',
          '#8172B3', '#937860', '#CC79A7']

EXT = 1.55  # how far lines are drawn beyond the unit circle
RADIUS = 1.0


def draw_unit_circle(ax, **kw):
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(phi), np.sin(phi), color=style.COLORS['gray'],
            lw=0.9, ls='--', zorder=1, **kw)


def style_axes(ax, lim=1.75):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color=style.COLORS['gray'], lw=0.6, zorder=0)
    ax.axvline(0, color=style.COLORS['gray'], lw=0.6, zorder=0)


fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))

PANEL_LETTERS = list('ab')
PANEL_SUBTITLES = [r'Projective space $\mathrm{RP}^1$', 'A projective map']
for _ax, _l, _s in zip(axes, PANEL_LETTERS, PANEL_SUBTITLES):
    style.panel_label(_ax, _l, subtitle=_s, fontsize=20, y=1.03)

# ═══ (a) Projective space: lines through the origin ═══
ax = axes[0]
draw_unit_circle(ax)

for theta, col in zip(thetas, COLORS):
    u = np.array([np.cos(theta), np.sin(theta)])
    ax.plot([-EXT * u[0], EXT * u[0]], [-EXT * u[1], EXT * u[1]],
            color=col, lw=1.7, zorder=2)
    for sign in (+1, -1):
        p = sign * RADIUS * u
        ax.plot(p[0], p[1], 'o', color=col, ms=8,
                markeredgecolor='k', markeredgewidth=0.6, zorder=3)

# Highlight one line explicitly: x and lambda*x -> same point [x]
i0 = 1
u0 = np.array([np.cos(thetas[i0]), np.sin(thetas[i0])])
x_pt = 0.55 * u0
mx_pt = -1.3 * u0
ax.plot(x_pt[0], x_pt[1], '*', color='k', ms=13, zorder=4)
ax.plot(mx_pt[0], mx_pt[1], '*', color='k', ms=13, zorder=4)
ax.annotate(r'$x$', xy=x_pt, xytext=x_pt + np.array([-0.32, 0.16]),
            fontsize=15)
ax.annotate(r'$\lambda x$', xy=mx_pt, xytext=mx_pt + np.array([-0.62, -0.05]),
            fontsize=15)
ax.text(0.0, -1.68, r'$[x] = \{\lambda x : \lambda \neq 0\}$',
        ha='center', va='top', fontsize=15)

style_axes(ax)

# ═══ (b) Projective map: [x] -> [Ax] ═══
ax = axes[1]
draw_unit_circle(ax)

# A mildly anisotropic, invertible linear map (positive determinant).
A = np.array([[1.6, 0.5],
              [0.3, 0.7]])

# Faint "before" lines for reference.
for theta in thetas:
    u = np.array([np.cos(theta), np.sin(theta)])
    ax.plot([-EXT * u[0], EXT * u[0]], [-EXT * u[1], EXT * u[1]],
            color=style.COLORS['gray'], lw=0.8, ls=':', alpha=0.6, zorder=1)

for theta, col in zip(thetas, COLORS):
    u = np.array([np.cos(theta), np.sin(theta)])
    Au = A @ u
    v = Au / np.linalg.norm(Au)  # renormalize: only the direction matters

    # curved arrow from the original point on the unit circle to its image,
    # tracing the shorter arc so the redistribution is easy to read
    ang0, ang1 = np.arctan2(u[1], u[0]), np.arctan2(v[1], v[0])
    d = (ang1 - ang0 + np.pi) % (2 * np.pi) - np.pi
    arc = np.linspace(ang0, ang0 + d, 30)
    r_arc = 1.18
    ax.plot(r_arc * np.cos(arc), r_arc * np.sin(arc), color=col, lw=1.1,
            alpha=0.85, zorder=2)
    ax.annotate('', xy=(r_arc * np.cos(arc[-1]), r_arc * np.sin(arc[-1])),
                xytext=(r_arc * np.cos(arc[-2]), r_arc * np.sin(arc[-2])),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1),
                zorder=2)

    # image line through the origin (solid) and its point markers
    ax.plot([-EXT * v[0], EXT * v[0]], [-EXT * v[1], EXT * v[1]],
            color=col, lw=1.7, zorder=3)
    for sign in (+1, -1):
        p = sign * RADIUS * v
        ax.plot(p[0], p[1], 'o', color=col, ms=8,
                markeredgecolor='k', markeredgewidth=0.6, zorder=4)
    # faint marker at the original (pre-image) position
    for sign in (+1, -1):
        p = sign * RADIUS * u
        ax.plot(p[0], p[1], 'o', color=col, ms=5, alpha=0.35, zorder=2)

ax.text(0.0, -1.68, r'$F([x]) = [Ax]$', ha='center', va='top', fontsize=15)

style_axes(ax)

fig.tight_layout(w_pad=2.0, pad=0.3)

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'projective_space_map.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out}")
