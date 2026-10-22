"""Fig 4: inverse problem, margin landscape and solver scaling, as ONE figure.

Replaces the former pair of separately saved PDFs (margin_landscape_paper.pdf and
inverse_scaling.pdf) that the manuscript stitched together with two minipages. Because the
two were rendered independently and then scaled to a common height in LaTeX, their panel
labels never landed on the same baseline. Drawing all three panels on a single set of axes
makes the alignment automatic: one figure, one top edge, one call to panel_label per panel.

(a),(b) admissibility margin m(alpha_1, alpha_2) for a short and a long target word, sharing
        one diverging colorbar centred on m = 0.
(c)     the largest attainable margin against network size K.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from scn import make_2d_system, admissibility_margin
import style

style.setup()

PSI = 0.4 * np.pi
THETA1 = 0.25
WORDS = [
    ([0, 0, 1], r'Short ($L=3$)'),
    ([0, 0, 1, 0, 0, 0, 1], r'Long ($L=7$)'),
]

# measured maximum margin per K (permutation words, simplex decoders; Appendix C)
K_OK = np.array([2, 3, 4, 5, 6])
M_OK = np.array([0.12, 0.039, 0.017, 0.008, 0.005])
K_FAIL = np.array([7, 8])          # solver returns m <= 0

VMAX, VMIN = 0.08, -0.04           # round limits so the colorbar ticks read cleanly
FS_PANEL, FS_AXIS, FS_TICK = 26, 26, 20

n_grid = 150
a1_range = np.linspace(0.1, 2.5, n_grid)
a2_range = np.linspace(0.1, 2.5, n_grid)

grids = []
for word, _ in WORDS:
    g = np.full((n_grid, n_grid), np.nan)
    for i, a1 in enumerate(a1_range):
        for j, a2 in enumerate(a2_range):
            g[j, i] = admissibility_margin(word, make_2d_system(THETA1, PSI, a1, a2))
    grids.append(g)
    if np.any(g > 0):
        print('%-26s max margin = %.4f' % (str(word), float(np.nanmax(g[g > 0]))))

# ── one figure: two landscapes, the shared colorbar, then the scaling panel ──
# The empty column is a spacer so the colorbar does not crowd panel (c).
fig = plt.figure(figsize=(15.5, 5.0))
gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.045, 0.14, 1.05], wspace=0.22,
                      left=0.045, right=0.985, top=0.86, bottom=0.17)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])
ax_c = fig.add_subplot(gs[0, 4])

norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)
im = None
for ax, grid, (word, subtitle), letter in zip((ax_a, ax_b), grids, WORDS, 'ab'):
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='RdBu_r', norm=norm,
                   extent=[a1_range[0], a1_range[-1], a2_range[0], a2_range[-1]],
                   interpolation='bilinear')
    ax.contour(a1_range, a2_range, grid, levels=[0], colors='k', linewidths=1.5)
    ok = np.where(np.isfinite(grid) & (grid > 0))
    if len(ok[0]):
        best = np.argmax(grid[ok])
        ax.plot(a1_range[ok[1][best]], a2_range[ok[0][best]], marker='o', color='k',
                ms=9, markeredgecolor='white', markeredgewidth=1.0, zorder=10)
    ax.set_xlabel(r'$\alpha_1$', fontsize=FS_AXIS)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=FS_TICK)
    style.panel_label(ax, letter, subtitle=subtitle, fontsize=FS_PANEL, y=1.02)
ax_a.set_ylabel(r'$\alpha_2$', fontsize=FS_AXIS)
ax_b.set_yticklabels([])

cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([VMIN, 0.0, VMAX])
cbar.set_ticklabels(['%.2f' % VMIN, '0', '%.2f' % VMAX])
cbar.ax.tick_params(labelsize=FS_TICK)
cax.set_title(r'$m$', fontsize=FS_AXIS, pad=12)

ax_c.semilogy(K_OK, M_OK, 'o-', color=style.COLORS['accent'], lw=1.6, ms=9,
              label='solver succeeds')
ax_c.axvspan(6.5, 8.5, alpha=0.10, color=style.COLORS['red'])
ax_c.scatter(K_FAIL, [5e-4] * len(K_FAIL), marker='s', color=style.COLORS['red'], s=80,
             label=r'infeasible ($m\leq 0$)')
ax_c.set_xlabel(r'number of neurons $K$', fontsize=FS_AXIS)
ax_c.set_ylabel(r'$m^*$', rotation=0, labelpad=20, fontsize=FS_AXIS + 2)
ax_c.set_xticks([2, 3, 4, 5, 6, 7, 8])
ax_c.set_xlim(1.5, 8.5)
ax_c.set_ylim(2e-4, 0.5)
ax_c.tick_params(labelsize=FS_TICK)
ax_c.legend(fontsize=17, loc='lower left', framealpha=0.85)
style.panel_label(ax_c, 'c', subtitle='Solver scaling', fontsize=FS_PANEL, y=1.02)

_here = os.path.dirname(os.path.abspath(__file__))
for ext in ('pdf', 'png'):
    out = os.path.normpath(os.path.join(_here, '..', 'figures', 'inverse_paper.' + ext))
    fig.savefig(out, dpi=300)
    print('Saved', out)
