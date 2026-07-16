"""Fig 4(a,b): Admissibility margin landscape m(alpha_1, alpha_2) for K=N=2.

Two side-by-side panels for two target words. A single shared colorbar on the
right uses a diverging colormap centered on m=0 with scientific notation and
three ticks (min-mid-max). "m" label sits above the colorbar.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
from scn import make_2d_system, admissibility_margin
import style

style.setup()

PSI = 0.4 * np.pi
THETA1 = 0.25

WORDS = [
    ([0, 0, 1],                   r'$\sigma = (1,1,2)$'),
    ([0, 0, 1, 0, 0, 0, 1],       r'$\sigma = (1,1,2,1,1,1,2)$'),
]
PANEL_SUBTITLES = ['Short word (L=3)', 'Long word (L=7)']

# Compute both margin grids first so we can share vmin/vmax on the colorbar.
n_grid = 150
a1_range = np.linspace(0.1, 2.5, n_grid)
a2_range = np.linspace(0.1, 2.5, n_grid)

grids = []
maxes = []
for word, title in WORDS:
    margin_grid = np.full((n_grid, n_grid), np.nan)
    for i, a1 in enumerate(a1_range):
        for j, a2 in enumerate(a2_range):
            sys_ = make_2d_system(THETA1, PSI, a1, a2)
            margin_grid[j, i] = admissibility_margin(word, sys_)
    grids.append(margin_grid)
    if np.any(margin_grid > 0):
        maxes.append(float(np.nanmax(margin_grid[margin_grid > 0])))
        print(f'{title}: max margin = {maxes[-1]:.4f}')

# Round the color limits to nice multiples of 10^-2 (measured max is ~0.0804,
# widest inadmissible region ~-0.03) so the colorbar ticks read cleanly.
vmax = 0.08   #  8e-2
vmin = -0.04  # -4e-2
norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

# Layout: two square panels + narrow colorbar on the right.
# Height 5.4in matches fig_inverse_scaling; explicit subplots_adjust below
# keeps the plot area top at the same fig-fraction, so panel labels align
# with (c) once both PDFs are scaled to the same height in the paper.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4),
                          gridspec_kw={'wspace': 0.20})

im = None
for panel, (word, title) in enumerate(WORDS):
    ax = axes[panel]
    im = ax.imshow(grids[panel], origin='lower', aspect='auto',
                    extent=[a1_range[0], a1_range[-1], a2_range[0], a2_range[-1]],
                    cmap='RdBu_r', norm=norm, interpolation='bilinear')
    try:
        ax.contour(a1_range, a2_range, grids[panel], levels=[0],
                   colors='k', linewidths=1.5)
    except Exception:
        pass

    valid = np.where(np.isfinite(grids[panel]) & (grids[panel] > 0))
    if len(valid[0]) > 0:
        best_idx = np.argmax(grids[panel][valid])
        best_a1 = a1_range[valid[1][best_idx]]
        best_a2 = a2_range[valid[0][best_idx]]
        # Star marker (dot with contrast edge — kept for max locator)
        ax.plot(best_a1, best_a2, marker='o', color='k', ms=10,
                markeredgecolor='white', markeredgewidth=1.0, zorder=10)

    ax.set_xlabel(r'$\alpha_1$', fontsize=19)
    if panel == 0:
        ax.set_ylabel(r'$\alpha_2$', fontsize=19)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=15)
    style.panel_label(ax, chr(ord('a') + panel),
                       subtitle=PANEL_SUBTITLES[panel], fontsize=16, y=1.02)

# ── Shared colorbar with 3 hand-labelled ticks and a separate scientific
# multiplier drawn above the "m" label so it never touches the ticks. ──
cax = fig.add_axes([0.92, 0.20, 0.020, 0.60])   # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cax)
ticks = [vmin, 0.0, vmax]
cbar.set_ticks(ticks)
# Hand-formatted labels expressed in units of 10^-2 (so the ×10^-2 is a
# single multiplier drawn once, well away from the tick labels).
cbar.set_ticklabels([f'{int(round(vmin*100))}',
                     '0',
                     f'{int(round(vmax*100))}'])
cbar.ax.tick_params(labelsize=15)
# "m" label above the colorbar; the multiplier sits just above "m".
cax.text(0.5, 1.15, r'$\times 10^{-2}$',
          transform=cax.transAxes, ha='center', va='bottom', fontsize=15)
cax.set_title(r'$m$', fontsize=19, pad=28)

# Same top/bottom fractions as fig_inverse_scaling — panel labels then land
# at the same axes-fraction across both PDFs.
fig.subplots_adjust(left=0.07, right=0.90, top=0.86, bottom=0.16)

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'margin_landscape.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f'Saved {_out}')
