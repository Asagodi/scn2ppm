"""Inverse solver scaling: admissibility margin as a function of network size K.

Data (from inverse_solver experiments on permutation words with simplex decoders,
documented in Appendix C of the paper):
    K = 2 .. 6 : solver succeeds, max margin m decreases monotonically
    K = 7, 8   : pairwise decoder cosines too large (>~0.83), no admissible drives
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt

import style
style.setup()

# Experimentally measured maximum admissibility margin per K
K_ok = np.array([2, 3, 4, 5, 6])
margin = np.array([0.12, 0.039, 0.017, 0.008, 0.005])

# K = 7, 8: solver returns m <= 0 (infeasible).
K_fail = np.array([7, 8])

# Same height as fig_margin_landscape so both panels' y-extents match when
# scaled to a common height in the paper.
fig, ax = plt.subplots(figsize=(6.0, 5.4))

# success region: simple dot marker (no ring around)
ax.semilogy(K_ok, margin, 'o-',
            color=style.COLORS['accent'], lw=1.6, ms=10,
            label='solver succeeds')

# infeasible region: shaded background + square markers (no ring around)
ax.axvspan(6.5, 8.5, alpha=0.10, color=style.COLORS['red'])
y_fail = 5e-4
ax.scatter(K_fail, [y_fail] * len(K_fail),
           marker='s', color=style.COLORS['red'], s=90,
           label=r'infeasible ($m\leq 0$)')

ax.set_xlabel('Number of neurons $K$', fontsize=28)
ax.set_ylabel(r'$m^*$', rotation=0, labelpad=20, fontsize=30)
ax.set_xticks([2, 3, 4, 5, 6, 7, 8])
ax.set_xlim(1.5, 8.5)
ax.set_ylim(2e-4, 0.5)
ax.tick_params(labelsize=22)
ax.legend(fontsize=20, loc='lower left', framealpha=0.85)
# Panel label (a,b are margin-landscape panels; this is panel c)
style.panel_label(ax, 'c', subtitle='Solver scaling', fontsize=28, y=1.02)

# Exactly the same top/bottom fractions as fig_margin_landscape so the panel
# labels sit at the same fig-height fraction after bbox_inches='tight'.
fig.subplots_adjust(left=0.18, right=0.95, top=0.86, bottom=0.16)

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'inverse_scaling.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.05)
print(f'Saved {_out}')
