"""Figure: Three-neuron staircases scanning each drive."""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scn import make_3d_symmetric_system, step, find_period
import style

style.setup()

ALPHAS_BASE = [1.0, 0.9, 0.7]
colors = style.NEURON_COLORS[:3]

def get_all_rhos(alphas):
    sys = make_3d_symmetric_system(alphas)
    for xi0, bi0 in [(np.array([0.1,0.1]),0),(np.array([0.5,-0.3]),1),
                      (np.array([-0.2,0.4]),2),(np.array([0.3,0.6]),0)]:
        xi, bi = xi0.copy(), bi0
        for _ in range(3000): xi, bi = step(xi, bi, sys)
        seq = []
        for _ in range(300): seq.append(bi); xi, bi = step(xi, bi, sys)
        if len(set(seq)) >= 2:
            p = find_period(seq)
            return [sum(1 for w in seq[:p] if w==j)/p for j in range(3)]
    return [1., 0., 0.]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)

scan_configs = [
    (0, np.linspace(0.4, 1.5, 300), r'Drive $\alpha_1$',
     r'(fixed $\alpha_2\!=\!0.9$, $\alpha_3\!=\!0.7$)'),
    (1, np.linspace(0.4, 1.5, 300), r'Drive $\alpha_2$',
     r'(fixed $\alpha_1\!=\!1.0$, $\alpha_3\!=\!0.7$)'),
    (2, np.linspace(0.4, 1.5, 300), r'Drive $\alpha_3$',
     r'(fixed $\alpha_1\!=\!1.0$, $\alpha_2\!=\!0.9$)'),
]

wp_rhos = get_all_rhos(ALPHAS_BASE)
global_max = 0

for panel, (vary_idx, a_range, xlabel, subtitle) in enumerate(scan_configs):
    ax = axes[panel]
    print(f"Computing staircase for alpha_{vary_idx+1}...")
    rhos_all = []
    for a_val in a_range:
        al = list(ALPHAS_BASE); al[vary_idx] = a_val
        rhos_all.append(get_all_rhos(al))
    global_max = max(global_max, max(max(r) for r in rhos_all))
    for j in range(3):
        axes[panel].plot(a_range, [r[j] for r in rhos_all], color=colors[j], lw=1.5,
                         label=f'$\\rho_{j+1}$')
    for j in range(3):
        ax.plot(ALPHAS_BASE[vary_idx], wp_rhos[j], '*', color=colors[j], ms=13,
                markeredgecolor='k', markeredgewidth=0.5, zorder=10)
    ax.set_xlabel(xlabel)
    ax.set_title(subtitle, fontsize=plt.rcParams['axes.titlesize'] - 1)
    ax.set_xlim(a_range[0], a_range[-1])
    if panel == 0: ax.set_ylabel(r'Firing ratio $\rho_j$')
    # Single shared legend at top centre (not in caption -- the figure is
    # busy enough that an explicit colour key is more useful here).
    if panel == 2:
        ax.legend(loc='upper right', framealpha=0.9,
                  fontsize=plt.rcParams['legend.fontsize'])

for ax in axes: ax.set_ylim(-0.02, global_max + 0.05)

fig.tight_layout(w_pad=1.5)
# Use per-axis panel_label (bold via fontweight='bold', placed above the axis
# via set_title(loc='left', y=1.02) — no overlap with y-tick labels).
for _ax, _l in zip(axes, 'abc'):
    style.panel_label(_ax, _l, fontsize=20, y=1.03)
fig.savefig('../figures/fig_3d_staircases.pdf', dpi=300, bbox_inches='tight')
print("Saved figures/fig_3d_staircases.pdf")
