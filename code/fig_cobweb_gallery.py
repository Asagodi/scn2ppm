"""Cobweb gallery: 8 panels across the devil's staircase (K=N=2).

One alpha_2 per distinct mode-locked plateau.
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scn import make_2d_system, first_return_B1, get_firing_ratios
import style

style.setup()

PSI = 0.4 * np.pi; THETA1 = 0.25; A1 = 1.0

# One alpha_2 per distinct plateau
A2_VALS = [0.20, 0.335, 0.38, 0.45, 0.49, 0.60, 0.73, 0.80]

fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

for idx, a2 in enumerate(A2_VALS):
    ax = axes[idx // 4][idx % 4]
    sys = make_2d_system(THETA1, PSI, A1, a2)

    ratios, word = get_firing_ratios(sys)
    rho = ratios[1]

    v21 = float((sys['ES'][0].T @ sys['DS'][1]).squeeze())
    xi_part = (a2*(A1+1) - A1*(A1+1)*sys['cos_psi']) / (A1 * v21)

    xi_range = np.linspace(-0.3, 2.5, 1000)
    f_vals = np.array([first_return_B1(xi, sys) for xi in xi_range])
    valid = np.isfinite(f_vals) & (np.abs(f_vals) < 5)
    mask_cross = xi_range < xi_part
    mask_self = xi_range >= xi_part

    ax.plot(xi_range[mask_cross & valid], f_vals[mask_cross & valid],
            color=style.COLORS['red'], lw=2.0)
    ax.plot(xi_range[mask_self & valid], f_vals[mask_self & valid],
            color=style.COLORS['blue'], lw=2.0)
    ax.plot([-1, 3], [-1, 3], 'k-', lw=0.4, alpha=0.3)
    ax.axvline(xi_part, color='gray', ls='--', lw=0.5, alpha=0.4)

    xi_sim = 1.3
    cx, cy = [xi_sim], [0]
    for _ in range(50):
        try:
            xn = first_return_B1(xi_sim, sys)
            if not np.isfinite(xn) or abs(xn) > 10: break
            cx.extend([xi_sim, xn]); cy.extend([xn, xn])
            xi_sim = xn
        except: break
    for i in range(len(cx)-1):
        t = min(i/30, 1.0)
        ax.plot([cx[i], cx[i+1]], [cy[i], cy[i+1]],
                color=style.COLORS['green'], lw=0.6, alpha=0.15+0.7*t)

    word_str = ''.join(str(w+1) for w in word)
    if len(word_str) > 12: word_str = word_str[:12] + '...'
    frac = Fraction(rho).limit_denominator(20)
    ax.set_title(f'$\\alpha_2={a2:.2f}$,  $\\rho={frac}$,  $\\sigma={word_str}$', fontsize=11)
    ax.set_xlim(-0.25, 2); ax.set_ylim(-0.25, 2)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_aspect('equal')
    if idx // 4 == 1: ax.set_xlabel('$\\xi_n$', fontsize=16)
    if idx % 4 == 0: ax.set_ylabel('$F(\\xi_n)$', fontsize=16)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
fig.savefig('../figures/cobweb_gallery.pdf', dpi=300, bbox_inches='tight')
print("Saved figures/cobweb_gallery.pdf")
