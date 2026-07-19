"""Figure 2: Overview of SCN dynamics (K=N=2), 6 panels on 2 rows.

Panels:
  (a) Geometry (converged orbit)
  (b) Theoretical orbit (eigenvector positions)
  (c) Cobweb of the first-return map F: B1 -> B1
  (d) Convergence to the limit cycle
  (e) Devil's staircase: firing ratio rho vs alpha_2
  (f) Firing counts per period vs alpha_2 (rate proxy)
"""

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scn import (make_2d_system, step, eigenvector_orbit, first_return_B1,
                 get_firing_ratios, dist_to_limit_cycle, simulate, find_period,
                 simulate_euler)
import style

style.setup()

# ── Parameters ──
# Matches interactive reference media/check/devil_slider.html on alpha_1=1
# (first-frame isolated-D1 ISI T = ln(1+1/alpha_1) = ln 2 = 0.693 tau) and
# alpha_2 sweep window [0.25, 1.0]. THETA1 is the decoder-1 orientation
# angle; it rotates panels (a)-(b) rigidly and has NO effect on the
# staircase rho(alpha_2), which depends only on psi and alpha_1.
PSI = 0.4 * np.pi
THETA1 = 0.25
A1, A2 = 1.0, 0.49
WORD = [0, 0, 1, 0, 0, 0, 1]

sys = make_2d_system(THETA1, PSI, A1, A2)
DS, ES = sys['DS'], sys['ES']
orbit, ev, xi_star = eigenvector_orbit(WORD, sys)
ext = 3.5

# 2 rows x 3 columns. Cells 5x5 so that panels (a)(b)(c) with
# aspect='equal' fill their cells entirely and their panel labels sit at
# the same figure-x as (d)(e)(f) below (no centering-induced shift).
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0), gridspec_kw={'width_ratios': [1.5, 1, 1]})

# Consistent panel labels via style.panel_label — bold "(letter)", upper-left
PANEL_LETTERS = list('abc')
for _ax, _l in zip(axes, PANEL_LETTERS):
    style.panel_label(_ax, _l)

# ═══ (a) Dynamics: simulate two neurons ═══
N, M, K = 2, 2, 2
D = np.zeros((K, N))
E = np.zeros((N, K))
T = np.zeros((N,))
for i, (col, lbl) in enumerate(zip(style.NEURON_COLORS[:2],
                                     [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$'])):
    d, a = DS[i], sys['alphas'][i]
    D[:, i] = d
    E[i, :] = -d
    T[i] = -a
F = D.T

Tend = 0.04
dt = 1e-4
time = np.arange(0, Tend, dt)
nT = len(time)
x = np.zeros((M, nT))
input = {'x': x, 'Tend': Tend, 'time': time}
init = {'y0': np.array([1.2, 0.6])}

results = simulate_euler(input, D, E, F, T, dt=dt, init=init)
V = results['V']
spk_times = results['spk_times']
y = results['y']

for i in range(N):
    axes[0].plot(time, V[i, :], color=style.NEURON_COLORS[i])
    axes[0].plot(spk_times[i], 1.075*np.ones_like(spk_times[i]), '|', color=style.NEURON_COLORS[i], markersize=10,
                 linewidth=15)
axes[0].axhline(y=1, color='gray', linestyle='--', linewidth=0.5)

axes[0].set_xticks([0, Tend/2, Tend])
axes[0].set_xticklabels([0, Tend/2, Tend], fontsize=13)
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels([0, 1], fontsize=13)
axes[0].set_ylabel('$V$', fontsize=15)
axes[0].set_xlabel('time (s)', fontsize=15)
axes[0].set_xlim([0, Tend])
axes[0].set_ylim([0, 1.1])

axes[1].plot(V[0, :], V[1, :], color='black')
axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
axes[1].axvline(x=1, color='gray', linestyle='--', linewidth=0.5)

axes[1].set_xticks([0, 0.5, 1])
axes[1].set_xticklabels([0, 0.5, 1], fontsize=13)
axes[1].set_yticks([0, 0.5, 1])
axes[1].set_yticklabels([0, 0.5, 1], fontsize=13)
axes[1].set_ylabel('$V_2$', fontsize=15)
axes[1].set_xlabel('$V_1$', fontsize=15)
axes[1].set_xlim([0, 1.1])
axes[1].set_ylim([0, 1.1])

axes[2].plot(y[0, :], y[1, :], color='black', linewidth=1.0)

# ═══ (a) Geometry: converged orbit ═══
ax = axes[2]
for i, (col, lbl) in enumerate(zip(style.NEURON_COLORS[:2],
                                     [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$'])):
    d, e, a = DS[i], ES[i].ravel(), sys['alphas'][i]
    foot = a * d
    ax.plot([foot[0] - ext*e[0], foot[0] + ext*e[0]],
            [foot[1] - ext*e[1], foot[1] + ext*e[1]], color=col, lw=1.5, label=lbl)
    ax.annotate('', xy=foot + 0.35*d, xytext=foot,
                arrowprops=dict(arrowstyle='->', color=col, lw=2.0))
    off = 0.12 * e.ravel()
    _tx = foot[0]+0.4*d[0]+off[0]
    _ty = 0.3 if i == 0 else foot[1]+0.4*d[1]+off[1]
    ax.text(_tx, _ty, f'$\\mathbf{{D}}_{i+1}$',
            fontsize=15, color=col, fontweight='bold')

xi_k, bi_k = xi_star.copy(), WORD[0]
_spike_dots = []   # (position, color) — plotted last so they sit ON the bound
for k in range(len(WORD)):
    y = sys['alphas'][bi_k]*DS[bi_k] + ES[bi_k] @ xi_k
    yp = y + DS[bi_k]
    col = style.NEURON_COLORS[bi_k]
    ax.annotate('', xy=yp, xytext=y,
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2, alpha=0.9))
    _spike_dots.append((y.copy(), col))
    xi_n, bi_n = step(xi_k, bi_k, sys)
    y_next = sys['alphas'][bi_n]*DS[bi_n] + ES[bi_n] @ xi_n
    ax.plot([yp[0], y_next[0]], [yp[1], y_next[1]],
            color='gray', lw=0.5, alpha=0.4, ls='--')
    xi_k, bi_k = xi_n, bi_n

# Spike-origin dots on the bounds — makes the "spike happens on the bound"
# geometry unambiguous when arrow tails and bound line overlap.
for pt, col in _spike_dots:
    ax.plot(pt[0], pt[1], 'o', color=col, ms=6,
            markeredgecolor='k', markeredgewidth=0.6, zorder=12)

ax.plot(0, 0, 'ko', ms=4)
ax.set_xlim(-0.25, 2); ax.set_ylim(-0.25, 2)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_aspect('equal')
leg = ax.legend(fontsize=13, loc='upper left', framealpha=0.85)
for txt, col in zip(leg.get_texts(), style.NEURON_COLORS[:2]):
    txt.set_color(col); txt.set_fontweight('bold')

fig.tight_layout(w_pad=1.2, h_pad=1.2, pad=0.1)
plt.show()

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'two_neuron_voltages.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out}")
