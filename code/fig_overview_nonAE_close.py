"""Figure 2 variant (close-D_1): non-autoencoder K=N=2, Convention B, D_1
only barely offset from E_1 (29.32° -> 27°, ~2.3°). Same E, and D_2 still
at 62° (9° off E_2=71.32°). Renders to a separate PDF pair.

Bound normal is the ENCODER e_i (V_i = e_i^T (x - y), so B_i = {y: e_i^T y = alpha_i}).
Post-spike readout jump is +D_i (decoder writes to the readout).
E and D play different roles: E defines the bound geometry; D defines the
readout write. Autoencoder is the special case E = D.

Runs a local Convention B step/orbit implementation so scn.py's autoencoder
step is not touched.
"""

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scn import (make_2d_system, step, eigenvector_orbit, first_return_B1,
                 get_firing_ratios, dist_to_limit_cycle, simulate, find_period)
import style

style.setup()

# ═════════════════════════════════════════════════════════════════════
# EDIT THIS BLOCK to tweak the figure — everything below is derived.
# ═════════════════════════════════════════════════════════════════════

# --- Encoder angles (degrees from +x axis). Under Convention B these define
#     the bound normals: B_i = {y : e_i^T y = alpha_i}. Kept at the same
#     directions as the previous figures (Convention A) so the bound
#     geometry is unchanged. ---
E_ANG_1 = 29.32     # ~ 0.512 rad (was the decoder angle under Convention A)
E_ANG_2 = 71.32     # ~ 1.245 rad

# --- Decoder angles (degrees from +x axis). Under Convention B these are the
#     post-spike jump directions (y -> y + D_i). Free (non-autoencoder if D_i
#     != e_i). Kept close-but-not-equal to E: D_1 well to the right of E_1
#     (29.32° -> 15°), D_2 a bit to the right of E_2 (71.32° -> 62°). ---
D_ANG_1 = 27.0     # only ~2.3° off E_1 = 29.32°
D_ANG_2 = 62.0     # ~9.3° off E_2 = 71.32°

# --- Drives ---
A1 = 1.0
A2 = 0.854          # rho = 3/7 plateau under D=(27,62), E=(29.32,71.32).
                    # word = [1, 0, 0, 1, 0, 1, 0]  (period 7).
# Other single-line swaps (Convention B; same D/E geometry):
#   rho = 1/3:  A2 = 0.30,   WORD = [0, 0, 1]
#   rho = 2/5:  A2 = 0.376,  WORD = [1, 0, 0, 1, 0]
#   rho = 4/7:  A2 = 0.560,  WORD = [1, 1, 0, 1, 0, 1, 0]
#   rho = 5/7:  A2 = 0.696,  WORD = [1, 1, 1, 0, 1, 1, 0]

WORD = [1, 0, 0, 1, 0, 1, 0]               # period 7, rho = 3/7

# --- Panel (a)+(b) axis limits + ticks ---
XLIM_A = (-0.05, 1.9)
YLIM_A = (-0.05, 1.9)
XTICKS_A = [0, 0.5, 1.0, 1.5]
YTICKS_A = [0, 0.5, 1.0, 1.5]

# --- Arrow lengths in panel (a). Convention B: E is the bound normal
#     (dashed line), D is the post-spike jump (solid). All four arrows are
#     drawn with the SAME norm so E_i, D_i sit on the unit circle. The
#     bound line lives at alpha_i * e_i and can be elsewhere on the E_i ray.
ARROW_NORM = 1.0

# --- Label positions in panel (a), data coords ---
# Close to each arrow tip, offset slightly perpendicular so text doesn't
# sit on the arrow shaft or any bound line.
LBL_POSITIONS = {
    # All arrows unit-norm. Under D=(27, 62), E_1 and D_1 tips nearly overlap:
    #   E_1 tip = (0.87, 0.49)   [angle 29.32]
    #   D_1 tip = (0.89, 0.45)   [angle 27]        <- basically overlaps E_1
    #   E_2 tip = (0.32, 0.95)   [angle 71.32]
    #   D_2 tip = (0.47, 0.88)   [angle 62]
    # Labels split above/below to separate the near-overlapping E_1 / D_1.
    'E_1': (0.60, 0.68),   # above E_1 arrow
    'E_2': (0.05, 1.15),   # above E_2 tip
    'D_1': (1.30, 0.15),   # far past D_1 tip, low right
    'D_2': (0.30, 1.30),   # above D_2 tip, well above bound_1
}
LBL_FONTSIZE = 14

# --- Spike-arrow style ---
# N_SPIKES_DRAWN: draw only the first N spike arrows (None => draw all L).
N_SPIKES_DRAWN = None    # draw all L arrows for a faithful geometric picture
SPIKE_ALPHA    = 0.35
SPIKE_LW       = 0.9

# --- Rational-lock star in panel (e): shown at (A2, RHO_STAR) ---
RHO_STAR       = 3/7
RHO_STAR_LABEL = r'$\rho=3/7$'

# --- Staircase (panels e, f) simulation budget ---
# Bigger n_transient + n_record => detect longer periods (avoid the rho drop
# at the right edge of the staircase, near alpha_2 = alpha_1 / cos(psi)).
STAIRCASE_N_TRANSIENT = 20000
STAIRCASE_N_RECORD    = 1500       # detects periods up to ~500

# ═════════════════════════════════════════════════════════════════════
# DERIVED CONSTANTS (do not edit)
# ═════════════════════════════════════════════════════════════════════

THETA1 = np.deg2rad(D_ANG_1)
D_THETA2 = np.deg2rad(D_ANG_2)
PSI = D_THETA2 - THETA1
_E1_ANG = np.deg2rad(E_ANG_1)
_E2_ANG = np.deg2rad(E_ANG_2)
E1 = np.array([np.cos(_E1_ANG), np.sin(_E1_ANG)])
E2 = np.array([np.cos(_E2_ANG), np.sin(_E2_ANG)])

D1 = np.array([np.cos(THETA1), np.sin(THETA1)])
D2 = np.array([np.cos(D_THETA2), np.sin(D_THETA2)])

# --- Convention B non-autoencoder K=N=2 system + step + orbit helpers ---
# Bound normal = encoder e_i; post-spike jump = decoder D_i.
def _tangent_2d(e):
    """Orthonormal tangent (2, N-1)=(2, 1) for encoder e (unit)."""
    return np.array([[-e[1]], [e[0]]])

def make_2d_convB(e_list, d_list, alphas):
    """K=N=2 system, Convention B."""
    return {
        'DS': [np.asarray(d, float) for d in d_list],           # decoders (jumps)
        'ES': [np.asarray(e, float) for e in e_list],           # encoders (bound normals)
        'U':  [_tangent_2d(e) for e in e_list],                 # tangent frames of encoders
        'alphas': list(alphas),
        'K': 2, 'N': 2,
        'E_dot_E': np.array([[float(e_list[i] @ e_list[j]) for j in range(2)] for i in range(2)]),
    }

def step_convB(xi, bi, sys_b):
    """One spike-then-decay step under Convention B.
    Bound i: e_i^T y = alpha_i. Position y = alpha_i e_i + U_i xi.
    Post spike: y+ = y + D_bi. Next firer: argmax_j alpha_j/(e_j^T y+)
    over positive denominators."""
    xi = np.atleast_1d(xi).ravel()
    e_i = sys_b['ES'][bi]
    U_i = sys_b['U'][bi]
    a   = sys_b['alphas']
    D   = sys_b['DS']
    yp  = a[bi] * e_i + (U_i @ xi).ravel() + D[bi]
    bj, bc = -1, -1.0
    for j in range(sys_b['K']):
        if a[j] <= 0:
            continue
        dn = sys_b['ES'][j] @ yp
        if dn < 1e-12:
            continue
        c = a[j] / dn
        if 0 < c <= 1.0 + 1e-10 and c > bc:
            bc, bj = c, j
    if bj < 0:
        return None, -1
    xi_next = (sys_b['U'][bj].T @ (bc * yp)).ravel()
    return xi_next, bj

def transition_matrix_convB(j, i, sys_b):
    """M_{j<-i} in homogeneous coords (xi, 1) -> (xi', 1) up to scale."""
    e_i, e_j = sys_b['ES'][i], sys_b['ES'][j]
    U_i, U_j = sys_b['U'][i], sys_b['U'][j]
    D_i = sys_b['DS'][i]
    a_i, a_j = sys_b['alphas'][i], sys_b['alphas'][j]
    R = float(U_j.T @ U_i)           # scalar in K=N=2
    v = float(U_i.T @ e_j)           # e_j in B_i tangent
    b = float(U_j.T @ e_i)           # e_i in B_j tangent
    w = float(U_j.T @ D_i)
    E_dot = float(e_j @ e_i)
    a_ji = float(e_j @ D_i)
    return np.array([[a_j * R,          a_j * (a_i * b + w)],
                     [v,                a_i * E_dot + a_ji]])

def eigenvector_orbit_convB(word, sys_b):
    """Return theoretical orbit (list of (y, bi, xi)), eigenvalues (mu1, mu2),
    and xi_star on B_{word[0]}."""
    L = len(word)
    # Compose return matrix on B_{word[0]}
    M = np.eye(2)
    for k in range(L):
        i, j = word[k], word[(k + 1) % L]
        M = transition_matrix_convB(j, i, sys_b) @ M
    eigvals, eigvecs = np.linalg.eig(M)
    idx = np.argsort(-np.abs(eigvals))
    mu1, mu2 = eigvals[idx[0]], eigvals[idx[1]]
    v1 = np.real(eigvecs[:, idx[0]])
    if abs(v1[1]) < 1e-12:
        raise RuntimeError('degenerate eigenvector')
    xi_star = np.array([v1[0] / v1[1]])
    # Play out the orbit
    orbit = []
    xi_k, bi_k = xi_star.copy(), word[0]
    for k in range(L):
        e_ki = sys_b['ES'][bi_k]
        U_ki = sys_b['U'][bi_k]
        y = sys_b['alphas'][bi_k] * e_ki + (U_ki @ xi_k).ravel()
        orbit.append((y.copy(), bi_k, xi_k.copy()))
        xi_k, bi_k = step_convB(xi_k, bi_k, sys_b)
    return orbit, (mu1, mu2), xi_star

def first_return_B1_convB(xi_scalar, sys_b, max_hops=10):
    """First return to B_1 (bi=0) starting from xi on B_1 under Convention B."""
    xi = np.array([float(xi_scalar)])
    bi = 0
    for _ in range(max_hops):
        xi, bi = step_convB(xi, bi, sys_b)
        if bi < 0:
            return float('nan')
        if bi == 0:
            return float(xi[0])
    return float('nan')

def simulate_convB(sys_b, xi0, bi0, n_transient, n_record):
    xi = np.atleast_1d(xi0).astype(float).copy()
    bi = int(bi0)
    for _ in range(n_transient):
        out = step_convB(xi, bi, sys_b)
        if out[1] < 0:
            return [], []
        xi, bi = out
    xis, bis = [], []
    for _ in range(n_record):
        xis.append(xi.copy())
        bis.append(int(bi))
        out = step_convB(xi, bi, sys_b)
        if out[1] < 0:
            break
        xi, bi = out
    return xis, bis

def get_firing_ratios_convB(sys_b, n_transient=5000, n_record=300, ics=None):
    """Multi-IC sweep. Returns the MAJORITY-VOTE mixed-orbit (ratios, word);
    if no IC finds a mixed orbit, returns the majority-vote trap. Voting on
    mixed rhos is what keeps the devil's staircase curve monotonic across
    parameter regions where multiple mixed orbits coexist (near saddle-node
    bifurcations)."""
    K = sys_b['K']
    if ics is None:
        ics = [(np.zeros(1), 0),           (np.array([0.5]),  1 % K),
               (np.ones(1),  0),           (np.array([1.5]),  0),
               (np.array([-1.0]), 0),      (np.array([-2.0]), 1 % K),
               (np.array([2.0]),  0),      (np.array([3.0]),  1 % K),
               (np.array([-0.5]), 1 % K),  (np.array([0.25]), 0),
               (np.array([1.25]), 1 % K),  (np.array([-1.5]), 0)]
    mixed_hits, trap_hits = {}, {}   # rho_rounded -> (count, word)
    for xi0, bi0 in ics:
        _, bis = simulate_convB(sys_b, xi0, bi0, n_transient, n_record)
        if not bis:
            continue
        word, ratios = None, None
        if len(set(bis)) < min(2, K):
            word = [bis[-1]]
            ratios = [1.0 if bis[-1] == j else 0.0 for j in range(K)]
        else:
            for p in range(1, len(bis) // 3):
                if bis[:p] == bis[p:2*p] == bis[2*p:3*p]:
                    word = bis[:p]
                    ratios = [sum(1 for w in word if w == j) / p for j in range(K)]
                    break
        if word is None:
            continue
        rho2 = ratios[1]
        bucket = mixed_hits if 0.001 < rho2 < 0.999 else trap_hits
        key = round(rho2, 4)
        cur = bucket.get(key, (0, word))
        bucket[key] = (cur[0] + 1, cur[1])
    # Majority vote among mixed; if none, fall back to majority trap.
    src = mixed_hits if mixed_hits else trap_hits
    if not src:
        return [1.0] + [0.0] * (K - 1), [0]
    winner_rho, (_, winner_word) = max(src.items(), key=lambda kv: kv[1][0])
    return [1.0 - winner_rho, winner_rho], winner_word

# Build the Convention-B system used everywhere below.
sys = make_2d_convB([E1, E2], [D1, D2], [A1, A2])
DS, ES = sys['DS'], sys['ES']
ENC = [E1, E2]
# Autoencoder-only autoencoder helpers (from scn.py) are NOT used in this file;
# panel (b) and the staircase use step_convB / eigenvector_orbit_convB.
orbit, ev, xi_star = eigenvector_orbit_convB(WORD, sys)
ext = 3.5

# 2 rows x 3 columns. Cells 5x5 so that panels (a)(b)(c) with
# aspect='equal' fill their cells entirely and their panel labels sit at
# the same figure-x as (d)(e)(f) below (no centering-induced shift).
fig, axes = plt.subplots(2, 3, figsize=(15.0, 10.0))
axes = axes.flatten()

# Consistent panel labels via style.panel_label — bold "(letter)", upper-left
PANEL_LETTERS = list('abcdef')
PANEL_SUBTITLES = ['Geometry', 'Eigenvector orbit', 'Cobweb',
                    'Convergence', r"Devil's staircase", 'Firing counts']
for _ax, _l, _s in zip(axes, PANEL_LETTERS, PANEL_SUBTITLES):
    style.panel_label(_ax, _l, subtitle=_s, fontsize=30, y=1.02)


# ═══ (a) Geometry: converged orbit ═══
ax = axes[0]
# Two arrows per neuron, both from the origin: decoder D_i (solid, longer)
# and encoder e_i (dashed, shorter). All tunable via config block above.
for i, (col, lbl) in enumerate(zip(style.NEURON_COLORS[:2],
                                     [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$'])):
    # Convention B: bound normal is the ENCODER e_i; foot is alpha_i * e_i;
    # tangent direction is U_i (orthonormal to e_i).
    d, e, a = DS[i], ES[i], sys['alphas'][i]
    u = sys['U'][i].ravel()
    foot = a * e
    ax.plot([foot[0] - ext*u[0], foot[0] + ext*u[0]],
            [foot[1] - ext*u[1], foot[1] + ext*u[1]],
            color=col, lw=1.5, label=lbl)
    # Decoder arrow (solid) — post-spike jump direction.
    ax.annotate('', xy=d * ARROW_NORM, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=2.2))
    # Encoder arrow (dashed) — bound normal.
    ax.annotate('', xy=e * ARROW_NORM, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.8,
                                linestyle='--', alpha=0.9))

# Labels at positions from config (LBL_POSITIONS).
for i, col in enumerate(style.NEURON_COLORS[:2]):
    for kind, style_kwargs in [('D', dict()), ('E', dict(style='italic'))]:
        lx, ly = LBL_POSITIONS[f'{kind}_{i+1}']
        ax.text(lx, ly, f'$\\mathbf{{{kind}}}_{i+1}$',
                fontsize=LBL_FONTSIZE, color=col, fontweight='bold',
                ha='center', va='center', **style_kwargs)

xi_k, bi_k = xi_star.copy(), WORD[0]
_spike_dots = []   # (position, color) — plotted last so they sit ON the bound
_n_arrows = len(WORD) if N_SPIKES_DRAWN is None else min(N_SPIKES_DRAWN, len(WORD))
for k in range(len(WORD)):
    # Convention B position: y = alpha_i * e_i + U_i * xi
    y = sys['alphas'][bi_k] * ES[bi_k] + (sys['U'][bi_k] @ xi_k).ravel()
    yp = y + DS[bi_k]                           # post-spike jump (decoder)
    col = style.NEURON_COLORS[bi_k]
    _spike_dots.append((y.copy(), col))
    xi_n, bi_n = step_convB(xi_k, bi_k, sys)
    if k < _n_arrows:
        ax.annotate('', xy=yp, xytext=y,
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=SPIKE_LW, alpha=SPIKE_ALPHA))
        y_next = sys['alphas'][bi_n] * ES[bi_n] + (sys['U'][bi_n] @ xi_n).ravel()
        ax.plot([yp[0], y_next[0]], [yp[1], y_next[1]],
                color='gray', lw=0.5, alpha=0.4, ls='--')
    xi_k, bi_k = xi_n, bi_n

# Spike-origin dots on the bounds — makes the "spike happens on the bound"
# geometry unambiguous when arrow tails and bound line overlap.
for pt, col in _spike_dots:
    ax.plot(pt[0], pt[1], 'o', color=col, ms=6,
            markeredgecolor='k', markeredgewidth=0.6, zorder=12)

ax.plot(0, 0, 'ko', ms=4)
ax.set_xlim(*XLIM_A); ax.set_ylim(*YLIM_A)
ax.set_xticks(XTICKS_A); ax.set_yticks(YTICKS_A)
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_aspect('equal')
leg = ax.legend(fontsize=13, loc='upper right', framealpha=1.0,
                facecolor='white', edgecolor='#bbb', fancybox=False)
leg.set_zorder(50)
for txt, col in zip(leg.get_texts(), style.NEURON_COLORS[:2]):
    txt.set_color(col); txt.set_fontweight('bold')


# ═══ (b) Theory orbit points ═══ (Convention B)
ax = axes[1]
for i, (col, lbl) in enumerate(zip(style.NEURON_COLORS[:2],
                                     [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$'])):
    a = sys['alphas'][i]
    e = ES[i]
    u = sys['U'][i].ravel()
    foot = a * e
    ax.plot([foot[0]-ext*u[0], foot[0]+ext*u[0]],
            [foot[1]-ext*u[1], foot[1]+ext*u[1]], color=col, lw=1.5, label=lbl)

for k in range(len(WORD)):
    y, bi_, _ = orbit[k]
    col = style.NEURON_COLORS[bi_]
    ax.plot(y[0], y[1], 'o', color=col, ms=7,
            markeredgecolor='k', markeredgewidth=0.7, zorder=10)

xi_k, bi_k = xi_star.copy(), WORD[0]
for k in range(len(WORD)):
    y = sys['alphas'][bi_k] * ES[bi_k] + (sys['U'][bi_k] @ xi_k).ravel()
    yp = y + DS[bi_k]
    ax.plot([y[0], yp[0]], [y[1], yp[1]], color=style.NEURON_COLORS[bi_k], lw=0.7, alpha=0.5)
    xi_n, bi_n = step_convB(xi_k, bi_k, sys)
    y_next = sys['alphas'][bi_n] * ES[bi_n] + (sys['U'][bi_n] @ xi_n).ravel()
    ax.plot([yp[0], y_next[0]], [yp[1], y_next[1]], color='gray', lw=0.4, alpha=0.3, ls='--')
    xi_k, bi_k = xi_n, bi_n

ax.plot(0, 0, 'ko', ms=4)
ax.set_xlim(*XLIM_A); ax.set_ylim(*YLIM_A)
ax.set_xticks(XTICKS_A); ax.set_yticks(YTICKS_A)
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_aspect('equal')
leg = ax.legend(fontsize=13, loc='upper right', framealpha=1.0,
                facecolor='white', edgecolor='#bbb', fancybox=False)
leg.set_zorder(50)
for txt, col in zip(leg.get_texts(), style.NEURON_COLORS[:2]):
    txt.set_color(col); txt.set_fontweight('bold')


# ═══ (c) Cobweb ═══ (Convention B first-return map on B_1)
ax = axes[2]
# Sweep xi and compute F(xi); split into "self" (first return is neuron 1) vs
# "cross" (first return is via neuron 2) by tracking the first-hop bound.
xi_range = np.linspace(-0.5, 3.0, 4000)
f_vals = np.zeros_like(xi_range) + np.nan
first_hop = np.zeros_like(xi_range, dtype=int)     # 0 = self, 1 = cross
for k, xi in enumerate(xi_range):
    xi_arr, bj = step_convB(np.array([xi]), 0, sys)
    if bj < 0:
        continue
    first_hop[k] = bj
    if bj == 0:                                    # self
        f_vals[k] = float(xi_arr[0])
    else:                                          # cross via B_2, then return
        xi_next, bj2 = step_convB(xi_arr, bj, sys)
        if bj2 == 0:
            f_vals[k] = float(xi_next[0])

_self = (first_hop == 0) & np.isfinite(f_vals)
_cross = (first_hop == 1) & np.isfinite(f_vals)
ax.plot(xi_range[_cross], f_vals[_cross], color=style.COLORS['green'], lw=2.0,
        label=r'Cross ($\mathbf{M}_{2\leftarrow 1}$)')
ax.plot(xi_range[_self], f_vals[_self], color=style.COLORS['dark'], lw=2.0,
        label=r'Self ($\mathbf{M}_{1\leftarrow 1}$)')
# Partition point: where first_hop flips from 0 to 1 (or vice versa).
_flip = np.where(np.diff(first_hop) != 0)[0]
xi_part = xi_range[_flip[0]] if len(_flip) else np.nan
ax.plot([-1, 3], [-1, 3], 'k-', lw=0.5, alpha=0.3)
if np.isfinite(xi_part):
    ax.axvline(xi_part, color='gray', ls='--', lw=0.8, alpha=0.4)

# Cobweb trajectory from a starting xi that lies inside the physical orbit range.
_finite_xi = xi_range[np.isfinite(f_vals)]
xi_sim = float(np.median(_finite_xi))
cx, cy = [xi_sim], [0]
for _ in range(50):
    xi_arr, bj = step_convB(np.array([xi_sim]), 0, sys)
    if bj < 0:
        break
    if bj == 0:
        xn = float(xi_arr[0])
    else:
        xi_next, bj2 = step_convB(xi_arr, bj, sys)
        if bj2 != 0:
            break
        xn = float(xi_next[0])
    if abs(xn) > 10:
        break
    cx.extend([xi_sim, xn]); cy.extend([xn, xn])
    xi_sim = xn
for i in range(len(cx)-1):
    t = min(i/30, 1.0)
    ax.plot([cx[i], cx[i+1]], [cy[i], cy[i+1]],
            color='#B78B27', lw=0.7, alpha=0.15+0.7*t)

from matplotlib.lines import Line2D as _L2D
cobweb_handle = _L2D([0], [0], color='#B78B27', lw=1.2, label='Cobweb')

ax.set_xlabel('$\\xi_n$'); ax.set_ylabel('$F(\\xi_n)$')
# Zoom in on the physical orbit range on B_1: fit to the xi values the
# limit cycle actually visits, plus a small margin (not to the full
# domain where F is defined).
_xi_orb = np.array([orbit[k][2][0] for k in range(len(WORD)) if orbit[k][1] == 0])
if len(_xi_orb):
    _margin = 0.10
    _lo = float(_xi_orb.min()) - _margin
    _hi = float(_xi_orb.max()) + _margin
else:
    _xi_defined = xi_range[np.isfinite(f_vals)]
    _lo, _hi = _xi_defined.min() - 0.10, _xi_defined.max() + 0.10
ax.set_xlim(_lo, _hi); ax.set_ylim(_lo, _hi)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
ax.set_aspect('equal')
h, l = ax.get_legend_handles_labels()
h.append(cobweb_handle); l.append('Cobweb')
leg = ax.legend(h, l, fontsize=13, loc='lower right',
                bbox_to_anchor=(1.03, 0.0), borderaxespad=0.0,
                framealpha=1.0, facecolor='white', edgecolor='#bbb',
                fancybox=False)
leg.set_zorder(100)   # sit on top of the cobweb
legend_colors = [style.COLORS['green'], style.COLORS['dark'], '#B78B27']
for txt, col in zip(leg.get_texts(), legend_colors):
    txt.set_color(col); txt.set_fontweight('bold')


# ═══ (d) Convergence to limit cycle ═══
ax = axes[3]
p = len(WORD)
ratio = abs(ev[1] / ev[0])
n_steps = 80
rng = np.random.RandomState(42)
converging = []
for trial in range(300):
    xi_sim = np.array([rng.uniform(-1.5, 1.5)])
    bi_sim = rng.randint(2)
    dists = []
    for s in range(n_steps):
        dists.append(dist_to_limit_cycle(xi_sim, bi_sim, orbit))
        _out = step_convB(xi_sim, bi_sim, sys)
        if _out[1] < 0:
            break
        xi_sim, bi_sim = _out
    if np.isfinite(dists[-1]) and dists[-1] < 1e-6:
        converging.append(dists)
converging = np.array(converging)
steps = np.arange(n_steps)
ax.fill_between(steps, np.percentile(converging, 5, axis=0),
                np.percentile(converging, 95, axis=0),
                color=style.COLORS['accent'], alpha=0.12, label='5-95th pctl.')
ax.fill_between(steps, np.percentile(converging, 25, axis=0),
                np.percentile(converging, 75, axis=0),
                color=style.COLORS['accent'], alpha=0.25, label='25-75th pctl.')
# Median (robust to log-space skew of positive iid perturbations; arithmetic
# mean lies above the geometric mean by Var(log C)/2 and bends the curve).
med_dist = np.median(converging, axis=0)
ax.plot(steps, med_dist, color=style.COLORS['accent'],
        lw=1.5, label='Median')
# Theoretical slope is asymptotic; anchor the dashed line to the median at
# the start of the linear-regime window (after 2 cycles of nonlinear
# alignment onto u_2) so slopes — not the transient amplitude — are compared.
anchor = 2 * p
ax.plot(steps, med_dist[anchor] * ratio**((steps - anchor)/p),
        'k--', lw=1.2, alpha=0.7,
        label=r'$|\mu_2/\mu_1|^{n/L}$')
ax.set_yscale('log')
ax.set_xlabel('Step $n$'); ax.set_ylabel('Dist. to limit cycle')
ax.set_xlim(0, n_steps); ax.set_ylim(1e-16, 1e1)
ax.legend(fontsize=13, loc='lower left', framealpha=0.85)


# ═══ (e) Devil's staircase ═══ (needs a2_range + rho_vals for panel f too)
# Staircase computation is the expensive step (~5 min at 750 alpha_2 samples).
# Cache to code/.cache_staircase.npz and reload if params match.
# Force recompute with `python fig_overview.py --recompute` or env SCN_RECOMPUTE=1.
ax = axes[4]
_code_dir = os.path.dirname(os.path.abspath(__file__))
_cache_path = os.path.join(_code_dir, '.cache_staircase_nonAE_convB_close.npz')
_force_recompute = ('--recompute' in _sys.argv) or (os.environ.get('SCN_RECOMPUTE') == '1')

# Convention B (post-swap, D close to E) staircase. Mixed regime spans
# A2 ~= [0.60, 1.22] with mixed rho ~0.14 -> ~0.86. Sweep just a small
# margin outside so we see the trap-driven rho=0 / rho=1 flanks; the
# plotting logic extends the curve to touch 0 on the left and 1 on the
# right. Force A2 exactly into the grid so the example star sits on the
# curve.
_a2_range_target = np.linspace(0.70, 1.20, 500)
_a2_range_target = np.sort(np.unique(np.concatenate([_a2_range_target, [A2]])))
_cache_hit = False
if not _force_recompute and os.path.exists(_cache_path):
    _d = np.load(_cache_path)
    _geom_ok = (float(_d.get('D_ANG_1', -1)) == D_ANG_1
                and float(_d.get('D_ANG_2', -1)) == D_ANG_2
                and float(_d.get('E_ANG_1', -1)) == E_ANG_1
                and float(_d.get('E_ANG_2', -1)) == E_ANG_2)
    if (_geom_ok and float(_d['A1']) == A1
            and _d['a2_range'].shape == _a2_range_target.shape
            and np.allclose(_d['a2_range'], _a2_range_target)):
        a2_range = _d['a2_range']
        rho_vals = _d['rho_vals']
        period_vals = _d['period_vals']
        _cache_hit = True
        print(f"Loaded staircase cache: {_cache_path}")
    else:
        print("Cache present but params differ; recomputing.")

if not _cache_hit:
    print("Computing staircase (~5 min at 750 samples; cache -> .cache_staircase.npz)...")
    a2_range = _a2_range_target
    rho_vals = []
    period_vals = []
    for a2_ in a2_range:
        sys_tmp = make_2d_convB([E1, E2], [D1, D2], [A1, a2_])
        ratios, word = get_firing_ratios_convB(
            sys_tmp, n_transient=STAIRCASE_N_TRANSIENT, n_record=STAIRCASE_N_RECORD)
        rho_vals.append(ratios[1])
        period_vals.append(len(word))
    rho_vals = np.array(rho_vals)
    period_vals = np.array(period_vals)
    np.savez(_cache_path, THETA1=THETA1, PSI=PSI, A1=A1,
             D_ANG_1=D_ANG_1, D_ANG_2=D_ANG_2,
             E_ANG_1=E_ANG_1, E_ANG_2=E_ANG_2,
             a2_range=a2_range, rho_vals=rho_vals, period_vals=period_vals)
    print(f"Saved staircase cache: {_cache_path}")

# Interpolate spurious transient-detection failures (short zero runs surrounded by non-zero)
i = 0
while i < len(rho_vals):
    if rho_vals[i] < 0.001:
        j = i
        while j < len(rho_vals) and rho_vals[j] < 0.001:
            j += 1
        left = rho_vals[i-1] if i > 0 else 0
        right = rho_vals[j] if j < len(rho_vals) else 0
        if left > 0.15 and right > 0.15:
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                rho_vals[k] = left * (1 - t) + right * t
        i = j
    else:
        i += 1

# Convention B: multi-IC pass to (a) fill in tail samples where the single-IC
# transient failed to lock, and (b) surface coexisting attractors (bistability).
# We recompute the primary rho with a big budget from many ICs, then also
# record any secondary attractor found by >= 2 ICs.
from collections import Counter

def _primary_rho_convB(sys_i, n_ics=25, n_transient=60000, n_record=3000,
                       rng_seed=42):
    """Primary rho (prefer mixed orbit rho in (0, 1); else majority-IC) and
    dict of all coexisting rho -> #ICs. Trap attractors (rho=0, rho=1)
    coexist everywhere under Convention B post-swap, so they aren't the
    "primary" branch — the mixed orbit is."""
    rng = np.random.default_rng(rng_seed)
    outcomes = []
    for k in range(n_ics):
        xi0 = rng.uniform(-3.0, 3.0, sys_i['N'] - 1)
        bi0 = k % sys_i['K']
        try:
            _, bis = simulate_convB(sys_i, xi0, bi0, n_transient, n_record)
        except Exception:
            continue
        if not bis:
            continue
        if len(set(bis)) < 2:
            outcomes.append(1.0 if bis[-1] == 1 else 0.0)
            continue
        p = find_period(bis)
        if p < 1 or p > n_record // 2:
            continue
        word = bis[:p]
        outcomes.append(sum(1 for w in word if w == 1) / p)
    if not outcomes:
        return None, {}
    rounded = [round(r, 4) for r in outcomes]
    counts = Counter(rounded)
    mixed = {r: n for r, n in counts.items() if 0.001 < r < 0.999}
    if mixed:
        primary = max(mixed, key=lambda r: mixed[r])
    else:
        primary = counts.most_common(1)[0][0]
    return primary, dict(counts)


# Multi-IC sweep to fix trap-fallback samples + find coexistence. Cached.
# Set SCN_SKIP_TAIL=1 to skip the multi-IC pass (for a quick figure sanity
# check; the main sweep still yields the primary curve, minus coexistence
# points).
_tail_cache_path = os.path.join(_code_dir, '.cache_tail_nonAE_convB_close.npz')
_a2_tail_grid = np.linspace(a2_range.min(), a2_range.max(), 60)
_tail_a2, _tail_rho, _bistable_pts = [], [], []
_skip_tail = os.environ.get('SCN_SKIP_TAIL') == '1'
if not _force_recompute and os.path.exists(_tail_cache_path):
    _c = np.load(_tail_cache_path, allow_pickle=True)
    if (float(_c['A1']) == A1 and _c['tail_a2_grid'].shape == _a2_tail_grid.shape
            and np.allclose(_c['tail_a2_grid'], _a2_tail_grid)
            and float(_c.get('E_ANG_1', -1)) == E_ANG_1
            and float(_c.get('E_ANG_2', -1)) == E_ANG_2
            and float(_c.get('D_ANG_1', -1)) == D_ANG_1
            and float(_c.get('D_ANG_2', -1)) == D_ANG_2):
        _tail_a2 = list(_c['tail_a2'])
        _tail_rho = list(_c['tail_rho'])
        _bistable_pts = [tuple(p) for p in _c['bistable_pts']]
        print(f"Loaded tail cache: {_tail_cache_path}")
if not _tail_a2 and _skip_tail:
    print("SCN_SKIP_TAIL=1: skipping multi-IC tail pass.")
if not _tail_a2 and not _skip_tail:
    print("Computing multi-IC tail (~10 min for 60 samples x 25 ICs)...")
    for _a2 in _a2_tail_grid:
        sys_i = make_2d_convB([E1, E2], [D1, D2], [A1, _a2])
        prho, counts = _primary_rho_convB(sys_i)
        if prho is None:
            continue
        _tail_a2.append(_a2)
        _tail_rho.append(prho)
        for r, n in counts.items():
            if r != prho and n >= 2:
                _bistable_pts.append((_a2, r))
    np.savez(_tail_cache_path, A1=A1, E_ANG_1=E_ANG_1, E_ANG_2=E_ANG_2,
             D_ANG_1=D_ANG_1, D_ANG_2=D_ANG_2,
             tail_a2_grid=_a2_tail_grid,
             tail_a2=np.asarray(_tail_a2), tail_rho=np.asarray(_tail_rho),
             bistable_pts=(np.asarray(_bistable_pts) if _bistable_pts
                            else np.empty((0, 2))))
    print(f"Saved tail cache: {_tail_cache_path}")

# Mask trap fallback samples (rho == 0 or 1); combine with tail multi-IC.
rho_plot = rho_vals.copy().astype(float)
_trap = (rho_plot < 0.001) | (rho_plot > 0.999)
rho_plot[_trap] = np.nan
_a2_all = np.concatenate([a2_range, np.asarray(_tail_a2)])
_rho_all = np.concatenate([rho_plot, np.asarray(_tail_rho)])
_order = np.argsort(_a2_all)
_a2_all, _rho_all = _a2_all[_order], _rho_all[_order]

# Two-pass cleanup so the plotted curve, the plateau markers, and the
# example star all agree:
#   (1) Median filter (window 3) on the mixed-orbit rhos to squash single-
#       sample outliers from majority-vote ties near saddle-node folds.
#   (2) Running max on the filtered signal to enforce the monotonic
#       rotation number expected by the devil's staircase.
def _median3(x):
    x = x.astype(float)
    y = x.copy()
    for i in range(1, len(x) - 1):
        w = [v for v in (x[i - 1], x[i], x[i + 1]) if np.isfinite(v)]
        if w:
            y[i] = np.median(w)
    return y

_rho_filt = _median3(_rho_all)
_mono = np.full_like(_rho_filt, np.nan)
_run = -np.inf
for _i, _r in enumerate(_rho_filt):
    if np.isfinite(_r):
        _run = max(_run, float(_r))
        _mono[_i] = _run

# Extend the curve so it spans firing ratio 0 -> 1: fill the left flank
# with rho=0 (neuron-1 trap dominates) and the right flank with rho=1
# (neuron-2 trap dominates). Result is a single connected devil's-
# staircase curve from (a2_min, 0) through the mixed staircase to
# (a2_max, 1), with saddle-node jumps at the mixed-regime boundaries.
_mono_ext = _mono.copy()
_finite = np.where(np.isfinite(_mono_ext))[0]
if len(_finite) > 0:
    _first, _last = _finite[0], _finite[-1]
    _mono_ext[:_first] = 0.0
    _mono_ext[_last + 1:] = 1.0

ax.plot(_a2_all, _mono_ext, color=style.COLORS['accent'], lw=1.2,
        label=r'$\rho(\alpha_2)$')

if _bistable_pts:
    _xb, _yb = zip(*_bistable_pts)
    ax.plot(_xb, _yb, 'o', ms=4, color=style.COLORS['orange'],
            markeredgecolor='k', markeredgewidth=0.4, alpha=0.85, zorder=6,
            label='Other stable orbit')

# Rational-lock plateau markers, computed from the FILTERED monotonic curve
# so they sit exactly on the plotted line (not on raw noisy samples).
plateaus = {}
for a2_, rho in zip(_a2_all, _mono):
    if not np.isfinite(rho): continue
    if not (0.001 < rho < 0.999): continue
    frac = Fraction(float(rho)).limit_denominator(30); key = str(frac)
    if key not in plateaus: plateaus[key] = {'rho': float(rho), 'vals': []}
    plateaus[key]['vals'].append(float(a2_))
plateau_label_added = False
for info in plateaus.values():
    ax.plot(np.mean(info['vals']), info['rho'], 'o',
            color=style.COLORS['purple'], ms=2, alpha=0.6, zorder=5,
            label='Rational locks' if not plateau_label_added else None)
    plateau_label_added = True

ax.plot(A2, RHO_STAR, '*', color=style.COLORS['red'], ms=12,
        markeredgecolor='k', markeredgewidth=0.5, zorder=10,
        label='Example ' + RHO_STAR_LABEL)
ax.set_xlabel(r'Drive $\alpha_2$'); ax.set_ylabel(r'Firing ratio $\rho$')
ax.set_xlim(0.70, 1.20); ax.set_ylim(-0.02, 1.02)
ax.set_xticks([0.7, 0.9, 1.1]); ax.set_yticks([0, 0.5, 1.0])
leg = ax.legend(fontsize=13, loc='lower right', framealpha=0.85)
_legend_colors = [style.COLORS['accent'], style.COLORS['purple'],
                    style.COLORS['red']]
for txt, col in zip(leg.get_texts(), _legend_colors):
    txt.set_color(col); txt.set_fontweight('bold')


# ═══ (f) Firing counts per period vs alpha_2 ═══
# For each alpha_2 in the sweep, we already recorded (rho, period).
# n_2 = rho * period   (spikes of neuron 2 per return period)
# n_1 = (1 - rho) * period
#
# Detection-failure interpolation: when find_period doesn't lock within the
# transient window it returns n_record=300, producing spurious spikes to
# ~300 in period_vals at scattered alpha_2. These sit outside the physical
# range of true rational locks (all L≲40 in this sweep); we replace runs of
# such outliers with a linear interpolant so panel (f) matches the HTML
# reference (media/check/devil_slider.html; L range [0, 42.9]).
ax = axes[5]
period_clean = period_vals.astype(float).copy()
OUTLIER_CAP = 50
i = 0
while i < len(period_clean):
    if period_clean[i] > OUTLIER_CAP:
        j = i
        while j < len(period_clean) and period_clean[j] > OUTLIER_CAP:
            j += 1
        left = period_clean[i-1] if i > 0 else np.nan
        right = period_clean[j] if j < len(period_clean) else np.nan
        if np.isfinite(left) and np.isfinite(right):
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                period_clean[k] = left * (1 - t) + right * t
        else:
            period_clean[i:j] = np.nan
        i = j
    else:
        i += 1

n2_vals = rho_vals * period_clean
n1_vals = (1 - rho_vals) * period_clean

# Total period L (matches the HTML reference's "period L" curve, max ~ 40).
ax.plot(a2_range, period_clean, color='#333333', lw=1.7,
        label=r'Period $L=n_1+n_2$')
ax.plot(a2_range, n1_vals, color=style.NEURON_COLORS[0], lw=1.3, alpha=0.85,
        label=r'Neuron 1: $n_1$')
ax.plot(a2_range, n2_vals, color=style.NEURON_COLORS[1], lw=1.3, alpha=0.85,
        label=r'Neuron 2: $n_2$')
ax.axvline(A2, color=style.COLORS['red'], ls=':', lw=1.0, alpha=0.7)
ax.set_xlabel(r'Drive $\alpha_2$')
ax.set_ylabel('Spikes per return period')
ax.set_xlim(0.70, 1.20)
ax.set_ylim(0, 15)
ax.set_xticks([0.7, 0.9, 1.1])
ax.set_yticks([0, 5, 10, 15])
leg = ax.legend(fontsize=13, loc='upper left', framealpha=0.85)
_period_col = ['#333333'] + list(style.NEURON_COLORS[:2])
for txt, col in zip(leg.get_texts(), _period_col):
    txt.set_color(col); txt.set_fontweight('bold')


fig.tight_layout(w_pad=1.2, h_pad=1.2, pad=0.1)

_here = os.path.dirname(os.path.abspath(__file__))

# POSTER version (panel labels at fontsize=30, already set)
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'scn_word_example_nonAE_close.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out}")

# PAPER version: shrink panel labels to 18pt so they match Fig 4's other
# subplots (fig_inverse_scaling, fig_margin_landscape_paper).
for _ax, _l, _s in zip(axes, PANEL_LETTERS, PANEL_SUBTITLES):
    style.panel_label(_ax, _l, subtitle=_s, fontsize=18, y=1.02)
_out_paper = os.path.normpath(os.path.join(_here, '..', 'figures', 'scn_word_example_nonAE_close_paper.pdf'))
fig.savefig(_out_paper, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out_paper}")
