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
_here = os.path.dirname(os.path.abspath(__file__))
_sys.path.insert(0, os.path.join(_here, "..", "code"))

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scn import (make_2d_system, step, eigenvector_orbit, first_return_B1,
                 get_firing_ratios, dist_to_limit_cycle, simulate, find_period)
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
fig, axes = plt.subplots(1, 6, figsize=(30.0, 5.0))
axes = axes.flatten()

# Consistent panel labels via style.panel_label — bold "(letter)" + subtitle,
# upper-left. Subtitles are only used in the poster flat variant.
PANEL_LETTERS = list('abcdef')
PANEL_SUBTITLES = ['Geometry', 'Eigenvector orbit', 'Cobweb',
                    'Convergence', r"Devil's staircase", 'Firing counts']
for _ax, _l, _s in zip(axes, PANEL_LETTERS, PANEL_SUBTITLES):
    # fontsize=30 chosen so this figure (2160pt wide) matches derivation_tikz's
    # \Large\bfseries (14.4pt / 1011pt \approx 1.4% of native fig width) once
    # both are scaled to a common \linewidth on the poster.
    style.panel_label(_ax, _l, subtitle=_s, fontsize=30, y=1.02)


# ═══ (a) Geometry: converged orbit ═══
ax = axes[0]
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


# ═══ (b) Theory orbit points ═══
ax = axes[1]
for i, (col, lbl) in enumerate(zip(style.NEURON_COLORS[:2],
                                     [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$'])):
    d, e, a = DS[i], ES[i].ravel(), sys['alphas'][i]
    foot = a * d
    ax.plot([foot[0]-ext*e[0], foot[0]+ext*e[0]],
            [foot[1]-ext*e[1], foot[1]+ext*e[1]], color=col, lw=1.5, label=lbl)

for k in range(len(WORD)):
    y, bi_, _ = orbit[k]
    col = style.NEURON_COLORS[bi_]
    ax.plot(y[0], y[1], 'o', color=col, ms=7,
            markeredgecolor='k', markeredgewidth=0.7, zorder=10)

xi_k, bi_k = xi_star.copy(), WORD[0]
for k in range(len(WORD)):
    y = sys['alphas'][bi_k]*DS[bi_k] + ES[bi_k] @ xi_k
    yp = y + DS[bi_k]
    ax.plot([y[0], yp[0]], [y[1], yp[1]], color=style.NEURON_COLORS[bi_k], lw=0.7, alpha=0.5)
    xi_n, bi_n = step(xi_k, bi_k, sys)
    y_next = sys['alphas'][bi_n]*DS[bi_n] + ES[bi_n] @ xi_n
    ax.plot([yp[0], y_next[0]], [yp[1], y_next[1]], color='gray', lw=0.4, alpha=0.3, ls='--')
    xi_k, bi_k = xi_n, bi_n

ax.plot(0, 0, 'ko', ms=4)
ax.set_xlim(-0.25, 2); ax.set_ylim(-0.25, 2)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_aspect('equal')
leg = ax.legend(fontsize=13, loc='upper left', framealpha=0.85)
for txt, col in zip(leg.get_texts(), style.NEURON_COLORS[:2]):
    txt.set_color(col); txt.set_fontweight('bold')


# ═══ (c) Cobweb ═══
ax = axes[2]
v21 = float((ES[0].T @ DS[1]).squeeze())
xi_part = (A2*(A1+1) - A1*(A1+1)*sys['cos_psi']) / (A1 * v21)
# Narrow range: user asked for -0.2 to 1
xi_range = np.linspace(-0.3, 1.2, 2000)
f_vals = np.array([first_return_B1(xi, sys) for xi in xi_range])

ax.plot(xi_range[xi_range < xi_part], f_vals[xi_range < xi_part],
        color=style.COLORS['green'], lw=2.0,
        label=r'Cross ($\mathbf{M}_{2\leftarrow 1}$)')
ax.plot(xi_range[xi_range >= xi_part], f_vals[xi_range >= xi_part],
        color=style.COLORS['dark'], lw=2.0,
        label=r'Self ($\mathbf{M}_{1\leftarrow 1}$)')
ax.plot([-1, 3], [-1, 3], 'k-', lw=0.5, alpha=0.3)
ax.axvline(xi_part, color='gray', ls='--', lw=0.8, alpha=0.4)

# Start close to the middle of the return-map fixed-point cluster so the
# cobweb stays in the top-right region, leaving the bottom-left corner
# free for the legend.
xi_sim = 0.35
cx, cy = [xi_sim], [0]
for _ in range(50):
    xn = first_return_B1(xi_sim, sys)
    if not np.isfinite(xn) or abs(xn) > 10: break
    cx.extend([xi_sim, xn]); cy.extend([xn, xn])
    xi_sim = xn
for i in range(len(cx)-1):
    t = min(i/30, 1.0)
    ax.plot([cx[i], cx[i+1]], [cy[i], cy[i+1]],
            color='#B78B27', lw=0.7, alpha=0.15+0.7*t)

from matplotlib.lines import Line2D as _L2D
cobweb_handle = _L2D([0], [0], color='#B78B27', lw=1.2, label='Cobweb')

ax.set_xlabel('$\\xi_n$'); ax.set_ylabel('$F(\\xi_n)$')
ax.set_xlim(-0.2, 1.0); ax.set_ylim(-0.2, 1.0)
ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
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
        xi_sim, bi_sim = step(xi_sim, bi_sim, sys)
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
_code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
_cache_path = os.path.join(_code_dir, '.cache_staircase.npz')
_force_recompute = ('--recompute' in _sys.argv) or (os.environ.get('SCN_RECOMPUTE') == '1')

_a2_range_target = np.linspace(0.25, 1.0, 750)   # matches devil_slider.html grid
_cache_hit = False
if not _force_recompute and os.path.exists(_cache_path):
    _d = np.load(_cache_path)
    if (float(_d['THETA1']) == THETA1 and float(_d['PSI']) == PSI
            and float(_d['A1']) == A1 and _d['a2_range'].shape == _a2_range_target.shape
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
        sys_tmp = make_2d_system(THETA1, PSI, A1, a2_)
        ratios, word = get_firing_ratios(sys_tmp)
        rho_vals.append(ratios[1])
        period_vals.append(len(word))
    rho_vals = np.array(rho_vals)
    period_vals = np.array(period_vals)
    np.savez(_cache_path, THETA1=THETA1, PSI=PSI, A1=A1,
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

ax.plot(a2_range, rho_vals, color=style.COLORS['accent'], lw=1.0,
        label=r'$\rho(\alpha_2)$')

plateaus = {}
for a2_, rho in zip(a2_range, rho_vals):
    if rho < 0.001: continue
    frac = Fraction(rho).limit_denominator(30); key = str(frac)
    if key not in plateaus: plateaus[key] = {'rho': rho, 'vals': []}
    plateaus[key]['vals'].append(a2_)
plateau_label_added = False
for info in plateaus.values():
    ax.plot(np.mean(info['vals']), info['rho'], 'o',
            color=style.COLORS['purple'], ms=2, alpha=0.6, zorder=5,
            label='Rational locks' if not plateau_label_added else None)
    plateau_label_added = True

ax.plot(A2, 2/7, '*', color=style.COLORS['red'], ms=12,
        markeredgecolor='k', markeredgewidth=0.5, zorder=10,
        label=r'Example $\rho=2/7$')
ax.set_xlabel(r'Drive $\alpha_2$'); ax.set_ylabel(r'Firing ratio $\rho$')
ax.set_xlim(0.25, 1); ax.set_ylim(-0.02, 0.55)
ax.set_xticks([0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5])
leg = ax.legend(fontsize=13, loc='lower right', framealpha=0.85)
legend_colors = [style.COLORS['accent'], style.COLORS['purple'], style.COLORS['red']]
for txt, col in zip(leg.get_texts(), legend_colors):
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
ax.set_xlim(0.25, 1)
ax.set_ylim(0, 40)
ax.set_xticks([0.25, 0.5, 0.75, 1])
ax.set_yticks([0, 20, 40])
leg = ax.legend(fontsize=13, loc='upper left', framealpha=0.85)
_period_col = ['#333333'] + list(style.NEURON_COLORS[:2])
for txt, col in zip(leg.get_texts(), _period_col):
    txt.set_color(col); txt.set_fontweight('bold')


fig.tight_layout(w_pad=1.5, h_pad=0.5, pad=0.1)

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'scn_word_example_flat.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved {_out}")
