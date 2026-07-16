"""Figure 3d: Three-neuron example (K=N=3).

Panels:
  (a) 3D geometry with bounds and orbit
  (b) Converged firing pattern (D_j^T y)
  (c) Convergence to limit cycle (distance vs step)
  (d) Devil's staircase with multistability
"""

import os
import sys; _here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "code"))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from scn import (make_3d_symmetric_system, step, eigenvector_orbit,
                 find_period, dist_to_limit_cycle, get_firing_ratios)
import style

style.setup()

ALPHAS = [1.0, 0.9, 0.7]
ELEV, AZIM = 70, 180  # near-top-down view: original; orbit reads as a planar cluster

sys = make_3d_symmetric_system(ALPHAS)
DS, ES, al = sys['DS'], sys['ES'], sys['alphas']
colors = style.NEURON_COLORS[:3]

# ─── Theory: eigenvector orbit ───
xi, bi = np.array([0.1, 0.1]), 0
for _ in range(5000): xi, bi = step(xi, bi, sys)
seq = []
for _ in range(300): seq.append(bi); xi, bi = step(xi, bi, sys)
p = find_period(seq); word = seq[:p]

orbit, ev, xi_star = eigenvector_orbit(word, sys)
ratio = abs(ev[1]/ev[0])


fig = plt.figure(figsize=(20, 5.0))

# ─── (a) 3D geometry: theory orbit + simulation overlay ───
ax1 = fig.add_subplot(141, projection='3d')

# Spike bound planes (more opaque + visible edges)
for i in range(3):
    d = DS[i]; foot = al[i]*d; t1 = ES[i][:,0]; t2 = ES[i][:,1]
    pts_on = [xi for y,bi_,xi in orbit if bi_==i]
    all_xi = [np.array([0.,0.])] + pts_on
    xis = np.array(all_xi); m = 0.25
    corners = np.array([foot+(xis[:,0].min()-m)*t1+(xis[:,1].min()-m)*t2,
                        foot+(xis[:,0].max()+m)*t1+(xis[:,1].min()-m)*t2,
                        foot+(xis[:,0].max()+m)*t1+(xis[:,1].max()+m)*t2,
                        foot+(xis[:,0].min()-m)*t1+(xis[:,1].max()+m)*t2])
    ax1.add_collection3d(Poly3DCollection([corners.tolist()], alpha=0.22,
                         facecolor=colors[i], edgecolor=colors[i], linewidth=1.2))

# Theory orbit points (eigenvector prediction)
for k, (y, bi_, _) in enumerate(orbit):
    ax1.scatter([y[0]],[y[1]],[y[2]], color=colors[bi_], s=70,
                edgecolors='k', linewidths=0.8, depthshade=False, zorder=20)

# Spike arrows (y -> y+) and decay segments (y+ -> next y) in firing order
xi_k = xi_star.copy(); bi_k = word[0]
for k in range(p):
    y = al[bi_k]*DS[bi_k] + ES[bi_k]@xi_k
    yp = y + DS[bi_k]
    # spike: y -> y+
    ax1.plot([y[0], yp[0]], [y[1], yp[1]], [y[2], yp[2]],
             color=colors[bi_k], lw=1.1, alpha=0.7, zorder=15)
    xi_n, bi_n = step(xi_k, bi_k, sys)
    y_next = al[bi_n]*DS[bi_n] + ES[bi_n]@xi_n
    # decay: y+ -> y_next
    ax1.plot([yp[0], y_next[0]], [yp[1], y_next[1]], [yp[2], y_next[2]],
             color='0.45', lw=0.8, alpha=0.55, ls='--', zorder=15)
    xi_k, bi_k = xi_n, bi_n

# Origin
_origin = ax1.scatter([0],[0],[0], color='k', s=80, depthshade=False,
                       edgecolors='white', linewidths=1.1, zorder=100)
_origin.set_clip_on(False)
ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
ax1.set_xlabel(''); ax1.set_ylabel(''); ax1.set_zlabel('')
ax1.view_init(elev=ELEV, azim=AZIM)
# Add margin so the origin at (0,0,0) sits inside the cube, not on its corner
_xl, _yl, _zl = ax1.get_xlim(), ax1.get_ylim(), ax1.get_zlim()
_pad = 0.10
ax1.set_xlim(_xl[0] - _pad*(_xl[1]-_xl[0]), _xl[1])
ax1.set_ylim(_yl[0] - _pad*(_yl[1]-_yl[0]), _yl[1])
ax1.set_zlim(_zl[0] - _pad*(_zl[1]-_zl[0]), _zl[1])
# panel (a) subtitle applied AFTER tight_layout via style.panel_label at bottom

# ─── (b) Converged firing pattern ───
ax2 = fig.add_subplot(142)
xi_conv, bi_conv = xi_star.copy(), word[0]
n_show = 2*p
conv_ys = []; conv_bis = []
for s in range(n_show):
    y = al[bi_conv]*DS[bi_conv]+ES[bi_conv]@xi_conv
    conv_ys.append(y); conv_bis.append(bi_conv)
    xi_conv, bi_conv = step(xi_conv, bi_conv, sys)
for j in range(3):
    dty = [DS[j]@y for y in conv_ys]
    ax2.plot(range(n_show), dty, color=colors[j], lw=1.2, alpha=0.70)
    ax2.axhline(al[j], color=colors[j], ls='--', lw=0.9, alpha=0.55)
for s in range(n_show):
    ax2.plot(s, DS[conv_bis[s]]@conv_ys[s], 'o', color=colors[conv_bis[s]], ms=5,
             markeredgecolor='k', markeredgewidth=0.4, zorder=5)
# Mark period boundary so reader sees the cycle repeats
ax2.axvline(p-0.5, color='0.65', lw=0.7, ls=':', alpha=0.7, zorder=1)
all_vals = [DS[j]@y for j in range(3) for y in conv_ys]
ax2.set_xlabel('Step $n$'); ax2.set_ylabel('$\\mathbf{D}_j^\\top \\mathbf{y}$')
ax2.set_xlim(-0.5, n_show-0.5); ax2.set_ylim(min(all_vals)-0.03, max(all_vals)+0.03)
ax2.set_xticks([0, p, 2*p])
# panel (b) subtitle applied below via style.panel_label

# ─── (c) Convergence ───
ax3 = fig.add_subplot(143)
n_steps = 120; rng = np.random.RandomState(42)
converging = []
for trial in range(500):
    xi_sim = rng.randn(2)*1.5; bi_sim = rng.randint(3)
    dists = []
    for s in range(n_steps):
        dists.append(dist_to_limit_cycle(xi_sim, bi_sim, orbit))
        xi_sim, bi_sim = step(xi_sim, bi_sim, sys)
    if dists[-1] < 1e-6: converging.append(dists)
converging = np.array(converging)
steps = np.arange(n_steps)
ax3.fill_between(steps, np.percentile(converging,5,axis=0), np.percentile(converging,95,axis=0),
                 color=style.COLORS['accent'], alpha=0.12, label='5--95th pctl.')
ax3.fill_between(steps, np.percentile(converging,25,axis=0), np.percentile(converging,75,axis=0),
                 color=style.COLORS['accent'], alpha=0.25)
ax3.plot(steps, np.mean(converging,axis=0), color=style.COLORS['accent'], lw=1.5)
ax3.plot(steps, converging[0][0]*ratio**(steps/p), 'k--', lw=1.2, alpha=0.7,
         label=f'$|\\mu_2/\\mu_1|^{{n/L}}$')
ax3.set_yscale('log'); ax3.set_xlabel('Spike $n$'); ax3.set_ylabel('Dist. to periodic orbit')
ax3.set_xlim(0,n_steps); ax3.set_ylim(1e-16,1e1)
ax3.legend(fontsize=plt.rcParams['axes.labelsize'], loc='upper right', framealpha=0.85)
# panel (c) subtitle applied below via style.panel_label

# ─── (d) Staircase with multistability ───
ax4 = fig.add_subplot(144)
print("Computing staircase...")
a3_range = np.linspace(0.6, 1.4, 100)
# Collect self-trap regions per neuron (which neuron is trapped at which alpha_3)
self_trap_by_a3_neuron = {}  # (a3, neuron) tuple set
for a3 in a3_range:
    sys_tmp = make_3d_symmetric_system([1.0, 0.9, a3])
    found = {}
    for trial in range(10):
        rng2 = np.random.RandomState(trial*13+7)
        xi_t = rng2.randn(2)*1.5; bi_t = rng2.randint(3)
        for _ in range(2000): xi_t, bi_t = step(xi_t, bi_t, sys_tmp)
        seq = []
        for _ in range(100): seq.append(bi_t); xi_t, bi_t = step(xi_t, bi_t, sys_tmp)
        p_ = find_period(seq)
        rhos = tuple(round(sum(1 for s in seq[:p_] if s==j)/p_, 3) for j in range(3))
        found[rhos] = found.get(rhos, 0) + 1
    for rhos, count in found.items():
        # True self-trap: one neuron has rho = 1 (all firings from that neuron).
        # Use a strict threshold rather than counting "active" neurons, so an
        # orbit like rho=(0.99, 0.005, 0.005) is NOT mislabeled as self-trap.
        if max(rhos) >= 0.999:
            j_dom = rhos.index(max(rhos))
            self_trap_by_a3_neuron[(a3, j_dom)] = True
        else:
            for j in range(3):
                if rhos[j] < 0.001: continue
                ax4.plot(a3, rhos[j], 'o', color=colors[j], ms=1.8, alpha=0.6)

# Self-trap markers placed at the actual value rho_j = 1.0 (NOT above it,
# since rho cannot exceed 1 by definition). Color = which neuron is trapped;
# overlapping trap ranges across neurons (rare here) darken naturally via
# semi-transparency.
for j in range(3):
    trap_a3s = sorted([a for (a, n) in self_trap_by_a3_neuron if n == j])
    if not trap_a3s:
        continue
    ax4.scatter(trap_a3s, [1.0]*len(trap_a3s), color=colors[j],
                marker='|', s=120, alpha=0.55, edgecolors='none', linewidths=2.4)
# (self-trap label removed -- explained in caption instead.)

rhos_wp = [sum(1 for w in word if w==j)/p for j in range(3)]
for j in range(3):
    ax4.plot(0.7, rhos_wp[j], '*', color=colors[j], ms=14,
             markeredgecolor='k', markeredgewidth=0.6, zorder=10)
ax4.set_xlabel(r'Drive $\alpha_3$'); ax4.set_ylabel(r'Firing ratio $\rho_j$')
ax4.set_xlim(0.6, 1.4)
ax4.set_ylim(-0.02, 1.02)
# panel (d) subtitle applied below via style.panel_label

fig.tight_layout(w_pad=2.5, pad=0.4)

# Legend moved to figure caption (was overlapping the staircase data).

# Zoom the 3D contents inside the axes' natural bbox WITHOUT expanding
# the bbox itself, so panel (a) visually matches the width/height of the
# 2D panels (b)/(c)/(d). Prior version enlarged the bbox by 0.08 in each
# dimension, making (a) noticeably bigger.
try:
    ax1.set_box_aspect((1, 1, 1), zoom=1.5)  # mpl >= 3.6
except TypeError:
    pass  # older mpl: zoom kwarg unsupported

# Panel subtitles in ABSOLUTE FIGURE COORDS. ax.set_title() on Axes3D
# anchors to the projected-cube region rather than the axes rectangle,
# so it ends up on a different figure-y than the 2D panels. Placing
# labels via fig.text at each ax's bbox top guarantees identical y across
# 3D and 2D axes. fontsize=30 matches derivation_tikz's \Large\bfseries
# once this figure is rendered at 0.70*linewidth in Box 4.
fig.canvas.draw()  # finalize axes positions before reading bboxes
_SUB = ['3D geometry', 'Return-map trace', 'Convergence', "Devil's staircase"]
_axes4 = [ax1, ax2, ax3, ax4]
_tops = [ax.get_position().y1 for ax in _axes4]
_ytxt = max(_tops) + 0.03  # common y just above the tallest axes bbox
for _ax, _l, _s in zip(_axes4, list('abcd'), _SUB):
    _bb = _ax.get_position()
    fig.text(_bb.x0, _ytxt,
             r'$\mathbf{(' + _l + r')}$\ \textbf{' + _s + '}',
             fontsize=30, fontweight='bold', va='bottom', ha='left')
import os
_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'fig_3d_example_flat.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight')
print(f"Saved {_out}")
