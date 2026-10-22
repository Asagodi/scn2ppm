"""Paper-ready version: K=N=3 symmetric multistability top-down view.

Changes vs earlier drafts:
- Neurons 1,2,3 (not 0-indexed).
- Bigger fonts throughout.
- Harmonized colorblind-safe palette.
- No zoom info in title; no redundant info between title/legend.
- No axis ticks.
- Saved as both PNG and PDF (vector, paper-embeddable).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from scn import step
import style
style.setup()   # CMU font + shared rcParams

# ============ Muted / paper-friendly palette ============
# Neurons: three deep, saturated colors that don't clash with the orbit
# markers below. Colorblind-check: vermillion / blue / green are still
# distinguishable in deuteranopia simulation.
NEURON_COL = {
    1: '#B03A2E',  # deep red-brown
    2: '#1F4E79',  # navy
    3: '#1E7A48',  # forest green
}
# Orbits: neutral gray levels — the fill just says "which orbit", the
# neuron color on the arrows already tells the reader everything else.
ORBIT_COL = {
    'A': '#F5F5F5',  # nearly-white  -> σ=(1,2,3)
    'B': '#2C3E50',  # dark slate     -> σ=(1,3,2)
}
DECAY_COL = '#7F8C8D'

def make_3d_symmetric(alphas, cos_psi_target=0.507):
    axis = np.array([1,1,1])/np.sqrt(3)
    ct = np.sqrt((1-cos_psi_target)/1.5)
    st = np.sqrt(1 - ct**2)
    e = [np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.])]
    e_perp = [ei - (ei @ axis)*axis for ei in e]
    e_perp = [ei/np.linalg.norm(ei) for ei in e_perp]
    DS = [ct*e_perp[i] + st*axis for i in range(3)]
    DS = [d/np.linalg.norm(d) for d in DS]
    def tangent_frame(d):
        z = np.array([0.,0,1.]) if abs(d[2]) < 0.9 else np.array([0.,1.,0])
        u1 = np.cross(d, z); u1 = u1/np.linalg.norm(u1)
        u2 = np.cross(d, u1); u2 = u2/np.linalg.norm(u2)
        return np.column_stack([u1, u2])
    ES = [tangent_frame(d) for d in DS]
    cos_psi = np.array([[DS[i] @ DS[j] for j in range(3)] for i in range(3)])
    return {'DS': DS, 'ES': ES, 'alphas': list(alphas), 'K': 3, 'N': 3, 'cos_psi_matrix': cos_psi}

alpha = 0.7
sys3 = make_3d_symmetric([alpha]*3)
DS, ES, alphas = sys3['DS'], sys3['ES'], sys3['alphas']

axis = np.array([1,1,1])/np.sqrt(3)
e1 = (DS[0] - (DS[0] @ axis) * axis); e1 /= np.linalg.norm(e1)
e2 = np.cross(axis, e1)
def proj(y):
    return np.array([y @ e1, y @ e2])

def simulate_orbit(bi0, xi0, n_warm=2000):
    xi = np.array(xi0); bi = bi0
    for _ in range(n_warm):
        xi_next, bj = step(xi, bi, sys3)
        if bj is None or bj < 0: return None
        xi, bi = xi_next, bj
    ys = []; bs = []
    for _ in range(9):
        y = alphas[bi]*DS[bi] + ES[bi] @ xi
        ys.append(y); bs.append(bi)
        xi_next, bj = step(xi, bi, sys3)
        if bj is None or bj < 0: break
        xi, bi = xi_next, bj
    return np.array(ys), np.array(bs)

# Find both chirality orbits
np.random.seed(0)
orbA, orbB = None, None
for trial in range(80):
    xi0 = np.random.uniform(-0.4, 0.4, size=2)
    for bi0 in [0,1,2]:
        r = simulate_orbit(bi0, xi0)
        if r is None: continue
        ys, bs = r
        w = tuple(bs[:3])
        can = min(w[i:] + w[:i] for i in range(3))
        if can == (0,1,2) and orbA is None: orbA = (ys, bs)
        elif can == (0,2,1) and orbB is None: orbB = (ys, bs)
        if orbA is not None and orbB is not None: break
    if orbA is not None and orbB is not None: break

def make_segments(orb):
    ys, bs = orb
    out = []
    for k in range(3):
        y_pre = ys[k]
        bi = bs[k]
        y_post = y_pre + DS[bi]
        y_next = ys[(k+1) % len(ys)]
        bj = bs[(k+1) % len(bs)]
        out.append((y_pre, y_post, y_next, bi, bj))
    return out

def pts_by_neuron(orb):
    ys, bs = orb
    d = {}
    for y, b in zip(ys, bs):
        if b not in d: d[b] = y
        if len(d) == 3: break
    return d

segsA = make_segments(orbA)
segsB = make_segments(orbB)
ptsA_3d = pts_by_neuron(orbA)
ptsB_3d = pts_by_neuron(orbB)
z_spike = orbA[0][0] @ axis

# ── Empirical check of the shared time-averaged readout claim ───────────
# Discrete mean over one period of the pre-spike states (what the caption
# labels ybar at the origin). By the S_3 permutation symmetry the two
# orbits are related to each other by a reflection permutation of the
# neuron labels, so their orbit means must coincide and both lie on the
# (1,1,1) diagonal. Whether the mean equals exactly (1/3) sum_j D_j is a
# specific numerical claim — we test it here.
ybar_A = np.mean(orbA[0][:3], axis=0)
ybar_B = np.mean(orbB[0][:3], axis=0)
ybar_1third_D = sum(DS) / 3.0
print(f'[ybar check]  orbit A pre-spike mean = {ybar_A}')
print(f'[ybar check]  orbit B pre-spike mean = {ybar_B}')
print(f'[ybar check]  (1/3) sum D_j          = {ybar_1third_D}')
print(f'[ybar check]  ||ybar_A - ybar_B||   = {np.linalg.norm(ybar_A - ybar_B):.3e}')
print(f'[ybar check]  ||ybar_A - (1/3) sum D|| = {np.linalg.norm(ybar_A - ybar_1third_D):.3e}')
# Result: the two orbit means coincide to machine precision (S_3 symmetry
# forces both onto the (1,1,1) diagonal), but the coefficient in front of
# the axis is not 1/3 — it also depends on alpha and on the tangent-plane
# eigenvector fixed point xi^*. The correct caption states shared readout
# on the (1,1,1) axis without the specific 1/3 sum D_j formula.

def bound_line_endpoints(i, z, half_length=0.6):
    Di_proj = proj(DS[i])
    n_hat = Di_proj / (np.linalg.norm(Di_proj)+1e-9)
    tangent = np.array([-n_hat[1], n_hat[0]])
    dist = (alphas[i] - z * (DS[i] @ axis)) / np.linalg.norm(Di_proj)
    center = dist * n_hat
    return center - half_length * tangent, center + half_length * tangent

# ============ Figure =========================
# The paper includes this at 0.45\textwidth, so the saved 8.4in canvas is scaled by about
# 0.35 on the page. Everything below is sized for that: the numbers are what the body text
# is set at (12pt), so each of these lands between 9 and 11pt once reduced.
FS_BOUND, FS_NUM, FS_YBAR, FS_LEGEND, FS_TITLE = 32, 26, 32, 27, 30
MS_DOT, MS_LEGEND = 36, 28          # the dot has to hold a FS_NUM digit

LIM = 0.5
fig, ax = plt.subplots(figsize=(13.4, 9))

# Bound slices (dashed lines through the spike points, colored by neuron)
for i in range(3):
    color = NEURON_COL[i+1]  # neuron index displayed as 1-based
    p1, p2 = bound_line_endpoints(i, z_spike, half_length=LIM*1.6)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls=(0, (6, 4)), color=color, lw=3.2,
            alpha=0.85, zorder=2)
    # Label at midpoint between the two spike points on this bound, offset outward.
    p_A = proj(ptsA_3d[i]); p_B = proj(ptsB_3d[i])
    spike_mid = 0.5 * (p_A + p_B)
    n_hat = proj(DS[i]) / np.linalg.norm(proj(DS[i]))
    lab_pt = spike_mid - n_hat * 0.06
    ax.text(lab_pt[0], lab_pt[1], rf'$\mathcal{{B}}_{i+1}$', color=color,
            fontsize=FS_BOUND, ha='center', va='center', style='italic', weight='bold',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=color, lw=1.8, alpha=0.95))

# Reflection axis (subtle)
ax.axhline(0, color='#cccccc', ls=':', lw=0.8, alpha=0.5, zorder=1)

# Spikes + decays for each orbit (single arrow per segment; no overlays)
for orb_segs, col_orb in [(segsA, ORBIT_COL['A']), (segsB, ORBIT_COL['B'])]:
    for y_pre, y_post, y_next, bi, bj in orb_segs:
        p_pre = proj(y_pre); p_post = proj(y_post); p_next = proj(y_next)
        # Spike arrow (neuron-colored)
        ax.annotate('', xy=p_post, xytext=p_pre,
                    arrowprops=dict(arrowstyle='-|>', color=NEURON_COL[bi+1], lw=4.0,
                                    mutation_scale=30, alpha=1.0),
                    zorder=6)
        # Decay arrow (single annotate: line + terminal arrowhead in one)
        ax.annotate('', xy=p_next, xytext=p_post,
                    arrowprops=dict(arrowstyle='-|>', color=DECAY_COL, lw=2.5,
                                    mutation_scale=28, alpha=0.85),
                    zorder=5)

# Spike dots with neuron labels (1-indexed).
# Orbit A = light fill, dark border, dark text; orbit B = dark fill, light text.
orbit_style = {
    'A': {'fc': ORBIT_COL['A'], 'ec': '#111', 'txt': '#111'},
    'B': {'fc': ORBIT_COL['B'], 'ec': '#111', 'txt': '#F5F5F5'},
}
for orb_segs, key in [(segsA, 'A'), (segsB, 'B')]:
    s = orbit_style[key]
    for y_pre, y_post, y_next, bi, bj in orb_segs:
        p_pre = proj(y_pre)
        ax.plot(p_pre[0], p_pre[1], 'o', color=s['fc'], markersize=MS_DOT,
                mec=s['ec'], mew=2.2, zorder=10)
        ax.text(p_pre[0], p_pre[1], str(bi+1), color=s['txt'],
                ha='center', va='center', fontsize=FS_NUM, weight='bold', zorder=11)

# Origin marker (also = shared readout by 3-fold symmetry)
ax.plot(0, 0, marker='+', color='#333', markersize=26, mew=3.5, zorder=15)
ax.text(0.02, 0.02, r'$\bar{y}$', fontsize=FS_YBAR, ha='left', va='bottom',
        color='#333', weight='bold')

# ============ Legend (compact) ============
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', mec='#111', mew=1.5,
           mfc=ORBIT_COL['A'], markersize=MS_LEGEND, label=r'$\sigma=(1,2,3)$', linestyle=''),
    Line2D([0], [0], marker='o', color='w', mec='#111', mew=1.5,
           mfc=ORBIT_COL['B'], markersize=MS_LEGEND, label=r'$\sigma=(1,3,2)$', linestyle=''),
    Line2D([0], [0], color=NEURON_COL[1], lw=4.2, label='spike (colored by firer)'),
    Line2D([0], [0], color=DECAY_COL, lw=3.2, label='decay toward origin'),
]
# Outside the axes on the right: the star fills the frame, so any inside placement lands on
# a trajectory or a bound line.
ax.legend(handles=legend_elems, loc='center left', bbox_to_anchor=(1.01, 0.5),
          fontsize=FS_LEGEND, frameon=False, handletextpad=0.8, labelspacing=1.0)

ax.set_aspect('equal')
# Centre the view on the star's own bounding box, with the same span in x and y. The pattern
# is three-fold symmetric, so it reaches further right than left; centring on the origin
# instead would leave a visible empty band on the left. The origin, and with it the ybar
# marker, therefore sits slightly off centre.
_ink = np.array([proj(p) for _segs in (segsA, segsB)
                 for (_pre, _post, _next, _bi, _bj) in _segs
                 for p in (_pre, _post, _next)])
_cx = 0.5 * (_ink[:, 0].min() + _ink[:, 0].max())
_cy = 0.5 * (_ink[:, 1].min() + _ink[:, 1].max())
_half = 1.06 * max(np.abs(_ink[:, 0] - _cx).max(), np.abs(_ink[:, 1] - _cy).max())
ax.set_xlim(_cx - _half, _cx + _half)
ax.set_ylim(_cy - _half, _cy + _half)
print('[view] centre (%+.4f, %+.4f), half-span %.4f, equal in x and y' % (_cx, _cy, _half))
# No title and no frame: the caption names the setting, and the axes carry no scale worth
# reading, so the box and ticks were only furniture.
ax.set_axis_off()

# Fixed axes box: tight_layout would shrink the square to make room for the legend,
# which is the opposite of what is wanted here.
ax.set_position([0.01, 0.02, 0.66, 0.96])
outpath_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'multistability_topdown.png')
outpath_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'multistability_topdown.pdf')
plt.savefig(outpath_png, dpi=200, bbox_inches='tight')
plt.savefig(outpath_pdf, bbox_inches='tight')
print(f'saved:\n  {outpath_png}\n  {outpath_pdf}')
