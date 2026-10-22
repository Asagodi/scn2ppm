"""Combined-noise robustness scan: drive noise + decoder noise (both multiplicative)
applied to the K=N=3 symmetric SCN with period-11 reference word.

For each (eps_alpha, eps_D) pair, we run n_trials noisy realizations, extract the
realized firing word, and compute the fraction that fail to recover the target word
(up to cyclic rotation). The resulting heatmap reveals whether the two noise
sources interact additively or super-additively.
"""

import os
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from scn import make_3d_symmetric_system, step, find_period

import style

style.setup()   # CMU font, text.usetex=True (needed for \boldsymbol)

# ---- Reference system + target word ----
ALPHAS = [1.0, 0.9, 0.7]
sys_base = make_3d_symmetric_system(ALPHAS)
DS_base = [d.copy() for d in sys_base['DS']]
al_base = list(sys_base['alphas'])

xi, bi = np.array([0.1, 0.1]), 0
for _ in range(5000):
    xi, bi = step(xi, bi, sys_base)
seq = []
xis_basin = []
for _ in range(300):
    seq.append(bi)
    xis_basin.append((xi.copy(), bi))
    xi, bi = step(xi, bi, sys_base)
p = find_period(seq)
target = tuple(seq[:p])
print(f"Target word (period {p}): {target}")
# Cache one canonical (xi, bi) on the target orbit to use as the seed
# starting point. Using a point on the orbit (not random) ensures the
# zero-noise cell has zero error rate.
xi_ref, bi_ref = xis_basin[0]
print(f"Reference orbit anchor: xi_ref={xi_ref}, bi_ref={bi_ref}")


def cyclic_match(w, t):
    """True iff w is a cyclic rotation of t."""
    if len(w) != len(t):
        return False
    s = list(t) + list(t)
    for k in range(len(t)):
        if tuple(s[k:k + len(t)]) == w:
            return True
    return False


def build_noisy_system(eps_alpha, eps_D, rng):
    """Apply multiplicative noise to drives and decoders (then renormalize)."""
    al_n = [a * (1.0 + eps_alpha * rng.randn()) for a in al_base]
    DS_n = []
    for d in DS_base:
        d_noisy = d * (1.0 + eps_D * rng.randn(len(d)))
        nrm = np.linalg.norm(d_noisy)
        if nrm < 1e-12:
            d_noisy = d.copy()
            nrm = 1.0
        DS_n.append(d_noisy / nrm)
    ES_n = []
    for d in DS_n:
        A = np.eye(3) - np.outer(d, d)
        Q, _ = np.linalg.qr(A)
        ES_n.append(Q[:, :2])
    cos_psi = np.array([[d1 @ d2 for d2 in DS_n] for d1 in DS_n])
    return {
        'DS': DS_n, 'ES': ES_n, 'alphas': al_n,
        'K': 3, 'N': 3,
        'cos_psi': cos_psi,
        'cos_psi_matrix': cos_psi,
    }


def realized_word(sys_, rng, n_warm=2500, n_word=80):
    # Start from a point ON the reference orbit (not random) so the zero-noise
    # cell gives zero error.  Tiny jitter so trials at the same noise level
    # explore slightly different orbit-phase entry points.
    xi = xi_ref + rng.randn(2) * 0.001
    bi = bi_ref
    for _ in range(n_warm):
        xi, bi = step(xi, bi, sys_)
    seq = []
    for _ in range(n_word):
        seq.append(bi)
        xi, bi = step(xi, bi, sys_)
    L = find_period(seq)
    return tuple(seq[:L])


eps_grid = list(np.linspace(0.0, 0.05, 21))
n_eps = len(eps_grid)
n_trials = 100
print(f"Scanning {n_eps}x{n_eps} grid, {n_trials} trials per cell ...")

error_rate = np.zeros((n_eps, n_eps))
for i, ea in enumerate(eps_grid):
    for j, ed in enumerate(eps_grid):
        errors = 0
        for t in range(n_trials):
            rng = np.random.RandomState(t * 31 + i * 7 + j * 13)
            try:
                sys_n = build_noisy_system(ea, ed, rng)
                w = realized_word(sys_n, rng)
                if not cyclic_match(w, target):
                    errors += 1
            except Exception:
                errors += 1
        error_rate[i, j] = errors / n_trials
    print(f"  eps_alpha={ea:.3f}: done.")

print("Error-rate matrix (rows = eps_alpha, cols = eps_D):")
print(np.round(error_rate, 2))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(4.6, 3.6))
im = ax.imshow(error_rate, origin='lower',
               extent=[eps_grid[0] - 0.005, eps_grid[-1] + 0.005,
                       eps_grid[0] - 0.005, eps_grid[-1] + 0.005],
               cmap='RdYlGn_r', vmin=0.0, vmax=1.0, aspect='auto')
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
cbar.set_label('word-error rate')

# Annotate cell values (skip when grid is too dense to keep readable)
if n_eps <= 8:
    for i, ea in enumerate(eps_grid):
        for j, ed in enumerate(eps_grid):
            val = error_rate[i, j]
            ax.text(ed, ea, f"{val:.2f}",
                    ha='center', va='center',
                    fontsize=8,
                    color='white' if 0.3 < val < 0.8 else 'k')


ax.set_xlabel(r'Decoder noise $\varepsilon_{\mathbf{D}}$')
ax.set_ylabel(r'Drive noise $\varepsilon_{\boldsymbol{\alpha}}$')
# Only ticks at endpoints: 0 and the max.
tick_vals = [0.0, 0.05]
ax.set_xticks(tick_vals)
ax.set_yticks(tick_vals)
ax.set_xticklabels([f"{e:.2f}" for e in tick_vals])
ax.set_yticklabels([f"{e:.2f}" for e in tick_vals])

fig.tight_layout()

_here = os.path.dirname(os.path.abspath(__file__))
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'combined_noise.pdf'))
fig.savefig(_out, dpi=300, bbox_inches='tight')
print(f"Saved {_out}")
