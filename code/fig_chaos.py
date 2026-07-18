"""
Fig 3 (chaos_figure.pdf) -- Chaos in the K=4, N=3 SCN.

Panels (new order after style pass):
  (a) Phase diagram over random (D, alpha) configurations at K=4, N=3.
      Decoders are rows of a 4x3 standard-normal matrix, unit-normalized;
      drives are i.i.d. Exp(0.8); random seed 42. Lyapunov spectrum per
      sample by QR accumulation. A large open-circle locator ring flags
      the specific chaotic configuration used for panels (b) and (c).
  (b) Sensitivity to initial conditions for that chaotic configuration:
      100 trajectory pairs at tangent-space separation 1e-12, tracked
      for 150 steps, median + IQR shown on a log distance axis. Dashed
      reference: epsilon * exp(lambda_max * t) with lambda_max = 0.282.
  (c) Strange attractor cross-section: 50,000 iterates of the transition
      map on the same chaotic configuration, plotted at the most-visited
      bound in tangent coordinates (xi_1, xi_2).

Numerics:
  - Per-step Jacobian by central finite differences on the current branch.
  - Lyapunov spectrum by QR accumulation over 50 segments of 1000 steps
    for the paper configuration; a shorter run per sample for the phase
    diagram (1500 transient + 3000 record).
  - Heavy computations (phase diagram + attractor iterates) are cached to
    .cache_chaos.npz next to this script so re-runs are fast.
"""

import os
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import scn
import style

style.setup()
# User-requested font sizes (match Fig 4)
mpl.rcParams.update({
    'axes.labelsize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})


# ============================================================================
# Paper chaotic (D, alpha) config -- app:fig-chaos, scn_supplementary.tex
# ============================================================================
D_CHAOS = np.array([
    [ 0.566, -0.791, -0.233],
    [ 0.748, -0.260,  0.611],
    [-0.296, -0.526, -0.798],
    [ 0.694, -0.601,  0.395],
])
D_CHAOS = D_CHAOS / np.linalg.norm(D_CHAOS, axis=1, keepdims=True)
ALPHAS_CHAOS = np.array([0.966, 0.342, 0.487, 0.653])


# ============================================================================
# Config knobs
# ============================================================================
# Paper spec is 800 samples for the phase diagram.
N_PHASE_SAMPLES = 800
PHASE_TRANSIENT = 1500
PHASE_RECORD    = 3000

# Paper Lyapunov for the reference (b),(c) config: QR over 50 segments of
# 1000 steps, well fed by 10,000 transient burn-in.
PAPER_TRANSIENT = 10_000
PAPER_SEGMENTS  = 50
PAPER_SEG_STEPS = 1000

ATTR_ITERATES   = 50_000
SENS_N_PERT     = 400        # more pairs, each anchored at a different point
SENS_EPS        = 1e-12      # along the converged chaotic orbit
SENS_TRANSIENT  = 10_000
SENS_N_TRACK    = 150
SENS_ANCHOR_SEP = 12         # spacing (steps) between successive anchors on
                             # the reference trajectory
# Set below to the QR-measured value from the paper config; keeps the
# dashed reference line self-consistent with the measured lambda_max.
LAMBDA_THEORY   = None  # filled in at runtime from paper_lam_max

FD_EPS          = 1e-6   # finite-difference step for the per-step Jacobian

# Sampler for the panel (a) phase diagram. 'iid' reproduces the paper
# (iid-D-on-S^2 x Exp(0.8)-alpha, 3/800 chaotic). 'apex' uses the
# apex-first sampler where all four bounds share a common apex on the
# y-axis and the polyhedron opens toward +y. Override via env var:
#   SCN_SAMPLER=apex python fig_chaos.py
SAMPLER    = os.environ.get('SCN_SAMPLER', 'iid').lower()
assert SAMPLER in ('iid', 'apex'), f'unknown SCN_SAMPLER={SAMPLER!r}'
_SUFFIX    = '' if SAMPLER == 'iid' else f'_{SAMPLER}'
CACHE_PATH = os.path.join(_HERE, f'.cache_chaos{_SUFFIX}.npz')
OUT_DIR    = os.path.join(_HERE, '..', 'figures')
OUT_PATH   = os.path.join(OUT_DIR, f'chaos_figure{_SUFFIX}.pdf')


# ============================================================================
# System builder for arbitrary K, N (paper spec is K=4, N=3)
# ============================================================================
def build_sys(D_mat, alphas):
    K, N = D_mat.shape
    DS = [D_mat[k].copy() for k in range(K)]
    ES = [scn.tangent_frame(d) for d in DS]
    return {'DS': DS, 'ES': ES, 'alphas': list(alphas), 'K': K, 'N': N}


# ============================================================================
# Principled apex-first sampler (Design A from chaos-sweep-audit workflow)
# ============================================================================
def sample_principled_KN(K=4, N=3, rng=None, y0=1.0,
                         gauge_fix_azimuth=True):
    """Apex-first sampler: all K bound hyperplanes share a common apex on
    the y-axis at (0, -y0, 0), and the admissible polyhedron opens toward
    +y by construction.

    INVARIANCE FIXED. The ambient SO(3) x R^3 rigid-motion gauge acting
    on (D_i, alpha_i) as (R D_i, alpha_i - D_i . t) is quotiented out by
      * anchoring the common apex to p = (0, -y0, 0) (kills R^3 and 2 of
        the 3 SO(3) dof by fixing which axis is the cone axis);
      * optionally rotating about +y so that D_1's (x, z) projection lies
        along +x (kills the residual SO(2)).
    State scale (D, alpha) ~ (D, c*alpha) is fixed by y0 = 1.

    Both sweep observables in `phase_diagram` -- min pairwise cosine and
    lambda_max -- are invariant under this gauge (min cos is rotation-
    invariant on unit rows; the SCN dynamics are equivariant under
    y -> R y + t so their Jacobian spectrum is preserved).

    CAVEAT (not a re-parameterization of the paper's iid prior). This
    ensemble is a codim-(K-N+2) slice of the joint (D, alpha) space with
    alpha_i = y0 * |D_i . e_y| deterministic in D_i. Chaos-fraction under
    this sampler is NOT comparable to the paper's iid-D x Exp(0.8)
    ensemble's 3/800 in Theorem cone(b); it is the correct sampler for
    the geometric question "canonical cone with apex on the y-axis,
    opening toward +y" only.

    With `alpha ~ Uniform(0, y0)` each decoder row is iid uniform on the
    OPEN LOWER hemisphere of S^2 (Archimedes' hat-box).

    Parameters
    ----------
    K : int
        Number of bound hyperplanes / decoders. Paper uses 4.
    N : int
        Ambient dimension. Specialized to N=3.
    rng : numpy Generator or None
        Randomness. If None, a fresh default_rng() is used.
    y0 : float > 0
        Apex distance below origin on the y-axis. Apex at (0, -y0, 0).
    gauge_fix_azimuth : bool
        If True, rotate about +y so D_1's (x, z) projection lies along +x.

    Returns
    -------
    D_mat : (K, N) float ndarray
        Unit-L2-row decoder matrix, D_{i,y} < 0 for every i.
    alphas : (K,) float ndarray
        Positive drives with alpha_i = y0 * |D_i . e_y|.
    """
    assert N == 3, ("Apex-first sampler is specialized to N=3 (the "
                    "y-axis is a canonical direction only in 3D).")
    if rng is None:
        rng = np.random.default_rng()

    alphas = rng.uniform(0.0, y0, size=K)
    Dy    = -alphas / y0
    r     = np.sqrt(np.clip(1.0 - Dy * Dy, 0.0, 1.0))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=K)
    D_mat = np.stack([r * np.cos(theta), Dy, r * np.sin(theta)], axis=1)

    if gauge_fix_azimuth and r[0] > 1e-12:
        phi = np.arctan2(D_mat[0, 2], D_mat[0, 0])
        c, s = np.cos(phi), np.sin(phi)
        Ry = np.array([[ c, 0.0,  s],
                       [0.0, 1.0, 0.0],
                       [-s, 0.0,  c]])
        D_mat = D_mat @ Ry.T

    apex = np.array([0.0, -y0, 0.0])
    assert np.allclose(np.linalg.norm(D_mat, axis=1), 1.0)
    assert np.allclose(D_mat @ apex, alphas)
    assert np.all(D_mat[:, 1] < 0.0)
    assert np.all(alphas > 0.0)
    return D_mat, alphas


# ============================================================================
# Finite-differenced per-step Jacobian on the currently active branch
# ============================================================================
def jacobian_fd(xi, bi, sys_dict, eps=FD_EPS):
    """Central-difference Jacobian of the transition map on the (bi -> bj)
    branch. Returns (xi_next, bj, J) or (None, -1, None) on step failure or
    branch flip inside the eps ball (in which case the map is not smooth
    at xi and the FD is meaningless).
    """
    xi = np.atleast_1d(xi).astype(float).ravel()
    m = len(xi)
    xi_next, bj = scn.step(xi, bi, sys_dict)
    if xi_next is None:
        return None, -1, None
    J = np.zeros((m, m))
    for k in range(m):
        e = np.zeros(m); e[k] = eps
        xi_p, bj_p = scn.step(xi + e, bi, sys_dict)
        xi_m, bj_m = scn.step(xi - e, bi, sys_dict)
        same_p = xi_p is not None and bj_p == bj
        same_m = xi_m is not None and bj_m == bj
        if same_p and same_m:
            J[:, k] = (xi_p - xi_m) / (2.0 * eps)
        elif same_p:
            J[:, k] = (xi_p - xi_next) / eps
        elif same_m:
            J[:, k] = (xi_next - xi_m) / eps
        else:
            return None, -1, None
    return xi_next, bj, J


# ============================================================================
# QR-based Lyapunov spectrum
# ============================================================================
def lyapunov_qr(sys_dict, n_transient, n_record, n_segments=1,
                xi0=None, bi0=0, seed=0):
    """Return (lyaps, xi_final, bi_final) or None on failure.

    lyaps has shape (N-1,) -- Lyapunov spectrum in nats/step, sorted
    descending so lyaps[0] is the largest exponent.
    """
    rng = np.random.default_rng(seed)
    m = sys_dict['N'] - 1
    if xi0 is None:
        xi = 0.1 * rng.standard_normal(m)
    else:
        xi = np.atleast_1d(xi0).astype(float).copy()
    bi = int(bi0)

    for _ in range(n_transient):
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            return None
        xi, bi = r

    Q = np.eye(m)
    total = np.zeros(m)
    total_steps = 0
    for _ in range(n_segments):
        for _ in range(n_record):
            xi_new, bi_new, J = jacobian_fd(xi, bi, sys_dict)
            if xi_new is None or J is None:
                return None
            Qnew, R = np.linalg.qr(J @ Q)
            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1.0
            Qnew = Qnew * signs
            R_diag = np.maximum(np.abs(np.diag(R)), 1e-300)
            total += np.log(R_diag)
            total_steps += 1
            Q = Qnew
            xi, bi = xi_new, bi_new
    lyaps = total / total_steps
    order = np.argsort(-lyaps)
    return lyaps[order], xi, bi


# ============================================================================
# Phase diagram: 800 random configs of (D, alpha) at K=4, N=3
# ============================================================================
def phase_diagram(n_samples=N_PHASE_SAMPLES, seed=42,
                  n_transient=PHASE_TRANSIENT, n_record=PHASE_RECORD,
                  sampler='iid'):
    """
    sampler : {'iid', 'apex'}
        'iid'  : paper's iid-D-on-S^2 x Exp(0.8)-alpha ensemble (default,
                 reproduces the numbers in Theorem cone(b)).
        'apex' : apex-first ensemble (see sample_principled_KN) with all
                 four bounds sharing a common apex on the y-axis and the
                 admissible polyhedron opening toward +y. NOT comparable
                 to the paper's 3/800 chaos count.
    """
    rng = np.random.default_rng(seed)
    min_cos = np.full(n_samples, np.nan)
    lam_max = np.full(n_samples, np.nan)
    lam_2   = np.full(n_samples, np.nan)
    t0 = time.time()
    for s in range(n_samples):
        if sampler == 'apex':
            D_mat, alphas = sample_principled_KN(K=4, N=3, rng=rng, y0=1.0)
        else:
            D_mat = rng.standard_normal((4, 3))
            D_mat = D_mat / np.linalg.norm(D_mat, axis=1, keepdims=True)
            alphas = rng.exponential(0.8, size=4)
        cos_mat = D_mat @ D_mat.T
        min_cos[s] = float(cos_mat[np.triu_indices(4, k=1)].min())
        sys_dict = build_sys(D_mat, alphas)
        res = lyapunov_qr(sys_dict, n_transient=n_transient,
                          n_record=n_record, seed=1000 + s)
        if res is not None:
            lyaps, _, _ = res
            lam_max[s] = float(lyaps[0])
            if len(lyaps) > 1:
                lam_2[s] = float(lyaps[1])
        if (s + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_samples - s - 1) / (s + 1)
            print(f'  phase diagram {s+1}/{n_samples}  '
                  f'elapsed {elapsed:.1f}s  eta {eta:.1f}s')
    return min_cos, lam_max, lam_2


# ============================================================================
# Sensitivity to initial conditions (panel b)
# ============================================================================
def sensitivity(sys_dict, n_pert=SENS_N_PERT, eps=SENS_EPS,
                n_transient=SENS_TRANSIENT, n_track=SENS_N_TRACK, seed=0):
    """Return (n_pert, n_track) matrix of ambient-space distances between
    each perturbed trajectory and the reference, per step.
    """
    rng = np.random.default_rng(seed)
    m = sys_dict['N'] - 1
    DS, ES, al = sys_dict['DS'], sys_dict['ES'], sys_dict['alphas']

    # Burn transient off a random seed on B_0.
    xi = 0.1 * rng.standard_normal(m)
    bi = 0
    for _ in range(n_transient):
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            raise RuntimeError('reference trajectory failed during transient')
        xi, bi = r

    # Pre-record a long stretch of the converged chaotic orbit so that we can
    # anchor successive perturbation pairs at DIFFERENT points along it.
    anchor_sep = int(SENS_ANCHOR_SEP)
    total_needed = n_pert * anchor_sep + 1
    anchors_xi = np.zeros((total_needed, m))
    anchors_bi = np.zeros(total_needed, dtype=int)
    anchors_xi[0] = xi
    anchors_bi[0] = int(bi)
    for k in range(1, total_needed):
        r = scn.step(anchors_xi[k-1], int(anchors_bi[k-1]), sys_dict)
        if r[0] is None:
            raise RuntimeError('reference trajectory failed while pre-recording anchors')
        anchors_xi[k], anchors_bi[k] = r

    dists = np.zeros((n_pert, n_track))
    for p in range(n_pert):
        # Anchor #p is at index p * anchor_sep along the converged orbit.
        k0 = p * anchor_sep
        xi_r, bi_r = anchors_xi[k0].copy(), int(anchors_bi[k0])
        d = rng.standard_normal(m); d /= np.linalg.norm(d)
        xi_p, bi_p = xi_r + eps * d, bi_r
        for t in range(n_track):
            y_r = al[bi_r] * DS[bi_r] + ES[bi_r] @ xi_r
            y_p = al[bi_p] * DS[bi_p] + ES[bi_p] @ xi_p
            dists[p, t] = max(float(np.linalg.norm(y_r - y_p)), 1e-300)
            r_next = scn.step(xi_r, bi_r, sys_dict)
            p_next = scn.step(xi_p, bi_p, sys_dict)
            if r_next[0] is None or p_next[0] is None:
                if t + 1 < n_track:
                    dists[p, t+1:] = dists[p, t]
                break
            xi_r, bi_r = r_next
            xi_p, bi_p = p_next
    return dists


# ============================================================================
# Strange attractor iterates (panel c)
# ============================================================================
def strange_attractor(sys_dict, n_iterates=ATTR_ITERATES,
                      n_transient=PAPER_TRANSIENT, seed=1):
    rng = np.random.default_rng(seed)
    m = sys_dict['N'] - 1
    xi = 0.1 * rng.standard_normal(m)
    bi = 0
    for _ in range(n_transient):
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            raise RuntimeError('transient failed')
        xi, bi = r

    xis = np.zeros((n_iterates, m))
    bis = np.zeros(n_iterates, dtype=int)
    n_actual = 0
    for i in range(n_iterates):
        xis[i] = xi
        bis[i] = bi
        n_actual = i + 1
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            break
        xi, bi = r
    xis = xis[:n_actual]
    bis = bis[:n_actual]
    counts = np.bincount(bis, minlength=sys_dict['K'])
    most_visited = int(np.argmax(counts))
    mask = bis == most_visited
    return xis[mask], most_visited, counts


def spike_sequence(sys_dict, n_steps=300,
                    n_transient=PAPER_TRANSIENT, seed=1):
    """Return the firing-neuron index at each of n_steps successive spikes
    starting from a post-transient state. Used for the raster in panel (d)."""
    rng = np.random.default_rng(seed)
    m = sys_dict['N'] - 1
    xi = 0.1 * rng.standard_normal(m)
    bi = 0
    for _ in range(n_transient):
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            raise RuntimeError('transient failed in spike_sequence')
        xi, bi = r
    seq = np.zeros(n_steps, dtype=int)
    seq[0] = int(bi)
    for k in range(1, n_steps):
        r = scn.step(xi, bi, sys_dict)
        if r[0] is None:
            seq = seq[:k]
            break
        xi, bi = r
        seq[k] = int(bi)
    return seq


# ============================================================================
# Main
# ============================================================================
def main():
    sys_paper = build_sys(D_CHAOS, ALPHAS_CHAOS)

    if os.path.exists(CACHE_PATH):
        print(f'Loading cache: {CACHE_PATH}')
        with np.load(CACHE_PATH) as z:
            min_cos       = z['min_cos']
            lam_max       = z['lam_max']
            paper_min_cos = float(z['paper_min_cos'])
            paper_lam_max = float(z['paper_lam_max'])
            paper_lam_2   = float(z['paper_lam_2'])
            attr_xi       = z['attr_xi']
            attr_bound    = int(z['attr_bound'])
            attr_counts   = z['attr_counts']
    else:
        print('No cache found. Computing heavy quantities...')

        print(f'Phase diagram over {N_PHASE_SAMPLES} random configs '
              f'(paper spec: 800), sampler={SAMPLER}...')
        min_cos, lam_max, _ = phase_diagram(sampler=SAMPLER)

        print('Lyapunov spectrum for paper chaotic config '
              '(50 segments x 1000 steps QR)...')
        res = lyapunov_qr(sys_paper, n_transient=PAPER_TRANSIENT,
                          n_record=PAPER_SEG_STEPS,
                          n_segments=PAPER_SEGMENTS, seed=7)
        assert res is not None, 'paper-config Lyapunov failed'
        lyaps_paper, _, _ = res
        paper_lam_max = float(lyaps_paper[0])
        paper_lam_2   = float(lyaps_paper[1]) if len(lyaps_paper) > 1 else np.nan
        cos_mat = D_CHAOS @ D_CHAOS.T
        paper_min_cos = float(cos_mat[np.triu_indices(4, k=1)].min())
        print(f'  lyapunov spectrum = {lyaps_paper}')
        print(f'  lambda_1 = {paper_lam_max:.3f} nats/step, '
              f'lambda_2 = {paper_lam_2:.3f} nats/step')
        print(f'  min pairwise cos psi = {paper_min_cos:.3f}')

        print('Strange attractor iterates...')
        attr_xi, attr_bound, attr_counts = strange_attractor(sys_paper)
        print(f'  most-visited bound: B_{attr_bound}, '
              f'{attr_counts[attr_bound]} of {int(attr_counts.sum())} points')
        print(f'  full counts = {attr_counts.tolist()}')

        np.savez(CACHE_PATH,
                 min_cos=min_cos, lam_max=lam_max,
                 paper_min_cos=paper_min_cos, paper_lam_max=paper_lam_max,
                 paper_lam_2=paper_lam_2,
                 attr_xi=attr_xi, attr_bound=attr_bound,
                 attr_counts=attr_counts)
        print(f'Cached to {CACHE_PATH}')

    # Sensitivity is cheap (~1s), always recompute
    print('Sensitivity to initial conditions...')
    dists = sensitivity(sys_paper)

    # Spike raster: 300 successive spikes from the post-transient state
    print('Spike raster for panel (d)...')
    raster_seq = spike_sequence(sys_paper, n_steps=300)

    # ========================================================================
    # Figure: 1x3, wider-than-tall
    # ========================================================================
    # Layout: three panels (a),(b),(c) on top row, single wide flat raster
    # (d) spanning all three column widths on the bottom row.
    fig = plt.figure(figsize=(17, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0],
                          hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, :])

    LABEL_FS = 19
    TICK_FS  = 15
    LEG_FS   = 14

    # ---- Panel (a): phase diagram -----------------------------------------
    finite = np.isfinite(lam_max) & np.isfinite(min_cos)
    chaotic = finite & (lam_max > 0)
    regular = finite & (lam_max <= 0)
    col_reg   = style.COLORS['dark']
    col_chaos = style.COLORS['red']

    ax_a.scatter(min_cos[regular], lam_max[regular],
                 marker='o', s=32, c=col_reg,
                 alpha=0.5, linewidths=0, label='regular')
    ax_a.scatter(min_cos[chaotic], lam_max[chaotic],
                 marker='s', s=56, c=col_chaos,
                 alpha=0.55, linewidths=0, label='chaotic')
    ax_a.axhline(0, color=style.COLORS['gray'], lw=0.7, ls='--', zorder=0)

    # The paper's constructed (D, alpha) case is NOT one of the 800 random
    # samples (it was chosen deliberately to prove chaos). Draw it explicitly
    # as another chaotic square, then overlay the accent-blue open-square
    # locator centered on it.
    ax_a.scatter([paper_min_cos], [paper_lam_max],
                 marker='s', s=56, c=col_chaos, linewidths=0, zorder=4)
    ax_a.scatter([paper_min_cos], [paper_lam_max],
                 marker='s', s=220, facecolors='none',
                 edgecolors=style.COLORS['accent'], linewidths=2.0,
                 zorder=5, label=r'panel (b),(c) case')

    ax_a.set_xlabel(r'$\min_{i<j}\ \cos\psi_{ij}$', fontsize=LABEL_FS)
    ax_a.set_ylabel(r'$\lambda_{\max}$ (nats/step)', fontsize=LABEL_FS)
    ax_a.tick_params(labelsize=TICK_FS)
    ax_a.legend(loc='lower right', bbox_to_anchor=(0.99, 0.01),
                fontsize=LEG_FS, framealpha=0.92,
                borderaxespad=0.0)
    style.panel_label(ax_a, 'a', subtitle='Regime survey', fontsize=27, y=1.02)

    # ---- Panel (b): sensitivity to IC -------------------------------------
    t = np.arange(dists.shape[1])
    med = np.median(dists, axis=0)
    q25 = np.quantile(dists, 0.25, axis=0)
    q75 = np.quantile(dists, 0.75, axis=0)
    ax_b.fill_between(t, q25, q75, color=style.COLORS['dark'],
                      alpha=0.22, linewidth=0,
                      label=f'IQR ({dists.shape[0]} pairs)')
    ax_b.plot(t, med, color=style.COLORS['dark'], lw=1.8, label='median')

    lam_ref = float(paper_lam_max)
    y_theory = SENS_EPS * np.exp(lam_ref * t)
    y_cap = q75.max() * 3.0
    mask_th = y_theory < y_cap
    ax_b.plot(t[mask_th], y_theory[mask_th],
              ls='--', color=style.COLORS['red'], lw=2.0,
              label=r'$\varepsilon\,e^{\lambda_{\max}t},\ \lambda_{\max}=%.2f$' % lam_ref)

    ax_b.set_yscale('log')
    ax_b.set_xlabel(r'step $t$', fontsize=LABEL_FS)
    ax_b.set_ylabel(r'$\|y_t^{\mathrm{ref}} - y_t^{\mathrm{pert}}\|$', fontsize=LABEL_FS)
    ax_b.tick_params(labelsize=TICK_FS)
    ax_b.legend(loc='lower right', fontsize=LEG_FS)
    style.panel_label(ax_b, 'b', subtitle='IC sensitivity', fontsize=27, y=1.02)

    # ---- Panel (c): strange attractor cross-section -----------------------
    ax_c.scatter(attr_xi[:, 0], attr_xi[:, 1],
                 s=0.7, c=style.COLORS['accent'],
                 alpha=0.45, linewidths=0, marker='o', rasterized=True)
    ax_c.set_xlabel(r'$\xi_1$', fontsize=LABEL_FS)
    ax_c.set_ylabel(r'$\xi_2$', fontsize=LABEL_FS)
    ax_c.tick_params(labelsize=TICK_FS)
    ax_c.set_aspect('equal', adjustable='datalim')
    style.panel_label(ax_c, 'c', subtitle='Strange attractor', fontsize=27, y=1.02)

    # ---- Panel (d): spike raster --------------------------------------------
    K = sys_paper['K']
    neuron_colors = style.NEURON_COLORS[:K]
    xs = np.arange(len(raster_seq))
    for k in range(K):
        m = raster_seq == k
        ax_d.vlines(xs[m], k - 0.42, k + 0.42,
                     color=neuron_colors[k], linewidth=1.2)
    ax_d.set_yticks(range(K))
    ax_d.set_yticklabels([f'{k+1}' for k in range(K)])
    ax_d.set_ylim(-0.6, K - 0.4)
    ax_d.set_xlabel(r'spike index $n$', fontsize=LABEL_FS)
    ax_d.set_ylabel('neuron', fontsize=LABEL_FS)
    ax_d.set_xlim(0, len(raster_seq))
    ax_d.tick_params(labelsize=TICK_FS)
    ax_d.invert_yaxis()
    style.panel_label(ax_d, 'd', subtitle='Chaotic spike raster', fontsize=27, y=1.02)

    # Gridspec already sets hspace/wspace, so skip tight_layout.
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f'Saved figure to {OUT_PATH}')


if __name__ == '__main__':
    main()
