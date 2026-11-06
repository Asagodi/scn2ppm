"""
Core Spike Coding Network (SCN) module.

Implements the piecewise projective map framework:
- Tangent frame construction
- Spike-then-decay transition map
- Return map composition
- Eigenvector orbit prediction
- Admissibility margin computation
- Period detection
- Forward simulation

All figures in the paper use this module.
"""

import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════

def tangent_frame(d: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the tangent space D_i^perp.

    Parameters
    ----------
    d : (N,) unit vector (decoder)

    Returns
    -------
    U : (N, N-1) orthonormal matrix with U^T d = 0
    """
    N = len(d)
    # N=2 is the one case where the paper fixes a sign convention:
    # U_i = (-sin theta_i, cos theta_i), the +90 degree rotation of D_i
    # (Prop. "first-return map", Step 4). Plain Gram-Schmidt below returns the
    # NEGATIVE of this, which silently flips the sign of every tangent
    # coordinate xi relative to the paper's formulas and to make_2d_system.
    # Special-case it so the whole module agrees with the derivation.
    if N == 2:
        return np.array([[-d[1]], [d[0]]])
    cols = []
    for i in range(N):
        v = np.zeros(N)
        v[i] = 1.0
        v -= (d @ v) * d
        for c in cols:
            v -= (c @ v) * c
        n = np.linalg.norm(v)
        if n > 1e-10:
            cols.append(v / n)
        if len(cols) == N - 1:
            break
    return np.column_stack(cols)


def make_2d_system(theta1: float, psi: float, alpha1: float, alpha2: float):
    """Construct a K=N=2 SCN system.

    Parameters
    ----------
    theta1 : angle of decoder 1 (radians)
    psi    : inter-decoder angle (radians)
    alpha1, alpha2 : suprathreshold drives

    Returns
    -------
    dict with keys: DS, ES, alphas, K, N, cos_psi
    """
    theta2 = theta1 + psi
    D1 = np.array([np.cos(theta1), np.sin(theta1)])
    D2 = np.array([np.cos(theta2), np.sin(theta2)])
    # Use explicit (-sin, cos) tangent convention (counter-clockwise 90° rotation)
    # so that sign convention matches the original derivation.
    E1 = np.array([[-np.sin(theta1)], [np.cos(theta1)]])
    E2 = np.array([[-np.sin(theta2)], [np.cos(theta2)]])
    return {
        'DS': [D1, D2], 'ES': [E1, E2],
        'alphas': [alpha1, alpha2],
        'K': 2, 'N': 2, 'cos_psi': D1 @ D2,
    }


def make_3d_symmetric_system(alphas, spread=0.7):
    """Construct K=N=3 system with symmetric decoders about [1,1,1]/sqrt(3).

    Parameters
    ----------
    alphas : (3,) drives
    spread : perturbation magnitude from central axis

    Returns
    -------
    dict with keys: DS, ES, alphas, K, N, cos_psi_matrix
    """
    c_axis = np.array([1, 1, 1]) / np.sqrt(3)
    u1 = np.array([1, -1, 0]) / np.sqrt(2)
    u2 = np.cross(c_axis, u1)
    u2 /= np.linalg.norm(u2)

    DS = []
    for phi in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        d = c_axis + spread * (np.cos(phi) * u1 + np.sin(phi) * u2)
        d /= np.linalg.norm(d)
        DS.append(d)

    ES = [tangent_frame(d) for d in DS]

    cos_psi = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cos_psi[i, j] = DS[i] @ DS[j]

    return {
        'DS': DS, 'ES': ES,
        'alphas': list(alphas),
        'K': 3, 'N': 3, 'cos_psi_matrix': cos_psi,
    }


# ═══════════════════════════════════════════════════════════════
# Forward simulation
# ═══════════════════════════════════════════════════════════════

# What to do when a reset leaves another neuron already past its bound, i.e.
# suprathreshold at tau = 0 with decay factor c_j > 1. Only reachable with an
# excitatory weight (a negative decoder cosine): in the cone regime the
# physical region is forward invariant, every c_j < 1, and the two rules
# coincide exactly.
#   'immediate' : it is over threshold now, so it fires now. Largest voltage
#                 excess first, re-checking after each spike because every
#                 spike moves the others. This is the greedy SCN rule, and
#                 the extra spikes share the instant of the reset.
#   'skip'      : ignore it and decay on to the next reachable bound. What
#                 this module did before 2026-11-04; kept so the published
#                 figures stay reproducible.
RESET_RULE = 'immediate'
MAX_CASCADE = 64


def step(xi: np.ndarray, bi: int, sys: dict, rule: str = None,
         return_spikes: bool = False):
    """One spike-then-decay step.

    Parameters
    ----------
    xi : tangent coordinate on bound bi
    bi : current bound index
    sys : system dict (DS, ES, alphas, K)
    rule : 'immediate' or 'skip'; defaults to the module-level RESET_RULE
    return_spikes : also return the neurons that fired at tau = 0 after the
        reset, in order, before the decay

    Returns
    -------
    xi_next : tangent coordinate on next bound
    bj : next bound index
    spikes : list of instantaneous spikes (only if return_spikes)

    Notes
    -----
    Each branch stays projective. The cascade adds fixed decoder vectors, so
    the post-cascade readout is still affine in xi, and the decay still
    divides by an affine function of it. What grows is the number of
    branches, not the form of any one of them.
    """
    DS, ES, al, K = sys['DS'], sys['ES'], sys['alphas'], sys['K']
    rule = RESET_RULE if rule is None else rule
    xi = np.atleast_1d(xi).ravel()
    yp = (al[bi] + 1) * DS[bi] + (ES[bi] @ xi).ravel()

    cascade = []
    if rule == 'immediate':
        for _ in range(MAX_CASCADE):
            best_exc, jmax = 1e-12, -1
            for j in range(K):
                if al[j] <= 0:
                    continue
                exc = al[j] - DS[j] @ yp        # voltage above threshold
                if exc > best_exc:
                    best_exc, jmax = exc, j
            if jmax < 0:
                break
            cascade.append(jmax)
            yp = yp + DS[jmax]
        else:
            # runaway avalanche: no sub-threshold state within MAX_CASCADE
            return (None, -1, cascade) if return_spikes else (None, -1)
    elif rule != 'skip':
        raise ValueError(f'unknown reset rule {rule!r}')

    bj, bc = -1, -1.0
    for j in range(K):
        if al[j] <= 0:
            continue
        dn = DS[j] @ yp
        if dn < 1e-12:
            continue
        c = al[j] / dn
        if 0 < c <= 1.0 + 1e-10 and c > bc:
            bc, bj = c, j
    if bj < 0:
        return (None, -1, cascade) if return_spikes else (None, -1)
    xi_next = (ES[bj].T @ (bc * yp)).ravel()
    return (xi_next, bj, cascade) if return_spikes else (xi_next, bj)


def simulate(sys: dict, xi0, bi0: int, n_transient: int, n_record: int):
    """Run forward simulation.

    Parameters
    ----------
    sys : system dict
    xi0 : initial tangent coordinate
    bi0 : initial bound index
    n_transient : steps to discard
    n_record : steps to record

    Returns
    -------
    xis : list of tangent coordinates
    bis : list of bound indices
    """
    xi, bi = np.atleast_1d(xi0).astype(float).copy(), bi0
    for _ in range(n_transient):
        xi, bi = step(xi, bi, sys)
    xis, bis = [], []
    for _ in range(n_record):
        xis.append(np.atleast_1d(xi).copy())
        bis.append(bi)
        xi, bi = step(xi, bi, sys)
    return xis, bis


# ═══════════════════════════════════════════════════════════════
# Period detection
# ═══════════════════════════════════════════════════════════════

def find_period(seq: list, min_repeats: int = 3) -> int:
    """Detect period of a firing sequence.

    Parameters
    ----------
    seq : list of bound indices
    min_repeats : minimum number of full repeats to confirm

    Returns
    -------
    p : detected period (len(seq) if no period found)
    """
    n = len(seq)
    for p in range(1, n // 3):
        checks = min(max(min_repeats * p, 30), n - p)
        if all(seq[k] == seq[k + p] for k in range(checks)):
            return p
    return n


# ═══════════════════════════════════════════════════════════════
# Return map and eigenvector orbit prediction
# ═══════════════════════════════════════════════════════════════

def transition_matrix(bi: int, bj: int, sys: dict, cascade=()) -> np.ndarray:
    """N×N transition matrix M_{j←i} in homogeneous coordinates.

    Parameters
    ----------
    bi : source bound index
    bj : target bound index
    sys : system dict
    cascade : neurons that fire instantaneously after the reset, in order,
        before the decay (see `step`). Empty in the cone regime.

    Returns
    -------
    M : (N, N) transition matrix

    Notes
    -----
    A cascade adds the fixed vector S = sum of the cascading decoders to the
    post-spike readout, so the readout stays affine in xi and the step stays
    projective; S enters only the last column. Without this argument the
    matrix assumes a direct decay, which is wrong for any branch where the
    reset pushed another neuron over threshold.
    """
    DS, ES, al = sys['DS'], sys['ES'], sys['alphas']
    N = sys['N']
    di, dj = DS[bi], DS[bj]
    Ei, Ej = ES[bi], ES[bj]
    ai, aj = al[bi], al[bj]
    S = np.zeros(N)
    for c in cascade:
        S = S + DS[c]

    M = np.zeros((N, N))
    M[:N-1, :N-1] = aj * (Ej.T @ Ei)
    M[:N-1, N-1] = aj * ((ai + 1) * (Ej.T @ di) + (Ej.T @ S)).ravel()
    M[N-1, :N-1] = (Ei.T @ dj).ravel()
    M[N-1, N-1] = (ai + 1) * (dj @ di) + (dj @ S)
    return M


def step_jacobian(xi: np.ndarray, bi: int, sys: dict, rule: str = None):
    """One step, plus the exact Jacobian of the chart map at that point.

    The step is projective, so its derivative is closed form. Writing the
    branch's transition matrix as M = [[A, b], [c^T, d]], the step is
    xi' = (A xi + b)/(c^T xi + d), and differentiating gives

        J = (A - xi' c^T) / (c^T xi + d).

    This is the same object the eigenvalue ratios of Theorem 2 describe for a
    periodic word, evaluated step by step instead of once per cycle, so a
    Lyapunov spectrum built from it comes from the transition matrices rather
    than from numerical differentiation.

    Returns (xi_next, bj, J, cascade); (None, -1, None, cascade) on failure.
    """
    xi = np.atleast_1d(xi).ravel()
    xn, bj, casc = step(xi, bi, sys, rule=rule, return_spikes=True)
    if xn is None:
        return None, -1, None, casc
    N = sys['N']
    M = transition_matrix(bi, bj, sys, cascade=casc)
    A, c, d = M[:N-1, :N-1], M[N-1, :N-1], M[N-1, N-1]
    w = float(c @ xi + d)
    if abs(w) < 1e-300:
        return None, -1, None, casc
    return xn, bj, (A - np.outer(xn, c)) / w, casc


def return_map(word: list, sys: dict) -> np.ndarray:
    """Compose transition matrices for a periodic word.

    Parameters
    ----------
    word : list of bound indices (one period)
    sys : system dict

    Returns
    -------
    M_sigma : (N, N) return map matrix
    """
    N = sys['N']
    M = np.eye(N)
    for k in range(len(word)):
        bi = word[k]
        bj = word[(k + 1) % len(word)]
        M = transition_matrix(bi, bj, sys) @ M
    return M


def eigenvector_orbit(word: list, sys: dict):
    """Predict the full periodic orbit from the return map eigenvector.

    Parameters
    ----------
    word : list of bound indices
    sys : system dict

    Returns
    -------
    orbit : list of (y, bi, xi) tuples — full orbit in R^N
    eigenvalues : sorted eigenvalues of M_sigma
    xi_star : seed tangent coordinate
    """
    N = sys['N']
    DS, ES, al = sys['DS'], sys['ES'], sys['alphas']

    M = return_map(word, sys)
    ev, evc = np.linalg.eig(M)
    idx = np.argsort(np.abs(ev))[::-1]
    ev = ev[idx]
    evc = evc[:, idx]

    # Dehomogenize
    xi_star = (evc[:N-1, 0] / evc[N-1, 0]).real
    if N == 2:
        xi_star = np.atleast_1d(xi_star)

    # Unfold orbit
    orbit = []
    xi_k = xi_star.copy()
    bi_k = word[0]
    for k in range(len(word)):
        xi_flat = np.atleast_1d(xi_k).ravel()
        y_k = al[bi_k] * DS[bi_k] + (ES[bi_k] @ xi_flat).ravel()
        orbit.append((y_k.copy(), bi_k, xi_flat.copy()))
        yp = (al[bi_k] + 1) * DS[bi_k] + ES[bi_k] @ xi_flat
        bj_n = word[(k + 1) % len(word)]
        c_n = al[bj_n] / (DS[bj_n] @ yp)
        xi_k = ES[bj_n].T @ (c_n * yp)
        bi_k = bj_n

    return orbit, ev, xi_star


# ═══════════════════════════════════════════════════════════════
# Admissibility margin
# ═══════════════════════════════════════════════════════════════

def admissibility_margin(word: list, sys: dict) -> float:
    """Compute the admissibility margin for a word.

    The margin is the minimum gap between the target decay factor
    and the best competitor, across all L steps.

    Parameters
    ----------
    word : list of bound indices
    sys : system dict

    Returns
    -------
    m : margin (positive = admissible, negative = inadmissible)
    """
    DS, ES, al, K = sys['DS'], sys['ES'], sys['alphas'], sys['K']
    L = len(word)

    try:
        orbit, ev, xi_star = eigenvector_orbit(word, sys)
    except Exception:
        return -1.0

    margins = []
    for k in range(L):
        y_k, bi_k, xi_k = orbit[k]
        xi_flat = np.atleast_1d(xi_k)
        yp = (al[bi_k] + 1) * DS[bi_k] + ES[bi_k] @ xi_flat

        # Compute all decay factors
        cs = {}
        for j in range(K):
            if al[j] <= 0:
                continue
            dn = DS[j] @ yp
            if dn < 1e-12:
                continue
            cs[j] = al[j] / dn

        target_j = word[(k + 1) % L]
        if target_j not in cs:
            return -1.0
        c_target = cs[target_j]

        for j, c_j in cs.items():
            if j != target_j:
                margins.append(c_target - c_j)

    if not margins:
        return -1.0
    return min(margins)


# ═══════════════════════════════════════════════════════════════
# Firing ratio computation
# ═══════════════════════════════════════════════════════════════

def get_firing_ratios(sys: dict, n_transient=5000, n_record=300,
                      ics=None):
    """Compute firing ratios by simulation, trying multiple ICs.

    Returns
    -------
    ratios : list of K floats (ρ_j for each neuron)
    word : detected periodic word
    """
    K, N = sys['K'], sys['N']
    if ics is None:
        ics = [(np.zeros(N - 1), 0),
               (0.5 * np.ones(N - 1), 1 % K),
               (np.ones(N - 1), 0),
               (1.5 * np.ones(N - 1), 0)]

    for xi0, bi0 in ics:
        xi0 = np.atleast_1d(xi0)[:N-1]
        _, bis = simulate(sys, xi0, bi0, n_transient, n_record)
        if len(set(bis)) >= min(2, K):
            p = find_period(bis)
            word = bis[:p]
            ratios = [sum(1 for w in word if w == j) / p for j in range(K)]
            return ratios, word

    return [1.0] + [0.0] * (K - 1), [0]


# ═══════════════════════════════════════════════════════════════
# First-return map (for K=2 cobweb diagrams)
# ═══════════════════════════════════════════════════════════════

def first_return_B1(xi_start, sys: dict, max_steps=50):
    """First-return map on B1: iterate until next visit to B1.

    Parameters
    ----------
    xi_start : scalar tangent coordinate on B1
    sys : system dict (K=2)
    max_steps : safety limit

    Returns
    -------
    xi_return : tangent coordinate at next B1 visit
    """
    xi = np.atleast_1d(xi_start).astype(float)
    xi_curr, bi = step(xi, 0, sys)
    for _ in range(max_steps):
        if bi == 0:
            return float(xi_curr[0])
        xi_curr, bi = step(xi_curr, bi, sys)
    return float(xi_curr[0])


# ═══════════════════════════════════════════════════════════════
# Distance to limit cycle (for convergence plots)
# ═══════════════════════════════════════════════════════════════

def dist_to_limit_cycle(xi_sim, bi_sim: int, orbit: list):
    """Minimum distance to any orbit point on the same bound.

    Parameters
    ----------
    xi_sim : current tangent coordinate
    bi_sim : current bound index
    orbit : list of (y, bi, xi) from eigenvector_orbit

    Returns
    -------
    d : minimum distance (accounts for phase automatically)
    """
    xi_sim = np.atleast_1d(xi_sim)
    pts = [np.atleast_1d(xi) for y, bi, xi in orbit if bi == bi_sim]
    if not pts:
        return np.nan
    return min(np.linalg.norm(xi_sim - pt) for pt in pts)

# ═══════════════════════════════════════════════════════════════
# Euler step (discrete time) simulation
# ═══════════════════════════════════════════════════════════════

def simulate_euler(input, D, E, F, T, dt=1e-4, leak=100., init=None):

    K, N = D.shape
    x = input['x']
    time = input['time']
    M, nT = x.shape

    # --- set all thresholds to 1 and introduce bias input currents ---
    T0 = np.ones((N,))
    b = T0 - T

    # --- set recurrent weights ---
    W = E @ D

    # initialize main vectors
    s = np.zeros((N, nT))  # spikes
    V = np.zeros((N, nT))  # voltages

    if init is not None:
        if 'y0' in init.keys():
            V[:, 0] = F @ x[:, 0] + E @ init['y0'] + 1 - T
        elif 'r0' in init.keys():
            V[:, 0] = F @ x[:, 0] + W @ init['r0'] - (T - 1)
        elif 'V0' in init.keys():
            V[:, 0] = init['V0']

    # run the Euler method to solve the differential equations
    for t in range(1, nT):

        # calculate feedforward inputs, bias currents and external inputs
        c = (x[:, t] - x[:, t - 1]) / (dt * leak) + x[:, t - 1]
        ff_inputs = F @ c + b

        # calculate recurrent inputs
        rec_inputs = (W @ s[:, t - 1]) / (dt * leak)

        # update membrane potentials
        V[:, t] = V[:, t - 1] + dt * leak * (-V[:, t - 1] + ff_inputs + rec_inputs)

        # Check for spiking - ONLY ONE SPIKE ALLOWED PER TIME STEP
        spiking_indices = np.arange(N)[V[:, t] >= T0]
        if spiking_indices.size > 0:

            spiking_index = np.argmax(V[:, t] - T0)
            if V[spiking_index, t] >= T0[spiking_index]:
                s[spiking_index, t] = 1
            # V[spiking_index, t] = T0[spiking_index]

    if init is not None and 'r0' in init.keys():
        r = exp_filter(s, time, leak=leak, r0=init['r0'], dt=dt)
    else:
        r = exp_filter(s, time, leak=leak, dt=dt)
    y = D @ r
    if init is not None and 'y0' in init.keys():
        y += init['y0'][:, None] * np.tile(np.exp(-leak * time), (K, 1))
    spk_times = get_spike_times(s, dt=dt)

    results = {'V': V,
               's': s,
               'r': r,
               'y': y,
    'spk_times': spk_times}

    return results

def exp_filter(s, times, leak=100., r0=None, dt=1e-4, incl_dt=False):
    if incl_dt:
        s *= dt
    r = np.zeros_like(s)
    if r0 is not None:
        r = r0[:, None] * np.tile(np.exp(-leak*times), (s.shape[0], 1))
    for n in range(s.shape[0]):
        r[n, :] += np.convolve(s[n, :], np.exp(-leak*times), mode='full')[:s.shape[1]]
    return r

def get_spike_times(s, dt=1e-4):
    spk_times = []
    for n in range(s.shape[0]):
        tmp = dt*np.where(s[n, :] == 1)[0]
        spk_times.append(list(tmp))
    return spk_times


# ─────────────────────────────────────────────────────────────────────────────
# Convention B: encoder and decoder as separate vectors
#
# Everything above treats one vector per neuron as both the bound normal and the
# post-spike jump, which is the autoencoder. Convention B separates the two:
#
#     bound      B_i = {y : ENC_i . y = alpha_i}      normal is the ENCODER
#     spike      y -> y + DS_i                        jump is the DECODER
#     weights    W_ij = -ENC_i . DS_j
#
# The autoencoder is the case ENC = DS, and to_convention_B() builds it from an
# existing system, so results obtained above are reproduced exactly.
#
# These are additive. Nothing above calls them, and no existing figure changes.
# They become the primary path once the text is written in the general case; the
# figure scripts that currently carry private copies of this algebra can then
# import from here instead.
#
# NOTE ON KEY NAMES. In the dicts built above, 'ES' holds the TANGENT FRAMES,
# not encoders. A Convention B system therefore uses different keys, so the two
# cannot be passed to each other's functions by accident:
#
#     Convention A (above)   DS = decoders,  ES = tangent frames
#     Convention B (here)    DS = decoders,  ENC = encoders,  U = tangent frames
# ─────────────────────────────────────────────────────────────────────────────


def make_system_B(encoders, decoders, alphas):
    """Construct a Convention B system from encoders, decoders and drives.

    Parameters
    ----------
    encoders : list of K unit vectors in R^N, the spike-bound normals
    decoders : list of K vectors in R^N, the post-spike jumps
    alphas   : list of K suprathreshold drives

    Returns
    -------
    dict with keys: DS, ENC, U, alphas, K, N, convention
    """
    enc = [np.asarray(e, dtype=float).ravel() for e in encoders]
    dec = [np.asarray(d, dtype=float).ravel() for d in decoders]
    if len(enc) != len(dec) or len(enc) != len(alphas):
        raise ValueError('encoders, decoders and alphas must have equal length')
    return {
        'DS': dec, 'ENC': enc, 'U': [tangent_frame(e) for e in enc],
        'alphas': list(alphas), 'K': len(enc), 'N': enc[0].size,
        'convention': 'B',
    }


def to_convention_B(sys: dict) -> dict:
    """Re-express an autoencoder system in Convention B, with ENC = DS.

    The resulting system is dynamically identical, which is what makes the two
    paths comparable: see the equivalence check in the test block below.
    """
    if sys.get('convention') == 'B':
        return sys
    return make_system_B(sys['DS'], sys['DS'], sys['alphas'])


def transition_matrix_B(bi: int, bj: int, sys: dict) -> np.ndarray:
    """N x N transition matrix M_{j<-i} in homogeneous coordinates, Convention B.

    The post-spike readout is y+ = alpha_i ENC_i + DS_i + U_i xi, which is affine
    in xi, so the step stays a ratio of affine maps and the matrix form survives.
    Relative to the autoencoder, the single quantity cos psi_ji splits into
    ENC_j . ENC_i and ENC_j . DS_i, and U_j^T d_i splits into U_j^T ENC_i and
    U_j^T DS_i. Setting ENC = DS collapses each pair and recovers the form above.
    """
    DS, ENC, U, al = sys['DS'], sys['ENC'], sys['U'], sys['alphas']
    N = sys['N']
    ai, aj = al[bi], al[bj]
    Ui, Uj = U[bi], U[bj]

    M = np.zeros((N, N))
    M[:N-1, :N-1] = aj * (Uj.T @ Ui)
    M[:N-1, N-1] = aj * (ai * (Uj.T @ ENC[bi]) + (Uj.T @ DS[bi])).ravel()
    M[N-1, :N-1] = (Ui.T @ ENC[bj]).ravel()
    M[N-1, N-1] = ai * (ENC[bj] @ ENC[bi]) + (ENC[bj] @ DS[bi])
    return M


def step_B(xi, bi: int, sys: dict):
    """One spike-then-decay step, Convention B. Mirrors step() above."""
    DS, ENC, U, al, K = sys['DS'], sys['ENC'], sys['U'], sys['alphas'], sys['K']
    xi = np.atleast_1d(xi).ravel()
    yp = al[bi] * ENC[bi] + DS[bi] + (U[bi] @ xi).ravel()
    bj, bc = -1, -1.0
    for j in range(K):
        if al[j] <= 0:
            continue
        dn = ENC[j] @ yp
        if dn < 1e-12:
            continue
        c = al[j] / dn
        if 0 < c <= 1.0 + 1e-10 and c > bc:
            bc, bj = c, j
    if bj < 0:
        return None, -1
    return (U[bj].T @ (bc * yp)).ravel(), bj


def return_map_B(word: list, sys: dict) -> np.ndarray:
    """Compose Convention B transition matrices around a periodic word."""
    L = len(word)
    M = np.eye(sys['N'])
    for k in range(L):
        M = transition_matrix_B(word[k], word[(k + 1) % L], sys) @ M
    return M


def eigenvector_orbit_B(word: list, sys: dict):
    """Orbit positions and eigenvalues for a word, Convention B.

    Returns (orbit, eigenvalues, xi_star) with the same shape of result as
    eigenvector_orbit() above: orbit is a list of (y, bound index, xi).
    """
    M = return_map_B(word, sys)
    vals, vecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    v = np.real(vecs[:, 0])
    if abs(v[-1]) < 1e-14:
        raise ValueError('dominant eigenvector is at infinity; no orbit point')
    xi_star = v[:-1] / v[-1]

    orbit, xi, bi = [], xi_star, word[0]
    for k in range(len(word)):
        y = sys['alphas'][bi] * sys['ENC'][bi] + (sys['U'][bi] @ xi).ravel()
        orbit.append((y, bi, xi))
        xi, bi = step_B(xi, bi, sys)
        if bi < 0:
            raise ValueError('orbit left the admissible region at step %d' % k)
    return orbit, vals, xi_star
