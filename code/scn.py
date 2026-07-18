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

def step(xi: np.ndarray, bi: int, sys: dict):
    """One spike-then-decay step.

    Parameters
    ----------
    xi : tangent coordinate on bound bi
    bi : current bound index
    sys : system dict (DS, ES, alphas, K)

    Returns
    -------
    xi_next : tangent coordinate on next bound
    bj : next bound index
    """
    DS, ES, al, K = sys['DS'], sys['ES'], sys['alphas'], sys['K']
    xi = np.atleast_1d(xi).ravel()
    yp = (al[bi] + 1) * DS[bi] + (ES[bi] @ xi).ravel()
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
        return None, -1
    xi_next = (ES[bj].T @ (bc * yp)).ravel()
    return xi_next, bj


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

def transition_matrix(bi: int, bj: int, sys: dict) -> np.ndarray:
    """N×N transition matrix M_{j←i} in homogeneous coordinates.

    Parameters
    ----------
    bi : source bound index
    bj : target bound index
    sys : system dict

    Returns
    -------
    M : (N, N) transition matrix
    """
    DS, ES, al = sys['DS'], sys['ES'], sys['alphas']
    N = sys['N']
    di, dj = DS[bi], DS[bj]
    Ei, Ej = ES[bi], ES[bj]
    ai, aj = al[bi], al[bj]

    M = np.zeros((N, N))
    M[:N-1, :N-1] = aj * (Ej.T @ Ei)
    M[:N-1, N-1] = aj * (ai + 1) * (Ej.T @ di)
    M[N-1, :N-1] = (Ei.T @ dj).ravel()
    M[N-1, N-1] = (ai + 1) * (dj @ di)
    return M


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
