"""Markov graph of the ONE-STEP map for trial 258, where it is continuous.

The pieces are the maximal runs of constant symbol on the landing interval of
each bound.  Each piece maps onto an interval of ONE bound, so the covering
relation is read off directly.  The bound this gives is PER SPIKE, with no
1/n: the covering is at the first iterate, not the third.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
D, AL = z['D'][0], z['alphas'][0]
S = {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(AL), 'K': 3, 'N': 2}


def one(xi, b):
    xn, bj, casc = scn.step(np.array([float(xi)]), b, S, return_spikes=True)
    if xn is None:
        return np.nan, -1, None
    return float(xn[0]), bj, (tuple(casc), bj)


xi, bi = np.array([0.05]), 0
for _ in range(20000):
    xi, bi = scn.step(xi, bi, S)
land = {0: [], 1: [], 2: []}
for _ in range(300000):
    land[bi].append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
L = {b: (min(v), max(v)) for b, v in land.items()}

NG = 20000
pieces = []
for b in (0, 1, 2):
    lo, hi = L[b]
    g = np.linspace(lo, hi, NG)
    out = [one(x, b) for x in g]
    sym = [o[2] for o in out]
    f = np.array([o[0] for o in out])
    cuts = [0] + [k for k in range(1, NG) if sym[k] != sym[k-1]] + [NG]
    for a, c in zip(cuts[:-1], cuts[1:]):
        if c - a < 10:
            continue
        xs, fs = g[a:c], f[a:c]
        pieces.append(dict(b=b, lo=xs[0], hi=xs[-1], tb=sym[a][1],
                           flo=min(fs[0], fs[-1]), fhi=max(fs[0], fs[-1]),
                           casc=sym[a][0]))

print(f"{len(pieces)} pieces of the one-step map:")
for n, p in enumerate(pieces):
    print(f"  P{n}: B_{p['b']} xi in [{p['lo']:+.6f}, {p['hi']:+.6f}]  ->  "
          f"B_{p['tb']} [{p['flo']:+.6f}, {p['fhi']:+.6f}]   cascade {p['casc']}")

n = len(pieces)
A = np.zeros((n, n), int)
M = np.full((n, n), -np.inf)
for a in range(n):
    for b in range(n):
        if pieces[a]['tb'] != pieces[b]['b']:
            continue
        m = min(pieces[b]['lo'] - pieces[a]['flo'],
                pieces[a]['fhi'] - pieces[b]['hi'])
        M[a, b] = m
        A[a, b] = 1 if m > 0 else 0
print("\ncovering matrix A[a][b] = 1 iff T(P_a) covers P_b:")
print(A)
print("\nmargins:")
print(np.where(np.isfinite(M), M, np.nan))

rho = float(np.abs(np.linalg.eigvals(A.astype(float))).max())
print(f"\nspectral radius rho = {rho:.6f}")
print(f"h_top >= log rho   = {np.log(max(rho,1e-12)):+.6f}  nats PER SPIKE")
print(f"compare: lambda    = {z['lam'][0]:+.6f}  nats per spike")
