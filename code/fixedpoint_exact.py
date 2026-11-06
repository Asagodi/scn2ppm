"""Does T have a fixed point at all?  Exact, from the piece data.

A fixed point needs a piece whose target bound equals its own bound, and whose
image interval meets its domain.  Only one of the five pieces maps a bound to
itself, so this is decidable without any grid.
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
    r = scn.step(np.array([float(xi)]), b, S, return_spikes=True)
    return (np.nan, -1, None) if r[0] is None else (float(r[0][0]), r[1],
                                                    (tuple(r[2]), r[1]))


xi, bi = np.array([0.05]), 0
for _ in range(50000):
    xi, bi = scn.step(xi, bi, S)
land = {0: [], 1: [], 2: []}
for _ in range(600000):
    land[bi].append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
L = {b: (min(v), max(v)) for b, v in land.items()}

NG = 40000
print("pieces of T, and whether a fixed point is possible on each:")
for b in (0, 1, 2):
    lo, hi = L[b]
    g = np.linspace(lo, hi, NG)
    out = [one(x, b) for x in g]
    sym = [o[2] for o in out]
    f = np.array([o[0] for o in out])
    cuts = [0] + [k for k in range(1, NG) if sym[k] != sym[k-1]] + [NG]
    for a, c in zip(cuts[:-1], cuts[1:]):
        if c - a < 6:
            continue
        xs, fs = g[a:c], f[a:c]
        tb = out[a][1]
        flo, fhi = min(fs[0], fs[-1]), max(fs[0], fs[-1])
        if tb != b:
            verdict = f"target B_{tb} != B_{b}, cannot be fixed"
        elif fhi < xs[0] or flo > xs[-1]:
            gap = max(xs[0] - fhi, flo - xs[-1])
            verdict = f"image DISJOINT from domain by {gap:.6f}, no fixed point"
        else:
            verdict = "image meets domain, fixed point possible"
        print(f"  B_{b} xi in [{xs[0]:+.6f}, {xs[-1]:+.6f}] -> B_{tb} "
              f"[{flo:+.6f}, {fhi:+.6f}]")
        print(f"     {verdict}")
