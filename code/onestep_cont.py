"""Is the ONE-STEP map continuous on the full landing region of the cage?

The first-return map to a single bound is discontinuous, but that may be an
artifact of the section: the number of steps needed to come back changes, and
that alone makes a return map jump even when every individual step is smooth.
So test the one-step map instead, in the PLANE, where continuity is
parametrisation-free.  Domain: the landing interval on each of the three
bounds.  At every point where the target bound or the cascade changes, measure
|y'(+) - y'(-)|.  Near zero everywhere means the map is continuous on the
glued boundary, and then the periodic orbits live on a connected 1-D set with
a continuous map, which is what Sharkovskii needs.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))


def make(i):
    D, AL = z['D'][i], z['alphas'][i]
    return {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
            'alphas': list(AL), 'K': 3, 'N': 2}


def plane(xi, b, S):
    return S['alphas'][b] * S['DS'][b] + (S['ES'][b] @ np.atleast_1d(xi)).ravel()


def one(xi, b, S):
    """One step: (xi', b', plane point, symbol = (cascade, b'))."""
    xn, bj, casc = scn.step(np.array([float(xi)]), b, S, return_spikes=True)
    if xn is None:
        return np.nan, -1, None, None
    return float(xn[0]), bj, plane(float(xn[0]), bj, S), (tuple(casc), bj)


for i in range(3):
    S = make(i)
    print(f"\n=== net {i} (trial {int(z['trial'][i])}) ===")

    # landing intervals on each bound, from a long orbit
    xi, bi = np.array([0.05]), 0
    for _ in range(20000):
        r = scn.step(xi, bi, S)
        if r[0] is None:
            break
        xi, bi = r
    land = {0: [], 1: [], 2: []}
    for _ in range(200000):
        land[bi].append(float(np.atleast_1d(xi)[0]))
        r = scn.step(xi, bi, S)
        if r[0] is None:
            break
        xi, bi = r
    L = {b: (min(v), max(v)) for b, v in land.items() if v}
    for b, (lo, hi) in L.items():
        print(f"  landing interval on B_{b}: [{lo:+.6f}, {hi:+.6f}]"
              f"   width {hi-lo:.6f}   ({len(land[b])} visits)")

    NG = 12000
    worst = 0.0
    for b, (lo, hi) in L.items():
        g = np.linspace(lo, hi, NG)
        sym = [one(x, b, S)[3] for x in g]
        sw = [k for k in range(1, NG) if sym[k] != sym[k-1]]
        print(f"  B_{b}: {len(sw)+1} pieces, {len(sw)} symbol changes")
        for k in sw:
            a_, c_ = g[k-1], g[k]
            s0 = sym[k-1]
            for _ in range(60):
                m = 0.5 * (a_ + c_)
                if one(m, b, S)[3] == s0:
                    a_ = m
                else:
                    c_ = m
            pl, pr = one(a_, b, S)[2], one(c_, b, S)[2]
            if pl is None or pr is None:
                print(f"     xi* = {0.5*(a_+c_):+.8f}   escapes, no comparison")
                continue
            d = float(np.linalg.norm(pr - pl))
            worst = max(worst, d)
            print(f"     xi* = {0.5*(a_+c_):+.8f}   |y'(+) - y'(-)| = {d:.3e}"
                  f"   {sym[k-1]} -> {sym[k]}")
    print(f"  --> largest plane jump of the one-step map: {worst:.3e}")
