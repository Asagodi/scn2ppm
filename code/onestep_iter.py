"""Trial 258: is the landing set connected, and where does T^n first cover?

T is the ONE-STEP map, continuous to machine precision on X = L_0 u L_1 u L_2
(shown by onestep_cont.py).  T(X) is inside X, so T^n is continuous on X for
EVERY n: no hypothesis has to be re-checked at any iterate.  Two questions:

  (1) do the three landing segments meet in the plane?  If they do, X is a
      topological arc or circle and the circle-map analogues of Sharkovskii
      are available.
  (2) what is the smallest n for which T^n has a horseshoe?  That bound is
      PER SPIKE, not per return to a section.
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


def plane(xi, b):
    return S['alphas'][b] * S['DS'][b] + (S['ES'][b] @ np.atleast_1d(xi)).ravel()


def Tn(xi, b, n):
    """n steps: (xi, b, symbol word) or (nan, -1, None)."""
    x, bb, w = float(xi), b, []
    for _ in range(n):
        xn, bj, casc = scn.step(np.array([x]), bb, S, return_spikes=True)
        if xn is None:
            return np.nan, -1, None
        w += list(casc) + [bj]
        x, bb = float(xn[0]), bj
    return x, bb, tuple(w)


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

print("(1) do the landing segments meet in the plane?")
ends = {}
for b, (lo, hi) in L.items():
    ends[(b, 'lo')] = plane(lo, b)
    ends[(b, 'hi')] = plane(hi, b)
    print(f"   B_{b}: {plane(lo,b)} .. {plane(hi,b)}")
ks = list(ends)
mind = np.inf
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        if ks[i][0] == ks[j][0]:
            continue
        d = float(np.linalg.norm(ends[ks[i]] - ends[ks[j]]))
        mind = min(mind, d)
        if d < 1e-6:
            print(f"   TOUCH: {ks[i]} and {ks[j]}  distance {d:.2e}")
print(f"   smallest distance between endpoints on different bounds: {mind:.6f}")
print(f"   --> X is {'CONNECTED' if mind < 1e-6 else 'three DISJOINT segments'}")

print("\n(2) covering graph of T^n  (per spike, no 1/n from a section)")
NG = 4000
for n in range(1, 11):
    pieces = []
    for b in (0, 1, 2):
        lo, hi = L[b]
        g = np.linspace(lo, hi, NG)
        out = [Tn(x, b, n) for x in g]
        w = [o[2] for o in out]
        f = np.array([o[0] for o in out])
        cuts = [0] + [k for k in range(1, NG) if w[k] != w[k-1]] + [NG]
        for a, c in zip(cuts[:-1], cuts[1:]):
            if c - a < 8 or w[a] is None:
                continue
            xs, fs = g[a:c], f[a:c]
            pieces.append(dict(b=b, lo=xs[0], hi=xs[-1], tb=out[a][1],
                               flo=min(fs[0], fs[-1]), fhi=max(fs[0], fs[-1])))
    m = len(pieces)
    A = np.zeros((m, m), int)
    MM = np.full((m, m), -np.inf)
    for a in range(m):
        for b2 in range(m):
            if pieces[a]['tb'] != pieces[b2]['b']:
                continue
            mg = min(pieces[b2]['lo'] - pieces[a]['flo'],
                     pieces[a]['fhi'] - pieces[b2]['hi'])
            MM[a, b2] = mg
            A[a, b2] = 1 if mg > 0 else 0
    rho = float(np.abs(np.linalg.eigvals(A.astype(float))).max()) if m else 0.0
    best = None
    for a in range(m):
        for b2 in range(a + 1, m):
            if A[a, a] and A[a, b2] and A[b2, a] and A[b2, b2]:
                mm = min(MM[a, a], MM[a, b2], MM[b2, a], MM[b2, b2])
                if best is None or mm > best[0]:
                    best = (mm, a, b2)
    tag = (f"horseshoe P{best[1]}/P{best[2]} margin {best[0]:.3e}"
           if best else "no horseshoe")
    hb = np.log(rho) / n if rho > 1 else float('nan')
    print(f"   n={n:2d}: {m:3d} pieces  rho={rho:8.5f}  "
          f"log(rho)/n={hb:+.5f}/spike   {tag}")
