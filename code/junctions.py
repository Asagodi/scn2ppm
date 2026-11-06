"""Do the three landing segments glue into ONE interval with ONE continuous map?

Two separate questions, and the second is the one that decides it.

  (a) Geometry: do the segment endpoints coincide in the plane?
  (b) Dynamics: at a shared endpoint, does T give the SAME image from both
      sides?  T takes (xi, b), and its first act is to undo neuron b's reset:
      y+ = (alpha_b + 1) D_b + E_b xi.  That depends on b, not on y alone.  So
      a corner lying on two bounds has two candidate images, and the glued map
      is well defined only if they agree.
"""
import os, sys, itertools
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


def T(xi, b):
    xn, bj = scn.step(np.array([float(xi)]), b, S)
    if xn is None:
        return None, -1
    return float(xn[0]), bj


# long orbit for tight endpoints
xi, bi = np.array([0.05]), 0
for _ in range(50000):
    xi, bi = scn.step(xi, bi, S)
land = {0: [], 1: [], 2: []}
for _ in range(2000000):
    land[bi].append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
L = {b: (min(v), max(v)) for b, v in land.items()}
print("landing intervals (2e6 steps):")
for b, (lo, hi) in L.items():
    print(f"  B_{b}: [{lo:+.9f}, {hi:+.9f}]")

print("\n(a) endpoint coincidences in the plane:")
ends = [(b, e, L[b][0 if e == 'lo' else 1]) for b in (0, 1, 2)
        for e in ('lo', 'hi')]
pairs = []
for (b1, e1, x1), (b2, e2, x2) in itertools.combinations(ends, 2):
    if b1 == b2:
        continue
    d = float(np.linalg.norm(plane(x1, b1) - plane(x2, b2)))
    if d < 1e-3:
        pairs.append((b1, e1, x1, b2, e2, x2, d))
        print(f"  B_{b1}.{e1}  ~  B_{b2}.{e2}   |dy| = {d:.3e}")
if not pairs:
    print("  none within 1e-3")

print("\n(b) is the map single-valued at those shared points?")
for b1, e1, x1, b2, e2, x2, d in pairs:
    (n1, t1), (n2, t2) = T(x1, b1), T(x2, b2)
    p1 = plane(n1, t1) if n1 is not None else None
    p2 = plane(n2, t2) if n2 is not None else None
    print(f"  corner B_{b1}.{e1} / B_{b2}.{e2}  (geometric gap {d:.2e})")
    print(f"     T from the B_{b1} side -> B_{t1}, y' = {p1}")
    print(f"     T from the B_{b2} side -> B_{t2}, y' = {p2}")
    if p1 is not None and p2 is not None:
        print(f"     |difference of the two images| = "
              f"{float(np.linalg.norm(p1 - p2)):.6f}")
