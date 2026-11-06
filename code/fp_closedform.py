"""Where CAN a fixed point of the one-step map sit?  Closed form, then check.

A fixed point on bound b with no cascade: undoing b's reset gives y+ = y + D_b,
then the decay to the next bound multiplies by c = alpha_j / (D_j . y+).  For
c*(y + D_b) = y we need (c-1) y = -c D_b, so y is PARALLEL to D_b.  On bound b
that forces xi = 0, i.e. y = alpha_b D_b, the foot of the perpendicular from
the origin.  So T can only be fixed at xi = 0 on a bound, which is the state
where neuron b fires tonically on its own.

So the question is not 'is there a fixed point somewhere unexamined', it is
'is xi = 0 fixed, and is it in the landing set'.  Both are checkable exactly.
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

print("(i) is xi = 0 a fixed point on each bound?")
for b in (0, 1, 2):
    r = scn.step(np.array([0.0]), b, S, return_spikes=True)
    if r[0] is None:
        print(f"   B_{b}: xi=0 escapes (runaway cascade)")
        continue
    xo, bj, casc = float(r[0][0]), r[1], list(r[2])
    print(f"   B_{b}: xi=0  ->  B_{bj}, xi' = {xo:+.9f}   cascade {casc}   "
          f"{'FIXED' if (bj == b and abs(xo) < 1e-12) else 'not fixed'}")

print("\n(ii) wide scan for fixed points, far outside the attractor")
for b in (0, 1, 2):
    g = np.linspace(-40, 40, 400001)
    hits = []
    prev = None
    for x in g:
        r = scn.step(np.array([float(x)]), b, S)
        v = (float(r[0][0]) - x) if (r[0] is not None and r[1] == b) else np.nan
        if prev is not None and np.isfinite(prev[1]) and np.isfinite(v) \
                and prev[1] * v < 0:
            lo, hi = prev[0], x
            for _ in range(60):
                m = 0.5 * (lo + hi)
                rm = scn.step(np.array([m]), b, S)
                if rm[0] is None or rm[1] != b:
                    break
                if (float(rm[0][0]) - m) * (prev[1]) > 0:
                    lo = m
                else:
                    hi = m
            hits.append(0.5 * (lo + hi))
        prev = (x, v)
    print(f"   B_{b}: {len(hits)} fixed point(s) in xi in [-40, 40]"
          + (f"  at {['%+.6f' % h for h in hits]}" if hits else ""))

print("\n(iii) where is the attractor's landing set, for comparison")
xi, bi = np.array([0.05]), 0
for _ in range(50000):
    xi, bi = scn.step(xi, bi, S)
land = {0: [], 1: [], 2: []}
for _ in range(400000):
    land[bi].append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
for b in (0, 1, 2):
    lo, hi = min(land[b]), max(land[b])
    print(f"   B_{b}: landing set [{lo:+.6f}, {hi:+.6f}]   "
          f"contains xi=0: {lo <= 0 <= hi}")
