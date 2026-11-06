"""Which periods does the one-step map T actually have, for trial 258?

T^n is continuous on each piece of constant word, and each piece is Mobius
hence monotone, so a fixed point of T^n lives on a piece whose target bound
equals its own bound and on which T^n(x) - x changes sign.  Enumerate those.

If Sharkovskii applied, a period 3 would force EVERY period.  Checking the
period set against that is a direct empirical test of the hypothesis.
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


def Tn(xi, b, n):
    x, bb, w = float(xi), b, []
    for _ in range(n):
        r = scn.step(np.array([x]), bb, S, return_spikes=True)
        if r[0] is None:
            return np.nan, -1, None
        w += list(r[2]) + [r[1]]
        x, bb = float(r[0][0]), r[1]
    return x, bb, tuple(w)


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

NG = 6000
found = {}
for n in range(1, 17):
    orbits = []
    for b in (0, 1, 2):
        lo, hi = L[b]
        g = np.linspace(lo, hi, NG)
        out = [Tn(x, b, n) for x in g]
        w = [o[2] for o in out]
        cuts = [0] + [k for k in range(1, NG) if w[k] != w[k-1]] + [NG]
        for a, c in zip(cuts[:-1], cuts[1:]):
            if c - a < 6 or w[a] is None or out[a][1] != b:
                continue
            xs = g[a:c]
            d = np.array([out[k][0] for k in range(a, c)]) - xs
            sg = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
            for k in sg:
                p, q = xs[k], xs[k+1]
                for _ in range(80):
                    m = 0.5 * (p + q)
                    r2 = Tn(m, b, n)
                    if r2[2] != w[a] or r2[1] != b:
                        break
                    if (r2[0] - m) * (Tn(p, b, n)[0] - p) <= 0:
                        q = m
                    else:
                        p = m
                x0 = 0.5 * (p + q)
                orbits.append((b, x0))
    # minimal period: drop points already periodic under a proper divisor
    prim = []
    for b, x0 in orbits:
        if any(abs(Tn(x0, b, m)[0] - x0) < 1e-9 and Tn(x0, b, m)[1] == b
               for m in range(1, n) if n % m == 0):
            continue
        prim.append((b, x0))
    # one representative per orbit
    reps = []
    for b, x0 in prim:
        if not any(any(abs(Tn(x0, b, k)[0] - y0) < 1e-7 and Tn(x0, b, k)[1] == c
                       for k in range(n))
                   for c, y0 in reps):
            reps.append((b, x0))
    found[n] = len(reps)
    print(f"  period {n:2d}: {len(reps)} orbit(s)" + ("" if reps else "   NONE"))

ps = sorted(k for k, v in found.items() if v)
print(f"\nperiods present up to 16: {ps}")
print(f"periods ABSENT up to 16:  {[k for k in range(1,17) if not found[k]]}")
