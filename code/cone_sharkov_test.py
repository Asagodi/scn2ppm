"""Direct Sharkovskii test on an inhibitory cone network with a period-3 orbit.

3 is the MINIMUM of Sharkovskii's order, so for a continuous interval map a
period-3 orbit forces a periodic point of every period.  This does not involve
entropy or Lyapunov exponents, so it cannot be dodged by coexistence of an
attracting cycle with a chaotic repeller.  Census the periods directly.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
ANG = [12.13, 49.59, 76.84]
ALP = [1.516, 1.361, 0.304]
D = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in ANG]
S = {'DS': D, 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(ALP), 'K': 3, 'N': 2}
W = -np.array([[D[i] @ D[j] for j in range(3)] for i in range(3)])
print("off-diagonal W:", [f"{W[i][j]:+.4f}" for i in range(3)
                          for j in range(3) if i != j])


def Tn(xi, b, n):
    x, bb, w = float(xi), b, []
    for _ in range(n):
        r = scn.step(np.array([x]), bb, S, return_spikes=True)
        if r[0] is None:
            return np.nan, -1, None
        w += list(r[2]) + [r[1]]
        x, bb = float(r[0][0]), r[1]
    return x, bb, tuple(w)


# admissible range on each bound: survives 60 steps
RNG = {}
for b in (0, 1, 2):
    g = np.linspace(-25, 25, 20001)
    ok = [x for x in g if Tn(x, b, 60)[2] is not None]
    RNG[b] = (min(ok), max(ok)) if ok else None
    print(f"  B_{b}: non-escaping xi in "
          f"{'none' if not ok else f'[{min(ok):+.4f}, {max(ok):+.4f}]'}")

NG = 20000
found = {}
for n in range(1, 9):
    reps = []
    for b in (0, 1, 2):
        if RNG[b] is None:
            continue
        lo, hi = RNG[b]
        g = np.linspace(lo, hi, NG)
        out = [Tn(x, b, n) for x in g]
        w = [o[2] for o in out]
        cuts = [0] + [k for k in range(1, NG) if w[k] != w[k-1]] + [NG]
        for a, c in zip(cuts[:-1], cuts[1:]):
            if c - a < 4 or w[a] is None or out[a][1] != b:
                continue
            xs = g[a:c]
            d = np.array([out[k][0] for k in range(a, c)]) - xs
            for k in np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
                p, q = xs[k], xs[k+1]
                for _ in range(70):
                    m = 0.5 * (p + q)
                    rm = Tn(m, b, n)
                    if rm[2] != w[a]:
                        break
                    if (rm[0] - m) * (Tn(p, b, n)[0] - p) <= 0:
                        q = m
                    else:
                        p = m
                x0 = 0.5 * (p + q)
                if any(abs(Tn(x0, b, mm)[0] - x0) < 1e-10 and Tn(x0, b, mm)[1] == b
                       for mm in range(1, n) if n % mm == 0):
                    continue
                if not any(any(abs(Tn(x0, b, k2)[0] - y0) < 1e-7
                               and Tn(x0, b, k2)[1] == c2 for k2 in range(n))
                           for c2, y0 in reps):
                    reps.append((b, x0))
    found[n] = len(reps)
    print(f"  period {n}: {len(reps)} orbit(s)" + ("" if reps else "   NONE"))

ps = [k for k, v in found.items() if v]
print(f"\nperiods present up to 8: {ps}")
print(f"periods absent  up to 8: {[k for k in range(1,9) if not found[k]]}")
if 3 in ps:
    missing = [k for k in range(1, 9) if not found[k]]
    print("\nperiod 3 is present.  Sharkovskii would force EVERY period.")
    print(f"  periods missing anyway: {missing}"
          if missing else "  no period missing in this range")
