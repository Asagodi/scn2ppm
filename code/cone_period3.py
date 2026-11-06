"""The inhibitory cone: a period-3 orbit in a provably regular network.

If Sharkovskii applied to the SCN one-step map, a period-3 orbit would force
periodic points of EVERY period, and through Block-Guckenheimer-Misiurewicz-
Young it would force h_top >= log((1+sqrt 5)/2) = 0.4812 per spike.  A cone
network is regular, so finding a period-3 orbit in one refutes the hypothesis
outright, with no reference to junctions or landing sets.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'


def build(angles_deg, alphas):
    D = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
         for a in angles_deg]
    return {'DS': D, 'ES': [scn.tangent_frame(d) for d in D],
            'alphas': list(alphas), 'K': len(D), 'N': 2}


for angles, alphas in (((20., 50., 80.), (0.8, 0.9, 1.0)),
                       ((10., 45., 80.), (1.0, 1.0, 1.0)),
                       ((30., 60., 90.), (0.7, 1.1, 0.9))):
    S = build(angles, alphas)
    D = S['DS']
    W = -np.array([[D[i] @ D[j] for j in range(3)] for i in range(3)])
    off = [W[i][j] for i in range(3) for j in range(3) if i != j]
    print(f"\n=== decoders at {angles} deg, alphas {alphas} ===")
    print(f"  off-diagonal W in [{min(off):+.4f}, {max(off):+.4f}]  "
          f"-> {'ALL INHIBITORY (cone)' if max(off) < 0 else 'has excitation'}")

    xi, bi = np.array([0.03]), 0
    ok = True
    for _ in range(20000):
        r = scn.step(xi, bi, S)
        if r[0] is None:
            ok = False
            break
        xi, bi = r
    if not ok:
        print("  orbit escapes, skipping")
        continue

    # period of the spike sequence after transient
    seq, states = [], []
    for _ in range(600):
        seq.append(bi)
        states.append((bi, float(np.atleast_1d(xi)[0])))
        xi, bi = scn.step(xi, bi, S)
    per = None
    for p in range(1, 60):
        if all(states[k][0] == states[k+p][0]
               and abs(states[k][1] - states[k+p][1]) < 1e-10
               for k in range(0, 200)):
            per = p
            break
    print(f"  spike word settles to period {per}:  "
          f"{seq[200:200+(per or 6)]}")

    lam = scn.lyapunov(xi, bi, S, n=200000) if hasattr(scn, 'lyapunov') else None
    if lam is None:
        J = np.eye(1)
        x, b = xi, bi
        tot, n = 0.0, 0
        for _ in range(200000):
            r = scn.step_jacobian(x, b, S)
            if r[0] is None:
                break
            x, b, Jk = r[0], r[1], r[2]
            tot += np.log(abs(float(np.atleast_2d(Jk)[0, 0])) + 1e-300)
            n += 1
        lam = tot / max(n, 1)
    print(f"  Lyapunov exponent = {lam:+.6f} per spike  "
          f"-> {'CHAOTIC' if lam > 1e-4 else 'REGULAR'}")

phi = 0.5 * (1 + 5 ** 0.5)
print(f"\nBGMY for an odd period q: h_top >= log(lambda_q).")
print(f"  q = 3  -> lambda_3 = {phi:.6f} (golden ratio), log = {np.log(phi):.4f} per spike")
for q in (15,):
    c = np.zeros(q + 1)
    c[0] = 1.0
    c[2] = -2.0
    c[q] = -1.0
    r = np.roots(c)
    lq = max(x.real for x in r if abs(x.imag) < 1e-9 and x.real > 0)
    print(f"  q = {q} -> lambda_{q} = {lq:.6f}, log = {np.log(lq):.4f} per spike")
