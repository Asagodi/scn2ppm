"""Find an inhibitory cone network with a period-3 orbit, and check its entropy.

Sharkovskii puts 3 at the BOTTOM of the order, so a period-3 orbit forces every
other period, and BGMY then forces h_top >= log(golden ratio) = 0.4812 per
spike.  A cone network with a period-3 orbit and a negative Lyapunov exponent
therefore refutes the hypothesis on its own.
"""
import os, sys, itertools
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'


def build(ang, al):
    D = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in ang]
    return {'DS': D, 'ES': [scn.tangent_frame(d) for d in D],
            'alphas': list(al), 'K': 3, 'N': 2}


def analyse(S, nburn=20000, nmeas=400):
    xi, bi = np.array([0.03]), 0
    for _ in range(nburn):
        r = scn.step(xi, bi, S)
        if r[0] is None:
            return None
        xi, bi = r
    st = []
    for _ in range(nmeas):
        st.append((bi, float(np.atleast_1d(xi)[0])))
        r = scn.step(xi, bi, S)
        if r[0] is None:
            return None
        xi, bi = r
    per = None
    for p in range(1, 40):
        if all(st[k][0] == st[k+p][0] and abs(st[k][1] - st[k+p][1]) < 1e-11
               for k in range(0, nmeas - 40)):
            per = p
            break
    if per is None:
        return None
    tot, n, x, b = 0.0, 0, xi, bi
    for _ in range(100000):
        r = scn.step_jacobian(x, b, S)
        if r[0] is None:
            break
        x, b = r[0], r[1]
        tot += np.log(abs(float(np.atleast_2d(r[2])[0, 0])) + 1e-300)
        n += 1
    return per, tot / max(n, 1), [s[0] for s in st[:per]]


best = []
rng = np.random.default_rng(0)
tried = 0
for _ in range(4000):
    a0 = rng.uniform(0, 20)
    spread = rng.uniform(10, 85)                 # keep all three in a cone
    ang = sorted([a0, a0 + rng.uniform(5, spread), a0 + spread])
    al = rng.uniform(0.3, 1.6, 3)
    S = build(ang, al)
    D = S['DS']
    off = [-(D[i] @ D[j]) for i in range(3) for j in range(3) if i != j]
    if max(off) >= 0:
        continue
    tried += 1
    r = analyse(S)
    if r and r[0] == 3:
        best.append((ang, al, r))
        if len(best) >= 4:
            break

print(f"cone networks tested: {tried};  with a period-3 orbit: {len(best)}")
for ang, al, (per, lam, word) in best:
    print(f"\n  decoders {np.round(ang,2)} deg, alphas {np.round(al,3)}")
    print(f"    spike word {word}, period {per}")
    print(f"    Lyapunov exponent {lam:+.6f} per spike  "
          f"-> {'CHAOTIC' if lam > 1e-4 else 'REGULAR'}")
    print(f"    BGMY from period 3 would demand h_top >= 0.4812 per spike")
