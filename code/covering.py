"""Semi-conjugacy onto a shift, by covering relations.

The Conley-index route in its simplest one-dimensional form. Partition the
trapping interval into the branches of the first-return map F: maximal
pieces on which F is continuous and monotone, so each image is an interval.
Build the transition matrix

    A[a][b] = 1   iff   F(branch a) covers branch b.

If A is irreducible with spectral radius rho > 1 then the dynamics admits a
semi-conjugacy onto the subshift of finite type with matrix A, giving
topological entropy at least log rho and uncountably many distinct orbits.
A 2x2 all-ones block is the classical horseshoe, rho = 2, entropy log 2.

This is a covering computation, not yet a proof: the images are evaluated
numerically. Each covering is reported with its MARGIN, the distance by
which the image overshoots the covered branch at each end. A covering with
a healthy margin is what an interval-arithmetic version would certify;
margins near zero are the ones that would fail.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
np.set_printoptions(precision=4, suppress=True)
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
D, AL = z['D'][0], z['alphas'][0]
K = 3
S = {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(AL), 'K': K, 'N': 2}
B0 = 0


ITER = int(__import__('os').environ.get('NITER', '1'))

def ret(xi0, maxit=400):
    """ITER-th return to B_0: (xi_out, itinerary) or (nan, None)."""
    x, b, it, nret = np.array([float(xi0)]), B0, [], 0
    for _ in range(maxit):
        xn, bj, casc = scn.step(x, b, S, return_spikes=True)
        if xn is None:
            return np.nan, None
        it += list(casc) + [bj]
        x, b = xn, bj
        if b == B0:
            nret += 1
            if nret == ITER:
                return float(x[0]), tuple(it)
    return np.nan, None


# trapping interval from the attractor
xi, bi = np.array([0.05]), 0
for _ in range(20000):
    xi, bi = scn.step(xi, bi, S)
vis = []
for _ in range(300000):
    if bi == B0:
        vis.append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
vis = np.array(vis)
LO, HI = vis.min(), vis.max()
print(f"trapping interval J = [{LO:.6f}, {HI:.6f}]\n")

# --- branches: maximal runs of constant itinerary ------------------------
NG = 40000
g = np.linspace(LO, HI, NG)
F = np.empty(NG)
IT = [None] * NG
for i, x in enumerate(g):
    F[i], IT[i] = ret(x)

edges, start = [], 0
for i in range(1, NG):
    if IT[i] != IT[start]:
        edges.append((start, i - 1))
        start = i
edges.append((start, NG - 1))
branches = []
for a, b in edges:
    if b - a < 12 or IT[a] is None:
        continue
    xs, fs = g[a:b + 1], F[a:b + 1]
    ok = np.isfinite(fs)
    if ok.sum() < 12:
        continue
    branches.append(dict(lo=xs[ok][0], hi=xs[ok][-1],
                         flo=min(fs[ok][0], fs[ok][-1]),
                         fhi=max(fs[ok][0], fs[ok][-1]),
                         word=IT[a], n=int(ok.sum())))
print(f"{len(branches)} branches of the first-return map on J:")
for i, br in enumerate(branches):
    print(f"  I{i}: xi in [{br['lo']:+.5f}, {br['hi']:+.5f}]  ->  "
          f"[{br['flo']:+.5f}, {br['fhi']:+.5f}]   word {br['word']}")

# --- covering matrix -----------------------------------------------------
n = len(branches)
A = np.zeros((n, n), dtype=int)
M = np.full((n, n), -np.inf)
for a in range(n):
    for b in range(n):
        m = min(branches[b]['lo'] - branches[a]['flo'],
                branches[a]['fhi'] - branches[b]['hi'])
        M[a, b] = m
        A[a, b] = 1 if m > 0 else 0
print("\ncovering matrix A[a][b] = 1 iff F(I_a) covers I_b:")
print(A)
print("\nmargins (how far the image overshoots each end; >0 means covered):")
print(np.where(np.isfinite(M), M, np.nan))

ev = np.abs(np.linalg.eigvals(A.astype(float)))
rho = ev.max() if len(ev) else 0.0
print(f"\nspectral radius of A: {rho:.5f}")
# The covering is a statement about F^ITER, and h(F^ITER) = ITER * h(F), so
# the bound on the FIRST-return map carries a factor 1/ITER.
_lr = np.log(max(rho, 1e-12))
print(f"entropy lower bound   log(rho)      = {_lr:+.5f}  per F^{ITER}")
print(f"                      log(rho)/{ITER}    = {_lr / ITER:+.5f}  "
      f"nats per RETURN to B_0")

# a 2x2 all-ones block is the classical horseshoe
best = None
for a in range(n):
    for b in range(a + 1, n):
        if A[a, a] and A[a, b] and A[b, a] and A[b, b]:
            mm = min(M[a, a], M[a, b], M[b, a], M[b, b])
            if best is None or mm > best[0]:
                best = (mm, a, b)
if best:
    mm, a, b = best
    print(f"\nHORSESHOE: I{a} and I{b} each cover both, min margin {mm:.3e}")
    print(f"  -> semi-conjugacy onto the full 2-shift: "
          f"h(F) >= log(2)/{ITER} = {np.log(2) / ITER:.4f} per return")
else:
    print("\nno 2x2 all-ones block among these branches")
