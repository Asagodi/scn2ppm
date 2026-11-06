"""Is the first-return map continuous?  Measure the jump at every branch boundary.

A branch is a maximal run of constant symbolic word.  Where the word changes the
map MAY still be continuous: the itinerary switches but the value can match.  So
we bisect each boundary to ~1e-13 and compare the one-sided values.  A jump at
machine precision would mean Sharkovskii's hypothesis is met after all.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
B0 = 0


def make(i):
    D, AL = z['D'][i], z['alphas'][i]
    return {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
            'alphas': list(AL), 'K': 3, 'N': 2}


def ret(xi0, S, maxit=400):
    """First return to B_0: (xi_out, word) or (nan, None)."""
    x, b, it = np.array([float(xi0)]), B0, []
    for _ in range(maxit):
        xn, bj, casc = scn.step(x, b, S, return_spikes=True)
        if xn is None:
            return np.nan, None
        it += list(casc) + [bj]
        x, b = xn, bj
        if b == B0:
            return float(x[0]), tuple(it)
    return np.nan, None


for i in range(len(z['D'])):
    S = make(i)
    print(f"\n=== net {i}  (trial {int(z['trial'][i])}, lambda={z['lam'][i]:+.5f}) ===")

    xi, bi = np.array([0.05]), 0
    for _ in range(20000):
        r = scn.step(xi, bi, S)
        if r[0] is None:
            break
        xi, bi = r
    vis = []
    for _ in range(200000):
        if bi == B0:
            vis.append(float(np.atleast_1d(xi)[0]))
        r = scn.step(xi, bi, S)
        if r[0] is None:
            break
        xi, bi = r
    vis = np.array(vis)
    LO, HI = vis.min(), vis.max()
    print(f"  trapping interval [{LO:.6f}, {HI:.6f}]   width {HI-LO:.6f}"
          f"   ({len(vis)} returns sampled)")

    NG = 20000
    g = np.linspace(LO, HI, NG)
    F = np.empty(NG); W = [None]*NG
    for k, x in enumerate(g):
        F[k], W[k] = ret(x, S)

    bnd = [k for k in range(1, NG) if W[k] != W[k-1]]
    print(f"  branches: {len(bnd)+1}   word changes: {len(bnd)}")

    jumps = []
    for k in bnd:
        lo, hi = g[k-1], g[k]
        wl = W[k-1]
        for _ in range(60):                      # localise the boundary
            mid = 0.5*(lo+hi)
            if ret(mid, S)[1] == wl:
                lo = mid
            else:
                hi = mid
        fl, fr = ret(lo, S)[0], ret(hi, S)[0]
        if np.isfinite(fl) and np.isfinite(fr):
            jumps.append((0.5*(lo+hi), abs(fr-fl), W[k-1] is None or W[k] is None))
    if jumps:
        j = np.array([a[1] for a in jumps])
        print(f"  |jump| at the word changes:  min {j.min():.3e}   "
              f"median {np.median(j):.3e}   max {j.max():.3e}")
        print(f"  jumps below 1e-9: {(j < 1e-9).sum()} of {len(j)}"
              f"   (interval width {HI-LO:.4f}; largest jump is "
              f"{100*j.max()/(HI-LO):.1f}% of it)")
        for xb, jj, esc in sorted(jumps, key=lambda a: -a[1])[:6]:
            print(f"     xi = {xb:.9f}   |jump| = {jj:.4e}")
    np.savez(os.path.join(here, f'jumps_net{i}.npz'),
             g=g, F=F, LO=LO, HI=HI,
             bx=np.array([a[0] for a in jumps]),
             bj=np.array([a[1] for a in jumps]))
