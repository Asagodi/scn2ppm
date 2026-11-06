"""Why can an SCN never be a continuous self-map of ONE interval?

There are only two ways to get a 1-D map out of an SCN, and each loses one of
Sharkovskii's two hypotheses:

  (A) the ONE-STEP map.  Domain = the three bounds, a DISCONNECTED set.  The
      state is a point plus a label saying which neuron just fired, so it is
      three segments, not one interval.
  (B) the FIRST-RETURN map to a single bound.  Domain is now one interval,
      but the number of spikes before the orbit comes back is not constant,
      and the map jumps exactly where that count changes.

The claim is that (B)'s return time is non-constant precisely when the bound
sequence branches, which is also what makes the dynamics non-trivial.  Test it
on both a chaotic network and a regular inhibitory cone network.
"""
import os, sys
import numpy as np

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn

scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))


def mk(D, AL):
    return {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
            'alphas': list(AL), 'K': 3, 'N': 2}


ANG = [12.13, 49.59, 76.84]
CONE = mk([np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
           for a in ANG], [1.516, 1.361, 0.304])
NETS = [("trial 258 (chaotic)", mk(z['D'][0], z['alphas'][0])),
        ("cone net (regular, period 3)", CONE)]


def ret(xi, b, S, maxit=200):
    """Return to bound b: (xi_out, n_spikes) or (nan, -1)."""
    x, bb, n = float(xi), b, 0
    for _ in range(maxit):
        r = scn.step(np.array([x]), bb, S)
        if r[0] is None:
            return np.nan, -1
        x, bb, n = float(r[0][0]), r[1], n + 1
        if bb == b:
            return x, n
    return np.nan, -1


for name, S in NETS:
    print(f"\n=== {name} ===")

    # domain: non-escaping xi on each bound
    RG = {}
    for b in (0, 1, 2):
        g = np.linspace(-12, 12, 4001)
        ok = [x for x in g if scn.step(np.array([x]), b, S)[0] is not None
              and ret(x, b, S)[1] > 0]
        RG[b] = (min(ok), max(ok)) if ok else None

    # (A) how many pieces does the ONE-STEP map have on each bound?
    tot = 0
    for b in (0, 1, 2):
        if RG[b] is None:
            continue
        lo, hi = RG[b]
        g = np.linspace(lo, hi, 4000)
        tgt = []
        for x in g:
            r = scn.step(np.array([float(x)]), b, S, return_spikes=True)
            tgt.append(None if r[0] is None else (tuple(r[2]), r[1]))
        np_ = 1 + sum(tgt[k] != tgt[k-1] for k in range(1, len(tgt)))
        tot += np_
        print(f"  (A) one-step map on B_{b}: {np_} piece(s) over "
              f"[{lo:+.3f}, {hi:+.3f}]")
    print(f"      domain is {3} disjoint segments, so NOT one interval")

    # (B) return map to B_0: how many distinct return times?
    lo, hi = RG[0]
    g = np.linspace(lo, hi, 6000)
    out = [ret(x, 0, S) for x in g]
    ns = np.array([o[1] for o in out])
    fs = np.array([o[0] for o in out])
    good = ns > 0
    vals, cnts = np.unique(ns[good], return_counts=True)
    print(f"  (B) return time to B_0 takes values {list(vals)} "
          f"(shares {np.round(cnts / cnts.sum(), 3).tolist()})")
    sw = [k for k in range(1, len(g)) if good[k] and good[k-1]
          and ns[k] != ns[k-1]]
    if not sw:
        print("      return time is CONSTANT -> return map can be continuous")
    else:
        jj = [abs(fs[k] - fs[k-1]) for k in sw]
        print(f"      return time changes at {len(sw)} point(s); "
              f"the return map jumps there by "
              f"{min(jj):.4f} to {max(jj):.4f}")
        print("      -> one interval, but NOT continuous")
