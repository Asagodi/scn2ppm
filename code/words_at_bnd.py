"""What changes across each branch boundary, and does it explain the jump size?

With RESET_RULE='immediate' a cascade is simultaneous, so merely REORDERING two
spikes inside one cascade cannot change the landing point.  A jump therefore
needs a real change: a spike that happens on one side and not the other, or a
different bound reached.  Print the two words side by side to see which it is.
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


def ret(x0, S, maxit=400):
    x, b, it = np.array([float(x0)]), B0, []
    for _ in range(maxit):
        xn, bj, casc = scn.step(x, b, S, return_spikes=True)
        if xn is None:
            return np.nan, None
        it += list(casc) + [bj]
        x, b = xn, bj
        if b == B0:
            return float(x[0]), tuple(it)
    return np.nan, None


for i in range(3):
    S = make(i)
    d = np.load(os.path.join(here, f'jumps_net{i}.npz'))
    LO, HI = float(d['LO']), float(d['HI'])
    print(f"\n=== net {i} (trial {int(z['trial'][i])})   interval width {HI-LO:.5f} ===")
    for xb, jj in zip(d['bx'], d['bj']):
        lo, hi = xb - 1e-7, xb + 1e-7
        wl = ret(lo, S)[1]
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            if ret(mid, S)[1] == wl:
                lo = mid
            else:
                hi = mid
        wl, wr = ret(lo, S)[1], ret(hi, S)[1]
        same_len = len(wl) == len(wr)
        same_set = sorted(wl) == sorted(wr)
        print(f"  xi* = {xb:+.9f}   |jump| = {jj:.5f} "
              f"({100*jj/(HI-LO):.1f}% of interval)")
        print(f"     left  ({len(wl)} symbols): {wl}")
        print(f"     right ({len(wr)} symbols): {wr}")
        print(f"     same length: {same_len}   same multiset (pure reorder): {same_set}")
