"""Sign of the slope on each branch.

If every branch is increasing AND the map were continuous, F would be a
continuous non-decreasing self-map of an interval.  Such a map has zero
topological entropy: every orbit converges to a fixed point.  So for these
networks the discontinuities are not a nuisance, they ARE the chaos.
"""
import os
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))

for i in range(3):
    d = np.load(os.path.join(here, f'jumps_net{i}.npz'))
    g, F, bx = d['g'], d['F'], np.sort(d['bx'])
    LO, HI = float(d['LO']), float(d['HI'])
    cuts = np.searchsorted(g, bx)
    print(f"\n=== net {i} (trial {int(z['trial'][i])}) ===")
    allpos = True
    for n, (a, b) in enumerate(zip(np.r_[0, cuts], np.r_[cuts, len(g)])):
        x, y = g[a:b], F[a:b]
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 8:
            print(f"  branch {n}: too few points ({len(x)})")
            continue
        x, y = x[2:-2], y[2:-2]                 # drop grid points adjacent to a cut
        s = np.diff(y) / np.diff(x)
        allpos &= bool((s > 0).all())
        print(f"  branch {n}: width {x[-1]-x[0]:.5f}  slope  min {s.min():+.4f}"
              f"  median {np.median(s):+.4f}  max {s.max():+.4f}"
              f"   {'increasing' if (s > 0).all() else 'HAS A TURNING POINT'}")
    print(f"  --> all branches increasing: {allpos}")
