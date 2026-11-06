"""Why no SCN is a continuous self-map of one interval.

Taking a return map to a single bound buys a connected domain and loses
continuity, because the number of spikes before the orbit comes back is not
constant.  Plot the return time above the return map: the map jumps at
exactly the points where the count changes, and nowhere else.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn
import style

style.setup()
scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
D, AL = z['D'][0], z['alphas'][0]
S = {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(AL), 'K': 3, 'N': 2}
NV, GY, CJ = style.COLORS['dark'], style.COLORS['gray'], '#D55E00'
B0 = 0


def ret(xi, maxit=200):
    x, b, n = float(xi), B0, 0
    for _ in range(maxit):
        r = scn.step(np.array([x]), b, S)
        if r[0] is None:
            return np.nan, -1
        x, b, n = float(r[0][0]), r[1], n + 1
        if b == B0:
            return x, n
    return np.nan, -1


xi, bi = np.array([0.05]), 0
for _ in range(50000):
    xi, bi = scn.step(xi, bi, S)
vis = []
for _ in range(400000):
    if bi == B0:
        vis.append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
LO, HI = min(vis), max(vis)

NG = 12000
g = np.linspace(LO, HI, NG)
out = [ret(x) for x in g]
F = np.array([o[0] for o in out])
N = np.array([o[1] for o in out])
cuts = [k for k in range(1, NG) if N[k] != N[k-1]]
print("return times present:", sorted(set(N.tolist())))
print("changes at:", [f"{0.5*(g[k-1]+g[k]):.6f}" for k in cuts])

fig, (axN, axF) = plt.subplots(2, 1, figsize=(8.6, 8.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.0, 1.9]))

for a, b in zip([0] + cuts, cuts + [NG]):
    axN.plot(g[a:b], N[a:b], color=NV, lw=2.6, solid_capstyle='butt',
             zorder=4)
for k in cuts:
    for ax in (axN, axF):
        ax.axvline(0.5 * (g[k-1] + g[k]), color=CJ, lw=1.2, ls=(0, (2, 2)),
                   zorder=2)
axN.set_ylabel('spikes to return')
axN.yaxis.set_major_locator(MaxNLocator(4, integer=True))

for a, b in zip([0] + cuts, cuts + [NG]):
    sl = slice(a, b)
    ok = np.isfinite(F[sl])
    if ok.sum() > 3:
        axF.plot(g[sl][ok], F[sl][ok], color=NV, lw=2.6,
                 solid_capstyle='butt', zorder=4)
for k in cuts:
    axF.plot([0.5 * (g[k-1] + g[k])] * 2, [F[k-1], F[k]], color=CJ, lw=1.6,
             ls=(0, (2, 2)), zorder=5)
    axF.plot([g[k-1], g[k]], [F[k-1], F[k]], 'o', ms=6, mfc='white', mec=CJ,
             mew=1.8, ls='none', zorder=6)
axF.plot([LO, HI], [LO, HI], color=GY, lw=1.0, ls='--', zorder=3)
axF.set_xlim(LO, HI)
axF.set_xlabel(r'$\xi$')
axF.set_ylabel(r'$F(\xi)$')
axF.xaxis.set_major_locator(MaxNLocator(4))
axF.yaxis.set_major_locator(MaxNLocator(4))

style.panel_label(axN, 'a', subtitle='The return time is not constant',
                  fontsize=17, y=1.04)
style.panel_label(axF, 'b', subtitle='The map jumps at those same points',
                  fontsize=17, y=1.03)

fig.tight_layout(h_pad=2.2, pad=0.6)
for ext in ('png', 'pdf'):
    out_ = os.path.join(here, '..', 'figures', f'why_discontinuous.{ext}')
    fig.savefig(out_, bbox_inches='tight')
    print('saved', out_)
