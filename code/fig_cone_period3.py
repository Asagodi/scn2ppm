"""A pure inhibitory cone network: period 3 exists, no other period does.

Sharkovskii puts 3 at the bottom of the order, so for a continuous interval
map a period-3 orbit forces a periodic point of EVERY period.  This network
has period 3 and nothing else up to 8, which refutes the hypothesis without
any appeal to entropy or to Lyapunov exponents.
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
BC = style.NEURON_COLORS[:3]
NV, GY, CJ = style.COLORS['dark'], style.COLORS['gray'], '#D55E00'

ANG = [12.13, 49.59, 76.84]
ALP = [1.516, 1.361, 0.304]
D = [np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))]) for a in ANG]
S = {'DS': D, 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(ALP), 'K': 3, 'N': 2}
PRESENT = {1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}


def plane(xi, b):
    return ALP[b] * D[b] + S['ES'][b].ravel() * float(xi)


xi, bi = np.array([0.03]), 0
for _ in range(20000):
    xi, bi = scn.step(xi, bi, S)
cyc = []
for _ in range(3):
    cyc.append((bi, float(np.atleast_1d(xi)[0])))
    xi, bi = scn.step(xi, bi, S)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.8))

# ---- (a) the cone and the period-3 cycle --------------------------------
P = np.array([plane(x, b) for b, x in cyc])
# zoom on the cycle: in the full plane it is a speck near the bound crossings
ctr = P.mean(axis=0)
half = 1.5 * max(np.abs(P - ctr).max(), 1e-3)
X0, X1 = ctr[0] - half, ctr[0] + half
Y0, Y1 = ctr[1] - half, ctr[1] + half
for b in (0, 1, 2):
    t = np.linspace(-40, 40, 2)
    seg = np.array([plane(tt, b) for tt in t])
    axA.plot(seg[:, 0], seg[:, 1], color=BC[b], lw=2.4, zorder=3)
    # label where the line leaves the zoom window, nudged inside
    d_ = seg[1] - seg[0]
    ts = [(X0 - seg[0][0]) / d_[0], (X1 - seg[0][0]) / d_[0],
          (Y0 - seg[0][1]) / d_[1], (Y1 - seg[0][1]) / d_[1]]
    hits = [seg[0] + tt * d_ for tt in ts
            if X0 - 1e-9 <= (seg[0] + tt * d_)[0] <= X1 + 1e-9
            and Y0 - 1e-9 <= (seg[0] + tt * d_)[1] <= Y1 + 1e-9]
    if len(hits) >= 2:
        # label a fifth of the way along the visible chord, offset sideways,
        # so it cannot be clipped by the axes edge
        h0 = max(hits, key=lambda q: q[1])
        h1 = min(hits, key=lambda q: q[1])
        lab = h0 + 0.20 * (h1 - h0)
        v = h1 - h0
        perp = np.array([-v[1], v[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-12)
        axA.annotate(rf'$\mathcal{{B}}_{b+1}$', xy=tuple(lab),
                     xytext=(16 * perp[0], 16 * perp[1]),
                     textcoords='offset points',
                     color=BC[b], fontsize=16, ha='center', va='center',
                     zorder=6)
for k in range(3):
    a, b2 = P[k], P[(k + 1) % 3]
    axA.annotate('', xy=tuple(b2), xytext=tuple(a),
                 arrowprops=dict(arrowstyle='-|>', color=NV, lw=2.0,
                                 shrinkA=9, shrinkB=9, mutation_scale=15),
                 zorder=5)
axA.plot(P[:, 0], P[:, 1], 'o', ms=11, mfc='white', mec=NV, mew=2.2, zorder=7)
axA.set_xlabel(r'$y_1$')
axA.set_ylabel(r'$y_2$')
axA.set_aspect('equal')
axA.set_xlim(X0, X1)
axA.set_ylim(Y0, Y1)
axA.xaxis.set_major_locator(MaxNLocator(3))
axA.yaxis.set_major_locator(MaxNLocator(3))
style.panel_label(axA, 'a', subtitle='A cone network cycles with period 3',
                  fontsize=17, y=1.03)

# ---- (b) which periods exist --------------------------------------------
ks = sorted(PRESENT)
for k in ks:
    have = PRESENT[k] > 0
    axB.plot([k], [PRESENT[k]], 'o', ms=13,
             mfc=CJ if have else 'white', mec=CJ if have else GY, mew=2.2,
             zorder=5)
axB.axhline(1, color=GY, lw=1.0, ls='--', zorder=2)
axB.annotate('Sharkovskii demands every period', xy=(5.0, 1.0),
             xytext=(0, 9), textcoords='offset points', color=GY,
             fontsize=15, ha='center', va='bottom', zorder=6)
axB.set_xlim(0.4, 8.6)
axB.set_ylim(-0.22, 1.5)
axB.set_xlabel('period')
axB.set_ylabel('orbits found')
axB.xaxis.set_major_locator(MaxNLocator(8, integer=True))
axB.yaxis.set_major_locator(MaxNLocator(2, integer=True))
style.panel_label(axB, 'b', subtitle='Only period 3 is present',
                  fontsize=17, y=1.03)

fig.tight_layout(w_pad=2.6, pad=0.6)
for ext in ('png', 'pdf'):
    out = os.path.join(here, '..', 'figures', f'cone_period3.{ext}')
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
