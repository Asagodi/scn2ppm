"""Trial 258: the landing set is one arc, but T is not a function on it.

(a) The three landing segments meet end to end in the plane, so the landing
    set is a single arc.  Walk it with an arc-length coordinate s.
(b) Plot T in that coordinate.  Above each junction the graph has TWO points,
    because T's first act undoes the reset of whichever neuron just fired and
    a junction lies on two bounds.  A function has one value per input, so the
    arc does not carry one.  The dashed grey diagonal is s' = s: the graph
    never meets it, so T has no fixed point, which a continuous self-map of an
    interval would be forced to have.
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
BC = style.NEURON_COLORS[:3]
NV, GY, CJ = style.COLORS['dark'], style.COLORS['gray'], '#D55E00'


def plane(xi, b):
    return S['alphas'][b] * S['DS'][b] + (S['ES'][b] @ np.atleast_1d(xi)).ravel()


def one(xi, b):
    r = scn.step(np.array([float(xi)]), b, S, return_spikes=True)
    if r[0] is None:
        return np.nan, -1, None
    return float(r[0][0]), r[1], (tuple(r[2]), r[1])


xi, bi = np.array([0.05]), 0
for _ in range(50000):
    xi, bi = scn.step(xi, bi, S)
land = {0: [], 1: [], 2: []}
for _ in range(600000):
    land[bi].append(float(np.atleast_1d(xi)[0]))
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
L = {b: (min(v), max(v)) for b, v in land.items()}

ORDER = [1, 0, 2]                       # B_1.hi = B_0.lo, B_0.hi = B_2.lo
ln = {b: L[b][1] - L[b][0] for b in ORDER}
off, acc = {}, 0.0
for b in ORDER:
    off[b] = acc
    acc += ln[b]
STOT = acc
JUNC = [off[0], off[2]]                 # the two shared endpoints


def s_of(xi, b):
    return off[b] + (xi - L[b][0])


fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.8, 6.0))

# ---- (a) the arc in the plane -------------------------------------------
for b in ORDER:
    p0, p1 = plane(L[b][0], b), plane(L[b][1], b)
    axA.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BC[b], lw=6,
             solid_capstyle='butt', zorder=3)
    mid = 0.5 * (p0 + p1)
    axA.annotate(rf'$B_{b}$', xy=(mid[0], mid[1]), xytext=(8, 8),
                 textcoords='offset points', color=BC[b], fontsize=17,
                 ha='left', va='bottom', zorder=6)
for b in (0, 2):
    c = plane(L[b][0], b)
    axA.plot(c[0], c[1], 'o', ms=12, mfc='white', mec=NV, mew=2.2, zorder=5)
axA.set_xlabel(r'$y_1$')
axA.set_ylabel(r'$y_2$')
axA.set_aspect('equal')
axA.xaxis.set_major_locator(MaxNLocator(4))
axA.yaxis.set_major_locator(MaxNLocator(4))
style.panel_label(axA, 'a', subtitle='The landing set is one arc',
                  fontsize=17, y=1.03)

# ---- (b) T in the arc-length coordinate ----------------------------------
NG = 8000
for b in ORDER:
    g = np.linspace(L[b][0], L[b][1], NG)
    out = [one(x, b) for x in g]
    sym = [o[2] for o in out]
    si = np.array([s_of(x, b) for x in g])
    so = np.array([s_of(o[0], o[1]) if o[2] else np.nan for o in out])
    cuts = [0] + [k for k in range(1, NG) if sym[k] != sym[k-1]] + [NG]
    for a, c in zip(cuts[:-1], cuts[1:]):
        if c - a < 4:
            continue
        axB.plot(si[a:c], so[a:c], color=BC[b], lw=2.6, zorder=4,
                 solid_capstyle='butt')
axB.plot([0, STOT], [0, STOT], color=GY, lw=1.1, ls='--', zorder=2)
for b in ORDER[1:]:
    axB.axvline(off[b], color=GY, lw=0.9, ls=':', zorder=1)

for prev, nxt in ((1, 0), (0, 2)):      # the two shared endpoints
    sj = off[nxt]
    v1 = s_of(*one(L[prev][1], prev)[:2])
    v2 = s_of(*one(L[nxt][0], nxt)[:2])
    axB.plot([sj, sj], [v1, v2], color=CJ, lw=1.6, ls=(0, (2, 2)), zorder=5)
    axB.plot([sj, sj], [v1, v2], 'o', ms=8, mfc='white', mec=CJ, mew=2.0,
             ls='none', zorder=6)
    axB.annotate(rf'two values', xy=(sj, 0.5 * (v1 + v2)), xytext=(10, 0),
                 textcoords='offset points', color=CJ, fontsize=15,
                 ha='left', va='center', zorder=7)

for b in ORDER:
    axB.annotate(rf'$B_{b}$', xy=(off[b] + 0.5 * ln[b], 0), xytext=(0, -34),
                 textcoords='offset points', color=BC[b], fontsize=16,
                 ha='center', va='top', annotation_clip=False, zorder=7)
axB.set_xlim(0, STOT)
axB.set_ylim(0, STOT)
axB.set_xlabel(r'arc length $s$')
axB.set_ylabel(r'$s$ of the next landing')
axB.xaxis.set_major_locator(MaxNLocator(4))
axB.yaxis.set_major_locator(MaxNLocator(4))
style.panel_label(axB, 'b', subtitle='Each junction carries two values',
                  fontsize=17, y=1.03)

fig.tight_layout(w_pad=2.6, pad=0.6)
for ext in ('png', 'pdf'):
    out = os.path.join(here, '..', 'figures', f'arc_junctions.{ext}')
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
