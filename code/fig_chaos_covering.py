"""What I_a and I_b are, in the picture the other figures use.

xi is the coordinate ALONG a bound, so an interval of xi is a SEGMENT of
that bound in the readout plane. I_a and I_b are two short segments of the
line B_1, 0.6% and 1.0% of the stretch of B_1 the attractor lands on.

F is the return map to B_1: leave B_1, spike around the cage (1 to 8
spikes, 4.2 on average), come back. F^3 is three of those, here 20 spikes.

  (a) one orbit that starts in I_a and, three returns later, lands in I_b
  (b) the covering as segments of B_1: each image overruns both intervals
  (c) the same covering as a graph, which is panel (b) read vertically

Run: python conley_bridge.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn
import style

style.setup()
plt.rcParams['agg.path.chunksize'] = 20000
scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
D, AL = z['D'][0], z['alphas'][0]
K, N = 3, 2
S = {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(AL), 'K': K, 'N': N}
B0 = 0
COLS = [style.NEURON_COLORS[i] for i in range(K)]
NV, GY = style.COLORS['dark'], style.COLORS['gray']
CA, CB = '#CC79A7', '#000000'          # I_a, I_b: clear of the three bounds
IA = (0.71744, 0.72021)
IB = (0.75530, 0.76047)
E0 = S['ES'][B0].ravel()
NRM = D[B0] / np.linalg.norm(D[B0])


def pt(b, xi):
    return AL[b] * D[b] + S['ES'][b].ravel() * xi


def orbit(x0, nret=3):
    """Path in the plane, the xi of every landing on B_1, spike count."""
    x, b, P, L, ns = np.array([float(x0)]), B0, [], [float(x0)], 0
    while len(L) <= nret:
        y = pt(b, float(np.atleast_1d(x)[0]))
        P += [y, y + D[b]]
        r = scn.step(x, b, S)
        if r[0] is None:
            break
        x, b = r
        ns += 1
        if b == B0:
            L.append(float(np.atleast_1d(x)[0]))
    P.append(pt(B0, L[-1]))
    return np.array(P), L, ns


def F3(x):
    return orbit(x)[1][-1]


# ---- curly brace, drawn in axes coordinates from a data-coordinate span --
_BR_U = np.array([0.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0])
_BR_V = np.array([0.0, 0.0, 0.25, 0.45, 0.5, 0.55, 0.75, 1.0, 1.0])
_BR_C = [Path.MOVETO] + [Path.CURVE3] * 8


def curly_brace(ax, span, pos, width, orient, color, lw=1.3, zorder=7):
    """Brace spanning `span` (data) at `pos` (data), tip `width` (axes frac).

    orient 'right'/'left' gives a vertical brace whose arms sit at x = pos,
    'up'/'down' a horizontal one whose arms sit at y = pos. Returns the tip
    position in axes coordinates, for placing the label.
    """
    vert = orient in ('right', 'left')
    s0 = ax.transLimits.transform((pos, span[0]) if vert else (span[0], pos))
    s1 = ax.transLimits.transform((pos, span[1]) if vert else (span[1], pos))
    u = _BR_U * width * (1.0 if orient in ('right', 'up') else -1.0)
    if vert:
        xy = np.c_[s0[0] + u, s0[1] + _BR_V * (s1[1] - s0[1])]
        tip = (s0[0] + u[4], 0.5 * (s0[1] + s1[1]))
    else:
        xy = np.c_[s0[0] + _BR_V * (s1[0] - s0[0]), s0[1] + u]
        tip = (0.5 * (s0[0] + s1[0]), s0[1] + u[4])
    ax.add_patch(PathPatch(Path(xy, _BR_C), fc='none', ec=color, lw=lw,
                           transform=ax.transAxes, clip_on=False,
                           zorder=zorder, joinstyle='round'))
    return tip


# ---- attractor, for the cage -------------------------------------------
xi, bi = np.array([0.05]), 0
for _ in range(20000):
    xi, bi = scn.step(xi, bi, S)
xs, bs, path = [], [], []
for _ in range(60000):
    x0 = float(np.atleast_1d(xi)[0])
    xs.append(x0)
    bs.append(bi)
    y = pt(bi, x0)
    path += [y, y + D[bi]]
    r = scn.step(xi, bi, S)
    if r[0] is None:
        break
    xi, bi = r
xs, bs, path = np.array(xs), np.array(bs), np.array(path)
land = {b: (xs[bs == b].min(), xs[bs == b].max()) for b in range(K)}
LO, HI = land[B0]

ga, gb = np.linspace(*IA, 700), np.linspace(*IB, 700)
fa = np.array([F3(x) for x in ga])
fb = np.array([F3(x) for x in gb])
IMA, IMB = (fa.min(), fa.max()), (fb.min(), fb.max())
print(f"B_1 landing interval  [{LO:.5f}, {HI:.5f}]   width {HI - LO:.5f}")
for nm, I in (('I_a', IA), ('I_b', IB)):
    print(f"  {nm} = [{I[0]:.5f}, {I[1]:.5f}]  width {I[1] - I[0]:.5f}"
          f"  = {100 * (I[1] - I[0]) / (HI - LO):.2f}% of it")
for nm, IM in (('I_a', IMA), ('I_b', IMB)):
    m = [min(T[0] - IM[0], IM[1] - T[1]) for T in (IA, IB)]
    print(f"  F3({nm}) = [{IM[0]:.5f}, {IM[1]:.5f}]   "
          f"margin over I_a {m[0]:+.5f},  over I_b {m[1]:+.5f}")

# the covering guarantees a point of I_a whose third return is in I_b
TGT = 0.5 * (IB[0] + IB[1])
lo_, hi_ = IA
for _ in range(80):
    mid = 0.5 * (lo_ + hi_)
    if (F3(mid) - TGT) * (F3(lo_) - TGT) <= 0:
        hi_ = mid
    else:
        lo_ = mid
X0 = 0.5 * (lo_ + hi_)
P, L, NS = orbit(X0)
print(f"\norbit drawn: xi_0 = {X0:.6f} in I_a,  {NS} spikes,  "
      f"landings {[round(v, 5) for v in L]}")
print(f"  xi_3 = {L[-1]:.5f} lies in I_b: "
      f"{IB[0] <= L[-1] <= IB[1]}  (itinerary 'ab')")

fig = plt.figure(figsize=(17.8, 6.1))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.22, 1.06], wspace=0.30)
axP, axR, axG = (fig.add_subplot(gs[0, i]) for i in range(3))
V0 = IMA[0] - 0.06 * (IMB[1] - IMA[0])
V1 = IMB[1] + 0.06 * (IMB[1] - IMA[0])

# ---- (a) one orbit, three returns, I_a -> I_b --------------------------
axP.plot(path[:14000, 0], path[:14000, 1], '-', color=NV, lw=0.25,
         alpha=0.09, zorder=1)
for b in range(K):
    lo, hi = land[b]
    axP.plot(*np.c_[pt(b, lo), pt(b, hi)], color=COLS[b], lw=3.4,
             alpha=0.85, zorder=3, solid_capstyle='butt')
    axP.plot(*np.c_[pt(b, lo) + D[b], pt(b, hi) + D[b]], color=COLS[b],
             lw=1.7, ls=(0, (4, 2.2)), alpha=0.6, zorder=3)
axP.plot(P[:, 0], P[:, 1], '-', color=NV, lw=0.95, alpha=0.85, zorder=6)
scale = max(np.ptp(path[:, 0]), np.ptp(path[:, 1]))
# the stretch of B_1 the inset magnifies
axP.plot(*np.c_[pt(B0, V0), pt(B0, V1)], color=COLS[B0], lw=6.0, alpha=1.0,
         zorder=4, solid_capstyle='butt')
for k, x in enumerate(L):
    first, last = k == 0, k == len(L) - 1
    axP.plot(*pt(B0, x), 'o', color=(CA if first else CB if last else NV),
             ms=7.0 if (first or last) else 4.0, mec='white', mew=1.1,
             zorder=8)
# the bound needs no label here: the inset below names it
pad = 0.09 * scale
axP.set_xlim(path[:, 0].min() - pad, path[:, 0].max() + pad)
axP.set_ylim(path[:, 1].min() - 5.4 * pad, path[:, 1].max() + pad)
axP.set_aspect('equal')
axP.set_xlabel('$y_1$')
axP.set_ylabel('$y_2$')
axP.xaxis.set_major_locator(MaxNLocator(4))
axP.yaxis.set_major_locator(MaxNLocator(4))

inA = axP.inset_axes([0.0, 0.015, 1.0, 0.145])
inA.plot([V0, V1], [0, 0], color=COLS[B0], lw=4.5, alpha=0.85,
         solid_capstyle='butt', zorder=2)
for I, c, nm in ((IA, CA, '$I_a$'), (IB, CB, '$I_b$')):
    inA.add_patch(plt.Rectangle((I[0], -0.85), I[1] - I[0], 1.70, color=c,
                                lw=0, zorder=4))
    inA.text(0.5 * (I[0] + I[1]), -1.05, nm, ha='center', va='top',
             fontsize=15, color=c, zorder=7)
for k, x in enumerate(L):
    first, last = k == 0, k == len(L) - 1
    inA.plot(x, 0, 'o', color=(CA if first else CB if last else NV),
             ms=6.0 if (first or last) else 4.0, mec='white', mew=1.3,
             zorder=6)
# only the start and the third return are labelled; the two intermediate
# returns fall within 0.001 of them and would simply overprint
inA.annotate('$\\xi_0$', xy=(L[0], 0), xytext=(4, 13),
             textcoords='offset points', ha='left', va='bottom',
             fontsize=15, color=CA, zorder=7)
inA.annotate('$F^3(\\xi_0)$', xy=(L[-1], 0), xytext=(-2, 13),
             textcoords='offset points', ha='right', va='bottom',
             fontsize=15, color=CB, zorder=7)
# The tangent frame for N=2 is the +90 degree rotation of D, so on this bound
# increasing xi runs LEFT in the plane (E_0 has x-component -0.950). Plotting
# xi left to right would mirror the magnified strip against the plane above it
# and make the two leader lines cross. Reverse the axis so the inset is a
# faithful magnification; panels (b) and (c) keep xi increasing rightwards,
# since they are coordinate plots rather than pictures of the plane.
inA.set_xlim(V1, V0)
inA.set_ylim(-2.6, 2.4)
inA.set_yticks([])
inA.set_xticks([])
for s_ in inA.spines.values():
    s_.set_visible(False)
inA.text(V1, -2.55, '$\\mathcal{B}_1$ magnified', fontsize=15, color=GY,
         ha='left', va='bottom')
for xe in (V0, V1):
    axP.add_artist(ConnectionPatch(
        xyA=pt(B0, xe), coordsA=axP.transData,
        xyB=(xe, 1.35), coordsB=inA.transData,
        color=GY, lw=0.8, ls=(0, (3, 2)), zorder=2))
style.panel_label(axP, 'a', subtitle='Chaotic orbit',
                  fontsize=16, y=1.02)

# ---- (b) the covering, as segments of B_1 ------------------------------
axR.plot([V0, V1], [3, 3], color=COLS[B0], lw=6, alpha=0.65,
         solid_capstyle='butt', zorder=2)
axR.text(V0 - 0.015 * (V1 - V0), 3, 'on $\\mathcal{B}_1$', ha='right',
         va='center', fontsize=15, color=COLS[B0])
for yv, (a, b), c, lab in ((2, IMA, CA, '$F^3(I_a)$'),
                           (1, IMB, CB, '$F^3(I_b)$')):
    axR.plot([a, b], [yv, yv], color=c, lw=10, alpha=0.85,
             solid_capstyle='butt', zorder=4)
    for e in (a, b):
        axR.plot([e, e], [yv - 0.32, yv + 0.32], color=c, lw=2.0, zorder=5)
    axR.text(V0 - 0.015 * (V1 - V0), yv, lab, ha='right', va='center',
             fontsize=15, color=c)
for I, c, nm in ((IA, CA, '$I_a$'), (IB, CB, '$I_b$')):
    axR.add_patch(plt.Rectangle((I[0], 2.72), I[1] - I[0], 0.56, color=c,
                                lw=0, zorder=6))
    axR.text(0.5 * (I[0] + I[1]), 3.42, nm, ha='center', va='bottom',
             color=c, fontsize=15)
    for e in I:
        axR.plot([e, e], [0.55, 3.28], color=c, lw=0.9, ls=':', zorder=3)
axR.set_xlim(V0, V1)
axR.set_ylim(0.45, 4.85)
axR.set_yticks([])
axR.set_xlabel('$\\xi$  along  $\\mathcal{B}_1$')
axR.xaxis.set_major_locator(MaxNLocator(5))
for s_ in ('top', 'right', 'left'):
    axR.spines[s_].set_visible(False)
ins = axR.inset_axes([0.50, 0.80, 0.48, 0.125])
ins.plot([LO, HI], [0, 0], color=COLS[B0], lw=4, alpha=0.65,
         solid_capstyle='butt')
ins.add_patch(plt.Rectangle((V0, -0.62), V1 - V0, 1.24, fc='none', ec=NV,
                            lw=1.0))
ins.set_xlim(LO - 0.03 * (HI - LO), HI + 0.03 * (HI - LO))
ins.set_ylim(-1, 1)
ins.set_yticks([])
ins.set_xticks([0.4, 0.6, 0.8])
ins.tick_params(labelsize=14, length=2, pad=1)
for s_ in ('top', 'right', 'left'):
    ins.spines[s_].set_visible(False)
ins.set_title('all of $\\mathcal{B}_1$', fontsize=15, pad=3)
style.panel_label(axR, 'b', subtitle='Each image overruns both intervals',
                  fontsize=16, y=1.02)

# ---- (c) the same, as a graph ------------------------------------------
GW = IB[1] - IA[0]
GX0 = IA[0] - 0.15 * GW
GX1 = IB[1] + 0.62 * GW        # room on the right for the image braces
for I, c in ((IA, CA), (IB, CB)):
    axG.axvspan(I[0], I[1], color=c, alpha=0.22, lw=0, zorder=1)
    axG.axhspan(I[0], I[1], color=c, alpha=0.22, lw=0, zorder=1)
axG.plot(ga, fa, color=CA, lw=3.0, zorder=5, solid_capstyle='butt')
axG.plot(gb, fb, color=CB, lw=3.0, zorder=5, solid_capstyle='butt')
ylo, yhi = min(IMA[0], IMB[0], IA[0]), max(IMA[1], IMB[1], IB[1])
ypad = 0.10 * (yhi - ylo)
axG.set_xlim(GX0, GX1)
axG.set_ylim(ylo - 3.3 * ypad, yhi + ypad)   # braces live under the data
# the two bands, named where they sit on the vertical axis
for I, c, nm in ((IA, CA, '$I_a$'), (IB, CB, '$I_b$')):
    axG.text(GX0 + 0.020 * (GX1 - GX0), 0.5 * (I[0] + I[1]), nm, color=c,
             fontsize=15, ha='left', va='center', zorder=7)
# domains below, images at the right: the vertical extent of each curve
for I, (a, b), c, nd, ni in ((IA, IMA, CA, '$I_a$', '$F^3(I_a)$'),
                             (IB, IMB, CB, '$I_b$', '$F^3(I_b)$')):
    t = curly_brace(axG, I, ylo - 0.035 * (yhi - ylo), 0.045, 'down', c)
    axG.text(t[0], t[1] - 0.045, nd, color=c, fontsize=15, ha='center',
             va='top', transform=axG.transAxes, zorder=7)
    t = curly_brace(axG, (a, b), I[1] + 0.075 * GW, 0.045, 'right', c)
    axG.text(t[0] + 0.062, t[1], ni, color=c, fontsize=15, ha='center',
             va='center', rotation=90, transform=axG.transAxes, zorder=7)
axG.set_xlabel('$\\xi$')
axG.set_ylabel('$F^3(\\xi)$')
axG.xaxis.set_major_locator(MaxNLocator(3))
axG.yaxis.set_major_locator(MaxNLocator(4))
style.panel_label(axG, 'c', subtitle='The graph shows the same covering',
                  fontsize=16, y=1.02)

fig.tight_layout(w_pad=2.4, pad=0.5)
out = os.path.join(here, '..', 'figures', 'conley_bridge.png')
fig.savefig(out, dpi=190, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('saved', out)
