"""First-return maps of the three chaotic K=3 networks, with their jumps.

Every one is a Lorenz-type map: increasing on each branch, with a finite set
of genuine discontinuities.  The one-sided limits are drawn as dots at true
scale, so the vertical gap between a pair IS the jump.  Sharkovskii's theorem
needs continuity on the whole interval and so has no purchase on any of these.
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
NV, GY = style.COLORS['dark'], style.COLORS['gray']
CJ = '#D55E00'
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


fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6))

for i, ax in enumerate(axes):
    S = make(i)
    d = np.load(os.path.join(here, f'jumps_net{i}.npz'))
    g, F, LO, HI, bx = d['g'], d['F'], float(d['LO']), float(d['HI']), d['bx']

    # one-sided limits at each boundary, by bisection
    sides = []
    for xb in bx:
        lo, hi = xb - 1e-7, xb + 1e-7
        wl = ret(lo, S)[1]
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            if ret(mid, S)[1] == wl:
                lo = mid
            else:
                hi = mid
        sides.append((float(lo), ret(lo, S)[0], float(hi), ret(hi, S)[0]))

    # plot each branch separately so nothing is drawn across a jump
    cuts = np.searchsorted(g, np.sort(bx))
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(g)]):
        s = slice(a, b)
        ok = np.isfinite(F[s])
        if ok.sum() > 3:
            ax.plot(g[s][ok], F[s][ok], color=NV, lw=2.2, zorder=4,
                    solid_capstyle='butt')
    for xb in bx:
        ax.axvline(xb, color=GY, lw=0.8, ls=':', zorder=2)
    ax.plot([LO, HI], [LO, HI], color=GY, lw=1.0, ls='--', zorder=3)

    jm = int(np.argmax([abs(s[3] - s[1]) for s in sides]))
    for k, (xl, fl, xr, fr) in enumerate(sides):
        # the gap at true scale: a segment joining the two one-sided limits
        ax.plot([0.5 * (xl + xr)] * 2, [fl, fr], color=CJ, lw=1.4,
                ls=(0, (2, 2)), zorder=5, clip_on=False)
        ax.plot([xl, xr], [fl, fr], 'o', ms=6, mfc=CJ if k == jm else 'none',
                mec=CJ, mew=1.6, color=CJ, ls='none', zorder=6, clip_on=False)
    xl, fl, xr, fr = sides[jm]
    right = 0.5 * (xl + xr) > 0.5 * (LO + HI)      # label on the roomier side
    ax.annotate(rf'$|\Delta| = {abs(fr-fl):.3f}$',
                xy=(0.5 * (xl + xr), 0.5 * (fl + fr)),
                xytext=(-9 if right else 9, 0),
                textcoords='offset points', color=CJ, fontsize=15,
                ha='right' if right else 'left', va='center', zorder=7)

    pad = 0.015 * (HI - LO)
    ax.set_xlim(LO - pad, HI + pad)
    ax.set_ylim(LO - pad, HI + pad)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$F(\xi)$')

    # the smallest jump of the three nets, magnified, to show it is real
    if i == 2:
        k = int(np.argmin([abs(s[3] - s[1]) for s in sides]))
        xl, fl, xr, fr = sides[k]
        xc, w = 0.5 * (xl + xr), 0.010
        yl, yh = min(fl, fr), max(fl, fr)
        yp = 0.55 * (yh - yl)
        axi = ax.inset_axes([0.55, 0.07, 0.42, 0.27])
        m = (g > xc - w) & (g < xc + w) & np.isfinite(F)
        for sel in ((g < xc) & m, (g > xc) & m):
            axi.plot(g[sel], F[sel], color=NV, lw=2.2, solid_capstyle='butt')
        axi.plot([xc, xc], [fl, fr], color=CJ, lw=1.4, ls=(0, (2, 2)))
        axi.plot([xl, xr], [fl, fr], 'o', ms=5, mfc='none', mec=CJ, mew=1.6,
                 ls='none')
        axi.set_xlim(xc - w, xc + w)
        axi.set_ylim(yl - yp, yh + yp)
        axi.xaxis.set_major_locator(MaxNLocator(2))
        axi.yaxis.set_major_locator(MaxNLocator(2))
        axi.tick_params(labelsize=14)
        axi.set_title(rf'$|\Delta| = {abs(fr - fl):.3f}$', fontsize=15,
                      color=CJ, pad=4)
        for sp in ('top', 'right'):
            axi.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    # lambda belongs in the caption; the trial id keeps the panels distinct
    style.panel_label(ax, 'abc'[i],
                      subtitle=f"Trial {int(z['trial'][i])}",
                      fontsize=17, y=1.03)

fig.tight_layout(w_pad=2.2, pad=0.6)
for ext in ('png', 'pdf'):
    out = os.path.join(here, '..', 'figures', f'return_jumps.{ext}')
    fig.savefig(out, bbox_inches='tight')
    print('saved', out)
