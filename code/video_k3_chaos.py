"""Chaotic K=3, N=2 network (trial 258), traced in time.

Layout fixes over v4, which ran long enough to break it:
  - voltages now scroll over a short window, since at 70 tau the full trace
    was a solid band with no resolvable threshold crossing;
  - the raster keeps the whole run, so fine detail and global structure are
    both on screen;
  - the connectivity graph gets its own grid cell instead of being squeezed
    under the raster, where its title overlapped the time axis and the
    weight labels sat on the nodes.

Geometry is exact: a spike happens at the landing point y' = c y+, on the
firing neuron's bound, and the path is built from event points (decay is a
straight ray) so it meets every bound precisely.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, r"C:\Users\abel_\Documents\Lab\Projects\scn_symbolic\scn2ppm\code")
import scn
import style

style.setup()
plt.rcParams['animation.ffmpeg_path'] = \
    r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\ffmpeg.exe"
scn.RESET_RULE = 'immediate'
here = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(here, 'k3_chaotic.npz'))
D, AL = z['D'][0], z['alphas'][0]
K = 3
S = {'DS': [d for d in D], 'ES': [scn.tangent_frame(d) for d in D],
     'alphas': list(AL), 'K': K, 'N': 2}
COLS = [style.NEURON_COLORS[i] for i in range(K)]
NV = style.COLORS['dark']
RED = style.COLORS['red']
BLUE = style.COLORS['accent']


def pt(b, xi):
    return AL[b] * D[b] + S['ES'][b].ravel() * xi


N_INT, FPS, SECONDS = 210, 30, 58
VWIN = 9.0                      # voltage window, in tau

xi, bi = np.array([0.05]), 0
for _ in range(4000):
    xi, bi = scn.step(xi, bi, S)

segs, t = [], 0.0
for _ in range(N_INT):
    yp = pt(bi, float(np.atleast_1d(xi)[0])) + D[bi]
    xn, bj, casc = scn.step(xi, bi, S, return_spikes=True)
    if xn is None:
        break
    for c_ in casc:
        segs.append((t, yp.copy(), t, yp.copy(), c_, True))
        yp = yp + D[c_]
    c = AL[bj] / float(D[bj] @ yp)
    tau = -np.log(max(c, 1e-12))
    segs.append((t, yp.copy(), t + tau, (c * yp).copy(), bj, False))
    t += tau
    xi, bi = xn, bj
T = t
err = max(abs(float(D[s[4]] @ s[3]) - AL[s[4]]) for s in segs if not s[5])
print(f"{len(segs)} intervals, T={T:.1f} tau, "
      f"max |D_j.y_land - alpha_j| = {err:.2e}")

spk_t = np.array([s[2] for s in segs])
spk_p = np.array([s[3] for s in segs])
spk_n = np.array([s[4] for s in segs])
NF = FPS * SECONDS
ts = np.linspace(0, T * 0.9995, NF)


def state(tq):
    i = min(np.searchsorted(spk_t, tq, side='right'), len(segs) - 1)
    s = segs[i]
    y = s[1].copy() if s[5] else s[1] * np.exp(-(tq - s[0]))
    P = [segs[0][1]]
    for k in range(i):
        P.append(segs[k][3])
        P.append(segs[k + 1][1])
    P.append(y)
    return y, np.array(P)


Y = np.array([state(q)[0] for q in ts])
V = np.array([[AL[j] - D[j] @ y for j in range(K)] for y in Y])

fig = plt.figure(figsize=(13.6, 6.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.0, 0.56],
                      height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.30,
                      left=0.06, right=0.985, top=0.93, bottom=0.10)
axL = fig.add_subplot(gs[:, 0])
axV = fig.add_subplot(gs[0, 1:])
axR = fig.add_subplot(gs[1, 1])
axC = fig.add_subplot(gs[1, 2])

allp = np.vstack([[s[1] for s in segs], [s[3] for s in segs]])
pad = 0.10 * max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1]))
X0, X1 = allp[:, 0].min() - pad, allp[:, 0].max() + pad
Y0, Y1 = allp[:, 1].min() - pad, allp[:, 1].max() + pad


def label_spot(b):
    e = np.linspace(-80, 80, 40001)
    P = AL[b] * D[b][None, :] + e[:, None] * S['ES'][b].ravel()[None, :]
    m = ((P[:, 0] >= X0) & (P[:, 0] <= X1) &
         (P[:, 1] >= Y0) & (P[:, 1] <= Y1))
    if not m.any():
        return None
    idx = np.where(m)[0]
    off = 0.055 * max(X1 - X0, Y1 - Y0)
    best, bd = None, -1.0
    for frac in (0.06, 0.94):
        p = P[idx[int(frac * (len(idx) - 1))]]
        for sgn in (+1, -1):
            q = p + sgn * off * D[b]
            if not (X0 + 0.03 <= q[0] <= X1 - 0.03
                    and Y0 + 0.03 <= q[1] <= Y1 - 0.03):
                continue
            d = float(np.min(np.linalg.norm(allp - q, axis=1)))
            if d > bd:
                bd, best = d, q
    return best


for b in range(K):
    e = np.array([-80.0, 80.0])
    L = AL[b] * D[b][None, :] + e[:, None] * S['ES'][b].ravel()[None, :]
    axL.plot(L[:, 0], L[:, 1], color=COLS[b], lw=1.8, alpha=0.7, zorder=2)
    q = label_spot(b)
    if q is not None:
        axL.text(q[0], q[1], rf'$\mathcal{{B}}_{b + 1}$', color=COLS[b],
                 fontsize=16, ha='center', va='center', zorder=9)

lc = LineCollection([], linewidths=1.4, zorder=3)
axL.add_collection(lc)
head, = axL.plot([], [], 'o', color=NV, ms=10, zorder=7)
flash, = axL.plot([], [], 'o', ms=21, mfc='none', mew=2.8, zorder=8)
axL.set_xlim(X0, X1)
axL.set_ylim(Y0, Y1)
axL.set_aspect('equal')
axL.set_xlabel(r'$y_1$')
axL.set_ylabel(r'$y_2$')
axL.xaxis.set_major_locator(MaxNLocator(4))
axL.yaxis.set_major_locator(MaxNLocator(4))
axL.set_title('readout', fontsize=15)

vl = [axV.plot([], [], lw=1.9, color=COLS[j])[0] for j in range(K)]
axV.axhline(0, color=NV, lw=1.3, ls='--')
axV.set_ylim(1.12 * V.min(), 0.26 * max(AL))
axV.set_ylabel(r'$V_j$')
axV.set_xlabel(r'time  ($\tau$)', labelpad=1)
axV.yaxis.set_major_locator(MaxNLocator(3))
axV.xaxis.set_major_locator(MaxNLocator(5))
axV.set_title(r'activity,  $\lambda_{\max}=+0.088$', fontsize=15)
axV.text(0.995, 0.05, 'threshold', transform=axV.transAxes, ha='right',
         va='bottom', fontsize=12, color=NV)

rast = [axR.plot([], [], '|', ms=12, mew=1.5, color=COLS[j])[0]
        for j in range(K)]
cursor = axR.axvline(0, color=NV, lw=1.1, alpha=0.6)
axR.set_xlim(0, T)
axR.set_ylim(-0.6, K - 0.4)
axR.set_yticks(range(K))
axR.set_yticklabels([str(j + 1) for j in range(K)])
axR.set_ylabel('neuron')
axR.set_xlabel(r'time  ($\tau$)')
axR.xaxis.set_major_locator(MaxNLocator(4))
for s_ in ('top', 'right'):
    axV.spines[s_].set_visible(False)
    axR.spines[s_].set_visible(False)

# ---- connectivity, W = -D D^T, in a cell of its own --------------------
W = -(D @ D.T)
ang = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                np.pi / 2 + 4 * np.pi / 3])
Pn = np.c_[np.cos(ang), np.sin(ang)]
wmax = max(abs(W[a, b]) for a in range(K) for b in range(a + 1, K))
for a in range(K):
    for b in range(a + 1, K):
        w = W[a, b]
        axC.plot(*np.c_[Pn[a], Pn[b]], color=(RED if w > 0 else BLUE),
                 lw=1.2 + 4.4 * abs(w) / wmax, alpha=0.9, zorder=1,
                 solid_capstyle='round')
        mid = 0.5 * (Pn[a] + Pn[b])
        axC.text(mid[0] * 1.62, mid[1] * 1.62, f'{w:+.2f}',
                 color=(RED if w > 0 else BLUE), fontsize=11.5,
                 ha='center', va='center', zorder=4)
for i in range(K):
    axC.plot(*Pn[i], 'o', ms=19, color='white', mec=COLS[i], mew=2.2, zorder=5)
    axC.text(Pn[i][0], Pn[i][1], str(i + 1), color=COLS[i], fontsize=12,
             ha='center', va='center', zorder=6)
axC.set_aspect('equal')
axC.set_xlim(-2.0, 2.0)
axC.set_ylim(-1.95, 2.05)
axC.set_xticks([])
axC.set_yticks([])
for sp in axC.spines.values():
    sp.set_visible(False)
axC.set_title(r'$W=-\mathbf{D}\mathbf{D}^{\!\top}$', fontsize=13, pad=3)
axC.text(0.5, -0.02, 'red: excitatory', transform=axC.transAxes,
         ha='center', va='top', fontsize=11, color=RED)

DT = ts[1] - ts[0]


def update(f):
    tq = ts[f]
    y, P = state(tq)
    if len(P) > 1:
        sg = np.stack([P[:-1], P[1:]], axis=1)
        a = np.linspace(0, 1, len(sg)) ** 3.0
        lc.set_segments(sg)
        lc.set_color([(0.082, 0.169, 0.290, 0.035 + 0.965 * v) for v in a])
    head.set_data([y[0]], [y[1]])
    j = np.searchsorted(spk_t, tq, side='right') - 1
    if j >= 0 and tq - spk_t[j] < 2.5 * DT:
        flash.set_data([spk_p[j][0]], [spk_p[j][1]])
        flash.set_markeredgecolor(COLS[spk_n[j]])
    else:
        flash.set_data([], [])
    a0, a1 = max(0.0, tq - VWIN), max(VWIN, tq)
    axV.set_xlim(a0, a1)
    m = (ts <= tq) & (ts >= a0 - DT)
    for jj in range(K):
        vl[jj].set_data(ts[m], V[m, jj])
        sm = (spk_t <= tq) & (spk_n == jj)
        rast[jj].set_data(spk_t[sm], np.full(sm.sum(), jj))
    cursor.set_xdata([tq, tq])
    return [lc, head, flash, cursor, *vl, *rast]


animation.FuncAnimation(
    fig, update, frames=NF, blit=False, interval=1000 / FPS
).save(os.path.join(here, '..', 'docs', 'video', 'k3_chaos.mp4'),
       writer=animation.FFMpegWriter(fps=FPS, bitrate=3200))
print(f'saved k3_chaos_v5.mp4  ({SECONDS}s, {NF} frames)')
