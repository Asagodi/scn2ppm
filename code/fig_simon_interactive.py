"""Interactive 3-D version of Fig 5: the Simon RBYGYB orbit, rotatable.

Same scene and same parameters as fig_simon_box6.py, in the visual vocabulary of
la_3d_causal.html: mesh3d bounds at opacity 0.15, spike jumps as colored lines with cone
heads anchored at the tip, decays in dashed grey, numbered markers, legend toggles and a
reset-camera button.

The static figure needed explicit z-orders to stop a translucent bound covering an arrow,
because a flat projection has to commit to one viewpoint. Here the reader rotates instead,
and planes can be switched off from the legend.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import plotly.graph_objects as go

import scn

# Joint (D, alpha) inverse-solve result, as in fig_simon_box6.py.
DECODERS = np.array([
    [0.917321, -0.120376, 0.379515],   # R
    [-0.011784, 0.872237, 0.488941],   # B
    [0.321296, -0.241496, 0.915668],   # Y
    [0.783224, 0.517016, 0.345332],    # G
])
DECODERS = np.array([d / np.linalg.norm(d) for d in DECODERS])
ALPHAS = np.array([0.826021, 0.790467, 0.968666, 1.060437])
HEX = ['#e74c3c', '#2980b9', '#f1c40f', '#27ae60']
NAMES = ['R', 'B', 'Y', 'G']
WORD = [0, 1, 2, 3, 2, 1]
MPAD = 0.28
DECAY_COL = '#3d3d3d'

FRAMES = [scn.tangent_frame(d) for d in DECODERS]
SYS = {'DS': [DECODERS[i] for i in range(4)], 'ES': FRAMES,
       'alphas': list(ALPHAS), 'K': 4, 'N': 3}
orbit, eigs, xi_star = scn.eigenvector_orbit(WORD, SYS)

y_pre = [y.copy() for (y, _, _) in orbit]
firers = [bi for (_, bi, _) in orbit]
L = len(WORD)
y_post = [y_pre[k] + DECODERS[firers[k]] for k in range(L)]
y_next = [y_pre[(k + 1) % L] for k in range(L)]

fig = go.Figure()

# ── bounds: one quadrilateral per neuron, sized around the spikes that land on it ──
for i in range(4):
    foot = ALPHAS[i] * DECODERS[i]
    t1, t2 = FRAMES[i][:, 0], FRAMES[i][:, 1]
    on_i = [p for p, b in zip(y_pre, firers) if b == i]
    xis = np.array([FRAMES[i].T @ (p - foot) for p in on_i]) if on_i else np.zeros((1, 2))
    u0, v0 = xis[:, 0].mean(), xis[:, 1].mean()
    du = 0.5 * float(np.ptp(xis[:, 0])) + MPAD
    dv = 0.5 * float(np.ptp(xis[:, 1])) + MPAD
    q = np.array([foot + (u0 - du) * t1 + (v0 - dv) * t2,
                  foot + (u0 + du) * t1 + (v0 - dv) * t2,
                  foot + (u0 + du) * t1 + (v0 + dv) * t2,
                  foot + (u0 - du) * t1 + (v0 + dv) * t2])
    fig.add_trace(go.Mesh3d(x=q[:, 0], y=q[:, 1], z=q[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                            color=HEX[i], opacity=0.15, flatshading=True, hoverinfo='skip',
                            showlegend=True, name='bound %s' % NAMES[i]))

# ── spikes and decays, numbered in firing order ──
for k in range(L):
    col = HEX[firers[k]]
    p, q, r = y_pre[k], y_post[k], y_next[k]
    fig.add_trace(go.Scatter3d(
        x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode='lines',
        line=dict(color=col, width=6), showlegend=(k == 0), legendgroup='spike',
        name='spike', hovertemplate='step %d: neuron %s fires<extra></extra>'
                                    % (k + 1, NAMES[firers[k]])))
    d = q - p
    fig.add_trace(go.Cone(x=[q[0]], y=[q[1]], z=[q[2]], u=[d[0]], v=[d[1]], w=[d[2]],
                          sizemode='absolute', sizeref=0.18, anchor='tip', showscale=False,
                          colorscale=[[0, col], [1, col]], hoverinfo='skip',
                          showlegend=False, legendgroup='spike'))
    fig.add_trace(go.Scatter3d(
        x=[q[0], r[0]], y=[q[1], r[1]], z=[q[2], r[2]], mode='lines',
        line=dict(color=DECAY_COL, width=3, dash='dash'), hoverinfo='skip',
        showlegend=(k == 0), legendgroup='decay', name='decay'))

fig.add_trace(go.Scatter3d(
    x=[p[0] for p in y_pre], y=[p[1] for p in y_pre], z=[p[2] for p in y_pre],
    mode='markers+text', text=[str(k + 1) for k in range(L)],
    textposition='middle center', textfont=dict(color='white', size=13),
    marker=dict(size=13, color=[HEX[b] for b in firers],
                line=dict(color='#111', width=2)),
    name='spike order', hovertemplate='step %{text}<extra></extra>'))

cam = dict(eye=dict(x=1.6, y=1.6, z=1.0))
fig.update_layout(
    title=dict(text='Simon sequence RBYGYB on a $K{=}4$, $N{=}3$ SCN'
                    '<br><sub>drag to rotate; click a legend entry to hide a bound</sub>'),
    scene=dict(aspectmode='data', camera=cam,
               xaxis=dict(title='y₁'), yaxis=dict(title='y₂'),
               zaxis=dict(title='y₃')),
    margin=dict(l=0, r=0, t=70, b=0), height=850, width=1100,
    updatemenus=[dict(type='buttons', showactive=False, x=0.02, y=0.98, buttons=[dict(
        label='↺ Reset camera', method='relayout', args=[{'scene.camera': cam}])])])

_here = os.path.dirname(os.path.abspath(__file__))
out = os.path.normpath(os.path.join(_here, '..', 'interactive_figures', 'simon_rbygyb_3d.html'))
# include_plotlyjs='cdn' matches the five existing files; pass True instead for a
# self-contained page, which costs about 4.6 MB but needs no network.
fig.write_html(out, include_plotlyjs='cdn', default_height='850px', default_width='1100px')
print('wrote', out, '(%.0f KB)' % (os.path.getsize(out) / 1024))

margin = scn.admissibility_margin(WORD, SYS)
ratio = np.sort(np.abs(eigs))[::-1]
print('check: margin %.6f, |mu_2/mu_1| %.6f' % (margin, ratio[1] / ratio[0]))
