"""Simon-style figure in K=3 RGB using the multistability_3d.html plotly aesthetic.

We use a K=3, N=3 example (word RGB) rather than K=4 RBYGYB because RGB is
admissible with a comfortable margin at a symmetric wide-cone geometry, so the
visualisation shows every neuron firing on its own bound with a positive margin;
the K=4 RBYGYB case is the analytic target discussed in Section~5 of the paper.
"""

import os
import numpy as np
import plotly.graph_objects as go

# Palette matching poster / multistability_3d style
NEURON_COL = ['#B03A2E', '#1F4E79', '#1E7A48']  # R, B, G (paper's SCN palette)
DECAY_COL = '#7F8C8D'


def make_system(spread=0.7):
    c_axis = np.array([1., 1., 1.]) / np.sqrt(3)
    u1 = np.array([1., -1., 0.]) / np.sqrt(2)
    u2 = np.cross(c_axis, u1); u2 /= np.linalg.norm(u2)
    DS = []
    for a in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        d = c_axis + spread * (np.cos(a) * u1 + np.sin(a) * u2)
        d /= np.linalg.norm(d)
        DS.append(d)
    ES = []
    for d in DS:
        z = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([0., 1., 0.])
        v1 = np.cross(d, z); v1 /= np.linalg.norm(v1)
        v2 = np.cross(d, v1); v2 /= np.linalg.norm(v2)
        ES.append(np.column_stack([v1, v2]))
    return DS, ES


DS, ES = make_system(spread=0.7)
ALPHAS = np.array([1.0, 1.0, 1.0])
WORD = [0, 1, 2]  # R, B, G — period 3


def transition(bi, bj):
    U_i, U_j = ES[bi], ES[bj]; D_i, D_j = DS[bi], DS[bj]
    a_i, a_j = ALPHAS[bi], ALPHAS[bj]
    M = np.zeros((3, 3))
    M[:2, :2] = a_j * (U_j.T @ U_i)
    M[:2, 2] = a_j * (a_i + 1.0) * (U_j.T @ D_i)
    M[2, :2] = D_j @ U_i
    M[2, 2] = (a_i + 1.0) * (D_j @ D_i)
    return M


def step(xi, bi):
    yp = (ALPHAS[bi] + 1.0) * DS[bi] + ES[bi] @ xi
    best_j, best_c = -1, -1.0
    for j in range(3):
        denom = DS[j] @ yp
        if denom <= 0: continue
        c = ALPHAS[j] / denom
        if c > best_c: best_c = c; best_j = j
    y_next = best_c * yp
    xi_next = ES[best_j].T @ (y_next - ALPHAS[best_j] * DS[best_j])
    return xi_next, best_j


def eigvec_orbit():
    M = np.eye(3)
    for k in range(len(WORD)):
        M = transition(WORD[k], WORD[(k + 1) % len(WORD)]) @ M
    w, V = np.linalg.eig(M)
    ordr = np.argsort(-np.abs(w))
    for idx in ordr:
        v = V[:, idx]
        if np.max(np.abs(v.imag)) < 1e-8 and abs(v.real[2]) > 1e-10:
            v = v.real / v.real[2]; return v[:2]
    return None


xi = eigvec_orbit()
orbit = []
xk, bk = xi.copy(), WORD[0]
for k in range(len(WORD)):
    y = ALPHAS[bk] * DS[bk] + ES[bk] @ xk
    orbit.append((y.copy(), bk))
    xk, bk = step(xk, bk)

segments = []
ys = np.array([o[0] for o in orbit])
bs = np.array([o[1] for o in orbit])
for k in range(len(orbit)):
    y_pre = ys[k]
    y_post = y_pre + DS[bs[k]]
    y_next = ys[(k + 1) % len(orbit)]
    segments.append({'y_pre': y_pre, 'y_post': y_post, 'y_next': y_next, 'neuron': int(bs[k])})

fig = go.Figure()

plane_r = 0.55   # shrink so the R/B bounds don't hang far below the orbit
for i in range(3):
    foot = ALPHAS[i] * DS[i]
    E = ES[i]
    corners = np.array([
        foot + E @ np.array([-plane_r, -plane_r]),
        foot + E @ np.array([+plane_r, -plane_r]),
        foot + E @ np.array([+plane_r, +plane_r]),
        foot + E @ np.array([-plane_r, +plane_r]),
    ])
    col = NEURON_COL[i]
    fig.add_trace(go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=col, opacity=0.12, flatshading=True,
        lighting=dict(ambient=1.0, diffuse=0),
        showlegend=False, hoverinfo='skip'))
    edge = np.vstack([corners, corners[[0]]])
    fig.add_trace(go.Scatter3d(x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
                               mode='lines', line=dict(color=col, width=2),
                               showlegend=False, hoverinfo='skip'))

fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                            marker=dict(size=6, color='black', symbol='cross'),
                            showlegend=False, hoverinfo='skip'))

for seg in segments:
    col = NEURON_COL[seg['neuron']]
    fig.add_trace(go.Scatter3d(
        x=[seg['y_pre'][0], seg['y_post'][0]],
        y=[seg['y_pre'][1], seg['y_post'][1]],
        z=[seg['y_pre'][2], seg['y_post'][2]],
        mode='lines', line=dict(color=col, width=10),
        showlegend=False, hoverinfo='skip'))
    v = seg['y_post'] - seg['y_pre']
    fig.add_trace(go.Cone(
        x=[seg['y_post'][0]], y=[seg['y_post'][1]], z=[seg['y_post'][2]],
        u=[v[0]], v=[v[1]], w=[v[2]],
        sizemode='absolute', sizeref=0.12, anchor='tip',
        colorscale=[[0, col], [1, col]], showscale=False,
        showlegend=False, hoverinfo='skip'))
    c = np.linalg.norm(seg['y_next']) / max(np.linalg.norm(seg['y_post']), 1e-6)
    ts = np.linspace(0, -np.log(max(c, 1e-6)), 24)
    curve = np.array([seg['y_post'] * np.exp(-t) for t in ts])
    fig.add_trace(go.Scatter3d(
        x=curve[:, 0], y=curve[:, 1], z=curve[:, 2],
        mode='lines', line=dict(color=DECAY_COL, width=5),
        showlegend=False, hoverinfo='skip'))
    v_d = curve[-1] - curve[-2]
    fig.add_trace(go.Cone(
        x=[curve[-1][0]], y=[curve[-1][1]], z=[curve[-1][2]],
        u=[v_d[0]], v=[v_d[1]], w=[v_d[2]],
        sizemode='absolute', sizeref=0.08, anchor='tip',
        colorscale=[[0, DECAY_COL], [1, DECAY_COL]], showscale=False,
        showlegend=False, hoverinfo='skip'))

for seg in segments:
    y = seg['y_pre']
    col = NEURON_COL[seg['neuron']]
    fig.add_trace(go.Scatter3d(
        x=[y[0]], y=[y[1]], z=[y[2]],
        mode='markers',
        marker=dict(size=8, color=col, line=dict(color='white', width=1.5)),
        showlegend=False, hoverinfo='skip'))

CMU = '"Computer Modern Serif", "CMU Serif", "Latin Modern Roman", Georgia, serif'
# Camera positioned closer + data-aspectmode to eliminate the padding that
# 'cube' introduces around a non-cubic scene.
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='data',
        # Tighter eye + smaller vertical center to shift the orbit up so the
        # top whitespace shrinks. z=0.55 lowers the horizon, y=1.35 closes in
        # a little, and center z=0.1 pushes the scene up in the frame.
        camera=dict(eye=dict(x=1.05, y=1.35, z=0.55),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0.15, y=0.15, z=0.10)),
    ),
    width=700, height=560,   # shorter frame kills the top blank strip
    margin=dict(l=0, r=0, t=0, b=0),
    paper_bgcolor='white',
    font=dict(family=CMU, size=14),
)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'simon_3d.pdf')
fig.write_image(out, engine='kaleido', width=700, height=560)
print(f"Saved {out}")
