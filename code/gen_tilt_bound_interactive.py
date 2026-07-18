"""Interactive: tilt the 4th bound about the horizontal line at y=c.

3 fixed decoders arranged symmetrically about the y-axis with mutual
angle `a` (find beta from a via cos a = 1.5 cos^2 beta - 0.5).
4th bound rotates about the line L = {(t, c, 0) : t in R}:
  D_4(psi) = (0, sin psi, cos psi)   (normal in y-z plane)
  alpha_4  = c * sin psi              (bound passes through (0, c, 0))

All four bounds meet at the common apex (0, c, 0) by construction, so
the "y=c axis" line lies in B_4 and the apex point lies in all four
bounds. Only psi varies in the slider; a and c are Python-side knobs.

Writes a Plotly HTML with drag-rotate 3D scene + psi slider.
"""

import os, sys, json
import numpy as np
import plotly.graph_objects as go

# ─── Knobs ─────────────────────────────────────────────────────────────────
A_DEG      = 100.0     # mutual angle between the 3 fixed decoders
C          = -1.0      # anchor height on y-axis; negative -> cone opens +y
PSI_LO_DEG = 5.0       # avoid degenerate ψ=0 (alpha_4=0)
PSI_HI_DEG = 85.0      # avoid degenerate ψ=π/2 (bound perp to xy)
N_FRAMES   = 41
PLANE_HALF = 1.4       # half-side of the drawn bound-plane quads

# ─── Geometry helpers ──────────────────────────────────────────────────────
def beta_from_a(a_deg):
    """cos a = 1.5 cos^2 beta - 0.5 (three-fold symmetric on cone about +y)."""
    cos_a = np.cos(np.deg2rad(a_deg))
    cos_b2 = (cos_a + 0.5) / 1.5
    if not (0 < cos_b2 <= 1):
        raise ValueError(f'a={a_deg}° gives cos^2 beta = {cos_b2:.3f}, out of range')
    return np.rad2deg(np.arccos(np.sqrt(cos_b2)))


def tangent_frame(D):
    ax = np.array([1.0, 0, 0]) if abs(D[0]) < 0.9 else np.array([0.0, 1, 0])
    u = ax - (ax @ D) * D; u /= np.linalg.norm(u)
    v = np.cross(D, u)
    return u, v


def bound_quad(D, alpha, half=PLANE_HALF):
    """Return (x, y, z, i, j, k) for a triangle-mesh quad on plane D.y = alpha."""
    foot = alpha * D
    u, v = tangent_frame(D)
    c00 = foot - half*u - half*v
    c01 = foot - half*u + half*v
    c11 = foot + half*u + half*v
    c10 = foot + half*u - half*v
    pts = np.array([c00, c01, c11, c10])
    return pts[:, 0], pts[:, 1], pts[:, 2], [0, 0], [1, 2], [2, 3]


# ─── 3 fixed decoders (with y-component sign matched to sign of C) ─────────
BETA = beta_from_a(A_DEG)
# We want alpha_i = C * D_{i,y} > 0. If C < 0, need D_{i,y} < 0 (cone opens +y).
# If C > 0, need D_{i,y} > 0 (cone opens -y). Sign is fixed by cos(beta) sign choice.
sign_y = -1.0 if C < 0 else 1.0
sin_b = float(np.sin(np.deg2rad(BETA)))
cos_b = float(np.cos(np.deg2rad(BETA))) * sign_y  # y-component sign

D_fixed = []
alpha_fixed = []
for phi_deg in (0.0, 120.0, 240.0):
    phi = np.deg2rad(phi_deg)
    d = np.array([sin_b*np.cos(phi), cos_b, sin_b*np.sin(phi)])
    D_fixed.append(d)
    alpha_fixed.append(float(C * d[1]))
D_fixed = np.array(D_fixed)
alpha_fixed = np.array(alpha_fixed)
apex = np.array([0.0, float(C), 0.0])

# Verify: D_i . apex = alpha_i for all i
for i, D in enumerate(D_fixed):
    assert abs(float(D @ apex) - alpha_fixed[i]) < 1e-10

# ─── Build Plotly figure ───────────────────────────────────────────────────
# Colors: red, amber, blue, green (D_1..D_3 fixed; D_4 always red-emphasized)
COLORS = ['#F39C12', '#2980B9', '#27AE60', '#E74C3C']

frames_plotly = []
psis = np.linspace(PSI_LO_DEG, PSI_HI_DEG, N_FRAMES)

# Static traces (fixed at every ψ): apex marker, y=c anchor line,
# coordinate axes, 3 fixed bounds + decoder arrows.
static_traces = []

# coordinate axes (short segments through origin)
ax_len = 1.6
for name, dir_v, col in [
    ('x', [ax_len, 0, 0], '#8F97AA'),
    ('y', [0, ax_len, 0], '#8F97AA'),
    ('z', [0, 0, ax_len], '#8F97AA'),
]:
    static_traces.append(go.Scatter3d(
        x=[-0.1*dir_v[0], dir_v[0]], y=[-0.1*dir_v[1], dir_v[1]], z=[-0.1*dir_v[2], dir_v[2]],
        mode='lines', line=dict(color=col, width=2), showlegend=False, hoverinfo='skip'))

# y=c anchor LINE (horizontal, direction x̂, through the apex)
static_traces.append(go.Scatter3d(
    x=[-1.8, 1.8], y=[C, C], z=[0, 0],
    mode='lines', line=dict(color='#152B4A', width=4, dash='dash'),
    name=f'anchor line y = {C:g}, z = 0', hoverinfo='skip'))

# apex point
static_traces.append(go.Scatter3d(
    x=[0], y=[float(C)], z=[0], mode='markers+text',
    marker=dict(size=8, color='#152B4A', line=dict(width=2, color='white')),
    text=[f'apex (0, {C:g}, 0)'], textposition='top right',
    name='apex', hoverinfo='skip'))

# origin
static_traces.append(go.Scatter3d(
    x=[0], y=[0], z=[0], mode='markers',
    marker=dict(size=4, color='#1F2A3A'), showlegend=False, hoverinfo='skip'))

# 3 fixed decoder arrows + planes
for i, (D, a) in enumerate(zip(D_fixed, alpha_fixed)):
    col = COLORS[i]
    # decoder arrow
    static_traces.append(go.Scatter3d(
        x=[0, D[0]], y=[0, D[1]], z=[0, D[2]],
        mode='lines+markers',
        line=dict(color=col, width=6),
        marker=dict(size=[0, 7], color=col, symbol='circle'),
        name=f'D_{i+1}',
        hovertemplate=f'D_{i+1} = ({D[0]:.2f}, {D[1]:.2f}, {D[2]:.2f})<br>α_{i+1} = {a:.3f}<extra></extra>'))
    # bound plane
    x, y, z, ii, jj, kk = bound_quad(D, a)
    static_traces.append(go.Mesh3d(
        x=x, y=y, z=z, i=ii, j=jj, k=kk,
        color=col, opacity=0.25, name=f'B_{i+1}',
        flatshading=True, hoverinfo='skip'))

# ─── Frames (moving D_4 + B_4) ─────────────────────────────────────────────
def moving_traces(psi_deg):
    """Traces for D_4 and B_4 at this psi."""
    psi = np.deg2rad(psi_deg)
    D4 = np.array([0.0, np.sin(psi), np.cos(psi)])
    a4 = float(C * D4[1])  # = C * sin psi
    col = COLORS[3]
    traces = []
    # decoder arrow (from origin to D_4)
    traces.append(go.Scatter3d(
        x=[0, D4[0]], y=[0, D4[1]], z=[0, D4[2]],
        mode='lines+markers',
        line=dict(color=col, width=7),
        marker=dict(size=[0, 8], color=col, symbol='circle'),
        name='D_4',
        hovertemplate=f'D_4 = ({D4[0]:.2f}, {D4[1]:.2f}, {D4[2]:.2f})<br>α_4 = {a4:+.3f}<br>ψ = {psi_deg:.1f}°<extra></extra>'))
    # plane
    x, y, z, ii, jj, kk = bound_quad(D4, a4)
    traces.append(go.Mesh3d(
        x=x, y=y, z=z, i=ii, j=jj, k=kk,
        color=col, opacity=0.35, name='B_4',
        flatshading=True, hoverinfo='skip'))
    # readout as an invisible 3D scatter with hover-only info
    cos_14 = float(D_fixed[0] @ D4)
    cos_24 = float(D_fixed[1] @ D4)
    cos_34 = float(D_fixed[2] @ D4)
    readout = (f'ψ = {psi_deg:.1f}°<br>'
               f'D_4 = ({D4[0]:.2f}, {D4[1]:.2f}, {D4[2]:.2f})<br>'
               f'α_4 = {a4:+.3f}<br>'
               f'cos Ψ_14 = {cos_14:+.3f}<br>'
               f'cos Ψ_24 = {cos_24:+.3f}<br>'
               f'cos Ψ_34 = {cos_34:+.3f}')
    return traces, readout


# initial frame at midpoint
mid_idx = N_FRAMES // 2
psi_init = psis[mid_idx]
moving_init, readout_init = moving_traces(psi_init)

# Number of static traces (needed for slider frame indexing)
n_static = len(static_traces)

# Build all frames
for k, psi_deg in enumerate(psis):
    mov, ro = moving_traces(psi_deg)
    # each frame REPLACES only the moving traces; static ones stay via data=[...]
    frames_plotly.append(go.Frame(
        data=static_traces + mov,
        name=f'{psi_deg:.1f}',
        layout=dict(title=dict(
            text=(f'<b>Tilt bound 4</b> — a = {A_DEG:.0f}°, β = {BETA:.1f}°, '
                  f'c = {C:g}, ψ = {psi_deg:.1f}°<br>'
                  f'<sub>{ro}</sub>'),
            font=dict(family='Georgia, serif', size=13, color='#152B4A')))))

# ─── Assemble figure ───────────────────────────────────────────────────────
fig = go.Figure(
    data=static_traces + moving_init,
    layout=dict(
        title=dict(
            text=(f'<b>Tilt bound 4</b> — a = {A_DEG:.0f}°, β = {BETA:.1f}°, '
                  f'c = {C:g}, ψ = {psi_init:.1f}°<br>'
                  f'<sub>{readout_init}</sub>'),
            font=dict(family='Georgia, serif', size=13, color='#152B4A')),
        scene=dict(
            xaxis=dict(range=[-1.8, 1.8], title='x', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
            yaxis=dict(range=[-1.8, 1.8], title='y', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
            zaxis=dict(range=[-1.8, 1.8], title='z', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.0, z=1.2), up=dict(x=0, y=1, z=0)),
            bgcolor='#FAF6EE',
        ),
        paper_bgcolor='#FAF6EE',
        margin=dict(l=0, r=0, t=80, b=0),
        width=1100, height=800,
        updatemenus=[dict(
            type='buttons', showactive=False,
            x=0.05, y=1.06, xanchor='left', yanchor='top',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label='❚❚ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate', transition=dict(duration=0))]),
            ])],
        sliders=[dict(
            active=mid_idx,
            currentvalue=dict(prefix='ψ = ', suffix='°',
                              font=dict(family='Georgia, serif', size=15, color='#00008C')),
            pad=dict(t=45, b=10),
            len=0.85, x=0.08,
            steps=[dict(method='animate', label=f'{p:.0f}',
                        args=[[f'{p:.1f}'],
                              dict(mode='immediate',
                                   frame=dict(duration=0, redraw=True),
                                   transition=dict(duration=0))])
                   for p in psis])],
    ),
    frames=frames_plotly,
)

out_dir = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'interactive_figures'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'tilt_bound.html')
fig.write_html(out_path,
               include_plotlyjs='cdn',
               include_mathjax=False,
               config=dict(displayModeBar=True, displaylogo=False))
print(f'Wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)')
print(f'  a = {A_DEG}°  ->  beta = {BETA:.2f}°')
print(f'  c = {C}   anchor at (0, {C}, 0)')
print(f'  ψ sweep: [{PSI_LO_DEG}, {PSI_HI_DEG}]° in {N_FRAMES} frames')
print(f'  Fixed decoders y-component sign: {int(sign_y):+d}  '
      f'(so alpha = c * D.y > 0)')
