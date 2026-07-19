"""Interactive: tilt the 4th bound about the horizontal line at y=c.

3 fixed decoders arranged symmetrically about the y-axis with mutual
angle `a` (find beta from a via cos a = 1.5 cos^2 beta - 0.5).
4th bound rotates about the line L = {(t, c, 0) : t in R}:
  D_4(psi) = (0, sign_y * sin psi, cos psi)   (normal in y-z plane)
  alpha_4  = c * D_4.y                         (bound passes through (0, c, 0))

All four bounds meet at the common apex (0, c, 0) by construction, so
the "y=c axis" line lies in B_4 and the apex point lies in all four
bounds. Only psi varies in the slider; a and c are Python-side knobs.

ADMISSIBILITY RIBBON (bottom panel): for each psi frame, we enumerate
primitive canonical periodic words up to length L_MAX over the K=4
alphabet, compute admissibility_margin(word, sys) for each, and colour
a strip at that psi by the MAX margin across all words. Blue means
"there is at least one admissible eigenvector periodic orbit at this
psi"; red means "no candidate word up to L_MAX is admissible"; grey
means the eigendecomposition failed everywhere (no real dominant
eigenvector). This is NOT chaos vs order per se -- it is the paper's
existence-of-admissible-UPO question restricted to short words.
"""

import os, sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from scn import eigenvector_orbit, admissibility_margin  # noqa: E402

# ─── Knobs ─────────────────────────────────────────────────────────────────
A_DEG      = 100.0     # mutual angle between the 3 fixed decoders
C          = -1.0      # anchor height on y-axis; negative -> cone opens +y
PSI_LO_DEG = 5.0       # avoid degenerate ψ=0 (alpha_4=0)
PSI_HI_DEG = 85.0      # avoid degenerate ψ=π/2 (bound perp to xy)
N_FRAMES   = 41
PLANE_HALF = 1.4       # half-side of the drawn bound-plane quads
L_MAX      = 7         # enumerate primitive canonical words up to this length
K          = 4         # 4 bounds

# ─── Geometry helpers ──────────────────────────────────────────────────────
def beta_from_a(a_deg):
    """cos a = 1.5 cos^2 beta - 0.5 (three-fold symmetric on cone about +y)."""
    cos_a = np.cos(np.deg2rad(a_deg))
    cos_b2 = (cos_a + 0.5) / 1.5
    if not (0 < cos_b2 <= 1):
        raise ValueError(f'a={a_deg} deg gives cos^2 beta = {cos_b2:.3f}, out of range')
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


# ─── Word enumeration + admissibility precompute ───────────────────────────
def enumerate_words(K, L_max):
    """All primitive canonical periodic words over alphabet {0..K-1} with
    lengths 1..L_max. Filters out adjacent repeats for L >= 2 (SCN dynamics
    can have bj==bi only at L=1 self-firings; longer words with bj==bi at
    the wrap-around are typically indistinguishable from a shorter word).
    Canonical = lexicographically smallest cyclic rotation.
    Primitive = not a repetition of any shorter word.
    """
    from itertools import product

    def canonical(w):
        return min(w[i:] + w[:i] for i in range(len(w)))

    def is_primitive(w):
        L = len(w)
        for d in range(1, L):
            if L % d == 0 and w[:d] * (L // d) == w:
                return False
        return True

    def has_adj_repeat(w):
        L = len(w)
        return any(w[k] == w[(k + 1) % L] for k in range(L))

    out = []
    # L=1: 4 period-1 self-firings (adjacent-repeat check would kill them, so
    # they get special handling — one per bound).
    for i in range(K):
        out.append((i,))

    for L in range(2, L_max + 1):
        seen = set()
        for w in product(range(K), repeat=L):
            if has_adj_repeat(w) or not is_primitive(w):
                continue
            c = canonical(w)
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def safe_margin_and_stability(word, sys_dict):
    """Return (margin, |mu_transverse_max/mu_1|) for this word.

    margin is admissibility_margin(word, sys) or NaN if eigendecomp fails or
    dominant eigenvalue is complex. stability_ratio is max over k>=1 of
    |ev[k]/ev[0]| where ev is sorted by magnitude descending by
    eigenvector_orbit; NaN if ev[0] is ~0 or if the computation fails.
    A word is admissible-and-stable iff margin > 0 AND stability_ratio < 1.
    A UPO is margin > 0 AND stability_ratio >= 1.
    """
    try:
        m = admissibility_margin(list(word), sys_dict)
    except Exception:
        return float('nan'), float('nan')
    if m is None:
        return float('nan'), float('nan')
    # scn.admissibility_margin returns -1.0 as sentinel on failure. Try
    # eigenvector_orbit to distinguish from a legit m == -1.0.
    try:
        _, ev, _ = eigenvector_orbit(list(word), sys_dict)
    except Exception:
        return float('nan'), float('nan')
    if not np.isreal(ev[0]):
        return float('nan'), float('nan')
    if abs(ev[0]) < 1e-15:
        return float(m), float('nan')
    # eigenvector_orbit sorts ev by |ev| descending; ev[0] is dominant.
    ratios = [abs(ev[k] / ev[0]) for k in range(1, len(ev))]
    stab = max(ratios) if ratios else 0.0
    return float(m), float(stab)


def build_sys_dict(D_mat, alphas):
    """SCN sys-dict compatible with scn.eigenvector_orbit / admissibility_margin."""
    from scn import tangent_frame as scn_tf
    DS = [D_mat[k].copy() for k in range(len(alphas))]
    ES = [scn_tf(d) for d in DS]
    return {'DS': DS, 'ES': ES, 'alphas': list(alphas), 'K': len(alphas), 'N': D_mat.shape[1]}


# ─── 3 fixed decoders (with y-component sign matched to sign of C) ─────────
BETA = beta_from_a(A_DEG)
sign_y = -1.0 if C < 0 else 1.0
sin_b = float(np.sin(np.deg2rad(BETA)))
cos_b = float(np.cos(np.deg2rad(BETA))) * sign_y

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

for i, D in enumerate(D_fixed):
    assert abs(float(D @ apex) - alpha_fixed[i]) < 1e-10

# ─── Precompute admissibility over the sweep ───────────────────────────────
psis = np.linspace(PSI_LO_DEG, PSI_HI_DEG, N_FRAMES)
words = enumerate_words(K, L_MAX)
print(f'Enumerated {len(words)} primitive canonical words up to L={L_MAX}.')

# Margin matrix M and stability-ratio matrix S: rows = words, cols = psi frames.
M = np.full((len(words), N_FRAMES), np.nan, dtype=float)
S = np.full((len(words), N_FRAMES), np.nan, dtype=float)
for j, psi_deg in enumerate(psis):
    psi = np.deg2rad(psi_deg)
    D4 = np.array([0.0, sign_y * np.sin(psi), np.cos(psi)])
    a4 = float(C * D4[1])
    D_mat = np.vstack([D_fixed, D4[None, :]])
    alphas = np.append(alpha_fixed, a4)
    sysd = build_sys_dict(D_mat, alphas)
    for i, w in enumerate(words):
        m, s = safe_margin_and_stability(w, sysd)
        M[i, j] = m
        S[i, j] = s

# Split M by stability class:
#   admissible & stable:  m > 0  and  s < 1
#   admissible & UPO:     m > 0  and  s >= 1
M_stable = np.where((M > 0) & (S < 1.0), M, np.nan)
M_upo    = np.where((M > 0) & (S >= 1.0), M, np.nan)

with np.errstate(all='ignore'):
    best_stable_psi = np.nanmax(M_stable, axis=0)
    best_upo_psi    = np.nanmax(M_upo,    axis=0)
    best_margin_psi = np.nanmax(M, axis=0)  # for the full-heatmap colorbar

# argmax word per frame within each stability class.
best_stable_idx = np.full(N_FRAMES, -1, dtype=int)
best_upo_idx    = np.full(N_FRAMES, -1, dtype=int)
for j in range(N_FRAMES):
    col_s = M_stable[:, j]
    col_u = M_upo[:, j]
    if np.any(np.isfinite(col_s)):
        best_stable_idx[j] = int(np.nanargmax(col_s))
    if np.any(np.isfinite(col_u)):
        best_upo_idx[j] = int(np.nanargmax(col_u))

n_stable = int(np.isfinite(best_stable_psi).sum())
n_upo    = int(np.isfinite(best_upo_psi).sum())
n_either = int(((best_margin_psi > 0)).sum())
print(f'Ribbon summary:')
print(f'  stable-admissible frames: {n_stable}/{N_FRAMES}')
print(f'  UPO-admissible frames:    {n_upo}/{N_FRAMES}')
print(f'  any admissible:           {n_either}/{N_FRAMES}')

# Row ordering (lex by (length, -coverage)):
lengths = np.array([len(w) for w in words])
coverage = np.array([np.mean(np.isfinite(M[i]) & (M[i] > 0)) for i in range(len(words))])
order = np.lexsort((-coverage, lengths))
words_sorted = [words[i] for i in order]
M_sorted = M[order]
labels_sorted = [''.join(str(x) for x in w) for w in words_sorted]

# ─── Build Plotly figure (subplots: 3D + ribbon + heatmap) ─────────────────
COLORS = ['#F39C12', '#2980B9', '#27AE60', '#E74C3C']

fig = make_subplots(
    rows=3, cols=1,
    row_heights=[0.62, 0.14, 0.24],
    vertical_spacing=0.05,
    specs=[[{'type': 'scene'}], [{'type': 'xy'}], [{'type': 'xy'}]],
    subplot_titles=('', 'Admissibility ribbon (top: stable-admissible, bottom: UPO-admissible)',
                    'Word × ψ margin heatmap (rows = periodic word σ)'),
)

# ─── static 3D content ─────────────────────────────────────────────────────
def add_static_3d():
    ax_len = 1.6
    for _name, dir_v, col in [
        ('x', [ax_len, 0, 0], '#8F97AA'),
        ('y', [0, ax_len, 0], '#8F97AA'),
        ('z', [0, 0, ax_len], '#8F97AA'),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[-0.1*dir_v[0], dir_v[0]], y=[-0.1*dir_v[1], dir_v[1]], z=[-0.1*dir_v[2], dir_v[2]],
            mode='lines', line=dict(color=col, width=2), showlegend=False, hoverinfo='skip'),
            row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[-1.8, 1.8], y=[C, C], z=[0, 0],
        mode='lines', line=dict(color='#152B4A', width=4, dash='dash'),
        name=f'anchor y={C:g}', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[float(C)], z=[0], mode='markers+text',
        marker=dict(size=8, color='#152B4A', line=dict(width=2, color='white')),
        text=[f'apex (0, {C:g}, 0)'], textposition='top right',
        name='apex', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=4, color='#1F2A3A'), showlegend=False, hoverinfo='skip'),
        row=1, col=1)
    for i, (D, a) in enumerate(zip(D_fixed, alpha_fixed)):
        col = COLORS[i]
        fig.add_trace(go.Scatter3d(
            x=[0, D[0]], y=[0, D[1]], z=[0, D[2]],
            mode='lines+markers',
            line=dict(color=col, width=6),
            marker=dict(size=[0, 7], color=col, symbol='circle'),
            name=f'D_{i+1}',
            hovertemplate=f'D_{i+1} = ({D[0]:.2f}, {D[1]:.2f}, {D[2]:.2f})<br>α_{i+1} = {a:.3f}<extra></extra>'),
            row=1, col=1)
        x, y, z, ii, jj, kk = bound_quad(D, a)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=ii, j=jj, k=kk,
            color=col, opacity=0.25, name=f'B_{i+1}',
            flatshading=True, hoverinfo='skip'), row=1, col=1)


add_static_3d()
n_static_3d = len(fig.data)

# ─── moving 3D content (D_4 + B_4), reused for the initial frame ───────────
def moving_traces(psi_deg):
    psi = np.deg2rad(psi_deg)
    D4 = np.array([0.0, sign_y * np.sin(psi), np.cos(psi)])
    a4 = float(C * D4[1])
    col = COLORS[3]
    arrow = go.Scatter3d(
        x=[0, D4[0]], y=[0, D4[1]], z=[0, D4[2]],
        mode='lines+markers',
        line=dict(color=col, width=7),
        marker=dict(size=[0, 8], color=col, symbol='circle'),
        name='D_4',
        hovertemplate=f'D_4 = ({D4[0]:.2f}, {D4[1]:.2f}, {D4[2]:.2f})<br>α_4 = {a4:+.3f}<br>ψ = {psi_deg:.1f}°<extra></extra>')
    x, y, z, ii, jj, kk = bound_quad(D4, a4)
    plane = go.Mesh3d(
        x=x, y=y, z=z, i=ii, j=jj, k=kk,
        color=col, opacity=0.35, name='B_4',
        flatshading=True, hoverinfo='skip')
    return [arrow, plane]


mid_idx = N_FRAMES // 2
psi_init = psis[mid_idx]
for t in moving_traces(psi_init):
    fig.add_trace(t, row=1, col=1)

# ─── admissibility RIBBON on xaxis1/yaxis1 (row 2) ────────────────────────
# Two-row heatmap: top row = best stable-admissible margin, bottom = best UPO
# margin. Where a class has no admissible orbit at this ψ, cell is NaN
# (rendered as the background). Both use the same diverging colorscale so
# the reader can compare magnitudes.
vmax = max(0.05, float(np.nanmax(np.abs(best_margin_psi))))
ribbon_Z = np.vstack([best_stable_psi, best_upo_psi])
ribbon_labels = ['stable', 'UPO']
fig.add_trace(go.Heatmap(
    x=psis, y=ribbon_labels, z=ribbon_Z,
    zmin=-vmax, zmax=+vmax, colorscale='RdBu', reversescale=False,
    zmid=0,
    colorbar=dict(
        len=0.18, x=1.01, y=0.42, thickness=12,
        title=dict(text='<i>m</i>*', side='right'),
        tickfont=dict(size=10),
    ),
    hovertemplate='ψ = %{x:.1f}°<br>class = %{y}<br>max margin = %{z:+.4f}<extra></extra>',
    name='ribbon',
    showscale=True,
), row=2, col=1)

# Moving cursor on the ribbon: a vertical dashed line spanning both rows.
fig.add_trace(go.Scatter(
    x=[psi_init, psi_init], y=['stable', 'UPO'],
    mode='lines', line=dict(color='#152B4A', width=1.6, dash='dash'),
    hoverinfo='skip', showlegend=False,
), row=2, col=1)

# ─── full word × ψ heatmap (row 3) ────────────────────────────────────────
fig.add_trace(go.Heatmap(
    x=psis, y=labels_sorted, z=M_sorted,
    zmin=-vmax, zmax=+vmax, colorscale='RdBu', reversescale=False,
    zmid=0,
    hovertemplate='σ = %{y}<br>ψ = %{x:.1f}°<br>margin = %{z:+.4f}<extra></extra>',
    showscale=False,
    name='word-heatmap',
), row=3, col=1)

# Moving vertical cursor on the heatmap
fig.add_trace(go.Scatter(
    x=[psi_init, psi_init], y=[labels_sorted[0], labels_sorted[-1]],
    mode='lines', line=dict(color='#152B4A', width=1.5, dash='dash'),
    hoverinfo='skip', showlegend=False,
), row=3, col=1)


# ─── Frame data list (must match figure trace ORDER) ──────────────────────
# figure trace order:
#   0..n_static_3d-1              : static 3D
#   n_static_3d, n_static_3d+1    : moving D_4 + B_4  (per-frame update)
#   n_static_3d+2                 : ribbon heatmap (static)
#   n_static_3d+3                 : ribbon cursor (per-frame update)
#   n_static_3d+4                 : word-heatmap (static)
#   n_static_3d+5                 : word-heatmap cursor (per-frame update)
static_ribbon = fig.data[n_static_3d + 2]
static_wordheat = fig.data[n_static_3d + 4]
static_3d = list(fig.data[:n_static_3d])


def title_for(psi_deg, idx):
    D4 = np.array([0.0, sign_y * np.sin(np.deg2rad(psi_deg)), np.cos(np.deg2rad(psi_deg))])
    a4 = float(C * D4[1])
    parts = []
    if best_stable_idx[idx] >= 0:
        i = best_stable_idx[idx]
        w = words[i]
        parts.append(
            f'<b style="color:#08519c">stable</b>: σ=({",".join(str(x) for x in w)}) '
            f'<i>m</i>={best_stable_psi[idx]:+.4f} '
            f'|μ<sub>2</sub>/μ<sub>1</sub>|={S[i, idx]:.3f}'
        )
    if best_upo_idx[idx] >= 0:
        i = best_upo_idx[idx]
        w = words[i]
        parts.append(
            f'<b style="color:#d97706">UPO</b>: σ=({",".join(str(x) for x in w)}) '
            f'<i>m</i>={best_upo_psi[idx]:+.4f} '
            f'|μ<sub>2</sub>/μ<sub>1</sub>|={S[i, idx]:.3f}'
        )
    if not parts:
        parts.append('<b style="color:#a50026">no admissible orbit at L≤' + str(L_MAX) + '</b>')
    adm_txt = '  ·  '.join(parts)
    return (
        f'<b>Tilt bound 4</b> — a = {A_DEG:.0f}°, β = {BETA:.1f}°, '
        f'c = {C:g}, ψ = {psi_deg:.1f}°<br>'
        f'<sub>D_4 = ({D4[0]:.2f}, {D4[1]:.2f}, {D4[2]:.2f}), α_4 = {a4:+.3f} · '
        f'{adm_txt}</sub>'
    )


frames = []
for k, psi_deg in enumerate(psis):
    mov = moving_traces(psi_deg)
    ribbon_cursor = go.Scatter(
        x=[psi_deg, psi_deg], y=['stable', 'UPO'],
        mode='lines', line=dict(color='#152B4A', width=1.6, dash='dash'),
        hoverinfo='skip', showlegend=False,
        xaxis='x', yaxis='y',
    )
    heat_cursor = go.Scatter(
        x=[psi_deg, psi_deg], y=[labels_sorted[0], labels_sorted[-1]],
        mode='lines', line=dict(color='#152B4A', width=1.5, dash='dash'),
        hoverinfo='skip', showlegend=False,
        xaxis='x2', yaxis='y2',
    )
    frames.append(go.Frame(
        name=f'{psi_deg:.1f}',
        data=static_3d + mov + [static_ribbon, ribbon_cursor, static_wordheat, heat_cursor],
        layout=dict(title=dict(
            text=title_for(psi_deg, k),
            font=dict(family='Georgia, serif', size=13, color='#152B4A'))),
    ))

fig.frames = frames

# ─── Layout ─────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=title_for(psi_init, mid_idx),
        font=dict(family='Georgia, serif', size=13, color='#152B4A')),
    paper_bgcolor='#FAF6EE',
    plot_bgcolor='#FAF6EE',
    margin=dict(l=60, r=80, t=90, b=40),
    width=1150, height=1050,
    scene=dict(
        xaxis=dict(range=[-1.8, 1.8], title='x', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
        yaxis=dict(range=[-1.8, 1.8], title='y', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
        zaxis=dict(range=[-1.8, 1.8], title='z', showgrid=True, gridcolor='#EEEAE0', zerolinecolor='#B4AC9C'),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.6, y=1.0, z=1.2), up=dict(x=0, y=1, z=0)),
        bgcolor='#FAF6EE',
    ),
    xaxis=dict(range=[PSI_LO_DEG - 1, PSI_HI_DEG + 1], title=None,
               showticklabels=True, tickfont=dict(size=10)),
    yaxis=dict(showticklabels=True, showgrid=False, zeroline=False,
               tickfont=dict(family='ui-monospace,Consolas,monospace', size=10)),
    xaxis2=dict(range=[PSI_LO_DEG - 1, PSI_HI_DEG + 1],
                title=dict(text='ψ (deg)', font=dict(size=11)),
                tickfont=dict(size=10)),
    yaxis2=dict(autorange='reversed', tickfont=dict(family='ui-monospace,Consolas,monospace', size=9)),
    updatemenus=[dict(
        type='buttons', showactive=False,
        x=0.05, y=1.05, xanchor='left', yanchor='top',
        buttons=[
            dict(label='▶ Play', method='animate',
                 args=[None, dict(frame=dict(duration=140, redraw=True),
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
)

out_dir = os.path.normpath(os.path.join(_HERE, '..', 'interactive_figures'))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'tilt_bound.html')
fig.write_html(out_path,
               include_plotlyjs='cdn',
               include_mathjax=False,
               config=dict(displayModeBar=True, displaylogo=False))
print(f'Wrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)')
print(f'  a = {A_DEG} deg  ->  beta = {BETA:.2f} deg')
print(f'  c = {C}   anchor at (0, {C}, 0)')
print(f'  psi sweep: [{PSI_LO_DEG}, {PSI_HI_DEG}] deg in {N_FRAMES} frames')
print(f'  admissibility: stable={n_stable}/{N_FRAMES}, UPO={n_upo}/{N_FRAMES}, '
      f'any={n_either}/{N_FRAMES}  across {len(words)} enumerated words (L in 1..{L_MAX})')
