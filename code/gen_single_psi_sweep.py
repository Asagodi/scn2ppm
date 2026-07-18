"""Precompute the symmetric-tilt (Option B) single-parameter sweep at
K=4, N=3.

Four decoders sit on a cone about the -y axis at polar angle theta,
equally spaced in azimuth (0, 90, 180, 270 deg). Apex-first is satisfied
automatically with alpha_i = y0 * cos(theta) for all i.

By azimuthal symmetry the six pairwise cosines take only TWO values:
  cos Psi_adj = cos^2(theta)                (four adjacent pairs, 90 deg apart)
  cos Psi_opp = cos(2 theta)                (two opposite pairs, 180 deg apart)

The sweep parameter is theta in (0, pi/2). At theta = 0 all decoders
degenerate onto -y (all cosines = 1). At theta = pi/2 the decoders lie
on the equator (cos_adj = 0, cos_opp = -1). The chaos-friendly negative-
cosine regime lives near large theta.

NOTE (why not "single Psi_ij"): moving one D_i continuously while
holding the other three fixed can preserve at most one of the three
inner products with them (fixed-dot-product circle intersections are
0-dimensional otherwise). A truly one-parameter sweep therefore has to
come from a symmetry that reduces the six pairwise cosines to fewer
independent values -- as symmetric-tilt does.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fig_chaos import build_sys, lyapunov_qr

APEX = np.array([0.0, -1.0, 0.0])
Y0   = 1.0
K    = 4


def decoders_at_theta(theta_deg):
    """Four unit decoders on cone about -y at polar angle theta_deg,
    azimuths 0, 90, 180, 270. Rows are unit-norm, y-component = -cos(theta),
    apex-first drive alpha = cos(theta) (same for all four)."""
    t = np.deg2rad(theta_deg)
    s = float(np.sin(t))
    c = float(np.cos(t))
    D = np.array([
        [ s * np.cos(np.deg2rad(  0)), -c, s * np.sin(np.deg2rad(  0))],
        [ s * np.cos(np.deg2rad( 90)), -c, s * np.sin(np.deg2rad( 90))],
        [ s * np.cos(np.deg2rad(180)), -c, s * np.sin(np.deg2rad(180))],
        [ s * np.cos(np.deg2rad(270)), -c, s * np.sin(np.deg2rad(270))],
    ])
    alphas = np.array([c, c, c, c])
    return D, alphas


# ─── Sweep theta ───────────────────────────────────────────────────────────
N_FRAMES     = 121
THETA_LO_DEG = 12.0        # avoid theta = 0 (degenerate)
THETA_HI_DEG = 87.0        # avoid theta = 90 (equator, alpha -> 0)
thetas       = np.linspace(THETA_LO_DEG, THETA_HI_DEG, N_FRAMES)

rows = []
for theta_deg in thetas:
    D, alphas = decoders_at_theta(theta_deg)
    cos_mat = D @ D.T

    sysd = build_sys(D, alphas)
    lam_max = float('nan')
    try:
        res = lyapunov_qr(sysd, n_transient=800, n_record=1500,
                          n_segments=1, seed=abs(int(1000 + 10 * theta_deg)))
        if res is not None:
            lyaps, _, _ = res
            lam_max = float(lyaps[0])
    except Exception as e:
        print(f'  theta={theta_deg:.1f} deg: Lyapunov failed ({e})')

    rows.append({
        'theta_deg':   round(float(theta_deg), 4),
        'D':           [[round(float(x), 6) for x in di] for di in D],
        'alpha':       round(float(alphas[0]), 6),   # all four equal
        'cos_adj':     round(float(cos_mat[0, 1]), 6),
        'cos_opp':     round(float(cos_mat[0, 2]), 6),
        'lam_max':     None if not np.isfinite(lam_max) else round(lam_max, 4),
    })

# Reporting
print(f'Sweep: theta in [{THETA_LO_DEG}, {THETA_HI_DEG}] deg, '
      f'{len(rows)} frames.')
lam = np.array([r['lam_max'] if r['lam_max'] is not None else np.nan
                for r in rows])
finite = np.isfinite(lam)
if finite.any():
    print(f'  lambda_max: min {np.nanmin(lam):+.3f}, '
          f'max {np.nanmax(lam):+.3f}, '
          f'chaotic frames (lam_max > 0): {(lam > 0).sum()}/{finite.sum()}')

# ─── Package for the HTML artifact ─────────────────────────────────────────
out = {
    'meta': {
        'K': 4, 'N': 3,
        'apex': APEX.tolist(),
        'y0':   Y0,
        'parameter': 'theta (polar angle of the decoder cone about -y)',
        'construction': (
            'K=4 decoders on cone about -y at polar angle theta, evenly '
            'spaced in azimuth (0, 90, 180, 270 deg). apex-first drive '
            'alpha_i = y0*cos(theta) for all i. Six pairwise cosines '
            'collapse to two by symmetry: cos_adj = cos^2(theta), '
            'cos_opp = cos(2 theta).'
        ),
        'theta_range_deg': [THETA_LO_DEG, THETA_HI_DEG],
        'n_frames': len(rows),
    },
    'frames': rows,
}
out_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'figures', 'single_theta_sweep.json'))
with open(out_path, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1)
print(f'\nWrote {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)')
