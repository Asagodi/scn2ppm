#!/usr/bin/env python3
"""Regenerate all figures for the paper.

Usage:
    python generate_all.py          # regenerate everything
    python generate_all.py fig1     # regenerate just one figure
    python generate_all.py list     # list available figures

All outputs go to ../figures/ (relative to this script).

Runtimes on a single CPU (informative only):
    fig1               ~5 s
    margin             ~10 s
    inverse_scaling    ~2 s
    3d_example         ~5 s
    3d_staircases      ~15 s
    cobweb_gallery     ~5 s
    combined_noise     ~2-3 min
    wm                 ~5 s
    simon_3d           ~10 s   (requires plotly + kaleido)
    multistability     ~10 s   (requires plotly + kaleido)
    orbit_zoo          ~5-15 min  (depends on cache)
    basin_map          ~25 min on first run  (caches to _basin_map_wide_cache.json)

The TikZ derivation figure is compiled separately:
    pdflatex -output-directory=../figures derivation_tikz.tex
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

FIGURES = {
    # ── Main text ──
    'fig1':               ('fig1_overview.py',               'Fig. 2 — Overview: geometry, orbit, cobweb, staircase'),
    'margin':             ('fig_margin_landscape.py',        'Fig. 4a,b — Admissibility-margin landscape'),
    'inverse_scaling':    ('fig_inverse_scaling.py',         'Fig. 4c — Inverse-solver scaling m*(K)'),
    'simon_3d':           ('fig_simon_3d.py',                'Fig. 5 — Simon RGB canonical exemplar (plotly)'),
    'multistability':     ('fig_multistability_topdown.py',  'Fig. 6 — Chirality multistability top-down'),
    # ── Working-memory storage regimes (optional, not in the current main draft) ──
    'wm':                 ('fig_wm.py',                      'Working-memory storage regimes (poster Box 5 layout)'),
    # ── Supplement ──
    '3d_example':         ('fig_3d_example.py',              'Suppl. S1 — K=N=3 worked example'),
    '3d_staircases':      ('fig_3d_staircases.py',           'Suppl. S2 — 3D staircases (scan each drive)'),
    'cobweb_gallery':     ('fig_cobweb_gallery.py',          'Suppl. S3 — Cobweb gallery across the staircase'),
    'combined_noise':     ('fig_combined_noise.py',          'Suppl. S4 — Combined drive/decoder noise'),
    'orbit_zoo':          ('fig_orbit_zoo.py',               'Suppl. — Attractor zoo (wide-cone N=50, K=10)'),
    'basin_map':          ('fig_basin_map.py',               'Suppl. — Basin map (wide-cone N=50, K=10)'),
    'multistability_hd':  ('fig_multistability_highD.py',    'Suppl. — Multistability atlas (23 attractors)'),
}

# These figures are shipped as PDFs but not yet ported to release-quality scripts:
NOT_YET_PORTED = [
    'chaos_figure.pdf — order/chaos transition (Fig. 3 main). Generation script pending.',
    'derivation_tikz.pdf — Fig. 1 main. Compile with: pdflatex -output-directory=../figures derivation_tikz.tex',
]


def run(script):
    path = os.path.join(HERE, script)
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=HERE, capture_output=False)
    dt = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {script} ({dt:.1f}s)")
    return result.returncode == 0


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'list':
            print("Available figures:")
            for key, (script, desc) in FIGURES.items():
                print(f"  {key:20s} {script:35s} {desc}")
            print("\nNot yet ported to a release-quality script:")
            for line in NOT_YET_PORTED:
                print(f"  {line}")
            return
        if arg in FIGURES:
            run(FIGURES[arg][0])
            return
        print(f"Unknown figure '{arg}'. Use 'list' to see options.")
        return

    # Generate all
    results = {}
    t_total = time.time()
    for key, (script, desc) in FIGURES.items():
        results[key] = run(script)
    dt_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"Summary ({dt_total:.0f}s total):")
    print(f"{'='*60}")
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {key:20s} {FIGURES[key][0]}")

    if NOT_YET_PORTED:
        print("\nNot regenerated (PDFs still present in ../figures/):")
        for line in NOT_YET_PORTED:
            print(f"  {line}")


if __name__ == '__main__':
    main()
