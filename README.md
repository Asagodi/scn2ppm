# Spike Coding Networks as Piecewise Projective Maps

**Closed-form stable periodic firing patterns and connectivity for working memory**

Reference implementation of the framework, the inverse-problem solver, one script per paper figure, and the full LaTeX source of the paper and supplement.

---

## Repository layout

```
scn2ppm/
├── paper/     LaTeX source of the paper + supplement (edit here to collaborate)
│              scn_paper_main.tex        main paper (with appendices inlined)
│              scn_appendices.tex        appendix bundle
│              scn_supplementary.tex     stand-alone supplement
│              _supp_defs_insert.tex, cascade_section.tex   fragments \input'd above
│              scn_paper_main.pdf, scn_supplementary.pdf    latest builds
├── bib/       Bibliography — all_ref.bib + catniplab.bib (self-contained)
├── code/      Python + TikZ generators for every paper figure, plus scn.py library
├── figures/   Rendered figure PDFs referenced by the paper
├── LICENSE, README.md, requirements.txt
```

To build the paper from source:

```bash
cd paper
pdflatex scn_paper_main       # produces the .aux
bibtex scn_paper_main         # resolves citations against ../bib/*.bib
pdflatex scn_paper_main       # (2×) fills cross-refs
pdflatex scn_paper_main
```

The supplement builds the same way (`scn_supplementary.tex`) and picks up cross-references to the main paper via `\externaldocument{scn_paper_main}` — build the main paper first.

---

## Overview

Under constant input, the spike-and-decay dynamics of a spike coding network (SCN) reduce **exactly** to a piecewise projective map on the disjoint union of the $(N-1)$-dimensional spike-bound hyperplanes: every spike corresponds to multiplication by a small fixed matrix, and every periodic firing pattern is the dominant eigenvector of the matrix product along its cycle. The core library `code/scn.py` exposes:

- **System construction** (`make_2d_system`, `make_3d_system`, `make_general_system`): decoders, drives, tangent frames on each bound (with the explicit `(-sin θ, cos θ)` convention that keeps the 2D cobweb and first-return-map formulas well-oriented).
- **Step map** (`step`, `simulate`): one spike-then-decay step in $\mathbb{R}^N$ and forward simulation of the continuous-time SCN.
- **Transition matrices** (`transition_matrix`, `return_matrix`): the per-pair $N\times N$ matrix $\mathbf{M}_{j\leftarrow i}$ and the return map $\mathbf{M}_\sigma$ for a firing word.
- **Eigenvector orbit** (`eigenvector_orbit`): given $(\mathbf{D}, \boldsymbol{\alpha}, \sigma)$, returns all $L$ orbit points from a single $N\times N$ eigendecomposition.
- **Admissibility margin** (`admissibility_margin`): the continuous score $m(\boldsymbol{\alpha})$ whose positivity is equivalent to $\sigma$ being dynamically realized.
- **Return-map diagnostics** (`get_firing_ratios`, `first_return_B1`, `dist_to_limit_cycle`): rotation number, first-return map on a chosen bound, and convergence diagnostics.

Everything else is a thin wrapper around these primitives.

---

## Quick start

```bash
git clone <ANONYMISED_REPO_URL>.git
cd scn2ppm
pip install -r requirements.txt

# Regenerate every figure into ../figures/
cd code
python generate_all.py list                 # see available targets
python generate_all.py                      # regenerate everything
python generate_all.py fig1                 # regenerate just one figure

# The geometric-derivation figure is compiled separately (needs TeX Live):
pdflatex -output-directory=../figures derivation_tikz.tex
```

Dependencies are intentionally minimal — see [`requirements.txt`](requirements.txt).

---

## Paper figure to script mapping

Main text:

| Paper figure | Output PDF | Generating script |
|---|---|---|
| **Fig. 1** — Geometric derivation of $\mathbf{M}_{j\leftarrow i}$ (6 panels) | `figures/derivation_tikz.pdf` | `code/derivation_tikz.tex` (`pdflatex`) |
| **Fig. 2** — $N{=}K{=}2$ overview: geometry, eigenvector orbit, cobweb, convergence, devil's staircase | `figures/fig1.pdf` | `code/fig1_overview.py` |
| **Fig. 3** — Order–chaos transition ($K{=}4$, $N{=}3$): sensitivity, phase diagram, strange attractor | `figures/chaos_figure.pdf` | PDF shipped; regeneration script pending |
| **Fig. 4** — Inverse problem: margin landscape (a,b) + solver scaling (c) | `figures/margin_landscape.pdf`, `figures/inverse_scaling.pdf` | `code/fig_margin_landscape.py`, `code/fig_inverse_scaling.py` |
| **Fig. 5** — Simon RGB canonical exemplar of the sequence-storage orbit (plotly render) | `figures/simon_3d.pdf` | `code/fig_simon_3d.py` |
| **Fig. 6** — Chirality multistability (two orbits related by $2\leftrightarrow 3$ reflection) | `figures/multistability_topdown.pdf` | `code/fig_multistability_topdown.py` |
| **Table 1** — Wall-clock benchmark: forward simulation vs. eigenvector method | (numeric only) | reproducible from `code/scn.py` primitives |

Supplement:

| Suppl. figure | Output PDF | Generating script |
|---|---|---|
| $K=3$, $N=3$ worked example (geometry + orbit + convergence) | `figures/fig_3d_example.pdf` | `code/fig_3d_example.py` |
| Devil's staircases along each drive of the 3-neuron system | `figures/fig_3d_staircases.pdf` | `code/fig_3d_staircases.py` |
| Cobweb gallery across eight staircase plateaus | `figures/cobweb_gallery.pdf` | `code/fig_cobweb_gallery.py` |
| Combined drive- and decoder-noise robustness | `figures/combined_noise.pdf` | `code/fig_combined_noise.py` |
| Attractor zoo for wide-cone $N=50$, $K=10$ (spike raster per stable orbit) | `figures/orbit_zoo_wide_N50.pdf` | `code/fig_orbit_zoo.py` |
| Basin map of the same network on a random tangent slice | `figures/basin_map_wide.pdf` | `code/fig_basin_map.py` |
| 23-attractor atlas (basins / periods / firers / representative spikes) | `figures/multistability_highD.png` | `code/fig_multistability_highD.py` |

Working-memory storage figure (poster Box 5 layout, kept for the revision cycle):

| | | |
|---|---|---|
| Persistent / multistable / continuous-attractor storage regimes | `figures/wm_figure.pdf` | `code/fig_wm.py` |

---

## Repository layout

```
scn2ppm/
  README.md               this file
  LICENSE                 MIT
  requirements.txt        numpy, scipy, matplotlib, plotly, kaleido, mpmath
  .gitignore
  code/
    scn.py                        core library
    style.py                      shared matplotlib style
    generate_all.py               runner (regenerate all or one figure)
    derivation_tikz.tex           Fig. 1 (standalone TikZ, compile with pdflatex)
    fig1_overview.py              Fig. 2 (5 panels)
    fig_margin_landscape.py       Fig. 4a,b
    fig_inverse_scaling.py        Fig. 4c
    fig_simon_3d.py               Fig. 5 (plotly)
    fig_multistability_topdown.py Fig. 6
    fig_wm.py                     Working-memory storage figure
    fig_3d_example.py             Suppl.
    fig_3d_staircases.py          Suppl.
    fig_cobweb_gallery.py         Suppl.
    fig_combined_noise.py         Suppl.
    fig_orbit_zoo.py              Suppl.
    fig_basin_map.py              Suppl.
    fig_multistability_highD.py   Suppl.
  figures/                pre-rendered PDFs of every figure (also the output dir)
```

---

## Reproducing individual figures

Every script is self-contained: it imports `scn` (and `style` for formatting), constructs the system, runs the analysis, and writes its output to `../figures/`. Run scripts from the `code/` directory:

```bash
cd code
python fig1_overview.py         # writes ../figures/fig1.pdf
python fig_margin_landscape.py  # writes ../figures/margin_landscape.pdf
python fig_simon_3d.py          # writes ../figures/simon_3d.pdf   (plotly+kaleido)
```

Runtimes on a single CPU are all under one minute except:
- `fig_combined_noise.py` (~2–3 min noise scan)
- `fig_basin_map.py` (~25 min on first run, caches to `_basin_map_wide_cache.json` on subsequent runs)
- `fig_orbit_zoo.py` (~5–15 min depending on cache availability)

---

## Cite this work

*A citation entry will be added at de-anonymisation.*


