# Figures

Rendered PDFs of every figure in the paper and supplement. This is also the output
directory: the scripts in [`../code/`](../code) write here.

Regenerate everything, or one figure:

```bash
cd ../code
python generate_all.py list
python generate_all.py
python generate_all.py fig_overview
```

Rasters (`*.png`), intermediate `*.svg` and HTML are not tracked here; the web-sized
previews the top-level README shows live in [`../docs/img/`](../docs/img).

## Main text

| Figure | Output | Script |
|---|---|---|
| **1** Geometric derivation of the transition matrix | `derivation_tikz.pdf` | `../code/derivation_tikz.tex` (`pdflatex`) |
| **2** Overview at N=K=2: geometry, orbit, cobweb, convergence, staircase | `scn_word_example_paper.pdf` | `../code/fig_overview.py` |
| **3** Order and chaos at K=4, N=3 | `chaos_figure.pdf` | `../code/fig_chaos.py` |
| **4** Inverse problem: margin landscapes and solver scaling | `inverse_paper.pdf` | `../code/fig_inverse.py` |
| **5** Simon sequence stored as a period-6 orbit | `simon_rbygyb_paper.pdf` | `../code/fig_simon_box6.py` |
| **6** Chirality multistability and the line attractor | `multistability_pair.pdf` | `../code/fig_multistability_pair.py` |

Figure 1 is standalone TikZ rather than Python, so it is compiled directly:

```bash
cd ../code
pdflatex -output-directory=../figures derivation_tikz.tex
```

## Supplement

| Figure | Output | Script |
|---|---|---|
| Worked K=3, N=3 example | `fig_3d_example.pdf` | `../code/fig_3d_example.py` |
| Devil's staircases along each drive | `fig_3d_staircases.pdf` | `../code/fig_3d_staircases.py` |
| Cobweb gallery across eight plateaus | `cobweb_gallery.pdf` | `../code/fig_cobweb_gallery.py` |
| Combined drive and decoder noise | `combined_noise.pdf` | `../code/fig_combined_noise.py` |
| Attractor zoo, wide cone at N=50, K=10 | `orbit_zoo_wide_N50.pdf` | `../code/fig_orbit_zoo.py` |
| Basin map on a random tangent slice | `basin_map_wide.pdf` | `../code/fig_basin_map.py` |
| Atlas of 23 coexisting attractors | `multistability_highD.pdf` | `../code/fig_multistability_highD.py` |

## Runtimes

Most scripts finish in under a minute on a single CPU. The exceptions:

| Script | Time |
|---|---|
| `fig_combined_noise.py` | 2 to 3 minutes, noise scan |
| `fig_orbit_zoo.py` | 5 to 15 minutes, depending on cache |
| `fig_basin_map.py` | about 25 minutes on first run, then cached to `_basin_map_wide_cache.json` |
