"""Map the basins of attraction of the wide-cone N=50, K=10 network on a
2D slice of state space."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import style
style.setup()   # CMU font + rcParams (see style.py)


def make_cone_system(K, N, seed, max_halfangle):
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal(N); ref /= np.linalg.norm(ref)
    A = np.eye(N) - np.outer(ref, ref)
    Q, _ = np.linalg.qr(A)
    U_ref = Q[:, :N - 1]
    DS = []
    for _ in range(K):
        u = rng.standard_normal(N - 1); u /= np.linalg.norm(u)
        theta = max_halfangle * np.sqrt(rng.random())
        d = np.cos(theta) * ref + np.sin(theta) * (U_ref @ u)
        d /= np.linalg.norm(d)
        DS.append(d)
    ES = []
    for d in DS:
        A = np.eye(N) - np.outer(d, d)
        Q, _ = np.linalg.qr(A)
        ES.append(Q[:, :N - 1])
    alphas = rng.uniform(0.5, 1.5, K).tolist()
    return {'DS': DS, 'ES': ES, 'alphas': alphas, 'K': K, 'N': N}


def scn_step(xi, bi, sys):
    DS, ES, al = sys['DS'], sys['ES'], sys['alphas']
    yp = (al[bi] + 1) * DS[bi] + ES[bi] @ xi
    best_j, best_c = -1, -np.inf
    for j in range(sys['K']):
        dyp = DS[j] @ yp
        if dyp <= 1e-12:
            continue
        c = al[j] / dyp
        if 0 < c <= 1.0 + 1e-12 and c > best_c:
            best_c, best_j = c, j
    if best_j < 0:
        return None, -1
    return ES[best_j].T @ (best_c * yp), best_j


def simulate_canonical(sys, xi0, bi0, n_warm=4000, n_record=80):
    xi, bi = xi0.copy(), bi0
    for _ in range(n_warm):
        out = scn_step(xi, bi, sys)
        if out[1] < 0:
            return None
        xi, bi = out
    seq = []
    for _ in range(n_record):
        seq.append(bi)
        out = scn_step(xi, bi, sys)
        if out[1] < 0:
            break
        xi, bi = out
    for p in range(1, min(len(seq) // 3, 35)):
        if seq[:p] == seq[p:2 * p] == seq[2 * p:3 * p]:
            word = seq[:p]
            # canonical form
            rots = [tuple(word[i:] + word[:i]) for i in range(len(word))]
            return min(rots)
    return None


# Rebuild the network
s = make_cone_system(K=10, N=50, seed=7974288, max_halfangle=1.1553566635680308)
print(f"N={s['N']}, K={s['K']}")

# Choose a 2D slice of the (N-1)=49-dim tangent space on bound 0.
# Use random orthonormal pair of directions for the slice.
rng = np.random.default_rng(2026)
v1 = rng.standard_normal(s['N'] - 1); v1 /= np.linalg.norm(v1)
v2 = rng.standard_normal(s['N'] - 1)
v2 -= (v2 @ v1) * v1
v2 /= np.linalg.norm(v2)

# 2D grid
RES = 80
SCALE = 3.0
xs = np.linspace(-SCALE, SCALE, RES)
ys = np.linspace(-SCALE, SCALE, RES)

# Cache file so we don't recompute the 25-minute simulation
_here = os.path.dirname(os.path.abspath(__file__))
_cache = os.path.normpath(os.path.join(_here, '_basin_map_wide_cache.json'))

if os.path.exists(_cache):
    print(f"Loading cached basin grid from {_cache}")
    with open(_cache) as f:
        cached = json.load(f)
    basin_grid = np.empty((RES, RES), dtype=object)
    for entry in cached['grid']:
        i, j, w = entry
        basin_grid[j, i] = tuple(w) if w is not None else None
    basin_count = {tuple(k): v for k, v in cached['count']}
    n_done = cached['n_done']
    print(f"  loaded {len(basin_count)} attractors, {n_done} admissible ICs")
else:
    basin_grid = np.empty((RES, RES), dtype=object)
    print(f"Mapping {RES}x{RES} basin grid (initial bound = 0)...")
    t0 = time.time()
    basin_count = {}
    n_done = 0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            xi0 = x * v1 + y * v2
            bi0 = 0
            word = simulate_canonical(s, xi0, bi0)
            if word is None:
                basin_grid[j, i] = None
                continue
            basin_grid[j, i] = word
            basin_count[word] = basin_count.get(word, 0) + 1
            n_done += 1
        if i % 10 == 0:
            print(f"  col {i+1}/{RES}, elapsed {time.time()-t0:.0f}s, "
                  f"{len(basin_count)} attractors found")
    print(f"Done in {time.time()-t0:.0f}s. {len(basin_count)} distinct attractors.")
    # Save cache
    grid_serial = []
    for i in range(RES):
        for j in range(RES):
            w = basin_grid[j, i]
            grid_serial.append([i, j, list(w) if w is not None else None])
    count_serial = [[list(k), v] for k, v in basin_count.items()]
    with open(_cache, 'w') as f:
        json.dump({'grid': grid_serial, 'count': count_serial,
                   'n_done': n_done}, f)
    print(f"  cached to {_cache}")

# --- Merge extended-budget classification of short-pass "unknowns" ------------
# The short-burn simulate_canonical above misses long-period orbits (period > 35
# or slow convergence) and leaves those cells as `None` — i.e. "diverged" in the
# colour map. The extended-budget pass (_diverged_classify.py, 20k burn-in,
# max period 130) reclassifies every one of those cells as a genuine long-period
# attractor. Loading its output collapses the "diverged" slice to zero and
# gives the merged basin map the paper reports.
_classify_path = os.path.normpath(os.path.join(_here, '_diverged_classify.json'))
if os.path.exists(_classify_path):
    with open(_classify_path) as f:
        classify_data = json.load(f)
    n_reclassified = 0
    for entry in classify_data['details']:
        i, j = entry['i'], entry['j']
        info = entry.get('info') or {}
        if entry['outcome'] == 'long_period' and 'canonical' in info:
            w = tuple(info['canonical'])
            if basin_grid[j, i] is None:
                basin_grid[j, i] = w
                basin_count[w] = basin_count.get(w, 0) + 1
                n_done += 1
                n_reclassified += 1
    print(f"  merged extended-budget classifier: "
          f"{n_reclassified} cells long-period, "
          f"{len(basin_count)} total attractors now")

print(f"\nTop by basin size:")
for w, c in sorted(basin_count.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  basin {c}/{n_done} ({100*c/n_done:.1f}%), period {len(w)}, "
          f"{len(set(w))} unique, head: {list(w[:15])}")

# Build colour array — only as many slots as actual attractors on this slice
sorted_attrs = sorted(basin_count.items(), key=lambda kv: -kv[1])
# Show ALL attractors individually (no "other" grouping); the wide-cone slice
# typically has ~10 attractors, all worth listing with their Fig 10 letter.
SHOW_TOP = len(sorted_attrs)
has_other = len(sorted_attrs) > SHOW_TOP
top_to_idx = {a[0]: i for i, a in enumerate(sorted_attrs[:SHOW_TOP])}

# value scheme:
# 0..SHOW_TOP-1: top attractors
# SHOW_TOP: "other" (if has_other)
# SHOW_TOP+1 (or SHOW_TOP if not has_other): "diverged"
DIVERGED = SHOW_TOP + (1 if has_other else 0)
OTHER = SHOW_TOP if has_other else None

color_grid = np.full((RES, RES), DIVERGED, dtype=int)
for i in range(RES):
    for j in range(RES):
        w = basin_grid[j, i]
        if w is None:
            color_grid[j, i] = DIVERGED
        elif w in top_to_idx:
            color_grid[j, i] = top_to_idx[w]
        else:
            color_grid[j, i] = OTHER
n_diverged = int(np.sum(color_grid == DIVERGED))
has_diverged = n_diverged > 0

# Colour list: SHOW_TOP attractor colors + optional gray + optional black
cmap_attractors = mpl.cm.tab20
cmap_list = [cmap_attractors(i) for i in range(SHOW_TOP)]
if has_other:
    cmap_list.append((0.6, 0.6, 0.6, 1.0))   # gray for "other"
if has_diverged:
    cmap_list.append((0, 0, 0, 1))           # black for "diverged"

# Use BoundaryNorm: each integer value gets one color slot.
n_colors = len(cmap_list)
boundaries = np.arange(-0.5, n_colors + 0.5)
norm = mpl.colors.BoundaryNorm(boundaries, n_colors)
custom_cmap = mpl.colors.ListedColormap(cmap_list)

# Plot — main basin map on the left, aligned table-legend on the right.
# Widened sidebar (was 6.5:3.0) so cells can fit larger fonts.
fig = plt.figure(figsize=(10.5, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[6.5, 4.0], wspace=0.05)
ax = fig.add_subplot(gs[0, 0])
ax_leg = fig.add_subplot(gs[0, 1])
# Matplotlib writes ≤256-colour images as 4-bit /Indexed with PNG predictor,
# yielding row widths of 1713*4=6852 bits (non-integer bytes). Acrobat
# rejects that with "Insufficient data for an image". Force an 8-bit RGB
# image by (i) building an explicit uint8 (H, W, 3) array and (ii) writing
# it directly as a PNG-in-PDF via ax.imshow with rasterized=True.
rgb_grid = np.zeros((RES, RES, 3), dtype=np.float32)
for jj in range(RES):
    for ii in range(RES):
        rgb_grid[jj, ii] = cmap_list[color_grid[jj, ii]][:3]
# One-LSB dithering (±1/510) breaks matplotlib's ≤256-colour detection so
# it writes an 8-bit /DeviceRGB image instead of 4-bit /Indexed. Invisible
# to the eye, decisive for Acrobat.
rgb_grid = np.clip(
    rgb_grid + np.random.default_rng(0).uniform(-1/510, 1/510, rgb_grid.shape),
    0.0, 1.0)
ax.imshow(rgb_grid, origin='lower',
          extent=[xs[0], xs[-1], ys[0], ys[-1]],
          interpolation='nearest', aspect='equal',
          rasterized=True)
# Figure is included at 0.75*textwidth in the paper; source-side fonts are
# scaled up (roughly 1/0.75 = 1.33x) so on-page rendering matches other
# figures (~16pt effective for ticks, ~20pt for labels).
ax.set_xlabel(r'$\xi$ component along $v_1$', fontsize=28)
ax.set_ylabel(r'$\xi$ component along $v_2$', fontsize=28)
ax.tick_params(labelsize=22)

# --- Right-side table-legend -------------------------------------------------
# Header + one row per attractor + optional "other" + "diverged". Column-1 is
# a filled colour swatch (BLANK text over a coloured background); remaining
# columns are rank, period L, distinct-firer count u, basin %.
ax_leg.axis('off')

# Load Fig 10's canonical-word -> panel-letter map (built by fig_orbit_zoo.py)
# so this legend can annotate which Fig 10 panel each attractor corresponds to.
_letter_map_path = os.path.normpath(os.path.join(_here, '_orbit_zoo_letters.json'))
letter_map = {}
if os.path.exists(_letter_map_path):
    with open(_letter_map_path) as f:
        letter_map = json.load(f)


def zoo_panel(word):
    """Return Fig 10 panel letter for a given canonical word, or '--'."""
    return letter_map.get(str(tuple(word)), '--')


# 5 columns: swatch | Fig10 panel | period L | unique-firer u | basin %.
# Dropped the rank column (#): Fig10 letter already identifies each attractor.
col_labels = ['', 'Fig10', r'$L$', r'$u$', 'basin \\%']

def _row(panel, period, uniq, pct, colour):
    return [' ', panel, period, uniq, pct], colour

table_rows = []
table_colours = []
for i, (w, count) in enumerate(sorted_attrs[:SHOW_TOP]):
    pct = 100 * count / n_done
    r, c = _row(zoo_panel(w), f'{len(w)}', f'{len(set(w))}',
                f'{pct:.1f}', cmap_list[i])
    table_rows.append(r); table_colours.append(c)
if has_other:
    n_other_attractors = len(sorted_attrs) - SHOW_TOP
    other_count = sum(c for _, c in sorted_attrs[SHOW_TOP:])
    pct = 100 * other_count / n_done
    r, c = _row('other', f'{n_other_attractors} attr', '', f'{pct:.1f}',
                cmap_list[SHOW_TOP])
    r[0] = ' '
    table_rows.append(r); table_colours.append(c)
if has_diverged:
    table_rows.append([' ', 'diverged', '', '', ''])
    table_colours.append(cmap_list[-1])

# Cell fill: only the leftmost column carries the attractor colour swatch.
cell_text    = [col_labels] + table_rows
cell_colours = [[(1, 1, 1, 0)] * len(col_labels)]
for c in table_colours:
    row = [c] + [(1, 1, 1, 0)] * (len(col_labels) - 1)
    cell_colours.append(row)

tbl = ax_leg.table(cellText=cell_text,
                    cellColours=cell_colours,
                    cellLoc='center', loc='center',
                    colWidths=[0.06, 0.24, 0.10, 0.10, 0.38])
tbl.auto_set_font_size(False)
tbl.set_fontsize(22)                # source-side; wider sidebar allows the bump
tbl.scale(1.0, 1.75)

# Left-align first column (swatch) and header; right-align numeric bodies.
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('0.85')
    cell.set_linewidth(0.5)
    if r == 0:                              # header row
        cell.get_text().set_weight('bold')
        cell.get_text().set_horizontalalignment('center')
        cell.set_edgecolor('0.4')
        cell.set_linewidth(0.9)
    elif c == 0:                            # colour swatch column
        cell.set_edgecolor('0.4')
        cell.set_linewidth(0.8)

fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.09, wspace=0.05)
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'basin_map_wide.pdf'))
plt.savefig(_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved {_out}")
