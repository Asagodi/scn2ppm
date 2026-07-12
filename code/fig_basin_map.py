"""Map the basins of attraction of the wide-cone N=50, K=10 network on a
2D slice of state space."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import style
style.setup()
mpl.rcParams['text.usetex'] = False


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

print(f"\nTop by basin size:")
for w, c in sorted(basin_count.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  basin {c}/{n_done} ({100*c/n_done:.1f}%), period {len(w)}, "
          f"{len(set(w))} unique, head: {list(w[:15])}")

# Build colour array — only as many slots as actual attractors on this slice
sorted_attrs = sorted(basin_count.items(), key=lambda kv: -kv[1])
# Show top 12 individually, group the rest as "other"; if <= 12, no "other" group
SHOW_TOP = min(12, len(sorted_attrs))
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

# Colour list: SHOW_TOP attractor colors + optional gray + black
cmap_attractors = mpl.cm.tab20
cmap_list = [cmap_attractors(i) for i in range(SHOW_TOP)]
if has_other:
    cmap_list.append((0.6, 0.6, 0.6, 1.0))   # gray for "other"
cmap_list.append((0, 0, 0, 1))               # black for "diverged"

# Use BoundaryNorm: each integer value gets one color slot.
n_colors = len(cmap_list)
boundaries = np.arange(-0.5, n_colors + 0.5)
norm = mpl.colors.BoundaryNorm(boundaries, n_colors)
custom_cmap = mpl.colors.ListedColormap(cmap_list)

# Plot
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.imshow(color_grid, origin='lower',
          extent=[xs[0], xs[-1], ys[0], ys[-1]],
          cmap=custom_cmap, norm=norm,
          interpolation='nearest', aspect='equal')
ax.set_xlabel('$\\xi$ component along $v_1$', fontsize=11)
ax.set_ylabel('$\\xi$ component along $v_2$', fontsize=11)
ax.set_title(f'Basins of attraction at $N=50$, $K=10$, half-angle 1.16 rad\n'
              f'{len(basin_count)} distinct attractors on slice; '
              f'2D slice of 49-D tangent space on $\\mathcal{{B}}_0$',
              fontsize=10)

# Legend — perfectly matched to the cmap_list ordering
from matplotlib.patches import Patch
legend_elements = []
for i, (w, count) in enumerate(sorted_attrs[:SHOW_TOP]):
    pct = 100 * count / n_done
    label = f'#{i+1}: period {len(w)}, {len(set(w))}u, {pct:.1f}%'
    legend_elements.append(Patch(facecolor=cmap_list[i], label=label))
if has_other:
    n_other_attractors = len(sorted_attrs) - SHOW_TOP
    other_count = sum(c for _, c in sorted_attrs[SHOW_TOP:])
    pct = 100 * other_count / n_done
    legend_elements.append(Patch(facecolor=cmap_list[SHOW_TOP],
                                  label=f'other ({n_other_attractors} attr, '
                                        f'{pct:.1f}%)'))
legend_elements.append(Patch(facecolor=cmap_list[-1], label='diverged'))
ax.legend(handles=legend_elements, loc='center left',
          bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.95)

plt.tight_layout()
_out = os.path.normpath(os.path.join(_here, '..', 'figures', 'basin_map_wide.pdf'))
plt.savefig(_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved {_out}")
