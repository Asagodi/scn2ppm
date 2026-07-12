"""Plot the entire zoo of attractors at the wide-cone N=50, K=10 network.
Short-period orbits come from a 400-IC random basin survey; long-period ones
from the _diverged_classify cache (if available).
Panels are width-scaled by period so long orbits don't get squashed."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import style
style.setup()
mpl.rcParams['text.usetex'] = False

from _orbit_raster_atlas import (
    make_cone_system, simulate_word, canonical, find_atlas,
    orbit_y_positions,
)


def render_one_orbit(ax, word, sys, label=None, n_periods=2):
    K = sys['K']
    L = len(word)
    n_show = L * n_periods
    for p in range(n_periods):
        for k in range(L):
            i = word[k]
            n = p * L + k
            ax.scatter(n, i, marker='|', s=80, color=mpl.cm.tab20(i / 20.0),
                       linewidths=2.0, zorder=5)
    for i in range(K):
        ax.axhline(i, color='0.92', lw=0.4, zorder=0)
    ax.set_xlim(-0.5, n_show - 0.5)
    ax.set_ylim(-0.7, K - 0.3)
    if L <= 20:
        ax.set_xticks(np.arange(0, n_show + 1, max(1, L // 2)))
    else:
        ax.set_xticks(np.arange(0, n_show + 1, L))
    ax.set_yticks(range(K))
    ax.set_yticklabels([str(i) for i in range(K)], fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlabel('step $n$', fontsize=7)
    if label:
        ax.set_title(label, fontsize=7, pad=2)


print("=== Wide-cone N=50, K=10: full zoo ===")
s = make_cone_system(K=10, N=50, seed=7974288,
                     max_halfangle=1.1553566635680308)
atlas = find_atlas(s, n_ics=400)
print(f"  Short-period atlas: {len(atlas)} attractors")

# Pull long-period attractors from classifier output (if present)
_here = os.path.dirname(os.path.abspath(__file__))
_classify = os.path.normpath(os.path.join(_here, '_diverged_classify.json'))
long_period_atlas = {}
if os.path.exists(_classify):
    with open(_classify) as f:
        classify_data = json.load(f)
    for d in classify_data['details']:
        if d['outcome'] == 'long_period' and 'canonical' in d.get('info', {}):
            w = tuple(d['info']['canonical'])
            long_period_atlas[w] = long_period_atlas.get(w, 0) + 1
    print(f"  Long-period atlas: {len(long_period_atlas)} attractors")
else:
    print(f"  (no classifier output yet)")

# Merge: convert short-period counts to fractions of own survey, long-period to
# fractions of their own survey, and combine via weighted basin estimate.
# Simpler approach: just label them with their survey-source counts.
zoo = []
for w, c in atlas.items():
    zoo.append({'word': list(w), 'count': c, 'src': 'short',
                'total': sum(atlas.values())})
for w, c in long_period_atlas.items():
    # skip ones already in short atlas (slow-convergence to known attractor)
    if w in atlas:
        continue
    zoo.append({'word': list(w), 'count': c, 'src': 'long',
                'total': sum(long_period_atlas.values()) if long_period_atlas else 1})
# sort by basin fraction within source, but bigger ones first overall
zoo.sort(key=lambda z: -z['count'] / z['total'])
n_orbits = len(zoo)
print(f"  Total distinct orbits in zoo: {n_orbits}")

# Layout: use gridspec so each panel's width ≈ period length
# Group orbits into rows; balance per-row total period to ~80-100 steps
ROW_BUDGET = 70 * 2   # 2 periods per panel, cap row width
rows = [[]]
row_widths = [0]
for z in zoo:
    w = len(z['word']) * 2
    if row_widths[-1] + w > ROW_BUDGET and len(rows[-1]) > 0:
        rows.append([])
        row_widths.append(0)
    rows[-1].append(z)
    row_widths[-1] += w
nrows = len(rows)
ncols_max = max(len(r) for r in rows)
print(f"  Layout: {nrows} rows, max {ncols_max} columns")

# Compute row heights from spec
fig_w = 16
fig_h = max(1.5 * nrows, 4)
fig = plt.figure(figsize=(fig_w, fig_h))
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
outer = GridSpec(nrows, 1, figure=fig, hspace=0.55,
                  top=0.95, bottom=0.05, left=0.03, right=0.99)
for ri, row in enumerate(rows):
    widths = [len(z['word']) * 2 for z in row]
    inner = GridSpecFromSubplotSpec(1, len(row), subplot_spec=outer[ri],
                                     width_ratios=widths, wspace=0.6)
    for ci, z in enumerate(row):
        ax = fig.add_subplot(inner[ci])
        word = z['word']
        pct = 100 * z['count'] / z['total']
        src_tag = 'S' if z['src'] == 'short' else 'L'
        label = (f'[{src_tag}] period {len(word)}, '
                  f'{len(set(word))} unique, basin {pct:.1f}%')
        render_one_orbit(ax, word, s, label=label)
fig.suptitle(f'Full attractor zoo at $N{{=}}50$, $K{{=}}10$ '
              f'(wide cone, half-angle 1.16 rad): {n_orbits} distinct orbits',
              fontsize=11, y=0.99)
_out = os.path.normpath(os.path.join(_here, '..', 'figures',
                                      'orbit_zoo_wide_N50.pdf'))
plt.savefig(_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {_out}")
