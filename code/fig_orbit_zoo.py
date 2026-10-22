"""Plot the entire zoo of attractors at the wide-cone N=50, K=10 network.
Short-period orbits come from a 400-IC random basin survey; long-period ones
from the _diverged_classify cache (if available).
Panels are width-scaled by period so long orbits don't get squashed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    # Source figsize is 16 inches wide but the paper embeds at ~6 inches
    # (\textwidth), so on-page fonts are scaled down by ~0.375x. To achieve
    # ~11pt effective on the page, source-side sizes need to be ~3x that.
    ax.set_yticklabels([str(i) for i in range(K)], fontsize=13)
    ax.tick_params(axis='x', labelsize=13)
    ax.set_xlabel('step $n$', fontsize=15)
    if label:
        # Panel letter only (see caption for period/unique/survey key).
        ax.set_title(label, fontsize=18, fontweight='bold', loc='left', pad=3)


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

# Layout: orbit rows on the left, key-table sidebar on the right (analogous
# to fig_basin_map's right-sidebar legend). Source figsize is chosen so
# on-page (\textwidth ~6") fonts render at ~10-13pt without cranking the
# source sizes past 22pt.
fig_w = 13
fig_h = max(1.4 * nrows, 4)
fig = plt.figure(figsize=(fig_w, fig_h))
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# Widened sidebar (was 10.5:2.5) so the key table has room for larger fonts.
outer = GridSpec(1, 2, figure=fig, width_ratios=[8.5, 4.5], wspace=0.05,
                 left=0.05, right=0.99, top=0.94, bottom=0.05)
orbits_gs = GridSpecFromSubplotSpec(nrows, 1, subplot_spec=outer[0, 0],
                                     hspace=1.15)   # per-panel titles need room
ax_key = fig.add_subplot(outer[0, 1]); ax_key.axis('off')

panel_idx = 0
key_rows = []                                       # (letter, L, u) tuples
for ri, row in enumerate(rows):
    widths = [len(z['word']) * 2 for z in row]
    inner = GridSpecFromSubplotSpec(1, len(row), subplot_spec=orbits_gs[ri],
                                     width_ratios=widths, wspace=0.6)
    for ci, z in enumerate(row):
        ax = fig.add_subplot(inner[ci])
        word = z['word']
        letter = chr(ord('A') + panel_idx)         # A, B, ..., Z, AA, AB, ...
        if panel_idx >= 26:
            letter = 'A' + chr(ord('A') + panel_idx - 26)
        render_one_orbit(ax, word, s, label=letter)
        key_rows.append((letter, len(word), len(set(word))))
        panel_idx += 1

# --- Right-side key table --------------------------------------------------
# Dropped the 'src' column: it recorded the survey pass (short-burn vs
# extended-burn) that first found the orbit -- a discovery-method fact
# not an attractor-quality fact -- and only added noise to the key.
cell_text = [['panel', 'L', 'u']] + [
    [ltr, str(L), str(u)] for (ltr, L, u) in key_rows]
tbl = ax_key.table(cellText=cell_text, cellLoc='center', loc='center',
                   colWidths=[0.36, 0.30, 0.28])
tbl.auto_set_font_size(False)
tbl.set_fontsize(20)                # up from 14; wider sidebar allows it
tbl.scale(1.0, 1.45)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('0.85')
    cell.set_linewidth(0.5)
    if r == 0:
        cell.get_text().set_weight('bold')
        cell.set_edgecolor('0.4')
        cell.set_linewidth(0.9)

# Save Fig 10's letter-word mapping to disk so fig_basin_map.py can look up
# which panel each basin_map attractor corresponds to.
_mapping_path = os.path.normpath(os.path.join(_here, '_orbit_zoo_letters.json'))
letter_map = {}
panel_idx = 0
for z in zoo:
    letter = chr(ord('A') + panel_idx)
    if panel_idx >= 26:
        letter = 'A' + chr(ord('A') + panel_idx - 26)
    letter_map[str(tuple(z['word']))] = letter
    panel_idx += 1
with open(_mapping_path, 'w') as f:
    json.dump(letter_map, f)
print(f"Saved letter map to {_mapping_path}")

fig.suptitle(f'Full attractor zoo at $N{{=}}50$, $K{{=}}10$ '
              f'(wide cone, half-angle 1.16 rad): {n_orbits} distinct orbits',
              fontsize=20, y=0.995)
_out = os.path.normpath(os.path.join(_here, '..', 'figures',
                                      'orbit_zoo_wide_N50.pdf'))
plt.savefig(_out, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {_out}")
