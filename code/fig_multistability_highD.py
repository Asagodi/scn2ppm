import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 23 attractors from _basin_check_wide.py output
raw = """
  count  36, period  18,  7 unique ([0, 1, 3, 4, 7, 8, 9]), word head: [0, 4, 7, 1, 9, 3, 8, 1, 9, 3, 1, 7, 8, 1, 9, 3, 1, 8]
  count  12, period  18,  7 unique ([0, 1, 3, 4, 7, 8, 9]), word head: [0, 4, 1, 9, 3, 8, 1, 7, 9, 1, 3, 8, 1, 9, 7, 1, 3, 8]
  count  10, period   1,  1 unique ([0]), word head: [0]
  count   8, period  31,  4 unique ([4, 5, 8, 9]), word head: [4, 5, 8, 9, 4, 8, 4, 9, 8, 4, 5, 8, 9, 4, 8, 4, 9, 8, 5, 4, 8, 9, 4, 8, 5]
  count   7, period  18,  7 unique ([0, 1, 3, 4, 7, 8, 9]), word head: [0, 4, 1, 9, 3, 8, 1, 7, 9, 1, 3, 8, 1, 9, 3, 1, 7, 8]
  count   7, period  19,  5 unique ([4, 5, 7, 8, 9]), word head: [4, 7, 5, 8, 4, 9, 7, 8, 4, 9, 7, 8, 4, 9, 8, 7, 4, 9, 8]
  count   7, period  15,  5 unique ([4, 5, 6, 7, 9]), word head: [4, 5, 7, 4, 9, 4, 7, 9, 4, 7, 9, 4, 6, 7, 9]
  count   6, period  17,  5 unique ([2, 4, 5, 6, 9]), word head: [2, 4, 9, 4, 9, 4, 6, 9, 4, 5, 9, 4, 6, 9, 4, 4, 9]
  count   6, period  19,  5 unique ([4, 5, 7, 8, 9]), word head: [4, 5, 8, 7, 4, 9, 8, 4, 7, 9, 8, 4, 7, 9, 8, 4, 7, 9, 8]
  count   6, period   1,  1 unique ([7]), word head: [7]
  count   6, period   8,  3 unique ([1, 4, 6]), word head: [1, 4, 1, 4, 1, 4, 1, 6]
  count   5, period   1,  1 unique ([8]), word head: [8]
  count   4, period  22,  5 unique ([0, 3, 4, 7, 9]), word head: [0, 3, 9, 4, 0, 7, 3, 9, 4, 0, 3, 9, 4, 7, 3, 9, 0, 4, 3, 9, 7, 4]
  count   4, period  23,  4 unique ([0, 5, 6, 9]), word head: [0, 5, 9, 0, 6, 9, 0, 5, 9, 6, 0, 9, 5, 0, 9, 6, 0, 9, 5, 0, 9, 6, 9]
  count   3, period  19,  5 unique ([4, 5, 7, 8, 9]), word head: [4, 7, 5, 8, 4, 9, 7, 8, 4, 9, 8, 7, 4, 9, 8, 4, 7, 9, 8]
  count   2, period  10,  3 unique ([1, 3, 4]), word head: [1, 3, 1, 4, 3, 1, 4, 1, 3, 4]
  count   2, period   7,  3 unique ([1, 4, 7]), word head: [1, 4, 1, 7, 4, 1, 7]
  count   2, period  19,  4 unique ([3, 4, 6, 7]), word head: [3, 4, 6, 3, 4, 7, 3, 4, 7, 4, 3, 6, 4, 3, 7, 4, 3, 4, 7]
  count   1, period   2,  2 unique ([0, 6]), word head: [0, 6]
  count   1, period  15,  5 unique ([4, 5, 7, 8, 9]), word head: [4, 8, 7, 5, 8, 4, 9, 7, 8, 4, 9, 7, 8, 4, 9]
  count   1, period   4,  4 unique ([4, 7, 8, 9]), word head: [4, 7, 9, 8]
  count   1, period  19,  5 unique ([4, 5, 7, 8, 9]), word head: [4, 7, 8, 5, 4, 9, 8, 7, 4, 9, 8, 4, 7, 9, 8, 4, 7, 9, 8]
  count   1, period  33,  5 unique ([0, 5, 6, 7, 9]), word head: [0, 7, 5, 9, 0, 7, 9, 6, 0, 7, 9, 5, 0, 7, 9, 6, 0, 7, 9, 5, 0, 7, 9, 6, 0]
"""

attrs = []
for line in raw.strip().split('\n'):
    m = re.search(r'count\s+(\d+), period\s+(\d+),\s+(\d+)\s+unique\s+\(\[([^\]]+)\]\), word head: \[([^\]]+)\]', line)
    if m:
        attrs.append({
            'count': int(m.group(1)),
            'period': int(m.group(2)),
            'n_unique': int(m.group(3)),
            'unique_neurons': [int(x) for x in m.group(4).split(',')],
            'word_head': [int(x) for x in m.group(5).split(',')],
        })

K = 10
for a in attrs:
    ratios = [0]*K
    for s in a['word_head'][:a['period']]:
        ratios[s] += 1
    a['ratios'] = np.array(ratios) / a['period']

print(f'{len(attrs)} attractors parsed, total ICs = {sum(a["count"] for a in attrs)}')

fig = plt.figure(figsize=(15.5, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 2.3, 1.2], hspace=0.55, wspace=0.28)

# Panel A: basin size distribution
axA = fig.add_subplot(gs[0, :])
counts = [a['count'] for a in attrs]
colors_A = plt.cm.viridis(np.linspace(0.15, 0.9, len(attrs)))
bars = axA.bar(range(len(attrs)), counts, color=colors_A, edgecolor='black', lw=0.5)
axA.set_xlabel('Attractor index (sorted by basin size)', fontsize=11)
axA.set_ylabel('IC count (basin size)', fontsize=11)
axA.set_title(f'$K=10, N=50$ wide-cone SCN: {len(attrs)} distinct stable attractors from 400 ICs',
              fontsize=13, weight='bold')
axA.set_xticks(range(len(attrs)))
for i, (a, c) in enumerate(zip(attrs, counts)):
    axA.text(i, c + max(counts)*0.02, f'p={a["period"]}', ha='center', fontsize=8)
axA.axhline(400/len(attrs), color='red', ls=':', alpha=0.6, label=f'uniform ({400/len(attrs):.1f})')
axA.legend(loc='upper right')

# Panel B: attractor x neuron heatmap
axB = fig.add_subplot(gs[1, :])
M = np.array([a['ratios'] for a in attrs])
im = axB.imshow(M, aspect='auto', cmap='hot', interpolation='nearest')
axB.set_xlabel(r'Neuron index $j$', fontsize=11)
axB.set_ylabel('Attractor index (sorted)', fontsize=11)
axB.set_title(r'Firing ratio $\rho_j$ per attractor (bright = fires often, dark = never fires)',
              fontsize=12)
axB.set_xticks(range(K))
axB.set_yticks(range(len(attrs)))
plt.colorbar(im, ax=axB, label=r'$\rho_j$')

# Panel C: n_unique_neurons
axC = fig.add_subplot(gs[2, 0])
weighted = np.zeros(K+1)
for a in attrs:
    weighted[a['n_unique']] += a['count']
axC.bar(range(K+1), weighted, color='#3388cc', edgecolor='black')
axC.set_xlabel('# distinct neurons in orbit', fontsize=11)
axC.set_ylabel('IC count (basin-weighted)', fontsize=11)
axC.set_title('Attractor "sparsity" (# firing neurons per attractor)', fontsize=12)
axC.set_xticks(range(K+1))

# Panel D: period distribution
axD = fig.add_subplot(gs[2, 1])
period_bins = {}
for a in attrs:
    p = a['period']
    period_bins[p] = period_bins.get(p, 0) + a['count']
periods_list = sorted(period_bins.keys())
axD.bar([str(p) for p in periods_list], [period_bins[p] for p in periods_list], color='#cc7733', edgecolor='black')
axD.set_xlabel('Period', fontsize=11)
axD.set_ylabel('IC count (basin-weighted)', fontsize=11)
axD.set_title('Attractor period distribution', fontsize=12)

plt.tight_layout()
_here = os.path.dirname(os.path.abspath(__file__))
for ext in ('png', 'pdf'):
    plt.savefig(os.path.join(_here, '..', 'figures', f'multistability_highD.{ext}'),
                dpi=150, bbox_inches='tight')
print('saved')

# Distinctness of readouts
rounded = set()
for a in attrs:
    rounded.add(tuple(np.round(a['ratios'], 3)))
print(f'{len(rounded)} out of {len(attrs)} attractors have distinct firing ratios')

# For each attractor, compute DC readout in R^N direction (which linear combo of D_j)
print()
print("All 23 attractors' firing ratios (neurons 0-9, - = 0):")
print(f"{'idx':>3} {'basin':>5} {'per':>4} " + " ".join(f'n{i}' for i in range(K)))
for i, a in enumerate(attrs):
    r = a['ratios']
    row = " ".join(f'{v:.2f}' if v > 0.001 else '  - ' for v in r)
    print(f'{i:>3} {a["count"]:>5} {a["period"]:>4}  {row}')
