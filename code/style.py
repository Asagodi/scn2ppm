"""Shared plotting style for all figures.

Uses seaborn's clean whitegrid context as the base, then layers paper-specific
overrides for math fonts, line weights, and consistent typography across all
figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def setup():
    """Apply seaborn-based clean style with paper-matching math fonts."""
    if _HAS_SEABORN:
        sns.set_theme(context='paper', style='ticks', palette='deep',
                      font='serif', rc={'axes.linewidth': 0.9})

    # Real LaTeX so calligraphic / fancy math glyphs match the paper exactly.
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times", "Computer Modern Roman", "DejaVu Serif"]
    mpl.rcParams["axes.formatter.use_mathtext"] = True

    # Consistent type sizes across every figure: bumped up vs the old defaults
    # so panels are readable at the printed scale.
    plt.rcParams.update({
        'font.size': 15,
        'axes.linewidth': 0.9,
        'axes.labelsize': 17,
        'axes.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 12,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.6,
        'lines.markersize': 5.5,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.4,
    })


# Standard colors (kept compatible with prior figures)
COLORS = {
    'red': '#e74c3c',
    'blue': '#2980b9',
    'green': '#27ae60',
    'orange': '#e67e22',
    'purple': '#8e44ad',
    'dark': '#2c3e50',
    'gray': '#95a5a6',
    'accent': '#00008C',
}

# Neuron colors (indexed 0, 1, 2, ...) -- pulled from a seaborn-deep-like palette
NEURON_COLORS = ['#4C72B0', '#DD8452', '#55A467', '#C44E52', '#8172B3']


def panel_labels(fig, axes, labels=None, fontsize=18, dx=-0.02, dy=0.02):
    """Legacy fig-level panel label. Prefer panel_label(ax, letter) instead."""
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]
    for ax, lbl in zip(axes, labels):
        bb = ax.get_position()
        fig.text(bb.x0 + dx, bb.y1 + dy, f'$\\mathbf{{({lbl})}}$',
                 fontsize=fontsize, fontweight='bold', va='top', ha='right')


def panel_label(ax, letter, subtitle=None, fontsize=18, y=1.05):
    """Consistent panel labels across all figures.

    Bold "(a)" positioned via axes-fraction coordinates so the label sits
    at the same fractional height regardless of the parent figure's aspect
    ratio (matters when figures with different native aspect ratios are
    scaled to a common paper height side-by-side).
    """
    title = f'$\\mathbf{{({letter})}}$'
    if subtitle:
        title = title + f'\\ \\textbf{{{subtitle}}}'
    # x=0 (left) puts the label at the axes' left edge; y>1 places it above
    # the axes. Using axes coords rather than the pad= (points) argument
    # of set_title keeps positioning invariant under aspect-ratio changes.
    ax.set_title(title, loc='left', y=y,
                 fontsize=fontsize, fontweight='bold')
