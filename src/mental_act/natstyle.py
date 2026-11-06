"""Nature-style matplotlib defaults.

Single column 89 mm, double column 183 mm, max height 247 mm.
Sans-serif, 7 pt base, hairline spines, no grid, panel letters bold lowercase.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM

# Okabe-Ito colourblind-safe palette
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
          purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#000000")
HUMAN = "#3B3B3B"
HUMAN_FILL = "#BFBFBF"
POLE = "#D55E00"

RC = {
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
    "axes.linewidth": 0.5, "grid.linewidth": 0.4,
    "lines.linewidth": 0.9, "lines.markersize": 3.2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0, "legend.labelspacing": 0.35,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
plt.rcParams.update(RC)


def panel(ax, letter, dx=-0.16, dy=1.04, size=8.5):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def tidy(ax, ygrid=False):
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#DDDDDD", lw=0.4)
    ax.tick_params(pad=1.8)


def save(fig, path, also_pdf=True):
    fig.savefig(path, dpi=400)
    if also_pdf:
        fig.savefig(str(path).replace(".png", ".pdf"))
    print("wrote", path)
