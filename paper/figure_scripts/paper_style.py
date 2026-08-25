"""Paper-figure matplotlib style for the GPU-hydrology papers.

Mirrors the Hydropower_production_reanalysis figure style
(figures/style.py + _validation_style.py) so the three papers here match the
reanalysis manuscript: Myriad Pro typography, 7-10 pt sizes, constrained
layout, thin spines with top/right removed, TrueType embedding, high-dpi raster.

Usage:
    from paper_style import setup_paper_style, OBS_COLOR, SIM_COLOR, SEQ_CMAP, DIV_CMAP
    setup_paper_style(base_size=8)
    fig, axes = plt.subplots(..., constrained_layout=True)  # or rely on rc default
    fig.savefig(out, dpi=300)   # NO bbox_inches="tight"
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_MYRIAD_DIR = Path("/usr/share/fonts/opentype/MyriadPro")

# --- shared palette (matches reanalysis _validation_style.py) ---
OBS_COLOR = "#0E3B5C"     # observed / reference / "truth": dark navy
SIM_COLOR = "#D32F2F"     # simulated / model: publication red
ACCENT    = "#1A7DB6"     # cool accent (blues)
ACCENT2   = "#E8A33D"     # warm accent (amber)
GOOD      = "#2E7D5B"     # green
MUTED     = "#8895A3"     # grey
PANEL_BG  = "#E8EEF5"     # light panel backdrop for multi-panel figures

# perceptually-uniform colormaps used throughout (sequential + diverging)
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"

CM2INCH = 1.0 / 2.54
WIDTH_SINGLE_IN = 8.5 * CM2INCH
WIDTH_DOUBLE_IN = 17.4 * CM2INCH

OBS_KW = dict(color=OBS_COLOR, linestyle="none", marker="o", markersize=3.0,
              markerfacecolor=OBS_COLOR, markeredgecolor="white",
              markeredgewidth=0.4, zorder=3, label="Observed")
SIM_KW = dict(color=SIM_COLOR, linewidth=1.4, linestyle="-", alpha=0.95,
              zorder=2, label="Simulated")
METRIC_BOX = dict(fontsize=7.5, va="top",
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.78,
                            boxstyle="round,pad=0.3"))


def _register_myriad():
    if not _MYRIAD_DIR.is_dir():
        return False
    for p in fm.findSystemFonts(fontpaths=str(_MYRIAD_DIR)):
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass
    return True


def setup_paper_style(base_size: float = 8.0) -> None:
    _register_myriad()
    plt.rcdefaults()
    mpl.rcParams.update({
        # paper1 figures: serif text matching the manuscript body (Palatino via
        # TeX Gyre Pagella) with Computer Modern mathtext, so figure lettering
        # matches the LaTeX text and equations exactly.
        "font.family":     "serif",
        "font.serif":      ["TeX Gyre Pagella", "P052", "DejaVu Serif"],
        "mathtext.fontset": "cm",

        "font.size":        base_size,
        "axes.titlesize":   base_size + 1,
        "axes.labelsize":   base_size,
        "xtick.labelsize":  base_size - 1,
        "ytick.labelsize":  base_size - 1,
        "legend.fontsize":  base_size - 1,
        "figure.titlesize": base_size + 1,

        "figure.constrained_layout.use":    True,
        "figure.constrained_layout.h_pad":  0.04,
        "figure.constrained_layout.w_pad":  0.04,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,

        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "svg.fonttype": "none",

        "axes.unicode_minus": False,
        "axes.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titleweight":  "bold",
        "axes.titlepad":     4.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "xtick.minor.size":  1.5,
        "ytick.minor.size":  1.5,
        "xtick.direction":   "out",
        "ytick.direction":   "out",

        "lines.linewidth":       1.2,
        "lines.markersize":      3.0,
        "lines.markeredgewidth": 0.4,
        "lines.solid_capstyle":  "round",

        "image.cmap":     SEQ_CMAP,
        "axes.grid":      False,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "grid.alpha":     0.3,

        "legend.frameon":     False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,

        "savefig.dpi":   300,
        "savefig.bbox":  "standard",   # constrained layout handles spacing
        "figure.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=[
            ACCENT, SIM_COLOR, GOOD, ACCENT2, "#6A4C93", MUTED]),
    })


def panel_tag(ax, tag, dx=-0.02, dy=1.04):
    """Bold lower-case panel tag (a), (b), ... at the top-left, journal style."""
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def finish(fig, out_paths, dpi=300):
    """Save a constrained-layout figure to one or more paths (no bbox=tight)."""
    if isinstance(out_paths, (str, Path)):
        out_paths = [out_paths]
    for p in out_paths:
        fig.savefig(p, dpi=dpi)
