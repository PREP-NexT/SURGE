"""Matplotlib style shared by the figure scripts: Myriad Pro typography, constrained
layout, thin spines with top/right removed, TrueType embedding.

Usage:
    from paper_style import setup_paper1_style
    setup_paper1_style()
"""
from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_MYRIAD_DIR = Path("/usr/share/fonts/opentype/MyriadPro")

# a perceptually uniform sequential colormap.
# Never a rainbow map.
SEQ_CMAP = "viridis"

# Colour-blind-safe categorical order (Okabe & Ito). The first three entries are
# what most panels use, and vermillion/bluish-green stay separable under
# deuteranopia in a way that pure red against green does not.
CB_BLUE   = "#0072B2"
CB_VERM   = "#D55E00"
CB_GREEN  = "#009E73"
CB_ORANGE = "#E69F00"
CB_PURPLE = "#CC79A7"
CB_GREY   = "#8895A3"
CATEGORICAL = [CB_BLUE, CB_VERM, CB_GREEN, CB_ORANGE, CB_PURPLE, CB_GREY]


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
        # sans-serif lettering (Myriad Pro) for accessibility. Mathtext uses a sans face
        # so symbols in axis labels match the surrounding text.
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Myriad Pro", "Lato", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",

        "font.size":        base_size,
        "axes.titlesize":   8.4,        # matches Figure 1/2 panel titles
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
        "axes.titleweight":  "normal",  # Figure 1/2 use unbolded panel titles
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
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


# ---- the paper figures -----------------------------------------------------------------
# The figures are built at the 166 mm text width, so a size set here is the size that
# prints. Words stay in Myriad Pro and equations are set in STIX, as in Figure 2, so a
# symbol in a figure looks the way it does in the text. Sizes follow
# the paper's figure style: 8 pt for ticks, axis labels and legends, 9 pt for
# titles inside a figure, and nothing smaller than 8 pt.


def setup_paper1_style() -> None:
    setup_paper_style(base_size=8.0)
    mpl.rcParams.update({
        "mathtext.fontset":      "stix",
        "font.size":             8.0,
        "axes.labelsize":        8.0,
        "axes.titlesize":        9.0,
        "xtick.labelsize":       8.0,
        "ytick.labelsize":       8.0,
        "legend.fontsize":       8.0,
        "legend.title_fontsize": 8.0,
        "figure.titlesize":      9.0,
    })


def panel_label(ax, letter, pad_pt=None, gap_mm=1.2):
    """Panel letter: bold 10 pt lower case,
    outside the frame at its top left. It sits on the baseline of the axes title, its right
    edge a small gap left of the frame, so the title keeps its place over the frame and no
    data moves. It is anchored to the axes, so a constrained layout makes room for it."""
    if pad_pt is None:
        pad_pt = mpl.rcParams["axes.titlepad"]
    return ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                       xytext=(-gap_mm * 72.0 / 25.4, pad_pt), textcoords="offset points",
                       ha="right", va="baseline", fontsize=10.0, fontweight="bold",
                       annotation_clip=False)


def myriad_ticks(fig) -> None:
    """Set every tick label of the figure, colour bars included, in Myriad Pro.

    Equations and symbols (k, KGE_c) stay in STIX, but numbers on the axes, also the
    powers of ten that mathtext builds, are lettering and use the text face. The tick
    labels are switched to the 'custom' mathtext set, whose faces (mathtext.rm and
    mathtext.it = 'sans') resolve to Myriad Pro. Call it just before saving: ticks made
    later copy the setting from the first tick of their axis."""
    mpl.rcParams.update({"mathtext.rm": "sans", "mathtext.it": "sans:italic",
                         "mathtext.bf": "sans:bold", "mathtext.sf": "sans"})
    import re
    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tick.label1, tick.label2):
                    # only numbers: a label with a symbol in it, such as '$k$ = 7', keeps STIX
                    maths = ''.join(re.findall(r'\$([^$]*)\$', lab.get_text()))
                    maths = re.sub(r'\\(mathdefault|times|minus)', '', maths)
                    if not re.search(r'[A-Za-z]', maths):
                        lab.set_math_fontfamily("custom")
