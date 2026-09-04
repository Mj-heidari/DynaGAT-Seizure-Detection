"""
Publication figure style for DynaGAT.

One place defines every colour and rule, so all figures read as one system.

Colour discipline (this is not decoration - it is what makes the figures
readable in print, in greyscale, and to colour-blind reviewers):

  * sequential magnitude  -> ONE hue, light to dark. Never a rainbow, never
    viridis/jet. A single-hue ramp survives greyscale conversion because
    lightness alone carries the value.
  * diverging (a signed difference such as ictal minus interictal) -> two
    opposed hues with a NEUTRAL GREY midpoint, equal steps per arm, and a
    symmetric colour limit so zero is always the neutral colour.
  * categorical identity -> a fixed slot order, assigned by entity and never
    by rank, so a figure that drops a series does not repaint the others.
  * status (pass/fail, above/below a cap) -> a reserved palette that is never
    reused for a data series, and always paired with a label or marker so the
    meaning never rests on hue alone.

Figure widths follow Elsevier: 90 mm single column, 140 mm 1.5 column,
190 mm double column.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "INK", "SERIES", "STATUS", "SEQ", "DIV", "SEQ_R",
    "W1", "W15", "W2", "apply_style", "save", "annotate_cells", "diverging_norm",
]

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
INK = {
    "surface":   "#fcfcfb",
    "primary":   "#0b0b0b",
    "secondary": "#52514e",
    "muted":     "#898781",
    "grid":      "#e1e0d9",
    "axis":      "#c3c2b7",
    "neutral":   "#f0efec",   # diverging midpoint
}

# Categorical slots, fixed order. Never cycle past these; fold to "other".
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Reserved for state, never for a data series. Always paired with a label.
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# Single-hue blue ramp, 100 -> 700.
_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Matching single-hue red arm for the diverging pair.
_RED = ["#fbdcdc", "#f7c3c3", "#f2a8a8", "#ed8d8d", "#e87070", "#e34948",
        "#d03b3b", "#b83030", "#9e2727", "#851f1f", "#6b1717"]

SEQ = LinearSegmentedColormap.from_list("dynagat_seq", _BLUE, N=256)
SEQ_R = LinearSegmentedColormap.from_list("dynagat_seq_r", _BLUE[::-1], N=256)
def _diverging(cool, warm, mid):
    """
    Build a diverging map with EQUAL arms and the neutral pinned at 0.5.

    Concatenating two lists of unequal length silently shifts the midpoint
    (13 blue + 1 + 11 red puts neutral at 0.54), which makes the "zero is
    neutral" reading a lie. Positions are therefore stated explicitly.
    """
    cool = list(cool)[::-1]
    stops = ([(0.5 * i / (len(cool) - 1), c) for i, c in enumerate(cool)]
             + [(0.5, mid)]
             + [(0.5 + 0.5 * (i + 1) / len(warm), c) for i, c in enumerate(warm)])
    return LinearSegmentedColormap.from_list("dynagat_div", stops, N=256)


DIV = _diverging(_BLUE, _RED, INK["neutral"])

# Elsevier column widths in inches
W1, W15, W2 = 3.54, 5.51, 7.48


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK["axis"],
        "axes.labelcolor": INK["primary"],
        "axes.titlecolor": INK["primary"],
        "text.color": INK["primary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        # Solid hairline grid. Dashed gridlines read as data.
        "grid.color": INK["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "pdf.fonttype": 42,      # embed TrueType, editable in Illustrator
        "ps.fonttype": 42,
        "figure.dpi": 120,
    })


def save(fig, name: str, outdir: Path) -> None:
    """Vector PDF for LaTeX, 600 dpi PNG for previews and Word."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", dpi=600)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _relative_luminance(rgb) -> float:
    c = np.asarray(rgb[:3], dtype=float)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def annotate_cells(ax, matrix, cmap, norm, fmt="{:.2f}", fontsize=6, mask=None):
    """
    Write each cell's value with ink chosen for contrast against that cell.

    Automatic, not eyeballed: cells whose fill is dark get white text, light
    cells get near-black. Guarantees legibility across the whole ramp.
    """
    m = np.asarray(matrix, dtype=float)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if mask is not None and mask[i, j]:
                continue
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=fontsize, color=INK["muted"])
                continue
            lum = _relative_luminance(cmap(norm(v)))
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize,
                    color="#ffffff" if lum < 0.45 else INK["primary"])


def diverging_norm(matrix) -> TwoSlopeNorm:
    """Symmetric limits so the neutral colour always means exactly zero."""
    m = np.asarray(matrix, dtype=float)
    v = np.nanmax(np.abs(m)) or 1.0
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
