"""Shared thesis figure style.

Register chosen by the author on 2026-08-07 after comparing three rendered
candidates side by side: larger sans-serif type, a full background grid, boxed
axes and a framed legend, in the register common to information-systems and
business-school dissertations.

Palette. Model hues are the Matplotlib tab10 blue/orange/green/red. This is a
deliberate change from the Okabe-Ito set introduced by the Phase A reporting
fixes, made on the author's explicit instruction and after the accessibility
cost was put to them: adjacent tab10 pairs are harder to separate under
deuteranopia than Okabe-Ito. The mitigation is that colour is NEVER the sole
carrier of model identity here — every model also has its own line style and
marker (see MODEL_LINESTYLES / MODEL_MARKERS), so each series remains
identifiable in greyscale and under any colour-vision deficiency. Those
redundant encodings, introduced with the Okabe-Ito palette, are retained
unchanged and must not be removed.

Typography is sans-serif and therefore does NOT match the Latin Modern body
text of the thesis; that too was the author's choice among the three
candidates, in exchange for the larger, more legible type. Figures are
reproduced on the page well below their canvas size, so base type is set high
enough to survive the reduction.
"""
from __future__ import annotations

import matplotlib as mpl

MODEL_COLORS = {
    "logistic_regression":     "#1F77B4",   # blue
    "random_forest":           "#FF7F0E",   # orange
    "xgboost":                 "#2CA02C",   # green
    "neural_network_balanced": "#D62728",   # red
}

# Redundant encodings preserve model identity in grayscale and for readers
# with colour-vision deficiencies. Colour groups the models but is not the
# sole carrier of information. Load-bearing for accessibility under the tab10
# palette — see the module docstring.
MODEL_LINESTYLES = {
    "logistic_regression":     "-",
    "random_forest":           "--",
    "xgboost":                 "-.",
    "neural_network_balanced": ":",
}

MODEL_MARKERS = {
    "logistic_regression":     "o",
    "random_forest":           "s",
    "xgboost":                 "^",
    "neural_network_balanced": "D",
}

SERIES_NAVY = "#1f3864"   # single-series lines (e.g. distress-rate path)
FIGURE_BG = "#FFFFFF"
GRID_COLOR = "#C8CDD4"


def apply_thesis_style() -> None:
    """Apply the shared rcParams. Call once, after ``matplotlib.use``."""
    mpl.rcParams.update({
        # typography
        "font.family":      "sans-serif",
        "font.sans-serif":  ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size":        11.5,
        "axes.titlesize":   13.0,
        # Bold, on the author's instruction, to set the title clearly apart
        # from the axis labels at a glance.
        "axes.titleweight": "bold",
        "axes.titlepad":    10,
        "axes.labelsize":   12.5,
        "xtick.labelsize":  11.0,
        "ytick.labelsize":  11.0,
        "legend.fontsize":  10.5,
        # boxed axes with a full background grid behind the data
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.linewidth":     1.0,
        "axes.grid":          True,
        "axes.grid.which":    "major",
        "grid.color":         GRID_COLOR,
        "grid.alpha":         0.35,
        "grid.linewidth":     0.7,
        "axes.axisbelow":     True,
        # Centred above the axes, on the author's instruction. Note this means
        # each figure states its subject both in the title and again in its
        # LaTeX caption; that duplication is intended here.
        "axes.titlelocation": "center",
        "legend.frameon":     True,
        "legend.framealpha":  1.0,
        "legend.edgecolor":   "0.6",
        "legend.title_fontsize": 10.0,
        "lines.linewidth":    1.8,
        "figure.facecolor":   FIGURE_BG,
        "axes.facecolor":     FIGURE_BG,
        "savefig.facecolor":  FIGURE_BG,
        "savefig.transparent": False,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })
