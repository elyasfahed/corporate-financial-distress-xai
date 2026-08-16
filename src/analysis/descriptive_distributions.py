"""
Predictor distributions by distress status — thesis figure (Chapter 4).
========================================================================
Produces ONE clean, publication-quality figure: the distribution of each
continuous predictor for distressed vs. healthy firm-years, shown as a tidy
grid of density histograms. This complements the required predictor
summary-statistics table (Pre-Specified Empirical Design §8.2) by showing the same group
differences visually.

This is a REAL thesis figure (unlike src/utils/scratch.py output):
  - academic serif style, 300 DPI, saved as .pdf (LaTeX) + .png (draft)
  - written to outputs/figures/descriptive/  (the Chapter 4 namespace)
  - reproducible: re-running this script regenerates the identical figure

Reads only the saved feature panel (features_all.parquet). It does NOT touch
the frozen models/SHAP/robustness pipeline.

NOTE (transparent deviation): this descriptive figure is a post-freeze
addition supporting §8.2. Disclose it as such in the thesis.

Run
---
    python -m src.analysis.descriptive_distributions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from src.config import DATA_FEATURES, OUT_FIGURES_DESCRIPTIVE
from src.utils.plots import set_academic_style, save_figure


# Human-readable axis labels (NEVER show raw codes like "TLTA" in a thesis figure).
FEATURE_LABELS: dict[str, str] = {
    "NITA":     "Net income / assets (ROA)",
    "TLTA":     "Total liabilities / assets",
    "WCTA":     "Working capital / assets",
    "CLCA":     "Current liab. / current assets",
    "NITA_LAG": "Lagged ROA (t–1)",
    "CHIN":     "Change in net income",
    "CASHTA":   "Cash / assets",
    "OCF_TA":   "Operating cash flow / assets",
    "LNTA":     "log(Total assets)",
    "EXRET":    "Excess return (12m)",
    "SIGMA":    "Return volatility",
    "LNMK":     "log(Market cap)",
    "RSIZE":    "Relative size",
    "MB":       "Market-to-book",
    "PRICE":    "log(Price per share)",
}

# Binary 0/1 flags — shown as bars (share of firms with flag = 1), not histograms.
BINARY_LABELS: dict[str, str] = {
    "OENEG": "Negative book equity",
    "INTWO": "Two consecutive loss years",
}

# Colourblind-safe pair: blue = healthy, red = distressed.
_PAL = sns.color_palette("colorblind")
_HEALTHY_C, _DISTRESS_C = _PAL[0], _PAL[3]


def plot_predictor_distributions_by_distress(
    df: pd.DataFrame,
    label: str = "distress",
    output_path=None,
) -> None:
    """
    Grid of density histograms: each continuous predictor, distressed vs healthy.

    The view of each panel is winsorised to the pooled 1st–99th percentile so the
    bulk of the distribution is visible (extreme tails would otherwise flatten it).
    Densities are normalised within group, so the two classes are comparable
    despite the strong class imbalance.
    """
    set_academic_style()

    continuous = [c for c in FEATURE_LABELS if c in df.columns]   # 15 histograms
    binary = [c for c in BINARY_LABELS if c in df.columns]        # 2 bar panels
    panels = continuous + binary
    n = len(panels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    # Bigger canvas + extra spacing so labels are large and never crowded.
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows))
    fig.set_constrained_layout_pads(w_pad=0.14, h_pad=0.20, hspace=0.06, wspace=0.05)
    axes = axes.ravel()

    healthy_mask = df[label] == 0
    distress_mask = df[label] == 1

    for ax, feat in zip(axes, panels):
        if feat in FEATURE_LABELS:
            # ── Continuous predictor: density histogram (view = pooled 1–99 pct) ──
            s = df[feat].replace([np.inf, -np.inf], np.nan)
            lo, hi = s.quantile([0.01, 0.99])
            bins = np.linspace(lo, hi, 36)

            ax.hist(s[healthy_mask].clip(lo, hi), bins=bins, density=True,
                    alpha=0.55, color=_HEALTHY_C, edgecolor="none")
            ax.hist(s[distress_mask].clip(lo, hi), bins=bins, density=True,
                    alpha=0.55, color=_DISTRESS_C, edgecolor="none")
            ax.axvline(s[healthy_mask].median(), color=_HEALTHY_C, lw=1.4, ls="--")
            ax.axvline(s[distress_mask].median(), color=_DISTRESS_C, lw=1.4, ls="--")

            ax.set_title(FEATURE_LABELS[feat], fontsize=14, pad=6)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=12)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))   # fewer, readable ticks
        else:
            # ── Binary flag: share of firms with flag = 1, by group ──
            h_rate = float(df.loc[healthy_mask, feat].mean()) * 100
            d_rate = float(df.loc[distress_mask, feat].mean()) * 100
            bars = ax.bar(["Healthy", "Distressed"], [h_rate, d_rate],
                          color=[_HEALTHY_C, _DISTRESS_C], alpha=0.75, width=0.6)
            for bar, v in zip(bars, [h_rate, d_rate]):
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.0f}%",
                        ha="center", va="bottom", fontsize=12)

            ax.set_title(BINARY_LABELS[feat], fontsize=14, pad=6)
            ax.set_ylabel("% with flag = 1", fontsize=11)
            ax.set_ylim(0, max(h_rate, d_rate) * 1.30)
            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=11)
            ax.margins(x=0.25)

    # Hide any unused panels in the last row.
    for ax in axes[n:]:
        ax.set_visible(False)

    # Single shared legend for the whole figure.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_HEALTHY_C, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=_DISTRESS_C, alpha=0.55),
    ]
    n_healthy = int(healthy_mask.sum())
    n_distress = int(distress_mask.sum())
    fig.legend(
        handles,
        [f"Healthy firm-years (n = {n_healthy:,})",
         f"Distressed firm-years (n = {n_distress:,})"],
        loc="lower center", ncol=2, frameon=False, fontsize=15,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle(
        "Distribution of Predictors by Distress Status\nUS CRSP/Compustat Panel, 1990–2024",
        fontsize=18, y=1.03,
    )

    if output_path is None:
        output_path = OUT_FIGURES_DESCRIPTIVE / "predictor_distributions_by_distress"
    save_figure(fig, output_path)


def main() -> None:
    df = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    print(f"Loaded features_all.parquet  ({len(df):,} rows)")
    plot_predictor_distributions_by_distress(df)
    print("\nThesis figure written to outputs/figures/descriptive/")


if __name__ == "__main__":
    main()
