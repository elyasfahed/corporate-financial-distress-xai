"""Refresh presentation-only descriptive artifacts for ``final_primary``.

This script reads the frozen train/validation/test parquets and does not fit,
tune, or overwrite any model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    DATA_SAMPLES_V2,
    OUT_TABLES_DESCRIPTIVE,
    V2_PROFILE,
)
from src.utils.fig_style import apply_thesis_style

apply_thesis_style()


FEATURES = list(V2_PROFILE["feature_set"])
TABLE_DIR = OUT_TABLES_DESCRIPTIVE / V2_PROFILE["spec"]


def _write_summary_table(panel: pd.DataFrame) -> None:
    prior_path = TABLE_DIR / "predictor_summary_stats.csv"
    prior = pd.read_csv(prior_path).set_index("feature") if prior_path.exists() else None
    distressed = panel.loc[panel["distress"].eq(1)]
    healthy = panel.loc[panel["distress"].eq(0)]
    rows: list[dict] = []
    for feature in FEATURES:
        d = distressed[feature].dropna().astype(float)
        h = healthy[feature].dropna().astype(float)
        # Preserve the previously verified Welch inference columns; this
        # presentation-only refresh adds group-specific quantiles and does not
        # change any inferential result.
        if prior is None or feature not in prior.index:
            raise FileNotFoundError(
                "Verified Welch statistics are required before presentation refresh."
            )
        t_stat = float(prior.loc[feature, "welch_t"])
        p_value = float(prior.loc[feature, "p_value"])
        rows.append({
            "feature": feature,
            "mean_D": d.mean(), "median_D": d.median(), "std_D": d.std(),
            "p10_D": d.quantile(.10), "p90_D": d.quantile(.90),
            "mean_N": h.mean(), "median_N": h.median(), "std_N": h.std(),
            "p10_N": h.quantile(.10), "p90_N": h.quantile(.90),
            "welch_t": float(t_stat), "p_value": float(p_value),
        })
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(TABLE_DIR / "predictor_summary_stats.csv", index=False)

    display = table.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "--"
        )
    display["p_value"] = table["p_value"].map(
        lambda value: "$<0.0001$" if value < .0001 else f"{value:.4f}"
    )
    header = (
        "\\begin{landscape}\n"
        "\\begin{table}[p]\n\\centering\n"
        "\\caption[Predictor summary statistics by outcome group]{Predictor summary "
        "statistics by outcome group, final modelling sample (110,837 firm-years). "
        "D denotes distressed and N non-distressed observations. The final columns "
        "report Welch's unequal-variance test of the difference in group means.}\n"
        "\\label{tab:predictor_stats}\n\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2.4pt}\n"
        "\\begin{tabular}{lrrrrrrrrrrrr}\n\\toprule\n"
        "& \\multicolumn{5}{c}{Distressed (D)} & "
        "\\multicolumn{5}{c}{Non-distressed (N)} & \\multicolumn{2}{c}{Mean test} \\\\\n+"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-13}\n"
        "Feature & Mean & Median & SD & P10 & P90 & Mean & Median & SD & P10 & P90 & $t$ & $p$ \\\\\n+"
        "\\midrule\n"
    )
    body_lines = []
    for _, row in display.iterrows():
        vals = [str(row["feature"]).replace("_", "\\_")] + [str(row[c]) for c in display.columns[1:]]
        body_lines.append(" & ".join(vals) + " \\\\")
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\end{landscape}\n"
    (TABLE_DIR / "predictor_summary_stats.tex").write_text(
        (header + "\n".join(body_lines) + footer).replace("\n+", "\n"),
        encoding="utf-8"
    )


def _write_correlation_figure(panel: pd.DataFrame) -> None:
    """Full-square correlation heatmap, drawn with the shared figure style.

    Replaces a hand-emitted TikZ picture. TikZ typeset the labels in the exact
    body font, but it could not share ``src.utils.fig_style`` and so had to be
    kept visually consistent with every other figure by hand. Rendering it here
    means the matrix inherits the same palette, type and grid settings as the
    rest of the thesis automatically.
    """
    corr = panel[FEATURES].corr()
    corr.round(3).to_csv(TABLE_DIR / "correlation_matrix.csv")

    names = list(corr.columns)
    n = len(names)
    values = corr.values

    fig, ax = plt.subplots(figsize=(0.62 * n + 2.0, 0.62 * n + 0.9))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    # Predictor names on both margins: the conventional arrangement, so a cell
    # can be read straight off either edge without tracing to a diagonal.
    ax.set_xticks(range(n), names, rotation=45, ha="right", fontsize=9.0)
    ax.set_yticks(range(n), names, fontsize=9.0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = float(values[i, j])
            # No leading zero (APA 6.02) and a true minus sign; at 18x18 the
            # saved character width is what keeps neighbouring entries apart.
            text = f"{v:.2f}".replace("0.", ".").replace("-.", "−.")
            if v >= 0.9995:
                text = "1"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.0,
                    color="white" if abs(v) > 0.55 else "black")

    # Short label inside the figure; the descriptive sentence belongs to the
    # "Fig. N." caption below it (Elsevier convention).
    ax.set_title("Pairwise Pearson Correlation Matrix", pad=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.02)
    cbar.set_label("Pearson r", fontsize=11.0)
    cbar.ax.tick_params(labelsize=9.5)

    figure_dir = ROOT / "outputs" / "figures" / "descriptive" / V2_PROFILE["spec"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figure_dir / f"correlation_matrix.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure -> {figure_dir / 'correlation_matrix'}.{{png,pdf}}")


def main() -> None:
    panel = pd.concat(
        [pd.read_parquet(DATA_SAMPLES_V2 / f"{split}.parquet")
         for split in ("train", "val", "test")],
        ignore_index=True,
    )
    _write_summary_table(panel)
    _write_correlation_figure(panel)
    print("Refreshed final_primary summary statistics and correlation matrix.")


if __name__ == "__main__":
    main()
