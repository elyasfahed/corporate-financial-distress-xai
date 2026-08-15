"""
Clean calibration figure — same structure as the original, quantile bins.
=========================================================================
Identical layout, title, legend and colours to the original
src/robustness/rc7_figures.py calibration plot, with ONE change:
bins are equal-COUNT (strategy="quantile") instead of equal-WIDTH.

Why: the neural-network vertical spike to 1.0 in the original figure is an
equal-width binning artifact. With test prevalence ~1.4% the high-probability
bins are nearly empty, so a single distressed firm in a bin drives the
observed fraction to 1.0. Quantile bins put an equal number of firms in every
bin, so no bin can be dominated by one firm — the spike disappears and the
curves reflect real calibration.

Read-only w.r.t. the frozen pipeline. Writes ONLY new files:
  outputs/figures/robustness/rc7_calibration_comparison_clean.{pdf,png}

Predictions use the IDENTICAL preprocessing as the deployed models
(X = test[features].astype(float).fillna(0)).

Run from the project root (or just press Run in PyCharm):
    python -m src.robustness.calibration_figure_clean
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

from src.config import ALL_FEATURES, OUT_FIGURES_ROBUSTNESS, MODEL_DISPLAY_NAMES, DATA_SAMPLES
from src.utils.io import load_model
from src.utils.plots import set_academic_style, save_figure
import pandas as pd

LABEL = "distress"
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "neural_network"]
N_BINS = 10


def main() -> None:
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X = test[ALL_FEATURES].astype(float).fillna(0).values
    y = test[LABEL].astype(int).values
    preds = {m: load_model(m).predict_proba(X)[:, 1] for m in MODEL_ORDER}

    palette = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))

    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, colour in zip(MODEL_ORDER, palette):
        # strategy="quantile" = equal-count bins (the only change vs the original)
        prob_true, prob_pred = calibration_curve(
            y, preds[name], n_bins=N_BINS, strategy="quantile"
        )
        ax.plot(prob_pred, prob_true, marker="s", lw=1.5, color=colour,
                label=MODEL_DISPLAY_NAMES.get(name, name))

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title("Calibration Plots: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_figure(fig, OUT_FIGURES_ROBUSTNESS / "rc7_calibration_comparison_clean")
    print("Saved -> outputs/figures/robustness/rc7_calibration_comparison_clean.{pdf,png}")


if __name__ == "__main__":
    main()
