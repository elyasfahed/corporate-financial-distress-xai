"""
4-model calibration figure — BALANCED neural network, Platt-scaled.
====================================================================
Consistency companion to curve_figures_4models.py. Rebuilds the 4-model
calibration (reliability) figure using the **balanced** neural network
(neural_network_balanced) instead of the raw RC7 NN, so every results figure in
the thesis uses the same headline NN.

Every model is shown after the IDENTICAL Platt scaling already used for the
headline LR (validation-fit; the test set is never used to fit calibration), so
the comparison is fair — this is the same correct treatment as
src/robustness/calibration_figure_4models.py, only with the balanced NN.

Two panels are written so the like-for-like point is explicit:
  raw        — every model's raw probabilities (oversampled/weighted models look
               overconfident). Appendix / context.
  calibrated — every model after Platt scaling. THIS is the recommended thesis
               calibration figure; it matches the calibrated headline LR.

Read-only: loads saved models + val/test splits, fits Platt on validation only,
writes ONLY new files. Never re-fits/overwrites a saved model.

Outputs (new files):
  outputs/figures/model/calibration_4models_raw.{pdf,png}
  outputs/figures/model/calibration_4models_calibrated.{pdf,png}

Run
---
    python -m src.analysis.calibration_figures_4models
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from src.config import ALL_FEATURES, DATA_SAMPLES, OUT_FIGURES_MODEL
from src.analysis.lr_calibration import apply_platt_scaling, PlattScaledModel
from src.utils.io import load_model
from src.utils.plots import set_academic_style, save_figure

LABEL = "distress"
N_BINS = 10

# Balanced NN is the headline; LR/RF/XGB keep their primary colours.
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "neural_network_balanced"]
DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest":        "Random Forest",
    "xgboost":              "XGBoost",
    "neural_network_balanced": "Neural Network (MLP)",
}


def ece_quantile(y_true, y_prob, n_bins: int = N_BINS) -> float:
    """Equal-count (quantile) ECE — identical definition to the other figures."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    ece, n = 0.0, len(y_true)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (y_prob >= lo) & (y_prob <= hi) if last else (y_prob >= lo) & (y_prob < hi)
        m = int(mask.sum())
        if m:
            ece += (m / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def _raw_base(model):
    return model._base if isinstance(model, PlattScaledModel) else model


def _plot_panel(y_test, prob_dict, palette, out_stem, zoom: bool) -> None:
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, y_prob), colour in zip(prob_dict.items(), palette):
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=N_BINS, strategy="quantile")
        ece = ece_quantile(y_test, y_prob)
        ax.plot(prob_pred, prob_true, marker="s", ms=4, lw=1.5, color=colour,
                label=f"{DISPLAY.get(name, name)} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title("Calibration Plots: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    if zoom:  # rare event → all curves sit near the origin; corner zoom keeps [0,1] axes readable
        zlim = 0.06
        axin = ax.inset_axes([0.46, 0.30, 0.50, 0.50])
        for (name, y_prob), colour in zip(prob_dict.items(), palette):
            pt, pp = calibration_curve(y_test, y_prob, n_bins=N_BINS, strategy="quantile")
            axin.plot(pp, pt, marker="s", ms=3, lw=1.2, color=colour)
        axin.plot([0, zlim], [0, zlim], "k--", lw=0.8)
        axin.set_xlim(0, zlim); axin.set_ylim(0, zlim)
        axin.set_title(f"zoom: [0, {zlim:g}]", fontsize=7)
        axin.tick_params(labelsize=6)
        ax.indicate_inset_zoom(axin, edgecolor="gray", lw=0.8)

    save_figure(fig, out_stem)


def main() -> None:
    OUT_FIGURES_MODEL.mkdir(parents=True, exist_ok=True)
    val = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X_test = test[ALL_FEATURES].astype(float).fillna(0).values
    y_test = test[LABEL].astype(int).values

    raw_probs, cal_probs = {}, {}
    for name in MODEL_ORDER:
        base = _raw_base(load_model(name))
        platt = apply_platt_scaling(base, val)
        raw_probs[name] = base.predict_proba(X_test)[:, 1]
        cal_probs[name] = platt.predict_proba(X_test)[:, 1]

    palette = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))
    _plot_panel(y_test, raw_probs, palette,
                OUT_FIGURES_MODEL / "calibration_4models_raw", zoom=False)
    _plot_panel(y_test, cal_probs, palette,
                OUT_FIGURES_MODEL / "calibration_4models_calibrated", zoom=True)
    print("Saved -> outputs/figures/model/calibration_4models_{raw,calibrated}.{pdf,png}")
    print("NN shown is the BALANCED neural network (headline). Use the "
          "'calibrated' version as the thesis calibration figure.")


if __name__ == "__main__":
    main()
