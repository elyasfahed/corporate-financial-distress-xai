"""
Academic calibration figure — like-for-like, all four models.
=============================================================
Replaces the apples-to-oranges calibration comparison (in which only the
Logistic Regression was Platt-scaled while RF/XGBoost/NN were shown raw)
with a defensible, internally consistent reliability diagram.

What this figure does differently
----------------------------------
1. **Same treatment for every model.** Post-hoc Platt scaling is fit on the
   VALIDATION set (2010–2014) and applied to ALL four models, using the
   IDENTICAL procedure already used for the headline LR
   (``src.analysis.lr_calibration.apply_platt_scaling``). The test set is
   never used to fit calibration. So the ECE values are comparable across
   models — the earlier figure's were not.
2. **Before vs. after.** Two panels: raw (as-trained) probabilities on the
   left, Platt-recalibrated probabilities on the right. This shows both the
   diagnosis (RF/XGBoost overconfidence from the rebalanced training prior)
   and the cure.
3. **Full [0,1] axes + score histogram.** Because distress is rare
   (test prevalence ~1.4%), every well-calibrated curve sits near the origin.
   A predicted-probability histogram (log count) under each panel shows where
   the probability mass actually is, so the near-empty upper range is honest
   rather than misleading. This is the standard reliability-diagram layout
   (Niculescu-Mizil & Caruana 2005; Guo et al. 2017).
4. **Rank metrics are invariant.** PR-AUC and ROC-AUC are unchanged by any
   monotone calibration map; the companion table reports both raw and
   calibrated values to make that explicit.

Read-only w.r.t. the frozen pipeline
------------------------------------
This module LOADS the saved models and the val/test splits and writes ONLY
new files. It never re-fits, re-tunes, or overwrites any saved model
(.joblib), config, or existing figure/table. Safe to run against the
do-not-re-run pipeline.

Outputs (new files only)
------------------------
  outputs/figures/robustness/calibration_academic_all_models.{pdf,png}
  outputs/tables/model_results/calibration_ece_brier_all_models.csv

Run from the project root:
    python -m src.robustness.calibration_figure_academic
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from src.config import (
    ALL_FEATURES,
    DATA_SAMPLES,
    OUT_FIGURES_ROBUSTNESS,
    OUT_TABLES_MODEL,
    MODEL_DISPLAY_NAMES,
)
from src.analysis.lr_calibration import apply_platt_scaling, PlattScaledModel
from src.utils.io import load_model
from src.utils.plots import set_academic_style, save_figure

LABEL = "distress"
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "neural_network"]
N_BINS = 10


# ---------------------------------------------------------------------------
# Expected Calibration Error — equal-count (quantile) bins.
# Identical definition to src/robustness/rc7_figures.py::_ece_quantile so the
# numbers are directly comparable to the existing figure.
# ---------------------------------------------------------------------------
def ece_quantile(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
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
    """Return the uncalibrated estimator. The saved LR is a PlattScaledModel;
    its ._base is the raw scaler+LR pipeline. All other saved models are raw."""
    return model._base if isinstance(model, PlattScaledModel) else model


def _reliability(ax, y_true, prob_dict, palette, title):
    """Draw a reliability diagram (quantile bins) for every model on one axes."""
    for (name, y_prob), colour in zip(prob_dict.items(), palette):
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=N_BINS, strategy="quantile"
        )
        ece = ece_quantile(y_true, y_prob)
        ax.plot(prob_pred, prob_true, marker="s", ms=4, lw=1.5, color=colour,
                label=f"{MODEL_DISPLAY_NAMES.get(name, name)} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def _hist(ax, prob_dict, palette):
    """Predicted-probability histogram (log count) under a reliability panel."""
    bins = np.linspace(0, 1, 51)
    for (name, y_prob), colour in zip(prob_dict.items(), palette):
        ax.hist(y_prob, bins=bins, histtype="step", lw=1.2, color=colour)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Count (log)")


def main() -> None:
    OUT_FIGURES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_MODEL.mkdir(parents=True, exist_ok=True)

    val = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X_test = test[ALL_FEATURES].astype(float).fillna(0).values
    y_test = test[LABEL].astype(int).values
    prevalence = float(y_test.mean())

    raw_probs: dict[str, np.ndarray] = {}
    cal_probs: dict[str, np.ndarray] = {}
    rows = []

    for name in MODEL_ORDER:
        base = _raw_base(load_model(name))                  # uncalibrated estimator
        platt = apply_platt_scaling(base, val)              # validation-fit, same as headline LR

        p_raw = base.predict_proba(X_test)[:, 1]
        p_cal = platt.predict_proba(X_test)[:, 1]
        raw_probs[name] = p_raw
        cal_probs[name] = p_cal

        rows.append({
            "model":            MODEL_DISPLAY_NAMES.get(name, name),
            "ece_raw":          round(ece_quantile(y_test, p_raw), 4),
            "ece_calibrated":   round(ece_quantile(y_test, p_cal), 4),
            "brier_raw":        round(brier_score_loss(y_test, p_raw), 4),
            "brier_calibrated": round(brier_score_loss(y_test, p_cal), 4),
            "mean_pred_raw":    round(float(p_raw.mean()), 4),
            "mean_pred_calib":  round(float(p_cal.mean()), 4),
            # rank metrics — identical raw vs calibrated by construction:
            "pr_auc_raw":       round(average_precision_score(y_test, p_raw), 4),
            "pr_auc_calib":     round(average_precision_score(y_test, p_cal), 4),
            "roc_auc_raw":      round(roc_auc_score(y_test, p_raw), 4),
            "roc_auc_calib":    round(roc_auc_score(y_test, p_cal), 4),
        })

    table = pd.DataFrame(rows)
    csv_path = OUT_TABLES_MODEL / "calibration_ece_brier_all_models.csv"
    table.to_csv(csv_path, index=False)
    print("\nCalibration summary (test set):")
    print(table.to_string(index=False))
    print(f"\nSaved table -> {csv_path}")

    # ---- Figure: 2 columns (raw | calibrated), reliability over histogram ----
    palette = sns.color_palette("colorblind", n_colors=len(MODEL_ORDER))
    set_academic_style()
    fig = plt.figure(figsize=(11, 6), constrained_layout=False)
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.18, figure=fig)

    ax_rel_raw = fig.add_subplot(gs[0, 0])
    ax_rel_cal = fig.add_subplot(gs[0, 1])
    ax_h_raw = fig.add_subplot(gs[1, 0], sharex=ax_rel_raw)
    ax_h_cal = fig.add_subplot(gs[1, 1], sharex=ax_rel_cal)

    _reliability(ax_rel_raw, y_test, raw_probs, palette,
                 "(a) Raw probabilities (as trained)")
    _reliability(ax_rel_cal, y_test, cal_probs, palette,
                 "(b) After Platt recalibration (validation-fit, all models)")

    # Zoom inset on the calibrated panel: at a ~1.4% base rate the well-calibrated
    # curves all live in [0, ~0.05], so a corner inset makes the agreement legible
    # without abandoning the full [0,1] axes.
    zlim = 0.06
    axin = ax_rel_cal.inset_axes([0.46, 0.10, 0.50, 0.50])
    for (name, y_prob), colour in zip(cal_probs.items(), palette):
        prob_true, prob_pred = calibration_curve(
            y_test, y_prob, n_bins=N_BINS, strategy="quantile"
        )
        axin.plot(prob_pred, prob_true, marker="s", ms=3, lw=1.2, color=colour)
    axin.plot([0, zlim], [0, zlim], "k--", lw=0.8)
    axin.set_xlim(0, zlim)
    axin.set_ylim(0, zlim)
    axin.set_title(f"zoom: [0, {zlim:g}]", fontsize=7)
    axin.tick_params(labelsize=6)
    ax_rel_cal.indicate_inset_zoom(axin, edgecolor="gray", lw=0.8)

    _hist(ax_h_raw, raw_probs, palette)
    _hist(ax_h_cal, cal_probs, palette)

    # Hide the reliability panels' x tick labels (shared with histograms below)
    plt.setp(ax_rel_raw.get_xticklabels(), visible=False)
    plt.setp(ax_rel_cal.get_xticklabels(), visible=False)
    ax_rel_raw.set_xlabel("")
    ax_rel_cal.set_xlabel("")

    fig.suptitle(
        "Calibration (Reliability Diagrams): Corporate Financial Distress — "
        f"Test Sample (2015–2024, prevalence = {prevalence:.3f})",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, OUT_FIGURES_ROBUSTNESS / "calibration_academic_all_models")
    print("Saved -> outputs/figures/robustness/calibration_academic_all_models.{pdf,png}")


if __name__ == "__main__":
    main()
