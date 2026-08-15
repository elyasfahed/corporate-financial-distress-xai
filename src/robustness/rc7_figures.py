"""
RC7 comparison figures — primary portfolio + neural-network extension.
======================================================================
Builds ROC, Precision-Recall, and calibration figures that overlay the
neural-network extension (RC7) on the three frozen primary models
(Logistic Regression, Random Forest, XGBoost).

These are EXTENSION figures and are kept strictly separate from the primary
``outputs/figures/model/`` figures (which show only the frozen 3-model
portfolio, Blueprint v4). They are written to ``outputs/figures/robustness/``
and are explicitly titled as the RC7 model-class sensitivity comparison. The
primary figures are never touched.

Correctness / consistency guarantees
------------------------------------
Predicted probabilities are computed with the IDENTICAL preprocessing used by
``src.models.train.evaluate_on_test`` —
``X = test[features].astype(float).fillna(0)`` followed by
``model.predict_proba(X)[:, 1]`` — so every curve reproduces exactly the
numbers already reported in
  - ``outputs/tables/model_results/model_performance_test.csv``  (LR/RF/XGB)
  - ``outputs/tables/robustness/rc7_neural_network_results.csv``  (NN).

Because robustness checks overwrite the primary ``{name}.joblib`` cache (see
``train_all_models``), this module verifies that each loaded primary model
accepts the full 17-feature matrix. If a model was last saved by a
reduced-feature robustness run (e.g. RC4 11-feature or RC6 16-feature), it
raises a clear instruction to re-run Stage 3 first, rather than silently
plotting wrong curves.

Reference: extension to Blueprint v4 §10 (documented, not frozen).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

from src.config import (
    ALL_FEATURES,
    OUT_FIGURES_ROBUSTNESS,
    MODEL_DISPLAY_NAMES,
    DATA_SAMPLES,
)
from src.utils.io import load_model
from src.utils.plots import set_academic_style, save_figure

LABEL_COL = "distress"

# Order fixes the colour mapping. The first three colours of the colorblind
# palette are identical to the primary figures (which use n_colors=3), so
# LR/RF/XGB keep their established colours and the neural network receives a
# distinct fourth colour.
_MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "neural_network"]


def _ece_quantile(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error using equal-count (quantile) bins —
    the scalar summary printed next to each model in the legend."""
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    ece, n = 0.0, len(y_true)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = (y_prob >= lo) & (y_prob <= hi) if last else (y_prob >= lo) & (y_prob < hi)
        m = int(mask.sum())
        if m:
            ece += (m / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


# ---------------------------------------------------------------------------
# Prediction loading (no leakage, no re-fit — read-only)
# ---------------------------------------------------------------------------

def _load_predictions(features: list[str] = ALL_FEATURES) -> tuple[dict, np.ndarray]:
    """
    Load all four saved models and compute test-set probabilities using the
    exact preprocessing of evaluate_on_test.

    Raises
    ------
    RuntimeError
        If a primary model on disk was last saved by a reduced-feature
        robustness run and therefore cannot consume the 17-feature matrix.
    """
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X_test = test[features].astype(float).fillna(0).values
    y_test = test[LABEL_COL].astype(int).values

    preds: dict[str, np.ndarray] = {}
    for name in _MODEL_ORDER:
        model = load_model(name)          # read-only load from outputs/models/saved/
        try:
            preds[name] = model.predict_proba(X_test)[:, 1]
        except ValueError as exc:
            raise RuntimeError(
                f"Model '{name}' does not accept the {len(features)}-feature matrix "
                f"({exc}). Its .joblib was last overwritten by a reduced-feature "
                f"robustness run (e.g. RC4/RC6). Re-run Stage 3 to restore the "
                f"primary 17-feature models, then re-run Stage 15:\n"
                f"    python run_pipeline.py --stages 3\n"
                f"    python run_pipeline.py --stages 15"
            ) from exc
    return preds, y_test


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_rc7_figures(features: list[str] = ALL_FEATURES) -> None:
    """
    Build ROC, PR, and calibration figures overlaying the neural network on the
    three primary models. Writes only to outputs/figures/robustness/; touches
    nothing existing.
    """
    print("\n  RC7 comparison figures (primary portfolio + neural network)")
    OUT_FIGURES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)

    preds, y_test = _load_predictions(features)
    palette = sns.color_palette("colorblind", n_colors=len(_MODEL_ORDER))
    prevalence = float(y_test.mean())

    def _label(name: str) -> str:
        return MODEL_DISPLAY_NAMES.get(name, name)

    # ---- ROC ----
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in zip(_MODEL_ORDER, palette):
        fpr, tpr, _ = roc_curve(y_test, preds[name])
        ax.plot(fpr, tpr, lw=1.8, color=colour,
                label=f"{_label(name)} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, OUT_FIGURES_ROBUSTNESS / "rc7_roc_comparison")

    # ---- Precision-Recall ----
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in zip(_MODEL_ORDER, palette):
        precision, recall, _ = precision_recall_curve(y_test, preds[name])
        # Use average_precision_score (the AP estimator) for the legend number so
        # it MATCHES outputs/tables/model_results/model_performance_test.csv exactly.
        # sklearn warns against auc(recall, precision) (trapezoidal) for PR curves —
        # it interpolates optimistically and would disagree with the results table.
        pr_auc = average_precision_score(y_test, preds[name])
        ax.plot(recall, precision, lw=1.8, color=colour,
                label=f"{_label(name)} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(y=prevalence, color="gray", linestyle="--", lw=0.8,
               label=f"No-skill baseline (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, OUT_FIGURES_ROBUSTNESS / "rc7_pr_comparison")

    # ---- Calibration ----
    # Bins are equal-COUNT (strategy="quantile") rather than equal-WIDTH.
    # Rationale: with test prevalence ~1.4% (434 positives / 30,585 firm-years),
    # equal-width bins leave the high-probability range almost empty. In the
    # earlier equal-width version the MLP's [0.7,0.8) bin held exactly ONE firm,
    # a true distress, so the observed fraction was 1/1 = 1.0 and the curve drew
    # a spurious vertical spike to the top. Quantile bins place an equal number
    # of firms in every bin, so no bin can be dominated by a single firm; the
    # spike disappears and the curves reflect genuine calibration. The figure
    # caption discloses the equal-count binning.
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in zip(_MODEL_ORDER, palette):
        prob_true, prob_pred = calibration_curve(
            y_test, preds[name], n_bins=10, strategy="quantile"
        )
        ece = _ece_quantile(y_test, preds[name], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="s", lw=1.5, color=colour,
                label=f"{_label(name)} (ECE={ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title("Calibration Plots: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper left", frameon=False)
    # y-axis zoomed to the populated range: with a 1.4% base rate all observed
    # frequencies are small, so [0,1] would leave the figure ~80% empty. The
    # diagonal is clipped to the zoom but still marks the perfect-calibration
    # reference; the x-axis stays full so the overconfident RF/XGB high-score
    # bins remain visible.
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.25)
    save_figure(fig, OUT_FIGURES_ROBUSTNESS / "rc7_calibration_comparison")

    print("  RC7 comparison figures saved (ROC, PR, calibration) "
          "-> outputs/figures/robustness/")


if __name__ == "__main__":
    run_rc7_figures()
