"""
PR and ROC curves for the 4-model portfolio — BALANCED neural network.
=======================================================================
Rebuilds the Precision–Recall and ROC curves for LR / RF / XGBoost / NN using
the **balanced** neural network (neural_network_balanced, the headline NN with
PR-AUC ≈ 0.102) instead of the raw RC7 NN (≈ 0.123). This makes the curve
figures consistent with the headline 4-model figure
(model_performance_comparison_4models), where the balanced NN
is the headline and the raw NN is only a one-line sensitivity.

Read-only: loads already-saved models and computes test predictions with the
exact preprocessing of evaluate_on_test. Nothing is re-trained. Writes only:
  outputs/figures/model/pr_curves_4models.{pdf,png}
  outputs/figures/model/roc_curves_4models.{pdf,png}

Run
---
    python -m src.analysis.curve_figures_4models
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.config import ALL_FEATURES, OUT_FIGURES_MODEL, DATA_SAMPLES
from src.utils.io import load_model
from src.utils.plots import set_academic_style, save_figure

LABEL_COL = "distress"

# LR/RF/XGB keep their primary colours (first 3 of the colorblind palette); the
# balanced NN takes the 4th colour — identical mapping to the headline figure.
_MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "neural_network_balanced"]
_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest":        "Random Forest",
    "xgboost":              "XGBoost",
    "neural_network_balanced": "Neural Network (MLP)",
}


def _load_predictions(features=ALL_FEATURES):
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X = test[features].astype(float).fillna(0).values
    y = test[LABEL_COL].astype(int).values
    preds = {name: load_model(name).predict_proba(X)[:, 1] for name in _MODEL_ORDER}
    return preds, y


def main(features=ALL_FEATURES) -> None:
    OUT_FIGURES_MODEL.mkdir(parents=True, exist_ok=True)
    preds, y = _load_predictions(features)
    palette = sns.color_palette("colorblind", n_colors=len(_MODEL_ORDER))
    prevalence = float(y.mean())

    # ---- ROC ----
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, colour in zip(_MODEL_ORDER, palette):
        fpr, tpr, _ = roc_curve(y, preds[name])
        ax.plot(fpr, tpr, lw=1.8, color=colour,
                label=f"{_DISPLAY[name]} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, OUT_FIGURES_MODEL / "roc_curves_4models")

    # ---- Precision-Recall ----
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, colour in zip(_MODEL_ORDER, palette):
        precision, recall, _ = precision_recall_curve(y, preds[name])
        pr_auc = average_precision_score(y, preds[name])   # matches the results table
        ax.plot(recall, precision, lw=1.8, color=colour,
                label=f"{_DISPLAY[name]} (PR-AUC = {pr_auc:.3f})")
    ax.axhline(prevalence, color="gray", ls="--", lw=0.8,
               label=f"No-skill baseline (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save_figure(fig, OUT_FIGURES_MODEL / "pr_curves_4models")

    print("Saved -> outputs/figures/model/pr_curves_4models.{pdf,png}, roc_curves_4models.{pdf,png}")
    print("NN shown is the BALANCED neural network (headline), consistent with "
          "model_performance_comparison_4models.")


if __name__ == "__main__":
    main()
