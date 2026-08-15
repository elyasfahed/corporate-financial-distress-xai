"""
Four-model headline performance table.
======================================
Assembles the headline out-of-sample table presenting FOUR co-primary models:
LR, RF, XGBoost, and the imbalance-matched neural network (RC7b).

The headline deliberately omits Brier score. The saved primary table contains
a Platt-calibrated LR Brier alongside raw RF/XGBoost Brier scores, while later
calibration extensions apply transformations to additional models. Mixing
those quantities in the main discrimination table invites an invalid
cross-model comparison. Calibration is therefore reported separately.

The imbalance-matched neural network is the co-primary fourth model under the
author's refined 2026-06-16 decision. The raw RC7 neural network remains a
sensitivity because it received no training-stage imbalance correction.

Pure read/assemble from already-saved artifacts: no model is fitted and no
frozen primary result is overwritten. Writes only:

    outputs/tables/model_results/model_performance_test_4models.{csv,tex}

Run:
    python -m src.analysis.headline_table_4models
"""
from __future__ import annotations

import pandas as pd

from src.config import OUT_TABLES_MODEL, OUT_TABLES_ROBUSTNESS
from src.utils.tables import save_table

DISPLAY = {
    "logistic_regression": "Logistic Regression (ridge)",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}

COLS = [
    "model",
    "prevalence_baseline_pr_auc",
    "pr_auc",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "ks_stat",
    "threshold",
]


def main() -> None:
    prim = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test.csv")
    nn = pd.read_csv(
        OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results.csv"
    ).iloc[0]

    rows = []
    for _, r in prim.iterrows():
        key = r["model"]
        rows.append({
            "model": DISPLAY[key],
            "prevalence_baseline_pr_auc": r["prevalence_baseline_pr_auc"],
            "pr_auc": r["pr_auc"],
            "pr_auc_ci_lower": r["pr_auc_ci_lower"],
            "pr_auc_ci_upper": r["pr_auc_ci_upper"],
            "roc_auc": r["roc_auc"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "ks_stat": r["ks_stat"],
            "threshold": r["threshold"],
        })

    rows.append({
        "model": "Neural Network (MLP)",
        "prevalence_baseline_pr_auc": nn["prevalence_baseline_pr_auc"],
        "pr_auc": nn["pr_auc"],
        "pr_auc_ci_lower": nn["pr_auc_ci_lower"],
        "pr_auc_ci_upper": nn["pr_auc_ci_upper"],
        "roc_auc": nn["roc_auc"],
        "precision": nn["precision"],
        "recall": nn["recall"],
        "f1": nn["f1"],
        "ks_stat": nn["ks_stat"],
        "threshold": nn["threshold"],
    })

    out = pd.DataFrame(rows)[COLS]

    save_table(
        out,
        OUT_TABLES_MODEL / "model_performance_test_4models",
        caption=(
            "Out-of-sample test-set performance (2015--2024), "
            "$n=30{,}585$ firm-years, 434 distressed (prevalence 1.42\\%; "
            "no-skill PR-AUC baseline $=0.0142$). Four co-primary models. "
            "PR-AUC is the primary metric; 95\\% bootstrap CIs use firm-block "
            "resampling (1{,}000 resamples). Brier score is omitted because "
            "the saved primary LR is Platt calibrated whereas the primary "
            "tree-model probabilities are raw; calibration is analysed "
            "separately. The neural network is the imbalance-matched RC7b "
            "variant; the raw threshold-only NN is reported "
            "as a sensitivity. Thresholds are validation-optimal F1 operating "
            "points and are not comparable across rows."
        ),
        label="tab:model_performance_4models",
    )
    print("\nFour-model headline table:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
