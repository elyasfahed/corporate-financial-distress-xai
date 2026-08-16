"""
Post-hoc LR calibration via Platt scaling.
===========================================
Addresses the calibration weakness identified in the critical assessment:
  LR Brier score = 0.2427 (very high — worst calibration of the three models)
  while LR PR-AUC = 0.1103 (best discrimination).

The model is a good ranker but a poor probability estimator. This matters
for any practical application (credit risk management, portfolio screening)
where predicted probabilities are interpreted as probability estimates.

Fix: Platt scaling (sigmoid calibration) using the held-out validation set
(2010–2014). This is the correct procedure — we do NOT use the test set.

  CalibratedClassifierCV(lr_pipeline, cv='prefit', method='sigmoid')
      .fit(X_val, y_val)

This wraps the already-fitted LR pipeline and learns a 2-parameter sigmoid
mapping from raw predicted probabilities to calibrated probabilities. The
underlying model weights are not changed.

The calibrated model is evaluated on the test set and compared to the
original LR on Brier score and calibration curve.

Post-specification extension — calibration robustness (design §6.1)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

from src.config import (
    ALL_FEATURES,
    OUT_TABLES_MODEL,
    OUT_FIGURES_MODEL,
    FIG_DPI,
    FIG_FORMATS,
)
from src.utils.tables import save_table
from src.utils.plots import set_academic_style, save_figure


# ---------------------------------------------------------------------------
# Platt scaling
# ---------------------------------------------------------------------------

class PlattScaledModel:
    """
    Thin wrapper: original model + a 2-parameter sigmoid (Platt scaler).

    Platt scaling fits a logistic regression on the original model's
    raw probability outputs using the validation set. The underlying
    model weights are not changed — only the output probabilities are
    re-mapped to be better calibrated.

    Compatible with sklearn's predict_proba interface.
    """

    def __init__(self, base_model, platt: LogisticRegression) -> None:
        self._base  = base_model
        self._platt = platt

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._base.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self._platt.predict_proba(raw)
        return cal   # shape (n, 2): [:, 1] is calibrated positive-class prob


def apply_platt_scaling(
    lr_model,
    val: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> PlattScaledModel:
    """
    Apply Platt scaling to an already-fitted LR pipeline.

    Procedure:
      1. Predict raw probabilities on the validation set.
      2. Fit a 2-parameter logistic regression on those raw probabilities.
      3. Wrap the original model + the sigmoid in PlattScaledModel.

    Parameters
    ----------
    lr_model : fitted sklearn Pipeline (scaler + LogisticRegression)
    val : pd.DataFrame
        Validation set (2010–2014) — used for calibration fitting only.
        NEVER use the test set for calibration fitting.
    features : list[str]

    Returns
    -------
    PlattScaledModel
    """
    X_val = val[features].astype(float).fillna(0).values
    y_val = val["distress"].values

    # Step 1: raw probabilities on validation set
    raw_val = lr_model.predict_proba(X_val)[:, 1].reshape(-1, 1)

    # Step 2: fit a logistic regression as the sigmoid (Platt, 1999)
    # Very high C = essentially no regularisation on the sigmoid layer
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1_000)
    platt.fit(raw_val, y_val)

    n_pos = int(y_val.sum())
    print(f"  Platt scaling fitted on validation set "
          f"({len(y_val):,} obs, {n_pos} distressed).")
    return PlattScaledModel(lr_model, platt)


# ---------------------------------------------------------------------------
# Comparison: original vs. calibrated LR
# ---------------------------------------------------------------------------

def compare_calibration(
    lr_model,
    calibrated_lr,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compare original and calibrated LR on test set.

    Produces:
      1. Numeric comparison table (Brier, PR-AUC, ROC-AUC)
      2. Calibration curve plot (original vs. calibrated vs. perfect)

    Parameters
    ----------
    lr_model : original fitted LR pipeline
    calibrated_lr : CalibratedClassifierCV wrapping lr_model
    test : pd.DataFrame
    features : list[str]
    n_bins : int

    Returns
    -------
    pd.DataFrame
        Comparison metrics.
    """
    OUT_TABLES_MODEL.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_MODEL.mkdir(parents=True, exist_ok=True)

    X_test = test[features].astype(float).fillna(0).values
    y_test = test["distress"].values

    y_prob_orig = lr_model.predict_proba(X_test)[:, 1]
    y_prob_cal  = calibrated_lr.predict_proba(X_test)[:, 1]

    rows = []
    for label, y_prob in [("LR (original)", y_prob_orig),
                           ("LR (Platt-scaled)", y_prob_cal)]:
        rows.append({
            "model":       label,
            "brier_score": round(brier_score_loss(y_test, y_prob), 4),
            "pr_auc":      round(average_precision_score(y_test, y_prob), 4),
            "roc_auc":     round(roc_auc_score(y_test, y_prob), 4),
            "mean_pred":   round(float(y_prob.mean()), 4),
            "max_pred":    round(float(y_prob.max()), 4),
        })

    comp_df = pd.DataFrame(rows)

    print("\n  LR calibration comparison (test set):")
    print(comp_df.to_string(index=False))

    improvement = (rows[0]["brier_score"] - rows[1]["brier_score"]) / rows[0]["brier_score"] * 100
    print(f"  Brier score improvement: {improvement:.1f}%")

    save_table(
        comp_df,
        OUT_TABLES_MODEL / "lr_calibration_comparison",
        caption=(
            "Post-hoc Platt scaling applied to Logistic Regression. "
            "Platt scaling uses the held-out validation set (2010--2014) "
            "to fit a two-parameter sigmoid transformation of the raw "
            "predicted probabilities. Brier score measures calibration "
            "quality (lower is better); PR-AUC and ROC-AUC measure "
            "discrimination (should be unchanged by calibration)."
        ),
        label="tab:lr_calibration",
    )

    # ── Calibration curve plot ────────────────────────────────────────────
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, y_prob, color, ls in [
        ("LR (original)",    y_prob_orig, "steelblue", "-"),
        ("LR (Platt-scaled)", y_prob_cal,  "darkorange", "--"),
    ]:
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="s", lw=1.5,
                    color=color, linestyle=ls, label=label)
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title(
        "Calibration Comparison: LR Original vs. Platt-Scaled\n"
        "Corporate Financial Distress Prediction — Test Sample (2015–2024)"
    )
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, OUT_FIGURES_MODEL / "calibration_lr_platt")

    return comp_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lr_calibration(
    trained_models: dict,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> dict:
    """
    Full LR calibration pipeline: fit Platt scaling, evaluate, save outputs.

    Parameters
    ----------
    trained_models : dict
        Must contain 'logistic_regression' key.
    val : pd.DataFrame
        Validation set (2010–2014) — used ONLY for calibration fitting.
    test : pd.DataFrame
        Test set — used for final evaluation.
    features : list[str]

    Returns
    -------
    dict
        Updated trained_models with 'logistic_regression_calibrated' added.
    """
    print("\n--- LR post-hoc calibration (Platt scaling) ---")

    lr_model = trained_models["logistic_regression"]["model"]

    if isinstance(lr_model, PlattScaledModel):
        # The headline LR is already Platt-scaled (fix D8 in Stage 3).
        # Compare the RAW base LR against the calibrated wrapper —
        # do NOT calibrate a second time.
        print("  Headline LR is already Platt-scaled — comparing raw vs calibrated.")
        calibrated_lr = lr_model
        lr_model      = lr_model._base
    else:
        calibrated_lr = apply_platt_scaling(lr_model, val, features)

    comp_df = compare_calibration(lr_model, calibrated_lr, test, features)

    trained_models["logistic_regression_calibrated"] = {
        "model":     calibrated_lr,
        "threshold": trained_models["logistic_regression"]["threshold"],
    }

    print("  LR calibration complete.")
    return trained_models, comp_df
