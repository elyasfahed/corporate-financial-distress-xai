"""
Sub-period performance stability analysis.
==========================================
Implements a post-specification extension: a test-period regime-stability check.

Splits the test set (2015–2024) into two sub-periods and evaluates
all three models in each:
  - Pre-COVID   : fyear 2015–2019  (stable expansion, low rates)
  - COVID+      : fyear 2020–2024  (shock + recovery + rate hikes)

Motivation (from critical assessment):
  A model trained on 1990–2009 may behave differently across regimes.
  If PR-AUC drops materially in one sub-period, this indicates regime
  non-stationarity — a genuine limitation requiring disclosure.

A 'material' drop is defined as:
  PR-AUC_{sub} < PR-AUC_{full} − 2 × bootstrap_se (roughly)
  In practice: flag if sub-period PR-AUC is more than 30% below
  the full-period PR-AUC for the same model.

Post-specification extension — sub-period stability (design §6.1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    ALL_FEATURES,
    OUT_TABLES_MODEL,
)
from src.utils.tables import save_table


# ---------------------------------------------------------------------------
# Sub-period definitions (frozen)
# ---------------------------------------------------------------------------

SUB_PERIODS: dict[str, tuple[int, int]] = {
    "2015–2019 (Pre-COVID)":  (2015, 2019),
    "2020–2024 (COVID+)":     (2020, 2024),
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_metrics_for_split(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
    model_name: str,
    period_label: str,
) -> dict:
    """Compute all key metrics for one model × one sub-period."""
    n_obs       = len(y)
    n_distress  = int(y.sum())
    prevalence  = float(y.mean())

    if n_obs == 0 or n_distress == 0 or y.var() == 0:
        return {
            "period": period_label, "model": model_name,
            "n_obs": n_obs, "n_distressed": n_distress,
            "prevalence_pct": round(prevalence * 100, 2),
            "pr_auc": float("nan"), "baseline_pr_auc": round(prevalence, 4),
            "roc_auc": float("nan"), "f1": float("nan"),
            "brier_score": float("nan"),
        }

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "period":          period_label,
        "model":           model_name,
        "n_obs":           n_obs,
        "n_distressed":    n_distress,
        "prevalence_pct":  round(prevalence * 100, 2),
        "pr_auc":          round(average_precision_score(y, y_prob), 4),
        "baseline_pr_auc": round(prevalence, 4),
        "roc_auc":         round(roc_auc_score(y, y_prob), 4),
        "f1":              round(f1_score(y, y_pred, zero_division=0), 4),
        "brier_score":     round(brier_score_loss(y, y_prob), 4),
    }


def run_subperiod_analysis(
    trained_models: dict,
    test: pd.DataFrame,
    thresholds: dict[str, float],
    features: list[str] = ALL_FEATURES,
) -> pd.DataFrame:
    """
    Compute performance metrics separately for each sub-period and model.

    Parameters
    ----------
    trained_models : dict
        {model_name: {'model': fitted_model, ...}}
    test : pd.DataFrame
        Full test set with 'fyear' and 'distress' columns.
    thresholds : dict
        {model_name: float} — validation-locked thresholds.
    features : list[str]

    Returns
    -------
    pd.DataFrame
        One row per model × sub-period.
    """
    print("\n--- Sub-period stability analysis ---")
    OUT_TABLES_MODEL.mkdir(parents=True, exist_ok=True)

    rows = []

    # Full test period baseline
    X_full = test[features].astype(float).fillna(0).values
    y_full = test["distress"].values
    for model_name, info in trained_models.items():
        row = _compute_metrics_for_split(
            info["model"], X_full, y_full,
            thresholds.get(model_name, 0.5),
            model_name, "2015–2024 (Full test)",
        )
        rows.append(row)

    # Sub-periods
    for period_label, (yr_from, yr_to) in SUB_PERIODS.items():
        mask = test["fyear"].between(yr_from, yr_to)
        sub  = test[mask]
        X_sub = sub[features].astype(float).fillna(0).values
        y_sub = sub["distress"].values

        for model_name, info in trained_models.items():
            row = _compute_metrics_for_split(
                info["model"], X_sub, y_sub,
                thresholds.get(model_name, 0.5),
                model_name, period_label,
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    # Regime stability flag: flag if sub-period PR-AUC < 70% of full-period PR-AUC
    full_pr = df[df["period"] == "2015–2024 (Full test)"].set_index("model")["pr_auc"]
    for i, row in df.iterrows():
        if row["period"] == "2015–2024 (Full test)":
            df.at[i, "stability_flag"] = "—"
            continue
        full = full_pr.get(row["model"], float("nan"))
        if pd.isna(full) or full == 0:
            df.at[i, "stability_flag"] = "—"
        elif row["pr_auc"] < 0.70 * full:
            df.at[i, "stability_flag"] = "UNSTABLE"
        else:
            df.at[i, "stability_flag"] = "stable"

    # Print summary
    print(f"{'Period':<28} {'Model':<24} {'PR-AUC':>8} {'ROC-AUC':>8} "
          f"{'Brier':>7} {'N_dist':>7} {'Flag'}")
    print("-" * 95)
    for _, row in df.iterrows():
        print(f"  {row['period']:<26} {row['model']:<24} "
              f"{row['pr_auc']:>8.4f} {row['roc_auc']:>8.4f} "
              f"{row['brier_score']:>7.4f} {int(row['n_distressed']):>7} "
              f"  {row.get('stability_flag', '')}")

    save_table(
        df,
        OUT_TABLES_MODEL / "subperiod_performance",
        caption=(
            "Sub-period performance stability: test sample split into "
            "pre-COVID (2015--2019) and COVID+ (2020--2024) regimes. "
            "PR-AUC is the primary metric; baseline PR-AUC equals the "
            "sub-period prevalence rate. "
            "Flag ``UNSTABLE'' indicates sub-period PR-AUC $< 70\\%$ "
            "of the full test-period PR-AUC for that model."
        ),
        label="tab:subperiod_performance",
    )
    print(f"  Sub-period table saved.")
    return df
