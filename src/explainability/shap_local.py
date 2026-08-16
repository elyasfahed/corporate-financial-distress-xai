"""
Local SHAP analysis — firm-level waterfall explanations.
=========================================================
Implements §10.4 of the Pre-Specified Empirical Design.

Produces SHAP waterfall plots for 3 substantively motivated firms:
  1. True positive from the held-out 2015-2024 test period
  2. False negative: low predicted score despite subsequent delisting
     — what concealed the distress signal?
  3. True negative: a financially healthy firm

Case selection criterion: economic insight, NOT statistical averageness.
The cases must be selected to illuminate the model's logic and illuminate
tensions between prediction and economic theory.

Design-specification reference: §10.4
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.config import (
    ALL_FEATURES,
    OUT_FIGURES_SHAP,
    FIG_DPI,
    MODEL_DISPLAY_NAMES,
)
from src.utils.plots import save_figure


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------

def select_cases(
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """
    Select three substantively motivated firm-year cases for local SHAP.

    Selection criteria (Pre-Specified Empirical Design §10.4):
      - True positive  : distress=1, y_prob >= threshold; choose the highest
                         predicted probability in the held-out test period
      - False negative : distress=1, y_prob < threshold
                         Choose the case with the LARGEST gap between
                         actual distress and predicted score (most surprising)
      - True negative  : distress=0, y_prob << threshold
                         Choose a firm with consistently healthy financials

    Parameters
    ----------
    test : pd.DataFrame
        Test set with 'distress', 'fyear', 'gvkey' columns.
    y_prob : np.ndarray
        Predicted distress probabilities.
    threshold : float
        Classification threshold (locked from validation).

    Returns
    -------
    dict
        {'true_positive': row_index, 'false_negative': row_index,
         'true_negative': row_index}
    """
    y_true = test["distress"].values
    fyear  = test["fyear"].values
    idx    = np.arange(len(test))

    # True positive: highest predicted probability among test-set positives.
    # The held-out test begins in 2015, so a GFC case cannot be selected here.
    tp_mask = (y_true == 1) & (y_prob >= threshold)
    if tp_mask.any():
        tp_idx = int(idx[tp_mask][np.argmax(y_prob[tp_mask])])
    else:
        raise ValueError("No true positives found in test set.")

    # False negative: distress=1, predicted negative — largest gap (lowest predicted prob)
    fn_mask = (y_true == 1) & (y_prob < threshold)
    if fn_mask.any():
        fn_idx = int(idx[fn_mask][np.argmin(y_prob[fn_mask])])
    else:
        raise ValueError("No false negatives found in test set.")

    # True negative: distress=0, predicted negative — lowest predicted probability
    tn_mask = (y_true == 0) & (y_prob < threshold)
    if tn_mask.any():
        tn_idx = int(idx[tn_mask][np.argmin(y_prob[tn_mask])])
    else:
        raise ValueError("No true negatives found in test set.")

    cases = {
        "true_positive":  tp_idx,
        "false_negative": fn_idx,
        "true_negative":  tn_idx,
    }
    for label, row_idx in cases.items():
        meta = test.iloc[row_idx]
        print(f"  {label:20s}: gvkey={meta.get('gvkey', '?')}  "
              f"fyear={meta.get('fyear', '?')}  "
              f"distress={meta.get('distress', '?')}  "
              f"y_prob={y_prob[row_idx]:.4f}")
    return cases


# ---------------------------------------------------------------------------
# Waterfall plots
# ---------------------------------------------------------------------------

def plot_waterfall(
    explainer,
    X_case: np.ndarray,
    feature_names: list[str],
    case_label: str,
    model_name: str = "xgboost",
) -> None:
    """
    SHAP waterfall plot for a single firm-year observation.

    The waterfall plot shows how each feature pushes the prediction
    from the base value (mean predicted probability) to the final
    prediction for this specific firm.

    Parameters
    ----------
    explainer : shap.TreeExplainer or shap.LinearExplainer
    X_case : np.ndarray
        Feature vector for the single firm-year (shape: (1, n_features)).
    feature_names : list[str]
    case_label : str
        Descriptive label for the plot title and filename
        (e.g., 'true_positive_GFC_2009').
    model_name : str
    """
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)

    # Compute SHAP values for the single observation
    sv = explainer.shap_values(X_case)
    if isinstance(sv, list):
        sv = sv[1]  # positive class
    sv_1d = sv[0] if sv.ndim == 2 else sv

    # Scalar base value (expected value for positive class)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])

    explanation = shap.Explanation(
        values=sv_1d,
        base_values=base_val,
        data=X_case[0],
        feature_names=feature_names,
    )

    plt.figure()
    # max_display=17 shows every predictor individually (no "N other features"
    # aggregate bar). Appropriate here because all 17 predictors are
    # theoretically motivated in the study's pre-specified predictor set, and
    # full
    # Displaying each predictor separately makes all contributions transparent.
    shap.plots.waterfall(explanation, max_display=17, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Waterfall — Corporate Financial Distress Prediction\n"
        f"{case_label.replace('_', ' ').title()} "
        f"({MODEL_DISPLAY_NAMES.get(model_name, model_name)})",
        fontsize=10,
        y=1.01,
    )
    save_figure(fig, OUT_FIGURES_SHAP / f"waterfall_{case_label}_{model_name}")


def run_local_shap(
    model,
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    features: list[str] = ALL_FEATURES,
    model_name: str = "xgboost",
) -> None:
    """
    Run the full local SHAP analysis: select cases, plot waterfalls.

    Parameters
    ----------
    model : fitted model
    test : pd.DataFrame
    y_prob : np.ndarray
    threshold : float
    features : list[str]
    model_name : str
    """
    cases = select_cases(test, y_prob, threshold)

    # Build explainer
    X_all = test[features].astype(float).fillna(0).values
    if model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(model)
    else:
        scaler = model.named_steps["scaler"]
        clf    = model.named_steps["clf"]
        X_scaled_bg = scaler.transform(X_all)
        explainer = shap.LinearExplainer(clf, X_scaled_bg)

    for case_label, row_idx in cases.items():
        meta    = test.iloc[row_idx]
        fyear   = int(meta.get("fyear", 0))
        gvkey   = meta.get("gvkey", "unknown")
        distress_val = int(meta.get("distress", -1))
        label   = f"{case_label}_fy{fyear}_{gvkey}"

        if model_name in ("xgboost", "random_forest"):
            X_case = X_all[row_idx : row_idx + 1]
        else:
            X_case = scaler.transform(X_all[row_idx : row_idx + 1])

        plot_waterfall(explainer, X_case, features, label, model_name)
        print(f"    {case_label}: gvkey={gvkey}  fyear={fyear}  "
              f"distress={distress_val}  y_prob={y_prob[row_idx]:.4f}")

    print(f"  Local SHAP analysis complete — {len(cases)} waterfall plots saved.")
