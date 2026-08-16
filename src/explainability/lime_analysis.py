"""
Supplementary XAI — LIME local explanations and SHAP-vs-LIME agreement.
=======================================================================
LIME (Local Interpretable Model-agnostic Explanations; Ribeiro, Singh &
Guestrin 2016) is presented as a co-primary explanation method following the
author's post-freeze decision. Its implemented scope is local: it independently
cross-checks SHAP on the same three XGBoost firm-years.

Rationale (research task description)
---------------------------------------
The task asks the student to "apply state-of-the-art XAI frameworks (such
as SHAP)" — plural, with SHAP as one example. Comparing SHAP and LIME
directly addresses the task's central question of whether XAI-derived
insights are "economically meaningful risk indicators rather than merely
statistical artifacts":
  - Where SHAP and LIME AGREE on a firm's drivers → the explanation is
    method-robust and more credible.
  - Where they DIVERGE → reported as XAI instability, a documented
    limitation (consistent with the known sensitivity of LIME to its
    local sampling; Ribeiro et al. 2016).

Design consistency
------------------
LIME explains the SAME best-of-portfolio model (XGBoost) on the SAME three
firm-years used for the SHAP waterfall plots (true positive, false
negative, true negative — selected by src.explainability.shap_local.
select_cases), so the two methods are compared like-for-like.

Outputs
-------
  - figures/shap/lime_{case}_{model}.{pdf,png}      one LIME bar plot per case
  - tables/shap/lime_shap_agreement.{csv,tex}       per-case SHAP-vs-LIME
                                                    rank/sign agreement

Reference: extension to the Pre-Specified Empirical Design §10 (documented, not frozen);
Ribeiro, Singh & Guestrin (2016), "Why Should I Trust You?", KDD.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from src.config import (
    ALL_FEATURES,
    OUT_FIGURES_SHAP,
    OUT_TABLES_SHAP,
    MODEL_DISPLAY_NAMES,
    RANDOM_SEED,
)
from src.utils.plots import save_figure, set_academic_style
from src.utils.tables import save_table

LABEL_COL = "distress"
_TOP_K = 5   # top-K features used for overlap / sign-agreement diagnostics


# ---------------------------------------------------------------------------
# LIME explanation plot
# ---------------------------------------------------------------------------

def _plot_lime(
    feature_names: list[str],
    lime_weights: np.ndarray,
    case_label: str,
    model_name: str,
) -> None:
    """Horizontal bar chart of signed LIME local weights for one firm-year."""
    set_academic_style()
    order = np.argsort(np.abs(lime_weights))          # ascending |weight|
    feats = [feature_names[i] for i in order]
    vals  = lime_weights[order]
    # Red = pushes prediction toward distress; blue = toward survival.
    colours = ["#c0392b" if v > 0 else "#2c7fb8" for v in vals]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(feats)), vals, color=colours)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("LIME local weight  (toward distress  →)")
    ax.set_title(
        f"LIME Local Explanation — Corporate Financial Distress\n"
        f"{case_label.replace('_', ' ').title()} "
        f"({MODEL_DISPLAY_NAMES.get(model_name, model_name)})",
        fontsize=10,
    )
    save_figure(fig, OUT_FIGURES_SHAP / f"lime_{case_label}_{model_name}")


# ---------------------------------------------------------------------------
# Agreement diagnostics between SHAP and LIME for one case
# ---------------------------------------------------------------------------

def _agreement(shap_vec: np.ndarray, lime_vec: np.ndarray, k: int = _TOP_K) -> dict:
    """
    Compare two signed attribution vectors over the same feature ordering.

    Returns
    -------
    dict with:
      kendall_tau   : rank concordance over signed attributions (all features)
      topk_overlap  : count of features shared by both methods' top-k by |value|
      sign_agree_topk : fraction of the union of top-k features whose
                        attribution sign matches across methods
    """
    tau, _ = kendalltau(shap_vec, lime_vec)

    shap_top = set(np.argsort(np.abs(shap_vec))[-k:])
    lime_top = set(np.argsort(np.abs(lime_vec))[-k:])
    overlap = len(shap_top & lime_top)

    union = list(shap_top | lime_top)
    if union:
        sign_agree = float(np.mean([
            np.sign(shap_vec[i]) == np.sign(lime_vec[i]) for i in union
        ]))
    else:
        sign_agree = float("nan")

    return {
        "kendall_tau":     round(float(tau), 4) if tau == tau else float("nan"),
        "topk_overlap":    f"{overlap}/{k}",
        "sign_agree_topk": round(sign_agree, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_lime_analysis(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    features: list[str] = ALL_FEATURES,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Run LIME on the three SHAP waterfall cases and build a SHAP-vs-LIME
    agreement table.

    Parameters
    ----------
    model : fitted estimator with predict_proba (the XGBoost model).
    train : pd.DataFrame
        Training set — supplies LIME's background sampling distribution.
    test : pd.DataFrame
        Test set (case rows are selected from here).
    y_prob : np.ndarray
        Test-set predicted probabilities (for case selection + reporting).
    threshold : float
        Validation-locked classification threshold (for case selection).
    features : list[str]
    model_name : str

    Returns
    -------
    pd.DataFrame
        One row per case with agreement diagnostics.
    """
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from src.explainability.shap_local import select_cases

    print("\n  LIME co-primary local analysis (cross-check against SHAP)")
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_SHAP.mkdir(parents=True, exist_ok=True)

    X_train = train[features].astype(float).fillna(0).values
    X_test  = test[features].astype(float).fillna(0).values

    # LIME explainer — background = training distribution (no leakage; train only).
    # Phase 5 audit fix: declare the binary indicators categorical so LIME's
    # perturbation samples them as {0, 1} instead of drawing continuous
    # values around them.
    categorical_idx = [i for i, f in enumerate(features)
                       if f in ("OENEG", "INTWO")]
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=features,
        class_names=["non_distress", "distress"],
        mode="classification",
        discretize_continuous=True,
        categorical_features=categorical_idx,
        random_state=RANDOM_SEED,
    )

    # SHAP explainer for the SAME model (exact, for tree models)
    shap_explainer = shap.TreeExplainer(model)

    # Identical case selection to the SHAP waterfall plots
    cases = select_cases(test, y_prob, threshold)

    rows = []
    for label, row_idx in cases.items():
        x = X_test[row_idx]

        # ---- LIME explanation (all 17 features, signed local weights) ----
        exp = explainer.explain_instance(
            x, model.predict_proba,
            num_features=len(features), labels=(1,),
        )
        lime_map = dict(exp.as_map()[1])                 # {feature_idx: weight}
        lime_vec = np.array([lime_map.get(i, 0.0) for i in range(len(features))])
        _plot_lime(features, lime_vec, label, model_name)

        # ---- SHAP values for the same instance ----
        sv = shap_explainer.shap_values(x.reshape(1, -1))
        if isinstance(sv, list):
            sv = sv[1]
        shap_vec = sv[0] if np.ndim(sv) == 2 else np.asarray(sv).ravel()

        diag = _agreement(shap_vec, lime_vec, k=_TOP_K)
        meta = test.iloc[row_idx]
        rows.append({
            "case":            label,
            "gvkey":           meta.get("gvkey", "?"),
            "fyear":           meta.get("fyear", "?"),
            "y_prob":          round(float(y_prob[row_idx]), 4),
            **diag,
        })
        print(f"    {label:15s}  kendall_tau={diag['kendall_tau']}  "
              f"top{_TOP_K}_overlap={diag['topk_overlap']}  "
              f"sign_agree={diag['sign_agree_topk']}")

    agreement_df = pd.DataFrame(rows)

    save_table(
        agreement_df,
        OUT_TABLES_SHAP / "lime_shap_agreement",
        caption=(
            "SHAP-vs-LIME explanation agreement for the three local cases "
            "(same firm-years as the SHAP waterfall plots), XGBoost model. "
            "\\textbf{kendall\\_tau} is the rank concordance between the two "
            "methods' signed local attributions across all 17 predictors; "
            "\\textbf{top5\\_overlap} counts how many of each method's five "
            "largest-magnitude drivers are shared; \\textbf{sign\\_agree\\_topk} "
            "is the fraction of the union of top-5 features whose attribution "
            "sign matches across methods. High agreement indicates the "
            "explanation is method-robust; divergence is reported as XAI "
            "instability (Ribeiro et al. 2016). LIME is a post-freeze "
            "co-primary local explanation method; SHAP additionally supplies "
            "the global and dependence analyses."
        ),
        label="tab:lime_shap_agreement",
    )
    print(f"  LIME analysis complete — {len(cases)} explanations + agreement table saved.")
    return agreement_df
