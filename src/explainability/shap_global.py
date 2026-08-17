"""
Global SHAP analysis — sample-wide feature importance.
=======================================================
Implements §10.2 of the pre-specified study design:

  1. SHAP beeswarm plot (summary plot) — features ordered by mean |SHAP|;
     each point = one firm-year; colour = raw feature value.
  2. Bar chart of mean |SHAP| for top 15 features.
  3. Kendall's tau concordance between SHAP ranking and logistic
     regression standardised-coefficient ranking.

Critical: SHAP values are model attributions, NOT causal effects.
The question asked is strictly:
  "Do the attributions align DESCRIPTIVELY with the sign, ranking,
   and pattern predictions of established financial distress theory?"
This is the three-layer rule (design §3.2, §10.1).

Design reference: §10.2
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, weightedtau
import matplotlib.pyplot as plt

from src.config import (
    ALL_FEATURES,
    OUT_FIGURES_SHAP,
    OUT_TABLES_SHAP,
    FIG_DPI,
    FIG_FORMATS,
    MODEL_DISPLAY_NAMES,
)
from src.utils.plots import save_figure
from src.utils.tables import save_table


# ---------------------------------------------------------------------------
# SHAP values computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_test: np.ndarray,
    model_name: str = "xgboost",
) -> np.ndarray:
    """
    Compute SHAP values for the test set using TreeExplainer.

    TreeExplainer is used for XGBoost and Random Forest — it computes
    exact Shapley values in polynomial time without approximation.
    For logistic regression, LinearExplainer is used.

    Parameters
    ----------
    model : fitted estimator
    X_test : np.ndarray
        Test set features.
    model_name : str
        Determines which SHAP explainer to use.

    Returns
    -------
    np.ndarray
        Shape (n_obs, n_features). SHAP values for the positive class.
    """
    if model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        # RF binary classification returns a list [neg_class, pos_class]
        # (older shap) or a 3-D array (newer shap): (n, feat, classes) or
        # (classes, n, feat). Normalise to the positive-class 2-D array.
        if isinstance(sv, list):
            return sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[1]
        if sv.shape != X_test.shape:
            raise ValueError(f"Unexpected SHAP shape {sv.shape} vs X {X_test.shape}")
        return sv
    else:  # logistic_regression — model is a Pipeline, possibly wrapped
        # D8 wraps the original Pipeline in a PlattScaledModel for calibration.
        # SHAP must operate on the underlying LR coefficients (raw scores),
        # not the Platt sigmoid output. Unwrap if necessary.
        pipeline = model
        if hasattr(model, "_base") and hasattr(model._base, "named_steps"):
            # PlattScaledModel from src/analysis/lr_calibration.py
            pipeline = model._base
        elif not hasattr(model, "named_steps"):
            raise TypeError(
                f"LR SHAP requires a Pipeline (or PlattScaledModel wrapping one); "
                f"got {type(model).__name__} with no `named_steps`."
            )

        scaler = pipeline.named_steps["scaler"]
        clf    = pipeline.named_steps["clf"]
        X_scaled = scaler.transform(X_test)
        explainer = shap.LinearExplainer(clf, X_scaled)
        sv = explainer.shap_values(X_scaled)
        # LinearExplainer for binary returns list [neg, pos] or single array
        if isinstance(sv, list):
            return sv[1]
        return sv


# ---------------------------------------------------------------------------
# Beeswarm plot
# ---------------------------------------------------------------------------

def plot_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    model_name: str,
    max_display: int = 17,
) -> None:
    """
    SHAP beeswarm (summary) plot.

    Features are ordered by mean |SHAP| value. Each point represents one
    firm-year. Colour indicates the raw feature value (red = high,
    blue = low), allowing simultaneous communication of importance
    magnitude, effect direction, and distribution.

    Parameters
    ----------
    shap_values : np.ndarray
    X_test : pd.DataFrame
        Feature values (used for colour coding).
    model_name : str
    max_display : int
        Number of features to display (default 17 = all).
    """
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # SHAP's summary_plot internally calls plt.tight_layout(), which throws
    # a RuntimeError in newer matplotlib when a colorbar is present
    # ("Colorbar layout of new layout engine not compatible with old engine").
    # The plot itself is fully rendered before tight_layout() fails, so we
    # catch and ignore the error to keep the beeswarm figure intact.
    try:
        shap.summary_plot(
            shap_values,
            X_test,
            plot_type="dot",
            max_display=max_display,
            show=False,
        )
    except RuntimeError as e:
        if "Colorbar layout" not in str(e):
            raise
        # Plot was drawn — just couldn't tighten. Continue.

    fig = plt.gcf()
    fig.suptitle(
        f"SHAP Beeswarm Plot: Corporate Financial Distress Prediction — "
        f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} (Test Sample)",
        fontsize=12,
    )
    save_figure(fig, OUT_FIGURES_SHAP / f"beeswarm_{model_name}")


# ---------------------------------------------------------------------------
# Bar chart of mean |SHAP|
# ---------------------------------------------------------------------------

def plot_mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Horizontal bar chart of mean |SHAP| values for the top-n features.

    Also returns the ranking DataFrame (used for Kendall's tau computation).

    Parameters
    ----------
    shap_values : np.ndarray
    feature_names : list[str]
    model_name : str
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap, rank.
        Sorted by mean_abs_shap descending.
    """
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
          .sort_values("mean_abs_shap", ascending=False)
          .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    top = ranking.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(
        f"Top {top_n} Features by Mean |SHAP|: Corporate Financial Distress Prediction\n"
        f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — Test Sample (2015–2024)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure(fig, OUT_FIGURES_SHAP / f"mean_abs_shap_{model_name}")

    return ranking


# ---------------------------------------------------------------------------
# Kendall's tau concordance with logistic regression
# ---------------------------------------------------------------------------

def compute_kendall_tau(
    shap_ranking: pd.DataFrame,
    logit_coef: pd.Series,
) -> tuple[float, float]:
    """
    Compute Kendall's tau rank correlation between SHAP importance ranking
    and logistic regression standardised coefficient |magnitude| ranking.

    High concordance (τ close to 1) suggests that ML and the linear model
    identify the same economic signals. Significant discordance requires
    substantive investigation (design §10.2).

    Parameters
    ----------
    shap_ranking : pd.DataFrame
        Output of plot_mean_abs_shap(). Must have 'feature' and 'rank'.
    logit_coef : pd.Series
        Absolute standardised logistic regression coefficients,
        indexed by feature name.

    Returns
    -------
    tau : float
    p_value : float

    Design reference: §10.2
    """
    # Align on common features
    common = shap_ranking["feature"][shap_ranking["feature"].isin(logit_coef.index)]
    if len(common) < 3:
        raise ValueError("Fewer than 3 features in common between SHAP ranking and logit coefficients.")

    shap_sub = shap_ranking.set_index("feature").loc[common, "rank"]

    # Rank logit by absolute coefficient magnitude (1 = most important)
    logit_abs = logit_coef.loc[common].abs()
    logit_ranks = logit_abs.rank(ascending=False)

    tau, p_value = kendalltau(shap_sub.values, logit_ranks.values)
    return float(tau), float(p_value)


# ---------------------------------------------------------------------------
# Kendall's tau — top-K and weighted variants
# ---------------------------------------------------------------------------

def _tau_on_subset(
    shap_ranking: pd.DataFrame,
    logit_coef: pd.Series,
    feature_subset: list[str],
) -> tuple[float, float, int]:
    """Compute (tau, p, n_features) on a restricted feature set."""
    sub = shap_ranking[shap_ranking["feature"].isin(feature_subset)]
    if len(sub) < 3:
        return float("nan"), float("nan"), int(len(sub))
    # Re-rank within the subset so τ measures concordance on this set only
    shap_ranks  = sub.set_index("feature").loc[feature_subset, "mean_abs_shap"] \
                     .rank(ascending=False)
    logit_abs   = logit_coef.loc[feature_subset].abs()
    logit_ranks = logit_abs.rank(ascending=False)
    tau, p = kendalltau(shap_ranks.values, logit_ranks.values)
    return float(tau), float(p), int(len(sub))


def compute_kendall_tau_variants(
    shap_ranking: pd.DataFrame,
    logit_coef: pd.Series,
) -> dict:
    """
    Compute Kendall's tau in four flavours:

      * tau_all       — Kendall τ_b across all features (original test)
      * tau_top10     — Kendall τ_b restricted to the union of top-10 features
                        in either ranking (SHAP top-10 ∪ logit top-10)
      * tau_top5      — same with top-5
      * weighted_tau  — Vigna (2015) weighted Kendall τ, which down-weights
                        concordance at the bottom of the ranking
                        (`scipy.stats.weightedtau`). This is appropriate
                        when low-ranked features are noise-dominated, as
                        is the case here for a heavily-shrunk LR.

    Parameters
    ----------
    shap_ranking : pd.DataFrame
        Must contain 'feature' and 'mean_abs_shap' columns.
    logit_coef : pd.Series
        Absolute standardised LR coefficients indexed by feature.

    Returns
    -------
    dict with keys: tau_all, p_all, n_all,
                    tau_top10, p_top10, n_top10,
                    tau_top5,  p_top5,  n_top5,
                    weighted_tau, weighted_tau_p, weighted_tau_n
    """
    # Align on common features
    common = [f for f in shap_ranking["feature"] if f in logit_coef.index]
    if len(common) < 3:
        raise ValueError("Fewer than 3 features in common between SHAP "
                         "ranking and logit coefficients.")

    shap_sub = (shap_ranking.set_index("feature").loc[common, "mean_abs_shap"]
                .rank(ascending=False))
    logit_abs   = logit_coef.loc[common].abs()
    logit_ranks = logit_abs.rank(ascending=False)

    # 1. Full ranking — Kendall τ_b
    tau_all, p_all = kendalltau(shap_sub.values, logit_ranks.values)

    # 2. Top-10 union
    top10_shap  = list(shap_ranking.nlargest(10, "mean_abs_shap")["feature"])
    top10_logit = list(logit_abs.sort_values(ascending=False).head(10).index)
    top10_union = [f for f in common if (f in top10_shap or f in top10_logit)]
    tau10, p10, n10 = _tau_on_subset(shap_ranking, logit_coef, top10_union)

    # 3. Top-5 union
    top5_shap  = list(shap_ranking.nlargest(5, "mean_abs_shap")["feature"])
    top5_logit = list(logit_abs.sort_values(ascending=False).head(5).index)
    top5_union = [f for f in common if (f in top5_shap or f in top5_logit)]
    tau5, p5, n5 = _tau_on_subset(shap_ranking, logit_coef, top5_union)

    # 4. Weighted Kendall τ — Vigna (2015)
    # weightedtau expects scores (high = important), not ranks; pass the raw
    # importances so the natural ordering of weights is "top is most important".
    shap_scores  = shap_ranking.set_index("feature").loc[common, "mean_abs_shap"].values
    logit_scores = logit_abs.loc[common].values
    wtau, wp = weightedtau(shap_scores, logit_scores)

    return {
        "tau_all":        float(tau_all),
        "p_all":          float(p_all),
        "n_all":          int(len(common)),
        "tau_top10":      float(tau10),
        "p_top10":        float(p10),
        "n_top10":        int(n10),
        "tau_top5":       float(tau5),
        "p_top5":         float(p5),
        "n_top5":         int(n5),
        "weighted_tau":   float(wtau),
        "weighted_tau_p": float(wp),
        "weighted_tau_n": int(len(common)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_global_shap(
    model,
    logit_model,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    model_name: str = "xgboost",
) -> None:
    """
    Run the full global SHAP analysis for the best-performing model.

    Parameters
    ----------
    model : fitted best model (typically XGBoost)
    logit_model : fitted logistic regression (for Kendall's tau)
    test : pd.DataFrame
    features : list[str]
    model_name : str
    """
    X_test = test[features]
    shap_values = compute_shap_values(model, X_test.astype(float).fillna(0).values, model_name)

    # Beeswarm plot
    plot_beeswarm(shap_values, X_test, model_name)

    # Bar chart + ranking
    shap_ranking = plot_mean_abs_shap(shap_values, features, model_name)

    # Kendall's tau vs logistic regression
    # Extract standardised logit coefficients from the Pipeline. D8 wraps the
    # LR Pipeline in a PlattScaledModel for calibration; the coefficients live
    # in the underlying pipeline (._base). Unwrap if necessary — mirrors the
    # logic in compute_shap_values() so the Kendall step survives calibration.
    lr_pipeline = logit_model
    if hasattr(logit_model, "_base") and hasattr(logit_model._base, "named_steps"):
        lr_pipeline = logit_model._base
    clf_lr    = lr_pipeline.named_steps["clf"]
    # The LR is fitted on scaled inputs (StandardScaler inside the pipeline),
    # so clf.coef_
    # is already the standardised coefficient vector. The previous code
    # multiplied it by the raw feature std a second time, which distorts
    # the |coefficient| ranking used for the Kendall concordance
    # (coef * std converts raw-input coefficients to standardised ones —
    # not applicable here).
    std_coef   = clf_lr.coef_[0]
    logit_coef = pd.Series(np.abs(std_coef), index=features)

    variants = compute_kendall_tau_variants(shap_ranking, logit_coef)
    print(
        f"  Kendall's tau (SHAP vs logit):"
        f"\n    all (n={variants['n_all']:>2}): tau = {variants['tau_all']: .3f}, "
        f"p = {variants['p_all']:.4f}"
        f"\n    top-10 union (n={variants['n_top10']:>2}): tau = {variants['tau_top10']: .3f}, "
        f"p = {variants['p_top10']:.4f}"
        f"\n    top-5  union (n={variants['n_top5']:>2}): tau = {variants['tau_top5']: .3f}, "
        f"p = {variants['p_top5']:.4f}"
        f"\n    weighted (Vigna 2015): tau = {variants['weighted_tau']: .3f} "
        f"(no p-value defined for weightedtau)"
    )

    tau_df = pd.DataFrame([{
        "tau_all":        round(variants["tau_all"],        4),
        "p_all":          round(variants["p_all"],          4),
        "n_all":          variants["n_all"],
        "tau_top10":      round(variants["tau_top10"],      4),
        "p_top10":        round(variants["p_top10"],        4),
        "n_top10":        variants["n_top10"],
        "tau_top5":       round(variants["tau_top5"],       4),
        "p_top5":         round(variants["p_top5"],         4),
        "n_top5":         variants["n_top5"],
        # scipy's weightedtau defines no p-value (returns NaN), so none is saved.
        "weighted_tau":   round(variants["weighted_tau"],   4),
    }])
    save_table(tau_df, OUT_TABLES_SHAP / f"kendall_tau_{model_name}")

    # Save ranking table
    OUT_TABLES_SHAP.mkdir(parents=True, exist_ok=True)
    save_table(shap_ranking, OUT_TABLES_SHAP / f"shap_ranking_{model_name}")
    print(f"  Global SHAP analysis complete for {model_name}.")
