"""
Non-linear SHAP analysis — dependence plots and interaction effects.
====================================================================
Implements Blueprint v4 §10.3.

Outputs:
  1. SHAP dependence plots for the 5 highest-importance features.
     Theoretical predictions are stated BEFORE results are examined.
  2. Leverage × Profitability interaction heatmap:
     3×3 heatmap of mean predicted distress probability by TLTA × NITA
     tertiles, testing the super-additive joint effect.

Pre-specified theoretical predictions (must be documented before running):
  TLTA  : SHAP increases with TLTA, potentially accelerating above 0.6
           (debt-overhang theory; Leland 1994; Garlappi et al. 2008)
  NITA  : SHAP decreases as NITA rises; steeper slope in negative NITA region
           (profitability → cash flow sufficiency; Altman 1968)
  SIGMA : SHAP increases with SIGMA (Merton 1974 distance-to-default)
  EXRET : SHAP decreases with EXRET (market anticipates distress; CHS 2008)
  LNTA  : SHAP decreases with size (smaller firms more vulnerable; Shumway 2001)

Patterns that CONTRADICT these predictions must be discussed critically,
not explained away (Blueprint v4 §10.3).

Blueprint v4 reference: §10.3
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    ALL_FEATURES,
    OUT_FIGURES_SHAP,
    OUT_TABLES_SHAP,
    FIG_DPI,
    MODEL_DISPLAY_NAMES,
)
from src.utils.plots import save_figure
from src.utils.tables import save_table

# Pre-specified support regions for the SHAP dependence support diagnostic
# (E4 fix). Edges chosen on theoretical grounds: TLTA=0.6 is the
# debt-overhang threshold (Leland 1994); TLTA=1.0 is the OENEG boundary.
DEPENDENCE_SUPPORT_BINS: dict[str, list[float]] = {
    "TLTA":  [-float("inf"), 0.6, 1.0,  float("inf")],
    "NITA":  [-float("inf"), -0.1, 0.0, float("inf")],
    "SIGMA": [-float("inf"), 0.05, 0.15, float("inf")],
    "EXRET": [-float("inf"), -0.5, 0.0, float("inf")],
    "LNTA":  [-float("inf"), 4.0, 8.0,  float("inf")],
}

# Features for which dependence plots are produced (top 5 by SHAP importance)
# These will be confirmed empirically; pre-specified priors below
DEPENDENCE_FEATURES = ["TLTA", "NITA", "SIGMA", "EXRET", "LNTA"]

# Pre-specified theoretical direction predictions (Blueprint v4 §10.3)
THEORETICAL_PREDICTIONS: dict[str, str] = {
    "TLTA":  "positive — SHAP increases with leverage (debt-overhang)",
    "NITA":  "negative — SHAP decreases with profitability",
    "SIGMA": "positive — SHAP increases with volatility (distance-to-default)",
    "EXRET": "negative — SHAP decreases with positive excess return",
    "LNTA":  "negative — SHAP decreases with size (smaller firms more distress-prone)",
}


# ---------------------------------------------------------------------------
# SHAP dependence plots
# ---------------------------------------------------------------------------

def _compute_dependence_support(
    feature_values: np.ndarray,
    shap_values_feat: np.ndarray,
    feature: str,
) -> pd.DataFrame:
    """
    Build the support-diagnostic table for a SHAP dependence plot (E4 fix).

    For each pre-specified region of the feature's value range, report:
      - n observations
      - share of test sample (%)
      - mean SHAP value in the region
      - mean feature value in the region

    The motivating example is TLTA: the SHAP dependence often shows a
    reversal at TLTA > 1.0 that is driven by very few observations.
    Examiners will press on the reversal — this table makes the supporting
    sample size explicit.
    """
    edges = DEPENDENCE_SUPPORT_BINS.get(feature)
    mask = np.isfinite(feature_values) & np.isfinite(shap_values_feat)
    fv = feature_values[mask]
    sv = shap_values_feat[mask]
    n_total = len(fv)

    if edges is None:
        # Fallback: simple tertiles
        if n_total == 0:
            return pd.DataFrame()
        q = np.quantile(fv, [0.0, 1/3, 2/3, 1.0])
        edges = [-float("inf"), float(q[1]), float(q[2]), float("inf")]

    labels = [f"({edges[i]:g}, {edges[i+1]:g}]" for i in range(len(edges) - 1)]
    bins = pd.cut(fv, bins=edges, labels=labels, include_lowest=True)
    df = pd.DataFrame({"region": bins, "value": fv, "shap": sv})
    grouped = df.groupby("region", observed=False).agg(
        n_obs=("value", "size"),
        mean_value=("value", "mean"),
        mean_shap=("shap", "mean"),
        median_shap=("shap", "median"),
    ).reset_index()
    grouped["share_pct"] = (100.0 * grouped["n_obs"] / max(n_total, 1)).round(2)
    grouped["mean_value"] = grouped["mean_value"].round(4)
    grouped["mean_shap"]  = grouped["mean_shap"].round(4)
    grouped["median_shap"] = grouped["median_shap"].round(4)
    return grouped[["region", "n_obs", "share_pct",
                    "mean_value", "mean_shap", "median_shap"]]


def plot_dependence(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature: str,
    interaction_feature: str | None = None,
    model_name: str = "xgboost",
) -> None:
    """
    SHAP dependence plot for a single feature, with a marginal histogram
    (E4 fix) that exposes tail thinning at the edges of the feature range.

    Each point is one firm-year. The x-axis is the raw feature value;
    the y-axis is the SHAP value for that feature. Optional colour coding
    by an interaction feature reveals interaction effects. A marginal
    histogram strip is drawn beneath the scatter so reviewers can see at
    a glance which regions of the dependence curve are supported by data.

    The plot is interpreted against the pre-specified theoretical prediction
    in THEORETICAL_PREDICTIONS[feature]. A support-diagnostic CSV is also
    saved to outputs/tables/shap/.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape (n_obs, n_features).
    X_test : pd.DataFrame
        Feature matrix.
    feature : str
        Feature to plot on x-axis.
    interaction_feature : str or None
        Feature to use for colour coding (auto-selects strongest interaction
        if None; pass 'auto' to use shap's built-in auto-selection).
    model_name : str
    """
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)
    OUT_TABLES_SHAP.mkdir(parents=True, exist_ok=True)
    feature_names = list(X_test.columns)
    if feature not in feature_names:
        print(f"  Warning: {feature} not in X_test columns — skipping dependence plot.")
        return

    feat_idx = feature_names.index(feature)
    interact_arg = interaction_feature if interaction_feature else "auto"

    # E4 fix: figure with two stacked axes — dependence scatter on top,
    # marginal histogram of feature support on the bottom.
    fig, (ax_dep, ax_hist) = plt.subplots(
        nrows=2, ncols=1, figsize=(7, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )

    # Switch SHAP's current axes to the top one before calling dependence_plot
    plt.sca(ax_dep)
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_test.astype(float).fillna(0).values,
        feature_names=feature_names,
        interaction_index=interact_arg,
        ax=ax_dep,
        show=False,
    )
    ax_dep.axhline(y=0, color="gray", linestyle="--", lw=0.8, alpha=0.7)

    theory_note = THEORETICAL_PREDICTIONS.get(feature, "no pre-specified prediction")
    ax_dep.set_title(
        f"SHAP Dependence — {feature}: Corporate Financial Distress Prediction\n"
        f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — Test Sample (2015–2024)",
        fontsize=11,
    )
    ax_dep.annotate(
        f"Theory: {theory_note}",
        xy=(0.02, 0.97), xycoords="axes fraction",
        ha="left", va="top", fontsize=8, color="dimgray",
        wrap=True,
    )
    ax_dep.spines["top"].set_visible(False)
    ax_dep.spines["right"].set_visible(False)
    # SHAP draws an x-label on the top axis; hide it since we share x
    ax_dep.set_xlabel("")

    # ── Marginal histogram (E4 fix) ───────────────────────────────────────
    feat_vals_raw = X_test[feature].astype(float).fillna(0).values
    shap_vals_feat = shap_values[:, feat_idx].astype(float)
    finite_vals = feat_vals_raw[np.isfinite(feat_vals_raw)]
    ax_hist.hist(
        finite_vals, bins=60, color="steelblue", alpha=0.7, edgecolor="none",
    )
    # Overlay vertical guides at pre-specified region edges (excluding ±inf)
    edges = DEPENDENCE_SUPPORT_BINS.get(feature, [])
    for e in edges:
        if np.isfinite(e):
            ax_hist.axvline(e, color="dimgray", linestyle=":", lw=0.8, alpha=0.6)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_xlabel(f"{feature} (raw value)", fontsize=10)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.tick_params(axis="y", labelsize=8)

    save_figure(fig, OUT_FIGURES_SHAP / f"dependence_{feature.lower()}_{model_name}")
    print(f"  Theoretical prediction for {feature}: {theory_note}")

    # ── Save support diagnostic table ─────────────────────────────────────
    support_df = _compute_dependence_support(feat_vals_raw, shap_vals_feat, feature)
    if not support_df.empty:
        save_table(
            support_df,
            OUT_TABLES_SHAP / f"dependence_support_{feature.lower()}_{model_name}",
            caption=(
                f"Support diagnostic for SHAP dependence of {feature} "
                f"({MODEL_DISPLAY_NAMES.get(model_name, model_name)}, test "
                "sample). Regions are pre-specified on theoretical grounds. "
                "\\texttt{n\\_obs} is the supporting sample size in each "
                "region; \\texttt{mean\\_shap} summarises the local SHAP "
                "attribution. Sparse regions (small \\texttt{share\\_pct}) "
                "indicate the dependence curve there is driven by few "
                "observations and should be interpreted conservatively."
            ),
            label=f"tab:dependence_support_{feature.lower()}_{model_name}",
        )
        print(f"    Support diagnostic: {support_df.to_string(index=False)}")


def select_dependence_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> list[str]:
    """
    Phase 5 audit fix — select the dependence-plot features EMPIRICALLY.

    The design says "dependence plots for the 5 highest-importance
    features", but the frozen implementation hard-coded
    DEPENDENCE_FEATURES; the model's actual top-5 by mean |SHAP| can
    differ. Returns the empirical top-k UNION the three pre-specified
    H4 hypothesis features (TLTA, NITA, SIGMA), which must always be
    plotted because directional predictions were pre-registered for
    them.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top_k = [feature_names[i] for i in order[:k]]
    h4 = [f for f in ("TLTA", "NITA", "SIGMA") if f in feature_names]
    selected = top_k + [f for f in h4 if f not in top_k]
    print(f"  Dependence features (empirical top-{k} ∪ H4 trio): {selected}")
    return selected


def plot_all_dependence_plots(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    features: list[str] | None = None,
    model_name: str = "xgboost",
) -> None:
    """
    Produce SHAP dependence plots for all features in `features`.

    Parameters
    ----------
    shap_values : np.ndarray
    X_test : pd.DataFrame
    features : list[str] or None
        None (default) selects the model's EMPIRICAL top-5 by mean
        |SHAP| plus the pre-specified H4 trio
        (select_dependence_features). Pass DEPENDENCE_FEATURES to
        reproduce the frozen hard-coded list.
    model_name : str
    """
    if features is None:
        features = select_dependence_features(
            shap_values, list(X_test.columns))
    for feat in features:
        plot_dependence(shap_values, X_test, feat, model_name=model_name)


# ---------------------------------------------------------------------------
# Leverage × Profitability interaction heatmap
# ---------------------------------------------------------------------------

def plot_lev_roa_heatmap(
    model,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_tertiles: int = 3,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    3×3 heatmap of mean predicted distress probability by TLTA × NITA tertiles.

    Tests whether the combination of high leverage AND low profitability
    generates a super-additive distress probability — an effect that
    logistic regression without interaction terms cannot capture.

    If the heatmap shows super-additivity (the high-TLTA, low-NITA cell
    is disproportionately elevated relative to the sum of individual
    effects), this confirms H₄ and demonstrates genuine ML value-add.

    Parameters
    ----------
    model : fitted model (XGBoost)
    test : pd.DataFrame
    features : list[str]
    n_tertiles : int
    model_name : str

    Returns
    -------
    pd.DataFrame
        n_tertiles × n_tertiles table of mean predicted probabilities.
    """
    OUT_FIGURES_SHAP.mkdir(parents=True, exist_ok=True)

    X_test_arr = test[features].astype(float).fillna(0).values
    y_prob = model.predict_proba(X_test_arr)[:, 1]

    # Assign tertiles for TLTA (leverage) and NITA (profitability)
    df = test[["TLTA", "NITA"]].copy()
    df["y_prob"] = y_prob
    df["lev_tertile"] = pd.qcut(df["TLTA"], q=n_tertiles, labels=["Low", "Mid", "High"])
    df["roa_tertile"] = pd.qcut(df["NITA"], q=n_tertiles, labels=["Low", "Mid", "High"],
                                duplicates="drop")

    # Mean predicted probability per cell
    heatmap_df = (
        df.groupby(["lev_tertile", "roa_tertile"], observed=True)["y_prob"]
          .mean()
          .unstack("roa_tertile")
    )
    # Reorder: leverage rows ascending (Low→High); profitability columns ascending (Low→High)
    heatmap_df = heatmap_df.loc[["Low", "Mid", "High"], ["Low", "Mid", "High"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        heatmap_df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Predicted Distress Prob."},
    )
    ax.set_xlabel("NITA Tertile (Profitability)")
    ax.set_ylabel("TLTA Tertile (Leverage)")
    ax.set_title(
        f"Leverage × Profitability Interaction: Corporate Financial Distress Prediction\n"
        f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — Test Sample (2015–2024)"
    )
    save_figure(fig, OUT_FIGURES_SHAP / f"lev_roa_heatmap_{model_name}")

    # Phase 5 audit fix — the raw heatmap cannot TEST super-additivity
    # (H4): elevated corner cells are equally consistent with additive
    # marginal effects. Save the interaction contrast alongside it.
    contrast = compute_interaction_contrast(heatmap_df)
    save_table(
        contrast.reset_index(),
        OUT_TABLES_SHAP / f"lev_roa_interaction_contrast_{model_name}",
        caption=(
            "Leverage $\\times$ profitability interaction contrast: observed "
            "cell mean minus the additive expectation (row margin + column "
            "margin $-$ grand mean). Positive values in the high-leverage, "
            "low-profitability corner indicate genuine super-additivity; "
            "values near zero indicate the heatmap pattern is explained by "
            "additive marginal effects."
        ),
        label=f"tab:lev_roa_contrast_{model_name}",
    )
    print("  Interaction contrast (observed − additive):")
    print(contrast.round(4).to_string())

    return heatmap_df


def compute_interaction_contrast(heatmap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Observed minus additive-expectation contrast for a 2-way cell-mean
    table. Additive expectation for cell (i, j) = row_i margin + col_j
    margin − grand mean (all margins computed as unweighted means of the
    cell means). A cell's positive contrast = super-additive joint
    effect beyond the two marginal effects — the quantity H4's
    "super-additivity" claim is actually about.
    """
    m = heatmap_df.astype(float)
    grand = m.to_numpy().mean()
    row_m = m.mean(axis=1)
    col_m = m.mean(axis=0)
    additive = pd.DataFrame(
        row_m.to_numpy()[:, None] + col_m.to_numpy()[None, :] - grand,
        index=m.index, columns=m.columns,
    )
    return m - additive
