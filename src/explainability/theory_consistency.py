"""
Theory-consistency validation table — design §10.5.
==========================================================
For each of the top 10 SHAP features, this module:
  (a) States the theoretical sign prediction (pre-specified, from the literature)
  (b) Reports the observed average SHAP direction on the test sample
  (c) Assigns a verdict: Consistent / Inconsistent / Ambiguous

CRITICAL: This table must be COMPLETED BEFORE the SHAP results are
discussed in the thesis. It must NOT be constructed retrospectively to
match observed findings. Pre-specification is what makes H₃ and H₄
falsifiable (design §3.3, §10.5).

The pre-specified theoretical predictions below are locked and must not
be changed after data work begins.

Design reference: §10.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUT_TABLES_SHAP
from src.utils.tables import save_table

# ---------------------------------------------------------------------------
# Pre-specified theoretical sign predictions (LOCKED — do not change)
# Reference: Altman (1968); Ohlson (1980); Shumway (2001);
#            Campbell, Hilscher & Szilagyi (2008); Merton (1974); Leland (1994)
# ---------------------------------------------------------------------------
THEORETICAL_SIGNS: dict[str, dict] = {
    "TLTA": {
        "sign":      "+",
        "direction": "Higher leverage → higher distress probability",
        "reference": "Ohlson (1980); Leland (1994)",
    },
    "NITA": {
        "sign":      "-",
        "direction": "Higher profitability → lower distress probability",
        "reference": "Altman (1968); Campbell et al. (2008)",
    },
    "SIGMA": {
        "sign":      "+",
        "direction": "Higher volatility → higher distress probability (distance-to-default)",
        "reference": "Merton (1974); Shumway (2001)",
    },
    "EXRET": {
        "sign":      "-",
        "direction": "Higher excess return → lower distress probability",
        "reference": "Campbell et al. (2008)",
    },
    "LNTA": {
        "sign":      "-",
        "direction": "Larger firms less distress-prone",
        "reference": "Shumway (2001)",
    },
    "WCTA": {
        "sign":      "-",
        "direction": "Higher working capital cushion → lower distress probability",
        "reference": "Altman (1968)",
    },
    "CASHTA": {
        "sign":      "-",
        "direction": "Higher cash reserve → lower distress probability",
        "reference": "Zmijewski (1984)",
    },
    "OCF_TA": {
        "sign":      "-",
        "direction": "Higher operating cash flow → lower distress probability",
        "reference": "Shumway (2001)",
    },
    "LNMK": {
        "sign":      "-",
        "direction": "Larger market cap → lower distress probability",
        "reference": "Campbell et al. (2008)",
    },
    "NITA_LAG": {
        "sign":      "-",
        "direction": "Higher lagged profitability → lower distress probability",
        "reference": "Campbell et al. (2008)",
    },
    "CLCA": {
        "sign":      "+",
        "direction": "Higher current liabilities / current assets → higher distress probability",
        "reference": "Ohlson (1980)",
    },
    "OENEG": {
        "sign":      "+",
        "direction": "Negative book equity → higher distress probability",
        "reference": "Ohlson (1980)",
    },
    "CHIN": {
        "sign":      "-",
        "direction": "Positive earnings change → lower distress probability",
        "reference": "Ohlson (1980)",
    },
    "INTWO": {
        "sign":      "+",
        "direction": "Sustained losses → higher distress probability",
        "reference": "Ohlson (1980)",
    },
    "MB": {
        "sign":      "-",
        "direction": "Higher M/B → lower distress probability",
        "reference": "Fama & French (1992)",
    },
    "RSIZE": {
        "sign":      "-",
        "direction": "Larger relative size → lower distress probability",
        "reference": "Campbell et al. (2008)",
    },
    "PRICE": {
        "sign":      "-",
        "direction": "Higher share price → lower distress probability (penny-stock signal)",
        "reference": "Shumway (2001)",
    },
}


# ---------------------------------------------------------------------------
# Observed SHAP direction computation
# ---------------------------------------------------------------------------

def compute_observed_shap_directions(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute the observed average SHAP direction for the top-n features.

    For each feature, compute the Pearson correlation between the raw
    feature value and the corresponding SHAP value across test observations.
    Positive correlation → SHAP increases with feature value ('+' direction).
    Negative correlation → SHAP decreases with feature value ('-' direction).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix (n_obs × n_features).
    X_test : pd.DataFrame
        Feature matrix (for feature values).
    top_n : int
        Number of top features to include (by mean |SHAP|).

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_abs_shap, rank, observed_direction,
                 observed_sign ('+'/'-'), corr_feature_shap.
    """
    from scipy.stats import pearsonr

    feature_names = list(X_test.columns)
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Select top_n features by mean |SHAP|
    ranked_idx = np.argsort(mean_abs)[::-1][:top_n]

    rows = []
    for rank_pos, feat_idx in enumerate(ranked_idx, start=1):
        feat = feature_names[feat_idx]
        feat_vals  = X_test.iloc[:, feat_idx].values.astype(float)
        shap_vals  = shap_values[:, feat_idx].astype(float)

        # Drop NaNs before correlation
        mask = np.isfinite(feat_vals) & np.isfinite(shap_vals)
        if mask.sum() < 10:
            corr, direction = 0.0, "Ambiguous"
        else:
            corr, _ = pearsonr(feat_vals[mask], shap_vals[mask])
            direction = "Increasing" if corr >= 0 else "Decreasing"

        obs_sign = "+" if corr >= 0 else "-"
        rows.append({
            "feature":           feat,
            "mean_abs_shap":     round(float(mean_abs[feat_idx]), 6),
            "rank":              rank_pos,
            "observed_direction": direction,
            "observed_sign":     obs_sign,
            "corr_feature_shap": round(float(corr), 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Verdict assignment
# ---------------------------------------------------------------------------

def assign_verdict(
    theoretical_sign: str,
    observed_sign: str,
    corr_abs: float,
    ambiguity_threshold: float = 0.1,
) -> str:
    """
    Assign a *raw* theory-consistency verdict (no collinearity adjustment).

    Consistent   : theoretical and observed sign agree AND |corr| > threshold
    Inconsistent : theoretical and observed sign disagree
    Ambiguous    : signs agree but |corr| <= threshold (weak relationship)

    Parameters
    ----------
    theoretical_sign : str '+' or '-'
    observed_sign : str '+' or '-'
    corr_abs : float
        Absolute value of the feature–SHAP correlation.
    ambiguity_threshold : float

    Returns
    -------
    str
        'Consistent', 'Inconsistent', or 'Ambiguous'.
    """
    if theoretical_sign == observed_sign:
        if corr_abs > ambiguity_threshold:
            return "Consistent"
        return "Ambiguous"
    return "Inconsistent"


# ---------------------------------------------------------------------------
# Collinearity diagnostics for the top-K features  (E1 fix)
# ---------------------------------------------------------------------------

def _topk_collinearity(
    X_test: pd.DataFrame,
    topk_features: list[str],
    corr_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    For each feature in `topk_features`, find its strongest correlated peer
    *within the same top-K set* and report the absolute correlation.

    This is the diagnostic that turns a naïve sign-test verdict into a
    collinearity-aware one (design §10.5 — "correlated features
    receive arbitrary SHAP splits — interpret pairs conservatively").

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature matrix.
    topk_features : list[str]
        The features under inspection.
    corr_threshold : float
        Absolute Pearson correlation above which a pair is flagged collinear.

    Returns
    -------
    pd.DataFrame
        Columns: feature, max_corr_with_other_top, collinear_partner,
                 is_collinear_in_topk (bool).
    """
    available = [f for f in topk_features if f in X_test.columns]
    if len(available) < 2:
        return pd.DataFrame({
            "feature": topk_features,
            "max_corr_with_other_top": 0.0,
            "collinear_partner":       "",
            "is_collinear_in_topk":    False,
        })

    sub = X_test[available].astype(float)
    corr = sub.corr(method="pearson")

    rows = []
    for f in topk_features:
        if f not in corr.columns:
            rows.append({
                "feature":                 f,
                "max_corr_with_other_top": 0.0,
                "collinear_partner":       "",
                "is_collinear_in_topk":    False,
            })
            continue
        others = [c for c in corr.columns if c != f]
        if not others:
            rows.append({
                "feature":                 f,
                "max_corr_with_other_top": 0.0,
                "collinear_partner":       "",
                "is_collinear_in_topk":    False,
            })
            continue
        abs_corrs = corr.loc[f, others].abs()
        partner = abs_corrs.idxmax()
        max_abs = float(abs_corrs.max())
        rows.append({
            "feature":                 f,
            "max_corr_with_other_top": round(max_abs, 3),
            "collinear_partner":       partner if max_abs > corr_threshold else "",
            "is_collinear_in_topk":    bool(max_abs > corr_threshold),
        })
    return pd.DataFrame(rows)


def adjust_verdict_for_collinearity(
    verdict_raw: str,
    is_collinear: bool,
) -> str:
    """
    Symmetric collinearity annotation.

    When a top-K feature is highly correlated (|ρ| > 0.7) with another
    top-K feature, SHAP splits attribution arbitrarily between the pair, so
    the observed sign of EITHER member is uninformative about whether the
    *model* is internally consistent with theory. The verdict is therefore
    annotated 'Ambiguous-collinear' whenever the feature is collinear —
    regardless of whether its raw sign happened to agree ('Consistent') or
    disagree ('Inconsistent') with theory.

    This is symmetric by construction: unlike the earlier one-way rule
    (which downgraded only 'Inconsistent' verdicts and could therefore only
    ever *raise* the consistency count), it never excuses a wrong-signed
    feature without equally discounting a right-signed collinear partner.
    The un-annotated pre-specified sign-test verdict is preserved separately
    in ``verdict_raw`` and remains the headline H₃ statistic.
    """
    if is_collinear:
        return "Ambiguous-collinear"
    return verdict_raw


# ---------------------------------------------------------------------------
# Build the full validation table
# ---------------------------------------------------------------------------

def build_theory_consistency_table(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    top_n: int = 10,
    write_table: bool = True,
) -> pd.DataFrame:
    """
    Build the pre-specified theory-consistency validation table.

    This is the operational test of H₃ and H₄ (design §10.5).
    It must be produced BEFORE the SHAP results section is written.

    Parameters
    ----------
    shap_values : np.ndarray
    X_test : pd.DataFrame
    top_n : int

    Returns
    -------
    pd.DataFrame
        Columns: rank, feature, theoretical_sign, theoretical_direction,
                 reference, observed_sign, corr_feature_shap, verdict.
    """
    observed = compute_observed_shap_directions(shap_values, X_test, top_n)

    # E1 fix: collinearity diagnostic across the top-K features
    topk_features = observed["feature"].tolist()
    collinearity = _topk_collinearity(X_test, topk_features, corr_threshold=0.7)
    collinearity = collinearity.set_index("feature")

    rows = []
    for _, row in observed.iterrows():
        feat = row["feature"]
        theory = THEORETICAL_SIGNS.get(feat, {})
        t_sign = theory.get("sign", "?")
        o_sign = row["observed_sign"]
        corr   = row["corr_feature_shap"]

        verdict_raw = assign_verdict(t_sign, o_sign, abs(corr))

        # Look up collinearity info; default to no collinearity
        max_corr_other = float(collinearity.loc[feat, "max_corr_with_other_top"]) \
            if feat in collinearity.index else 0.0
        partner = str(collinearity.loc[feat, "collinear_partner"]) \
            if feat in collinearity.index else ""
        is_col = bool(collinearity.loc[feat, "is_collinear_in_topk"]) \
            if feat in collinearity.index else False

        # Symmetric collinearity annotation (disclosed as a separate column);
        # the headline `verdict` is the pre-specified raw sign test so the
        # H₃ count is never inflated by a one-way rescue.
        verdict_adj = adjust_verdict_for_collinearity(verdict_raw, is_col)

        rows.append({
            "rank":                        int(row["rank"]),
            "feature":                     feat,
            "theoretical_sign":            t_sign,
            "theoretical_direction":       theory.get("direction", ""),
            "reference":                   theory.get("reference", ""),
            "observed_sign":               o_sign,
            "corr_feature_shap":           round(corr, 3),
            "max_corr_with_other_top":     round(max_corr_other, 3),
            "collinear_partner":           partner,
            "is_collinear":                is_col,
            "verdict_raw":                 verdict_raw,
            "verdict_collinearity_adjusted": verdict_adj,
            # Headline verdict = the honest, pre-specified sign test.
            "verdict":                     verdict_raw,
        })

    table = pd.DataFrame(rows).sort_values("rank")

    # Summary — report the honest raw sign test as the headline, with the
    # collinearity caveat disclosed alongside (never used to inflate the count).
    n_consistent   = (table["verdict_raw"] == "Consistent").sum()
    n_inconsistent = (table["verdict_raw"] == "Inconsistent").sum()
    n_ambiguous    = (table["verdict_raw"] == "Ambiguous").sum()
    n_collinear    = int(table["is_collinear"].sum())
    n_amb_coll     = (table["verdict_collinearity_adjusted"]
                      == "Ambiguous-collinear").sum()
    print(f"  Theory-consistency (raw sign test, top {top_n}): "
          f"{n_consistent} Consistent | {n_inconsistent} Inconsistent | "
          f"{n_ambiguous} Ambiguous")
    print(f"    collinearity disclosure: {n_collinear} of top {top_n} are "
          f"collinear (|r|>0.7); conservative symmetric annotation flags "
          f"{n_amb_coll} as Ambiguous-collinear")

    # Save — write_table=False lets an alternative-specification caller
    # (the v2 parity batch) save its own prefixed copy WITHOUT overwriting
    # the frozen v1 table (2026-07-15: a v2 run silently rewrote it;
    # restored from git — same guard as h2_leakage_sensitivity).
    if write_table:
        OUT_TABLES_SHAP.mkdir(parents=True, exist_ok=True)
        save_table(table, OUT_TABLES_SHAP / "theory_consistency_table")

    return table
