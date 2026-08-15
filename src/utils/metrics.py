"""
Shared metric utilities — descriptive statistics and group-level metrics.
=========================================================================
Provides helper functions for the descriptive statistics chapter (Chapter 4)
and for subsample performance analysis (Chapter 6).

The evaluation metrics used in model comparison are in src/models/evaluate.py.
This module covers descriptive statistics, group comparisons, and
diagnostic plots required by Blueprint v4 §8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summary_stats_by_group(
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "distress",
) -> pd.DataFrame:
    """
    Compute summary statistics separately for distressed and non-distressed firms.

    Required output for Blueprint v4 §8.2:
    mean, median, std, 10th percentile, 90th percentile for each feature,
    separately by group, with a t-statistic and Wilcoxon z-statistic for
    the between-group difference.

    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]
    group_col : str
        Column containing the binary group indicator (0/1).

    Returns
    -------
    pd.DataFrame
        One row per feature; columns: mean_0, median_0, std_0, p10_0, p90_0,
        mean_1, median_1, std_1, p10_1, p90_1, t_stat, t_pvalue,
        wilcoxon_z, wilcoxon_pvalue.
    """
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue

        g0 = df.loc[df[group_col] == 0, feat].dropna()
        g1 = df.loc[df[group_col] == 1, feat].dropna()

        # Group statistics
        def _stats(s: pd.Series) -> dict:
            return {
                "mean":   s.mean(),
                "median": s.median(),
                "std":    s.std(),
                "p10":    s.quantile(0.10),
                "p90":    s.quantile(0.90),
            }

        s0 = _stats(g0)
        s1 = _stats(g1)

        # Two-sample t-test (Welch, unequal variance)
        t_stat, t_pval = stats.ttest_ind(g0, g1, equal_var=False)

        # Wilcoxon rank-sum test (Mann-Whitney U)
        try:
            mwu_stat, mwu_pval = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            # Convert U-statistic to approximate z-score
            n0, n1 = len(g0), len(g1)
            mu_u   = n0 * n1 / 2
            sigma_u = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
            z_mwu  = (mwu_stat - mu_u) / sigma_u if sigma_u > 0 else 0.0
        except Exception:
            z_mwu, mwu_pval = np.nan, np.nan

        rows.append({
            "feature":        feat,
            "mean_nondist":   round(s0["mean"],   4),
            "median_nondist": round(s0["median"], 4),
            "std_nondist":    round(s0["std"],    4),
            "p10_nondist":    round(s0["p10"],    4),
            "p90_nondist":    round(s0["p90"],    4),
            "mean_dist":      round(s1["mean"],   4),
            "median_dist":    round(s1["median"], 4),
            "std_dist":       round(s1["std"],    4),
            "p10_dist":       round(s1["p10"],    4),
            "p90_dist":       round(s1["p90"],    4),
            "t_stat":         round(float(t_stat),  3),
            "t_pvalue":       round(float(t_pval),  4),
            "wilcoxon_z":     round(float(z_mwu),   3),
            "wilcoxon_pvalue": round(float(mwu_pval), 4),
        })

    return pd.DataFrame(rows)


def annual_distress_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual distress rate by fiscal year.

    Required output for Blueprint v4 §8.1 (Sample Composition Table).
    Sanity check: rates should spike in 2001–02, 2008–09, and 2020.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'fyear' and 'distress' columns.

    Returns
    -------
    pd.DataFrame
        Columns: fyear, n_total, n_distressed, distress_rate_pct.
    """
    agg = (
        df.groupby("fyear")
          .agg(
              n_total      =("distress", "count"),
              n_distressed =("distress", "sum"),
          )
          .reset_index()
    )
    agg["distress_rate_pct"] = (
        agg["n_distressed"] / agg["n_total"] * 100
    ).round(3)
    return agg


def correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
    high_corr_threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson correlation matrix and flag highly correlated pairs.

    Required output for Blueprint v4 §8.3.
    Pairs with |r| > high_corr_threshold are flagged — relevant for
    interpreting SHAP values for correlated feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]
    high_corr_threshold : float
        Default 0.70 per Blueprint v4 §8.3.

    Returns
    -------
    corr_matrix : pd.DataFrame
        Full correlation matrix.
    high_corr_pairs : pd.DataFrame
        Pairs with |r| > threshold. Columns: feature_a, feature_b, correlation.
    """
    available = [f for f in features if f in df.columns]
    corr_matrix = df[available].corr(method="pearson")

    # Extract upper triangle (avoid duplicates and self-correlations)
    rows = []
    for i, fa in enumerate(available):
        for j, fb in enumerate(available):
            if j <= i:
                continue
            r = corr_matrix.loc[fa, fb]
            if abs(r) > high_corr_threshold:
                rows.append({"feature_a": fa, "feature_b": fb, "correlation": round(r, 4)})

    high_corr_pairs = (
        pd.DataFrame(rows)
          .sort_values("correlation", key=abs, ascending=False)
          .reset_index(drop=True)
    )
    return corr_matrix, high_corr_pairs
