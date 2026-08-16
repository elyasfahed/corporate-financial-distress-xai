"""
SHAP feature ranking stability analysis.
=========================================
Addresses the SHAP-instability limitation stated in §10.5 of the
pre-specified study design:

  "SHAP instability: test by bootstrapping the test period; material
   ranking changes must be reported, not suppressed."

Method
------
Bootstrap the test set 100 times using block resampling by firm.
For each bootstrap sample, compute SHAP values (TreeExplainer for
XGBoost) and rank features by mean |SHAP|. Report:
  - Mean rank across bootstrap samples
  - Standard deviation of rank
  - 95% confidence interval of rank (2.5th and 97.5th percentiles)

A feature whose rank CI spans more than 3 positions is flagged as
UNSTABLE and must be interpreted conservatively in the thesis.

Design reference: §10.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.config import OUT_TABLES_SHAP, RANDOM_SEED, ALL_FEATURES
from src.utils.tables import save_table

N_BOOTSTRAP = 100


def run_shap_stability(
    model,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    model_name: str = "xgboost",
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """
    Bootstrap SHAP feature ranking to assess stability.

    Parameters
    ----------
    model : fitted XGBoost or tree model
    test : pd.DataFrame
        Test set with 'gvkey' column for firm-level block bootstrap.
    features : list[str]
    model_name : str
    n_bootstrap : int
        Number of bootstrap resamples (default 100).

    Returns
    -------
    pd.DataFrame
        Stability table: feature, mean_rank, std_rank, ci_lower, ci_upper,
        stable (bool).
    """
    print(f"\n  SHAP stability analysis ({n_bootstrap} bootstrap samples) ...")

    rng = np.random.default_rng(seed=RANDOM_SEED)
    unique_firms = test["gvkey"].unique()
    n_firms = len(unique_firms)
    firm_to_idx = {f: np.where(test["gvkey"].values == f)[0] for f in unique_firms}

    X_full = test[features].astype(float).fillna(0).values
    explainer = shap.TreeExplainer(model)

    rank_matrix = []   # shape: (n_bootstrap, n_features)
    mag_matrix  = []   # shape: (n_bootstrap, n_features)  — mean |SHAP| per feature (E2 fix)

    for b in range(n_bootstrap):
        sampled = rng.choice(unique_firms, size=n_firms, replace=True)
        idx = np.concatenate([firm_to_idx[f] for f in sampled])

        X_b = X_full[idx]
        sv  = explainer.shap_values(X_b)
        if isinstance(sv, list):
            sv = sv[1]

        mean_abs = np.abs(sv).mean(axis=0)   # shape: (n_features,)
        mag_matrix.append(mean_abs)
        # Rank 1 = highest importance
        ranks = pd.Series(mean_abs, index=features).rank(ascending=False).values
        rank_matrix.append(ranks)

        if (b + 1) % 20 == 0:
            print(f"    Bootstrap {b+1}/{n_bootstrap} done.")

    rank_matrix = np.array(rank_matrix)   # (n_bootstrap, n_features)
    mag_matrix  = np.array(mag_matrix)    # (n_bootstrap, n_features)

    # Also compute primary (full test set) rank for reference
    sv_full = explainer.shap_values(X_full)
    if isinstance(sv_full, list):
        sv_full = sv_full[1]
    primary_rank = (
        pd.Series(np.abs(sv_full).mean(axis=0), index=features)
        .rank(ascending=False)
        .astype(int)
    )

    # Primary magnitude on full test set (for reference)
    primary_mag = pd.Series(np.abs(sv_full).mean(axis=0), index=features)

    rows = []
    for i, feat in enumerate(features):
        ranks_i = rank_matrix[:, i]
        mags_i  = mag_matrix[:, i]

        # Rank stability (existing)
        rank_lo = float(np.percentile(ranks_i, 2.5))
        rank_hi = float(np.percentile(ranks_i, 97.5))

        # Magnitude stability across bootstrap samples
        mag_mean = float(mags_i.mean())
        mag_std  = float(mags_i.std())
        mag_lo   = float(np.percentile(mags_i, 2.5))
        mag_hi   = float(np.percentile(mags_i, 97.5))
        mag_cv   = (mag_std / mag_mean) if mag_mean > 0 else float("nan")

        rows.append({
            "feature":               feat,
            "primary_rank":          int(primary_rank[feat]),
            "primary_mean_abs_shap": round(float(primary_mag[feat]), 6),
            # Rank stability
            "mean_rank":             round(float(ranks_i.mean()), 2),
            "std_rank":              round(float(ranks_i.std()),  2),
            "ci_lower_rank":         round(rank_lo, 1),
            "ci_upper_rank":         round(rank_hi, 1),
            "ci_width_rank":         round(rank_hi - rank_lo, 1),
            "rank_stable":           bool((rank_hi - rank_lo) <= 3.0),
            # Magnitude stability (E2 fix)
            "mean_abs_shap_mean":    round(mag_mean, 6),
            "mean_abs_shap_std":     round(mag_std,  6),
            "mean_abs_shap_ci_lo":   round(mag_lo,   6),
            "mean_abs_shap_ci_hi":   round(mag_hi,   6),
            "magnitude_cv":          round(mag_cv,   3),
            "magnitude_stable":      bool(np.isfinite(mag_cv) and mag_cv <= 0.30),
        })

    stability_df = pd.DataFrame(rows).sort_values("primary_rank").reset_index(drop=True)

    # Flag rank-unstable features (legacy)
    rank_unstable = stability_df[~stability_df["rank_stable"]]
    if len(rank_unstable) > 0:
        print(f"\n  Rank CI width > 3 in {len(rank_unstable)} features "
              f"(treat ranks conservatively for these):")
        print(rank_unstable[["feature", "primary_rank", "ci_lower_rank",
                             "ci_upper_rank"]].to_string(index=False))
    else:
        print(f"\n  All {len(features)} features have rank CI width <= 3. "
              "Note: rank ties are mechanical when the magnitude gap between "
              "adjacent features is large, so rank stability alone is not "
              "informative — see magnitude_cv below.")

    # Flag magnitude-unstable features (E2 fix — the substantive signal)
    mag_unstable = stability_df[~stability_df["magnitude_stable"]]
    if len(mag_unstable) > 0:
        print(f"\n  Magnitude CV > 0.30 in {len(mag_unstable)} features "
              f"(SHAP magnitude varies substantially across bootstrap samples):")
        print(mag_unstable[["feature", "primary_mean_abs_shap",
                            "mean_abs_shap_ci_lo", "mean_abs_shap_ci_hi",
                            "magnitude_cv"]].to_string(index=False))
    else:
        print(f"\n  All {len(features)} features have magnitude CV <= 0.30.")

    save_table(
        stability_df,
        OUT_TABLES_SHAP / f"shap_stability_{model_name}",
        caption=(
            f"SHAP feature stability ({model_name.replace('_', ' ').title()}, "
            f"{n_bootstrap} block-bootstrap resamples by firm). "
            "Rank stability columns measure positional movement; "
            "magnitude stability columns measure variation in mean $|$SHAP$|$ "
            "(coefficient of variation \\texttt{magnitude\\_cv}). "
            "Rank ties are mechanical when adjacent magnitude gaps are large; "
            "magnitude stability is the more informative criterion."
        ),
        label=f"tab:shap_stability_{model_name}",
    )

    return stability_df
