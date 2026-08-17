"""
Fold-safe rolling-origin CV — in-fold preprocessing + label-maturity purging.
=============================================================================
This module addresses two problems in the original hyperparameter-tuning CV:

1. **In-fold preprocessing leak.** Winsorisation thresholds and imputation
   medians are fitted ONCE on all 1990–2009 observations before the rolling
   folds are formed (build_features stage 6/7), so a fold validating on 2005
   was preprocessed with statistics that include 2005–2009 data.
   -> fold_safe_preprocess() re-fits winsorisation + imputation inside each
   fold using ONLY that fold's training years.

2. **Label-maturity leakage across the fold boundary.** Splitting folds by
   fyear alone is not point-in-time: a training firm-year filed in late 2004
   has a distress window reaching into late 2005 — beyond the earliest
   validation filing date — so its LABEL encodes post-origin information.
   -> purge_fold_train() drops from the fold's training set every row whose
   label window [fdate, fdate + horizon] extends past the fold's prediction
   origin (the earliest validation filing date), the standard purged-CV rule
   (Lopez de Prado 2018).

Both behaviours are OFF by default: tune.py's original path is unchanged
and the original results remain byte-reproducible. The v2 rebuild passes
fold_safe=True + purge_horizon_days=DISTRESS_HORIZON_DAYS (V2_PROFILE
["cv_fold_safe"]) and tunes on the RAW (pre-winsorisation, pre-imputation)
training frame.
"""

from __future__ import annotations

import pandas as pd

from src.config import DISTRESS_HORIZON_DAYS


def purge_fold_train(
    train_df: pd.DataFrame,
    tr_idx: pd.Index,
    val_idx: pd.Index,
    horizon_days: int = DISTRESS_HORIZON_DAYS,
    date_col: str = "fdate",
) -> pd.Index:
    """
    Purge label-immature rows from one rolling-origin fold's training set.

    The fold's prediction origin is the earliest filing date in the
    validation fold. A training row whose outcome window
    [fdate, fdate + horizon_days] ends at or after that origin has a label
    that encodes information from the validation period, and is dropped.
    (This also removes any training row filed at/after the origin itself,
    since its window trivially extends past it.)

    Parameters
    ----------
    train_df : pd.DataFrame
        Frame the fold indices refer to. Must contain `date_col`.
    tr_idx, val_idx : pd.Index
        One (train, validation) fold from rolling_origin_cv_splits().
    horizon_days : int
        Label horizon (primary specification: 365).
    date_col : str
        Filing-date column anchoring the label window.

    Returns
    -------
    pd.Index — the purged training index (subset of tr_idx).
    """
    if date_col not in train_df.columns:
        raise KeyError(
            f"purge_fold_train needs {date_col!r} in the training frame; "
            "tune on a frame that retains the filing date."
        )
    dates = pd.to_datetime(train_df[date_col])
    origin = dates.loc[val_idx].min()
    if pd.isna(origin):
        return tr_idx
    window_end = dates.loc[tr_idx] + pd.Timedelta(days=horizon_days)
    keep = window_end < origin
    # NaT filing dates cannot be shown mature -> purged (conservative)
    keep = keep.fillna(False)
    purged = int((~keep).sum())
    if purged:
        print(f"    Purged {purged:,} of {len(tr_idx):,} fold-train rows "
              f"(label window crosses fold origin {origin.date()})")
    return tr_idx[keep.values]


def fold_safe_preprocess(
    raw_tr: pd.DataFrame,
    raw_val: pd.DataFrame,
    features: list[str] | None = None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
    impute_features: list[str] | None = None,
    coverage_filter: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit winsorisation + coverage filter + imputation INSIDE one CV fold.

    Thresholds and medians are computed on the fold's training rows only
    and applied unchanged to the fold's validation rows — mirroring, at
    fold level, exactly what the outer pipeline does at split level:
    winsorise → ≥8-of-11 coverage filter → impute (build_features
    stages 6 / 6b / 7).

    Parameters
    ----------
    raw_tr, raw_val : pd.DataFrame
        RAW (pre-winsorisation, pre-imputation) fold frames.
    features : list[str] or None
        Winsorisation features, passed to compute_thresholds
        (None = its frozen default: ALL_FEATURES minus bounded ones).
    sic_col, peer_rule : str
        Imputation options (frozen defaults "sich"/"rows"; the v2 profile
        passes "_sic"/"per_feature" — see src/features/impute.py).
    impute_features : list[str] or None
        Features routed through the imputation hierarchy (None = the
        frozen accounting-only default; the v2 profile passes
        ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES — §8).
    coverage_filter : bool
        Apply the post-winsorisation ≥8-of-11 accounting-feature filter
        inside the fold. The raw splits are saved BEFORE the outer filter,
        so without this the tuning folds
        include firm-years the final training set drops). Default True;
        the frozen tuning path never calls this function, so frozen
        results are unaffected.

    Returns
    -------
    (tr, val) — preprocessed copies.
    """
    from src.features.winsorize import apply_thresholds, compute_thresholds
    from src.features.impute import apply_imputation, compute_imputation_medians

    thresholds = compute_thresholds(raw_tr, features=features)
    tr  = apply_thresholds(raw_tr,  thresholds)
    val = apply_thresholds(raw_val, thresholds)

    if coverage_filter:
        from src.features.build_features import apply_post_winsor_missingness_filter
        tr  = apply_post_winsor_missingness_filter(tr,  "fold-train")
        val = apply_post_winsor_missingness_filter(val, "fold-val")

    sic2_med, sic2_only_med, annual_med, global_med = compute_imputation_medians(
        tr, features=impute_features, sic_col=sic_col, peer_rule=peer_rule
    )
    tr  = apply_imputation(tr,  sic2_med, sic2_only_med, annual_med,
                           global_med, features=impute_features, sic_col=sic_col)
    val = apply_imputation(val, sic2_med, sic2_only_med, annual_med,
                           global_med, features=impute_features, sic_col=sic_col)
    return tr, val
