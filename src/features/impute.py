"""
Missing-value imputation for the 11 accounting predictors.
============================================================
Pre-Specified Empirical Design §6.4 imputation protocol:

  1. Primary: SIC-2-digit industry cross-sectional median for the same
     fiscal year.
  2. Fall-back: full-sample annual median if fewer than 5 industry peers
     are available for a given SIC-2 × fyear cell.
  3. Market capitalisation (me_dec) and PRICE are NEVER imputed —
     firm-years with missing values for these are dropped upstream.

Imputation is applied separately to each split (train / val / test) using
ONLY the medians from the training sample. This prevents leakage.

Design-specification reference: §6.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ACCOUNTING_FEATURES

# Minimum number of industry peers required for SIC-2 median imputation
MIN_INDUSTRY_PEERS = 5


def compute_imputation_medians(
    train_df: pd.DataFrame,
    features: list[str] | None = None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute imputation medians from the training sample ONLY.

    F2 fix — the imputation hierarchy now has four levels (Pre-Specified Empirical Design §6.4):
      1. SIC-2 × fyear median (primary; requires ≥5 industry peers in
         the same fiscal year)
      2. SIC-2 median across all training years (NEW — F2 fix)
         For val/test years (2010+) the fyear lookup at Level 1 misses
         every time because no training data exists past 2009. This level
         provides industry-specific imputation for those years.
      3. Annual full-sample median for the same fyear (fall-back inside
         the training period)
      4. Global training-period median (final fall-back)

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set. Must contain `sic_col` and 'fyear'.
    features : list[str] or None
        Features to impute. Defaults to ACCOUNTING_FEATURES.
    sic_col : str, default "sich"
        Column holding the historical SIC code. The frozen default "sich"
        reproduces the original run byte-for-byte — but in the frozen data
        extract `sich` is zero for every row, so
        the industry conditioning silently collapsed to one pooled group.
        The corrected rebuild must pass sic_col="_sic" (the valid
        historical code produced by the merge).
    peer_rule : str, default "rows"
        How the ≥MIN_INDUSTRY_PEERS support rule counts peers.
        "rows" (frozen): counts firm-year ROWS in the SIC-2 × fyear cell,
        regardless of whether the feature being imputed is present.
        "per_feature" (corrected): counts NON-MISSING values of each
        feature separately, so a cell qualifies feature-by-feature.

    Returns
    -------
    sic2_medians : pd.DataFrame
        Level 1 — SIC-2 × fyear medians; ≥MIN_INDUSTRY_PEERS cells only.
    sic2_only_medians : pd.DataFrame
        Level 2 (F2 fix) — SIC-2 medians averaged across all training years.
    annual_medians : pd.DataFrame
        Level 3 — Annual full-sample medians (training years only).
    global_medians : dict
        Level 4 — Overall training-period medians (final fall-back).
    """
    if features is None:
        features = ACCOUNTING_FEATURES

    available = [f for f in features if f in train_df.columns]

    if sic_col not in train_df.columns:
        raise KeyError(f"sic_col={sic_col!r} not found in training frame")
    if peer_rule not in ("rows", "per_feature"):
        raise ValueError(f"peer_rule must be 'rows' or 'per_feature', got {peer_rule!r}")

    train_df = train_df.copy()
    sic = pd.to_numeric(train_df[sic_col], errors="coerce").astype("Float64")
    # CRSP/Compustat use zero as an unknown placeholder in some extracts;
    # it is not a genuine SIC industry and must not create a pseudo-industry.
    if sic_col == "_sic":
        sic = sic.mask(sic <= 0)
    train_df["sic2"] = (sic // 100).astype("Int64")

    if peer_rule == "rows":
        # Frozen behaviour: peer support counts firm-year ROWS in the cell.
        # SIC-2 × fyear peer counts (needed to enforce ≥ MIN_INDUSTRY_PEERS)
        peer_counts = (
            train_df.groupby(["sic2", "fyear"])["sic2"]
            .count()
            .reset_index(name="_peer_count")
        )

        # SIC-2 × fyear medians (all cells, before peer filter)
        sic2_medians_all = (
            train_df.groupby(["sic2", "fyear"])[available]
            .median()
            .reset_index()
        )

        # Keep only cells with enough peers — cells with < MIN_INDUSTRY_PEERS firms
        # fall back to annual median (applied in apply_imputation)
        sic2_medians = sic2_medians_all.merge(peer_counts, on=["sic2", "fyear"], how="left")
        n_before = len(sic2_medians)
        sic2_medians = sic2_medians[sic2_medians["_peer_count"] >= MIN_INDUSTRY_PEERS].copy()
        sic2_medians = sic2_medians.drop(columns=["_peer_count"])
        n_dropped = n_before - len(sic2_medians)
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} SIC-2×fyear cells with < {MIN_INDUSTRY_PEERS} peers "
                  f"(will use annual median fallback).")
    else:
        # Corrected behaviour: a cell qualifies feature-by-feature, counting
        # only NON-MISSING values of the feature being imputed. Disqualified
        # (feature, cell) medians are masked to NaN, so the Level-1 lookup
        # falls through to the next level for exactly those features.
        grouped = train_df.groupby(["sic2", "fyear"])[available]
        cell_medians = grouped.median()
        cell_support = grouped.count()
        masked = cell_medians.where(cell_support >= MIN_INDUSTRY_PEERS)
        n_masked = int((cell_medians.notna() & masked.isna()).sum().sum())
        sic2_medians = masked.dropna(how="all").reset_index()
        if n_masked > 0:
            print(f"  Masked {n_masked} (feature × SIC-2 × fyear) medians with "
                  f"< {MIN_INDUSTRY_PEERS} non-missing peers "
                  f"(fall back to lower levels).")

    # F2 fix — SIC-2 medians across all training years (Level 2)
    # Built from cells that already pass the ≥MIN_INDUSTRY_PEERS filter to
    # avoid noise from sparse industry-years. For each SIC-2, we take the
    # median across the (qualifying) per-year medians, which is robust to
    # year-by-year outliers.
    if not sic2_medians.empty:
        sic2_only_medians = (
            sic2_medians.groupby("sic2")[available]
            .median()
            .reset_index()
        )
        print(f"  F2 fix: built SIC-2 (all-train-years) fallback for "
              f"{len(sic2_only_medians)} industries")
    else:
        sic2_only_medians = pd.DataFrame(columns=["sic2"] + available)

    # Annual full-sample medians (fall-back for training years)
    annual_medians = (
        train_df.groupby("fyear")[available]
        .median()
        .reset_index()
    )

    # Global training-period medians (final fall-back for val/test years
    # that have no matching fyear in annual_medians)
    global_medians = train_df[available].median().to_dict()

    print(f"  Imputation medians computed for {len(available)} features.")
    return sic2_medians, sic2_only_medians, annual_medians, global_medians


def apply_imputation(
    df: pd.DataFrame,
    sic2_medians: pd.DataFrame,
    sic2_only_medians: pd.DataFrame | None = None,
    annual_medians: pd.DataFrame | None = None,
    global_medians: dict | None = None,
    features: list[str] | None = None,
    sic_col: str = "sich",
) -> pd.DataFrame:
    """
    Impute missing values using a four-level hierarchy (all from training):
      1. SIC-2 × fyear median (primary)
      2. SIC-2 median across all training years (F2 fix — NEW)
      3. Annual full-sample median for the same fyear (training years only)
      4. Overall training-period median (final fall-back)

    Market cap and PRICE are never imputed (they must already be present
    or the observation dropped upstream).

    Parameters
    ----------
    df : pd.DataFrame
    sic2_medians : pd.DataFrame
        Level 1 — SIC-2 × fyear medians.
    sic2_only_medians : pd.DataFrame or None
        Level 2 (F2 fix) — SIC-2 medians across all training years.
        Pass None to fall straight to Level 3 (legacy 3-level behaviour).
    annual_medians : pd.DataFrame or None
        Level 3 — Annual full-sample medians.
    global_medians : dict or None
        Level 4 — overall training-period medians {feature: scalar}.
    features : list[str] or None
    sic_col : str, default "sich"
        Column holding the historical SIC code; must match the sic_col
        used in compute_imputation_medians (frozen default "sich";
        corrected rebuild "_sic").

    Returns
    -------
    pd.DataFrame
        Copy of df with missing accounting features imputed.
    """
    if features is None:
        features = ACCOUNTING_FEATURES

    available = [f for f in features if f in df.columns]
    df = df.copy()

    # Derive SIC-2 digit code from the historical SIC column
    if sic_col in df.columns:
        sic = pd.to_numeric(df[sic_col], errors="coerce")
        if sic_col == "_sic":
            sic = sic.mask(sic <= 0)
        df["sic2"] = (sic.fillna(-1).astype(float) // 100).astype(int)
    else:
        df["sic2"] = -1

    # Determine which SIC-2 × fyear cells have enough peers in training data
    # We use the peer counts embedded in sic2_medians (any row present = has data)
    peer_counts = (
        sic2_medians.groupby(["sic2", "fyear"])
        .size()
        .reset_index(name="_peer_count")
    ) if "sic2" in sic2_medians.columns else pd.DataFrame(
        columns=["sic2", "fyear", "_peer_count"]
    )

    # Build flat lookups
    sic2_med_lookup   = sic2_medians.set_index(["sic2", "fyear"]) \
        if not sic2_medians.empty else None
    sic2_only_lookup  = sic2_only_medians.set_index("sic2") \
        if sic2_only_medians is not None and not sic2_only_medians.empty else None
    annual_med_lookup = annual_medians.set_index("fyear") \
        if annual_medians is not None and not annual_medians.empty else None

    # F2 fix — tally how many imputations each level resolves
    level_counts = {"L1_sic2_fyear": 0, "L2_sic2_only": 0,
                    "L3_annual": 0, "L4_global": 0}

    for feat in available:
        if df[feat].isna().sum() == 0:
            continue  # no missing values for this feature

        missing_mask = df[feat].isna()
        if not missing_mask.any():
            continue

        # Vectorised imputation using map. Work in plain float64: nullable
        # integer feature columns (e.g. binary indicators round-tripped
        # through parquet as Int64) cannot accept the object-dtype median
        # series produced by the lookups below (Series.update -> putmask
        # coercion TypeError). No-op for the frozen float64 panel.
        imputed = df[feat].astype("float64")

        # Level 1: SIC-2 × fyear median
        if sic2_med_lookup is not None and feat in sic2_med_lookup.columns:
            before = imputed.isna().sum()
            idx_key = list(zip(df.loc[missing_mask, "sic2"],
                               df.loc[missing_mask, "fyear"]))
            sic2_vals = pd.to_numeric(pd.Series(
                [sic2_med_lookup[feat].get(k, np.nan) for k in idx_key],
                index=df.index[missing_mask],
            ), errors="coerce")
            imputed.update(sic2_vals)
            level_counts["L1_sic2_fyear"] += int(before - imputed.isna().sum())

        # Level 2 (F2 fix): SIC-2 median across all training years
        still_missing = imputed.isna() & missing_mask
        if (still_missing.any()
                and sic2_only_lookup is not None
                and feat in sic2_only_lookup.columns):
            before = imputed.isna().sum()
            sic2_only_vals = pd.to_numeric(pd.Series(
                [sic2_only_lookup[feat].get(s, np.nan)
                 for s in df.loc[still_missing, "sic2"]],
                index=df.index[still_missing],
            ), errors="coerce")
            imputed.update(sic2_only_vals)
            level_counts["L2_sic2_only"] += int(before - imputed.isna().sum())

        # Level 3: annual full-sample median for the same fyear
        still_missing = imputed.isna() & missing_mask
        if (still_missing.any()
                and annual_med_lookup is not None
                and feat in annual_med_lookup.columns):
            before = imputed.isna().sum()
            fyear_vals = pd.to_numeric(pd.Series(
                [annual_med_lookup[feat].get(fy, np.nan)
                 for fy in df.loc[still_missing, "fyear"]],
                index=df.index[still_missing],
            ), errors="coerce")
            imputed.update(fyear_vals)
            level_counts["L3_annual"] += int(before - imputed.isna().sum())

        # Level 4: overall training-period median (final fall-back)
        still_missing = imputed.isna() & missing_mask
        if still_missing.any() and global_medians and feat in global_medians:
            before = imputed.isna().sum()
            imputed.loc[still_missing] = global_medians[feat]
            level_counts["L4_global"] += int(before - imputed.isna().sum())

        df[feat] = imputed

    # Category-safe imputation for the binary accounting indicators.  The
    # hierarchy above may return fractional medians (especially 0.5).  Those
    # are not valid states.  Snap to the nearest category with an explicit
    # conservative tie rule: exactly 0.5 becomes 0.
    for feat in ("OENEG", "INTWO"):
        if feat in available and df[feat].notna().any():
            df[feat] = (pd.to_numeric(df[feat], errors="coerce") > 0.5).astype("int8")

    n_remaining = sum(df[f].isna().sum() for f in available)
    print(f"  Imputation complete. Remaining NaN across {len(available)} features: "
          f"{n_remaining:,}")
    print(f"  Imputation hierarchy usage: "
          f"L1(SIC2×year)={level_counts['L1_sic2_fyear']:,}  "
          f"L2(SIC2-only)={level_counts['L2_sic2_only']:,}  "
          f"L3(annual)={level_counts['L3_annual']:,}  "
          f"L4(global)={level_counts['L4_global']:,}")

    df = df.drop(columns=["sic2"], errors="ignore")
    return df
