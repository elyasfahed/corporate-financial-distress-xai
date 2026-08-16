"""
Feature engineering pipeline — orchestrates all feature construction.
======================================================================
Reads the raw merged panel and produces the final feature matrix split
into train / validation / test sets with no data leakage.

Pipeline
--------
  1. Load panel_raw.parquet and GDP deflator
  2. Build 11 accounting features (src/features/accounting_features.py)
  3. Build 6 market features (src/features/market_features.py)
  4. Chronological split into train / val / test
  5. Compute winsorisation thresholds on TRAIN ONLY → save to YAML
  6. Apply winsorisation to all three splits
  7. Compute imputation medians on TRAIN ONLY
  8. Apply imputation to all three splits
  9. Save feature matrices and split metadata

Saves
-----
    data/processed/features/features_all.parquet   — full panel (pre-split)
    data/processed/samples/train.parquet
    data/processed/samples/val.parquet
    data/processed/samples/test.parquet
    data/processed/samples/split_summary.csv        — N, distress rate per split
    outputs/models/configs/winsor_thresholds.yaml
    outputs/models/configs/imputation_medians.parquet

Run
---
    python -m src.features.build_features
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    DATA_MERGED,
    DATA_FEATURES,
    DATA_SAMPLES,
    DATA_FF_FACTORS,
    DATA_RAW_CRSP,
    OUT_MODELS_CONFIGS,
    ALL_FEATURES,
    ACCOUNTING_FEATURES,
    MARKET_IMPUTE_FEATURES,
    MB_MISSING_COL,
    MIN_ACCOUNTING_NONMISSING,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    SAMPLE_END_YEAR,
)
from src.features.accounting_features import build_accounting_features
from src.features.market_features import build_market_features
from src.features.winsorize import compute_thresholds, apply_thresholds, save_thresholds
from src.features.impute import compute_imputation_medians, apply_imputation


# ---------------------------------------------------------------------------
# F4 fix — post-winsorisation ≥8-of-11 features filter
# ---------------------------------------------------------------------------

def apply_post_winsor_missingness_filter(
    df: pd.DataFrame,
    split_name: str,
    min_nonmissing: int = MIN_ACCOUNTING_NONMISSING,
) -> pd.DataFrame:
    """
    F4 fix — strict ≥`min_nonmissing`-of-11 ACCOUNTING_FEATURES filter
    applied *after winsorisation* (design §6.4).

    A firm-year is dropped if fewer than `min_nonmissing` of the 11
    constructed accounting predictors (NITA, TLTA, WCTA, ...) are
    non-missing after winsorisation. This matches the design prose;
    the merge-layer filter is a lenient pre-screen on raw Compustat
    items only.

    Parameters
    ----------
    df : pd.DataFrame
    split_name : str
        Name of the split ('train', 'val', 'test') for logging.
    min_nonmissing : int
        Default = MIN_ACCOUNTING_NONMISSING = 8.

    Returns
    -------
    pd.DataFrame
        Filtered copy of `df`.
    """
    available = [f for f in ACCOUNTING_FEATURES if f in df.columns]
    n_features = len(available)
    n_required = min(min_nonmissing, n_features)
    nonmissing = df[available].notna().sum(axis=1)
    keep_mask = nonmissing >= n_required
    n_before = len(df)
    out = df.loc[keep_mask].copy()
    n_dropped = n_before - len(out)
    pct = (100.0 * n_dropped / n_before) if n_before else 0.0
    print(f"  F4 post-winsor filter ({split_name}): require ≥{n_required}/{n_features} "
          f"non-missing accounting features → dropped {n_dropped:,} of {n_before:,} "
          f"({pct:.2f}%)")
    return out


# ---------------------------------------------------------------------------
# GDP Deflator loader
# ---------------------------------------------------------------------------

def load_gdp_deflator() -> pd.DataFrame:
    """
    Load the GDP deflator index for constant-dollar deflation.

    The deflator is used to convert nominal total assets and market cap
    to constant 2012 USD (LNTA, LNMK).

    Returns
    -------
    pd.DataFrame
        Columns: ['year', 'deflator'] where deflator = 1.0 in base year.

    Notes
    -----
    Source: US Bureau of Economic Analysis (BEA) GDP implicit price
    deflator. Download from FRED series 'GDPDEF' or BEA Table 1.1.9.
    Save to data/external/macro/gdp_deflator.csv with columns: year, deflator.
    """
    from src.config import DATA_MACRO, DEFLATOR_BASE_YEAR

    deflator_path = DATA_MACRO / "gdp_deflator.csv"

    if not deflator_path.exists():
        # Provide clear guidance instead of a cryptic FileNotFoundError
        raise FileNotFoundError(
            f"GDP deflator not found at:\n  {deflator_path}\n\n"
            "Download the BEA GDP implicit price deflator (FRED series 'GDPDEF')\n"
            "and save as a CSV with columns: year (int), deflator (float, index).\n"
            "The 2012 value should equal 100 (raw FRED format).\n"
            "Example first rows:\n  year,deflator\n  1990,72.753\n  ...  2012,100.0"
        )

    df_raw = pd.read_csv(deflator_path)

    # Handle FRED format: observation_date + GDPDEF  (quarterly → annual average)
    if "observation_date" in df_raw.columns and "GDPDEF" in df_raw.columns:
        df_raw["year"] = pd.to_datetime(df_raw["observation_date"]).dt.year
        df_raw["deflator"] = pd.to_numeric(df_raw["GDPDEF"], errors="coerce")
        df_raw = (
            df_raw.groupby("year", as_index=False)["deflator"]
                  .mean()
        )

    # Normalise so the base year has deflator = 1.0
    if "year" not in df_raw.columns or "deflator" not in df_raw.columns:
        raise ValueError(
            "gdp_deflator.csv must have columns 'year' (int) and 'deflator' (float)."
        )

    df_raw["year"] = df_raw["year"].astype(int)
    df_raw["deflator"] = df_raw["deflator"].astype(float)

    base_val = df_raw.loc[df_raw["year"] == DEFLATOR_BASE_YEAR, "deflator"].values
    if len(base_val) == 0:
        raise ValueError(
            f"Base year {DEFLATOR_BASE_YEAR} not found in gdp_deflator.csv."
        )

    df_raw["deflator"] = df_raw["deflator"] / base_val[0]
    print(f"  GDP deflator loaded ({len(df_raw)} years; base year={DEFLATOR_BASE_YEAR}, value=1.0)")
    return df_raw[["year", "deflator"]]


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------

def split_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the panel into chronological train / validation / test sets.

    Split boundaries (design §3.4, frozen):
      Train      : fyear <= TRAIN_END_YEAR  (1990–2009)
      Validation : TRAIN_END_YEAR < fyear <= VAL_END_YEAR  (2010–2014)
      Test       : fyear > VAL_END_YEAR  (2015–2024)  ← evaluate ONCE

    Parameters
    ----------
    panel : pd.DataFrame
        Full feature panel with 'fyear' column.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    train = panel[panel["fyear"] <= TRAIN_END_YEAR].copy()
    val   = panel[(panel["fyear"] > TRAIN_END_YEAR) & (panel["fyear"] <= VAL_END_YEAR)].copy()
    test  = panel[panel["fyear"] > VAL_END_YEAR].copy()

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n = len(split)
        d = split["distress"].sum() if "distress" in split.columns else "?"
        rate = f"{100*d/n:.2f}%" if isinstance(d, int) else "?"
        print(f"  {name:8s}: {n:,} obs | distress={d} ({rate})")

    return train, val, test


def purge_outer_split_labels(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    horizon_days: int,
    date_col: str = "fdate",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove labels that were not mature at the next split's origin.

    The feature-year split is chronological, but a 365-day forward label can
    cross the boundary.  For a genuine real-time design, training labels must
    be fully observed before validation starts, and validation labels must be
    fully observed before the held-out test starts.
    """
    def _purge(source: pd.DataFrame, next_split: pd.DataFrame, name: str):
        if source.empty or next_split.empty:
            return source
        if date_col not in source or date_col not in next_split:
            raise KeyError(f"Outer purge requires {date_col!r} in both splits")
        origin = pd.to_datetime(next_split[date_col], errors="coerce").min()
        label_end = pd.to_datetime(source[date_col], errors="coerce") + pd.Timedelta(
            days=horizon_days
        )
        keep = label_end.lt(origin)
        n_drop = int((~keep).sum())
        events_drop = int(source.loc[~keep, "distress"].sum())
        print(f"  Outer label-maturity purge ({name}): dropped {n_drop:,} rows "
              f"and {events_drop:,} events; next origin={origin.date()}")
        return source.loc[keep].copy()

    return (
        _purge(train, val, "train before validation"),
        _purge(val, test, "validation before test"),
        test,
    )


def _split_summary(train, val, test) -> pd.DataFrame:
    """Build a summary DataFrame of split sizes and distress rates."""
    rows = []
    for name, df in [("train", train), ("val", val), ("test", test)]:
        n = len(df)
        d = int(df["distress"].sum()) if "distress" in df.columns else None
        rows.append({
            "split": name,
            "fyear_min": int(df["fyear"].min()),
            "fyear_max": int(df["fyear"].max()),
            "n_obs": n,
            "n_distress": d,
            "distress_rate_pct": round(100 * d / n, 3) if d is not None else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    panel_path=None,
    features_path=None,
    samples_dir=None,
    save_winsor_imputation_configs: bool = True,
    market_min_obs: int = 6,
    eligible_permnos=None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
    configs_out_dir=None,
    save_raw_splits: bool = False,
    missing_policy: str = "frozen",
    lag_donor: pd.DataFrame | None = None,
    impute_market: bool = False,
    add_mb_missing: bool = False,
    outer_purge_horizon_days: int | None = None,
) -> None:
    """
    Build features and chronological splits from a merged panel.

    Parameterised so the same pipeline can produce the primary dataset,
    the RC5 alternative dataset, and the v2 corrected dataset from
    different input panels into different output directories. All
    defaults reproduce the frozen primary behaviour byte-for-byte.

    Parameters
    ----------
    panel_path : Path or None
        Input merged panel. None = DATA_MERGED/panel_raw.parquet (primary).
    features_path : Path or None
        Output features parquet path. None = DATA_FEATURES/features_all.parquet.
    samples_dir : Path or None
        Output directory for train/val/test splits. None = DATA_SAMPLES.
    save_winsor_imputation_configs : bool
        If True (default — primary), saves winsorisation thresholds and
        imputation medians. Set False for RC5 so it doesn't overwrite
        the primary's saved configs.
    market_min_obs : int
        Trailing-window minimum for EXRET/SIGMA (6 frozen / 12 v2, §9).
    eligible_permnos : set or None
        Restrict the index + RSIZE denominator to the eligible universe
        (None frozen; v2 passes the universe-filter PERMNOs, §17(ii)).
    sic_col, peer_rule : str
        Imputation options ("sich"/"rows" frozen; "_sic"/"per_feature"
        v2, §18b).
    configs_out_dir : Path or None
        Where winsor thresholds + imputation medians are saved when
        save_winsor_imputation_configs is True. None = the frozen
        outputs/models/configs/ (primary). The v2 rebuild passes
        outputs/models/configs/v2/ so primary artifacts stay untouched.
    save_raw_splits : bool
        If True, additionally writes train_raw/val_raw/test_raw.parquet
        (post-split, PRE-winsorisation, PRE-imputation) to samples_dir —
        required input for fold-safe hyperparameter tuning
        (src/models/fold_safe_cv.py).
    missing_policy : str
        "frozen" (default) or "strict" — OENEG/INTWO handling of missing
        raw inputs (src/features/accounting_features.py; v2 "strict").
    lag_donor : pd.DataFrame or None
        Raw Compustat rows of the pre-sample fiscal year (FY1989), used
        only as lag donors for the first sample year and removed before
        any split (v2 only).
    impute_market : bool
        If True (v2), EXRET/SIGMA/MB are routed through the standard
        train-fitted imputation hierarchy instead of reaching the models
        as fillna(0) (§8 principled fix). PRICE / market cap are still
        never imputed. False (frozen) imputes accounting features only.
    add_mb_missing : bool
        If True (v2), adds the MB_MISSING indicator (1 when MB is
        undefined, i.e. book equity <= 0) BEFORE splitting/imputation —
        row-local, leak-free. The v2 model feature set is then
        ALL_FEATURES_V2 (18). False (frozen) adds nothing.
    """
    # Default paths reproduce primary behaviour exactly
    panel_path    = panel_path    if panel_path    is not None else DATA_MERGED / "panel_raw.parquet"
    features_path = features_path if features_path is not None else DATA_FEATURES / "features_all.parquet"
    samples_dir   = samples_dir   if samples_dir   is not None else DATA_SAMPLES

    # ── 1. Load inputs ────────────────────────────────────────────────────
    print(f"Loading panel from {panel_path} ...")
    panel = pd.read_parquet(panel_path)
    msf   = pd.read_parquet(DATA_RAW_CRSP / "crsp_monthly_raw.parquet")
    gdp_deflator = load_gdp_deflator()

    # ── 2. Accounting features ────────────────────────────────────────────
    panel = build_accounting_features(panel, gdp_deflator,
                                      missing_policy=missing_policy,
                                      lag_donor=lag_donor)

    # ── 3. Market features ────────────────────────────────────────────────
    panel = build_market_features(panel, msf, gdp_deflator,
                                  market_min_obs=market_min_obs,
                                  eligible_permnos=eligible_permnos)

    # ── 3b. MB missingness indicator (v2 — §8 principled fix) ─────────────
    # Row-local (depends only on the row's own MB), so it is leak-free to
    # compute before the chronological split. Must be built BEFORE
    # imputation fills MB, or the indicator would be all-zero.
    if add_mb_missing:
        panel[MB_MISSING_COL] = panel["MB"].isna().astype(int)
        print(f"  {MB_MISSING_COL}: {int(panel[MB_MISSING_COL].sum()):,} of "
              f"{len(panel):,} firm-years flagged (MB undefined)")

    # ── 4. Save full feature panel ────────────────────────────────────────
    features_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(features_path, index=False)
    print(f"\nFull feature panel saved ({len(panel):,} obs, {panel.shape[1]} cols) -> {features_path}")

    # ── 5. Chronological split ────────────────────────────────────────────
    print("\nSplitting into train / val / test ...")
    train, val, test = split_panel(panel)
    if outer_purge_horizon_days is not None:
        train, val, test = purge_outer_split_labels(
            train, val, test, horizon_days=outer_purge_horizon_days
        )

    # ── 5b. Optional raw (pre-winsor, pre-impute) splits ──────────────────
    # Needed by fold-safe hyperparameter tuning, which re-fits the
    # preprocessing inside each CV fold (src/models/fold_safe_cv.py).
    if save_raw_splits:
        samples_dir.mkdir(parents=True, exist_ok=True)
        train.to_parquet(samples_dir / "train_raw.parquet", index=False)
        val.to_parquet(samples_dir   / "val_raw.parquet",   index=False)
        test.to_parquet(samples_dir  / "test_raw.parquet",  index=False)
        print(f"  Raw (pre-preprocessing) splits saved -> {samples_dir}")

    # ── 6. Winsorisation (thresholds from TRAIN only) ─────────────────────
    print("\nComputing winsorisation thresholds (train only) ...")
    thresholds = compute_thresholds(train)
    if save_winsor_imputation_configs:
        if configs_out_dir is not None:
            configs_out_dir = Path(configs_out_dir)
            configs_out_dir.mkdir(parents=True, exist_ok=True)
            save_thresholds(thresholds, path=configs_out_dir / "winsor_thresholds.yaml")
        else:
            save_thresholds(thresholds)

    train = apply_thresholds(train, thresholds)
    val   = apply_thresholds(val,   thresholds)
    test  = apply_thresholds(test,  thresholds)

    # ── 6b. F4 fix — post-winsorisation ≥8-of-11 features filter ──────────
    # Design §6.4 requires the strict missingness filter to be applied AFTER
    # winsorisation on the constructed accounting features (NITA, TLTA, ...).
    print("\nApplying post-winsorisation ≥8-of-11 features filter (F4 fix) ...")
    train = apply_post_winsor_missingness_filter(train, "train")
    val   = apply_post_winsor_missingness_filter(val,   "val")
    test  = apply_post_winsor_missingness_filter(test,  "test")

    # ── 7. Imputation (medians from TRAIN only) ───────────────────────────
    # Frozen: the 11 accounting features only; market features reach the
    # models after fillna(0). v2 (impute_market=True) routes
    # EXRET/SIGMA/MB through the same train-fitted hierarchy; the
    # MB_MISSING indicator (stage 3b) preserves the MNAR signal.
    impute_features = list(ACCOUNTING_FEATURES)
    if impute_market:
        impute_features += MARKET_IMPUTE_FEATURES
    print("\nComputing imputation medians (train only) ...")
    sic2_med, sic2_only_med, annual_med, global_med = compute_imputation_medians(
        train, features=impute_features, sic_col=sic_col, peer_rule=peer_rule)

    # Save imputation medians for reproducibility (primary only)
    if save_winsor_imputation_configs:
        med_dir = Path(configs_out_dir) if configs_out_dir is not None else OUT_MODELS_CONFIGS
        med_dir.mkdir(parents=True, exist_ok=True)
        sic2_med.to_parquet(med_dir / "imputation_sic2_medians.parquet", index=False)
        sic2_only_med.to_parquet(med_dir / "imputation_sic2_only_medians.parquet", index=False)
        annual_med.to_parquet(med_dir / "imputation_annual_medians.parquet", index=False)
        # Persist the global (Level-4) fallback medians; without them a
        # saved model's preprocessing cannot be reproduced exactly.
        import yaml
        with open(med_dir / "imputation_global_medians.yaml", "w") as f:
            yaml.dump({k: float(v) for k, v in global_med.items()}, f)

    train = apply_imputation(train, sic2_med, sic2_only_med, annual_med, global_med,
                             features=impute_features, sic_col=sic_col)
    val   = apply_imputation(val,   sic2_med, sic2_only_med, annual_med, global_med,
                             features=impute_features, sic_col=sic_col)
    test  = apply_imputation(test,  sic2_med, sic2_only_med, annual_med, global_med,
                             features=impute_features, sic_col=sic_col)

    # ── 8. Save splits ────────────────────────────────────────────────────
    samples_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(samples_dir / "train.parquet", index=False)
    val.to_parquet(samples_dir   / "val.parquet",   index=False)
    test.to_parquet(samples_dir  / "test.parquet",  index=False)

    summary = _split_summary(train, val, test)
    summary.to_csv(samples_dir / "split_summary.csv", index=False)
    print(f"\nSplit summary:\n{summary.to_string(index=False)}")
    print(f"\nSplits saved -> {samples_dir}")


def main() -> None:
    """Primary feature pipeline entry point — design defaults."""
    run_feature_pipeline()
    print("\nNext step: python -m src.models.train")


if __name__ == "__main__":
    main()
