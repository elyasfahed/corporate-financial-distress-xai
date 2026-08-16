"""
Merge Compustat Annual + Filing Dates + CRSP into a clean panel.
=================================================================
Implements the two-layer CCM + CUSIP merge (Pre-Specified Empirical Design §6.2) and
the leakage-free distress label construction (Pre-Specified Empirical Design §5.2).

Pipeline
--------
  1.  Load all raw files
  2.  Apply Compustat universe restrictions (SIC, at > 0, fyear range)
  3.  Attach 10-K filing dates; apply fall-back lag for missing dates
  4.  Drop obs with filing-date lag > 180 days
  5.  CCM primary merge (GVKEY → PERMNO, date-range validated)
  6.  CUSIP cross-validation (flag mismatches; do not silently drop)
  7.  Attach December market cap from CRSP monthly (never imputed)
  8.  Drop obs with missing market cap
  9.  Construct distress label D_{i,t} anchored to filing date
  10. Apply ≥2 consecutive fiscal years filter
  11. Apply ≥8-of-11 accounting items non-missing filter
  12. Record attrition at every step → attrition_table.csv

Timing convention (critical):
  D_{i,t} = 1  iff  any DLSTCD 400–499 in [F_{i,t}, F_{i,t} + 365]
  where F_{i,t} is the actual 10-K filing date, NOT fiscal year-end.

Design-specification reference: §5.2, §6.1–6.4

Saves
-----
    data/processed/merged/panel_raw.parquet
    data/processed/merged/attrition_table.csv
    data/processed/merged/cusip_mismatches.csv   (for manual review)

Run
---
    python -m src.data.merge_crsp_compustat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_COMPUSTAT,
    DATA_RAW_CRSP,
    DATA_MERGED,
    EXCLUDED_SIC_RANGES,
    VALID_SHRCDS,
    MAX_FILING_LAG_DAYS,
    FALLBACK_FILING_LAG_DAYS,
    DISTRESS_CODES_PRIMARY,
    DISTRESS_HORIZON_DAYS,
    SAMPLE_START_YEAR,
    SAMPLE_END_YEAR,
    MIN_CONSECUTIVE_YEARS,
    MIN_ACCOUNTING_NONMISSING,
    FILING_DATE_UNMATCHED_POLICY,
)

# Whether firm-years with no real filing date are dropped (design-faithful)
# or kept with a +lag estimate. "broaden" fills 20-F/AR at build time, then
# drops any residual; "estimate" keeps and imputes. See config.py.
_DROP_MISSING_FDATES = FILING_DATE_UNMATCHED_POLICY in ("drop", "broaden")

# Raw Compustat items whose availability determines the ≥8-of-11 filter
_ACCOUNTING_ITEMS_RAW = [
    "ib", "at", "lt", "act", "lct",
    "che", "oancf", "ni", "ceq", "re", "sale",
]


# ===========================================================================
# Attrition tracker
# ===========================================================================

class AttritionTracker:
    """Records firm-year counts at each sample restriction step."""

    def __init__(self) -> None:
        self._steps: list[dict] = []

    def record(self, df: pd.DataFrame, step: str) -> None:
        """Log the current row count and unique firm count."""
        n       = len(df)
        n_firms = df["gvkey"].nunique() if "gvkey" in df.columns else None
        self._steps.append({"step": step, "firm_years": n, "unique_firms": n_firms})
        tag = f"  ({n_firms:,} firms)" if n_firms else ""
        print(f"  [{step}]  {n:,} firm-years{tag}")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._steps)


# ===========================================================================
# Helper utilities
# ===========================================================================

def _load(path) -> pd.DataFrame:
    print(f"  Loading  {path.name} ...")
    return pd.read_parquet(path)


def _is_excluded_sic(
    sic: pd.Series,
    excluded_sic_ranges: list[tuple[int, int]] | None = None,
) -> pd.Series:
    """
    Return True for SIC codes in the excluded ranges.

    Parameters
    ----------
    sic : pd.Series
    excluded_sic_ranges : list of (low, high) inclusive ranges
        If None, uses the design default from config (financials + utilities).
        RC5 passes [(6000, 6999)] to keep utilities in the sample.
    """
    if excluded_sic_ranges is None:
        excluded_sic_ranges = EXCLUDED_SIC_RANGES
    mask = pd.Series(False, index=sic.index)
    for lo, hi in excluded_sic_ranges:
        mask |= sic.between(lo, hi)
    return mask


# ===========================================================================
# Step 1 — Load raw data
# ===========================================================================

def load_raw(comp_path=None, delist_path=None,
             secnames_path=None) -> tuple[pd.DataFrame, ...]:
    """
    Load all raw parquet files from Seafile data directory.

    The three optional path overrides let the v2 rebuild read its
    corrected raw_v2/ extracts (standard FYE dating, corrected delisting
    mapping, letter-field security names) while every other input — and
    the default behaviour — stays the frozen file.
    """
    print("\nLoading raw data files ...")
    comp     = _load(comp_path     or DATA_RAW_COMPUSTAT / "compustat_annual_raw.parquet")
    filing   = _load(DATA_RAW_COMPUSTAT / "compustat_filing_dates.parquet")
    msf      = _load(DATA_RAW_CRSP      / "crsp_monthly_raw.parquet")
    delist   = _load(delist_path   or DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
    ccm      = _load(DATA_RAW_CRSP      / "ccm_linktable_raw.parquet")
    secnames = _load(secnames_path or DATA_RAW_CRSP / "crsp_security_names.parquet")
    return comp, filing, msf, delist, ccm, secnames


# ===========================================================================
# Step 2 — Compustat universe restrictions
# ===========================================================================

def apply_compustat_filters(
    comp: pd.DataFrame,
    tracker: AttritionTracker,
    excluded_sic_ranges: list[tuple[int, int]] | None = None,
    sample_end_year: int | None = None,
) -> pd.DataFrame:
    """
    Apply sample restrictions to Compustat:
      - Drop financials (SIC 6000–6999) and utilities (SIC 4900–4999)
      - Drop non-positive total assets
      - Restrict to sample fiscal year range
      - Deduplicate to one observation per GVKEY-fiscal year

    Parameters
    ----------
    comp : pd.DataFrame
        Raw Compustat fundamentals.
    tracker : AttritionTracker

    Returns
    -------
    pd.DataFrame
    """
    print("\nApplying Compustat universe restrictions ...")
    tracker.record(comp, "01  Raw Compustat")

    # Prefer historical SIC from companyhistory.HSIC (already merged upstream
    # in load_local_rds.build_compustat_annual as the `sic` column); fall
    # back to legacy `sich` where companyhistory had no matching interval.
    # The local CIZ source populates `sich` = 0 for ~79% of rows, so 0 is
    # treated as missing (not a valid SIC).
    comp = comp.copy()
    sich_num = pd.to_numeric(comp["sich"], errors="coerce").replace(0, np.nan)
    if "sic" in comp.columns:
        sic_num = pd.to_numeric(comp["sic"], errors="coerce")
        comp["_sic"] = sic_num.fillna(sich_num)
    else:
        comp["_sic"] = sich_num

    n_missing_sic = comp["_sic"].isna().sum()
    if n_missing_sic > 0:
        print(f"  WARN: {n_missing_sic:,} firm-years lack any SIC — kept (not excluded)")

    # 1. Drop excluded SIC ranges (primary: financials 6000-6999 + utilities 4900-4999;
    #    RC5: only financials 6000-6999 — utilities kept in sample)
    exclude_mask = _is_excluded_sic(comp["_sic"].fillna(-1), excluded_sic_ranges)
    _ranges_label = (
        "financials+utilities" if excluded_sic_ranges is None
        else f"SIC {excluded_sic_ranges}"
    )
    print(f"  Excluding {_ranges_label}: {int(exclude_mask.sum()):,} firm-years")
    comp = comp[~exclude_mask].copy()
    tracker.record(comp, f"02  Excl. {_ranges_label}")

    # 2. Drop non-positive or missing total assets
    comp = comp[comp["at"].notna() & (comp["at"] > 0)].copy()
    tracker.record(comp, "03  Excl. at <= 0 or missing")

    # 3. Restrict to sample period. sample_end_year=None keeps the frozen
    #    SAMPLE_END_YEAR (2024); the v2 rebuild passes SAMPLE_END_YEAR_V2
    #    (2023) because FY2024 outcomes are right-censored by the delisting
    #    extract.
    end_year = sample_end_year if sample_end_year is not None else SAMPLE_END_YEAR
    comp = comp[
        comp["fyear"].between(SAMPLE_START_YEAR, end_year)
    ].copy()
    tracker.record(comp, "04  Restrict to sample period")

    # 4. Deduplicate — keep latest datadate per gvkey-fyear
    comp = (
        comp.sort_values("datadate")
            .drop_duplicates(subset=["gvkey", "fyear"], keep="last")
            .reset_index(drop=True)
    )
    # Retain the harmonised `_sic` column for downstream descriptive grouping
    # (sample_composition_by_sic table). Do not drop.
    tracker.record(comp, "05  Deduplicated gvkey-fyear")

    return comp


# ===========================================================================
# Step 3 — Attach filing dates
# ===========================================================================

def attach_filing_dates(
    comp: pd.DataFrame,
    filing: pd.DataFrame,
    drop_missing_fdates: bool = True,
) -> pd.DataFrame:
    """
    Merge actual 10-K filing dates onto Compustat panel.

    F1 fix — design fidelity (Pre-Specified Empirical Design §5.2):
      "Observations with missing filing dates are dropped (not imputed)."
    When `drop_missing_fdates=True` (the default), firm-years that lack an
    actual 10-K filing date are dropped at this step rather than receiving
    a `datadate + FALLBACK_FILING_LAG_DAYS` fallback. The count of dropped
    rows is logged for transparency. Set drop_missing_fdates=False to
    retain the previous lenient behaviour (kept only for diagnostic runs).

    The default was False until 2026-07-28 while this docstring already said
    "(default)" for True, so a bare call silently took the legacy fallback
    path. Every caller in the tree passes the flag explicitly, so aligning the
    default with the documented design changes no existing behaviour.

    Parameters
    ----------
    comp : pd.DataFrame
    filing : pd.DataFrame
        Output of extract_compustat.extract_filing_dates() / load_local_rds.
    drop_missing_fdates : bool
        F1 fix flag. True = drop missing-fdate rows (design-faithful);
        False = legacy fallback behaviour.

    Returns
    -------
    pd.DataFrame
        With columns fdate (datetime), fdate_source (str),
        filing_lag_days (int).
    """
    print("\nAttaching 10-K filing dates ...")

    # Drop fdate if already present in comp (comes from load_local_rds) to
    # avoid _x/_y suffixes when merging the dedicated filing-dates table.
    comp = comp.drop(columns=["fdate", "fdate_source"], errors="ignore")

    # Merge actual filing dates on gvkey + datadate
    comp = comp.merge(
        filing[["gvkey", "datadate", "fdate", "fdate_source"]],
        on=["gvkey", "datadate"],
        how="left",
    )

    # Classify rows. Two paths produce a non-actual filing date:
    #   1. NaN fdate after the merge (local-RDS extract path; source="missing")
    #   2. fdate_source already labelled "fallback_*" by extract_compustat
    src = comp["fdate_source"].astype(str)
    is_fallback = src.str.startswith("fallback") | comp["fdate"].isna()
    n_actual   = int((~is_fallback).sum())
    n_fallback = int(is_fallback.sum())
    print(f"  Filing dates — actual: {n_actual:,}  |  fallback/missing: {n_fallback:,}")

    if drop_missing_fdates:
        # F1 fix: drop rows that do not have an actual filing date
        before = len(comp)
        comp = comp.loc[~is_fallback].copy()
        print(f"  F1 fix: dropped {before - len(comp):,} firm-years lacking an "
              "actual 10-K filing date (design-faithful behaviour).")
    else:
        # Legacy fallback path — apply +FALLBACK_FILING_LAG_DAYS to missing fdates
        missing = comp["fdate"].isna()
        comp.loc[missing, "fdate"] = (
            comp.loc[missing, "datadate"] + pd.Timedelta(days=FALLBACK_FILING_LAG_DAYS)
        )
        comp.loc[missing, "fdate_source"] = f"fallback_{FALLBACK_FILING_LAG_DAYS}d_lag"

    # Compute filing lag in calendar days (now defined for all surviving rows)
    comp["filing_lag_days"] = (comp["fdate"] - comp["datadate"]).dt.days
    if (comp["filing_lag_days"] < 0).any():
        n_negative = int((comp["filing_lag_days"] < 0).sum())
        raise ValueError(
            f"Found {n_negative:,} filing dates before fiscal year-end; "
            "negative filing lags violate the point-in-time design."
        )

    # Drop observations where filing lag is implausibly large (>180 days)
    n_before = len(comp)
    comp = comp[comp["filing_lag_days"] <= MAX_FILING_LAG_DAYS].copy()
    n_dropped = n_before - len(comp)
    print(f"  Dropped {n_dropped:,} obs with filing lag > {MAX_FILING_LAG_DAYS} days")

    return comp


# ===========================================================================
# Step 4 — CCM primary merge (GVKEY → PERMNO)
# ===========================================================================

def merge_ccm_primary(
    comp: pd.DataFrame,
    ccm: pd.DataFrame,
    tracker: AttritionTracker,
) -> pd.DataFrame:
    """
    Join Compustat to CRSP PERMNO via CCM link table.

    Date-range constraint: datadate must fall within [linkdt, linkenddt].

    The CCM file is produced by load_local_rds.build_ccm_linktable() from
    the local linkhistory.rds (Compustat CIZ) and is pre-filtered to
    linktype IN ('LC','LU') and linkprim IN ('P','C') per the Pre-Specified Empirical Design
    §6.2 at extraction time.

    Parameters
    ----------
    comp : pd.DataFrame
    ccm : pd.DataFrame
    tracker : AttritionTracker

    Returns
    -------
    pd.DataFrame
        With permno attached to each firm-year.
    """
    print("\nCCM primary merge (GVKEY → PERMNO) ...")

    # Drop permno/permco if already present in comp (from load_local_rds lpermno)
    # to avoid _x/_y suffixes after the CCM merge.
    comp = comp.drop(columns=["permno", "permco"], errors="ignore")

    # Left-join Compustat panel to CCM link records on GVKEY
    panel = comp.merge(
        ccm[["gvkey", "permno", "permco", "linktype", "linkprim", "linkdt", "linkenddt"]],
        on="gvkey",
        how="left",
    )
    tracker.record(panel, "06  After CCM left-join (pre-date-filter)")

    # Apply date-range validity constraint: datadate must fall within [linkdt, linkenddt]
    valid_date = (
        panel["datadate"].ge(panel["linkdt"])
        & panel["datadate"].le(panel["linkenddt"])
    )
    panel = panel[valid_date].copy()
    tracker.record(panel, "07  CCM date-range filter applied")

    # If a GVKEY-fyear still has multiple links after date filtering, keep the
    # primary link (linkprim='P') before 'C', then the first by linkdt
    panel["_linkprim_rank"] = panel["linkprim"].map({"P": 0, "C": 1}).fillna(2)
    panel = panel.sort_values(
        ["gvkey", "fyear", "_linkprim_rank", "linkdt"],
        ascending=[True, True, True, True],
    )
    panel = panel.drop_duplicates(subset=["gvkey", "fyear"], keep="first")
    panel = panel.drop(columns=["_linkprim_rank"])
    tracker.record(panel, "08  Deduplicated gvkey-fyear post-CCM")

    # Drop obs without a valid PERMNO
    panel = panel[panel["permno"].notna()].copy()
    panel["permno"] = panel["permno"].astype(int)
    tracker.record(panel, "09  Excl. missing PERMNO")

    return panel.reset_index(drop=True)


# ===========================================================================
# Step 5 — CUSIP cross-validation (second layer)
# ===========================================================================

def validate_cusip(
    panel: pd.DataFrame,
    secnames: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-validate the CCM merge using CUSIP codes (Pre-Specified Empirical Design §6.2).

    Match first 8 characters of Compustat CUSIP against NCUSIP from CRSP
    Security Names (date-range validated). Firm-years where the CCM link
    and CUSIP disagree are flagged and saved to cusip_mismatches.csv for
    manual review. They are NOT silently dropped.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel after CCM primary merge.
    secnames : pd.DataFrame
        CRSP security names (extract_crsp.extract_security_names()).

    Returns
    -------
    panel : pd.DataFrame
        With column cusip_match (bool).
    mismatches : pd.DataFrame
        Firm-years where CCM and CUSIP links disagree.
    """
    print("\nCUSIP cross-validation (second merge layer) ...")

    # If cusip column is absent (e.g. companydescription join produced no matches),
    # skip validation and continue — documented as a limitation.
    panel = panel.copy()
    if "cusip" not in panel.columns:
        print("  ⚠  No 'cusip' column in panel — skipping CUSIP cross-validation.")
        panel["cusip_match"] = False
        return panel.reset_index(drop=True), pd.DataFrame()

    # Standardise Compustat CUSIP to first 8 characters (uppercase, stripped)
    panel["cusip8_comp"] = (
        panel["cusip"]
        .fillna("")
        .str.strip()
        .str.upper()
        .str[:8]
    )

    # Build a point-in-time NCUSIP lookup from CRSP security names.
    # For each permno × fiscal year-end date, find the NCUSIP valid on that date.
    secnames = secnames.copy()
    secnames["ncusip8"] = (
        secnames["ncusip"]
        .fillna("")
        .str.strip()
        .str.upper()
        .str[:8]
    )

    # Merge secnames onto panel on permno with date-range validity
    panel = panel.merge(
        secnames[["permno", "ncusip8", "namedt", "nameendt"]],
        on="permno",
        how="left",
    )
    valid_name = (
        panel["datadate"].ge(panel["namedt"])
        & panel["datadate"].le(panel["nameendt"].fillna(pd.Timestamp("2099-12-31")))
    )
    panel_with_ncusip = panel[valid_name].copy()

    # Deduplicate — keep one NCUSIP per firm-year
    panel_with_ncusip = panel_with_ncusip.drop_duplicates(
        subset=["gvkey", "fyear"], keep="first"
    )

    # For firm-years that fell outside all namedt/nameendt windows, fill with NaN
    panel = panel.drop_duplicates(subset=["gvkey", "fyear"], keep="first")
    panel = panel.drop(columns=["ncusip8", "namedt", "nameendt"], errors="ignore")
    panel = panel.merge(
        panel_with_ncusip[["gvkey", "fyear", "ncusip8"]],
        on=["gvkey", "fyear"],
        how="left",
    )

    # CUSIP match: both non-empty and equal
    both_have_cusip = panel["cusip8_comp"].ne("") & panel["ncusip8"].notna()
    panel["cusip_check_available"] = both_have_cusip
    panel["cusip_match"] = (
        both_have_cusip & (panel["cusip8_comp"] == panel["ncusip8"])
    )
    panel["cusip_mismatch"] = both_have_cusip & ~panel["cusip_match"]

    n_matched   = panel["cusip_match"].sum()
    n_mismatch  = both_have_cusip.sum() - n_matched
    n_no_check  = (~both_have_cusip).sum()
    print(f"  CUSIP match : {n_matched:,}  |  mismatch: {n_mismatch:,}  |  no CUSIP: {n_no_check:,}")

    # Isolate mismatches for manual review (NOT silently dropped)
    mismatches = panel[both_have_cusip & ~panel["cusip_match"]].copy()
    if len(mismatches) > 0:
        print(f"  ⚠  {len(mismatches):,} CUSIP mismatches flagged → cusip_mismatches.csv for review")

    # Drop temporary columns
    panel = panel.drop(columns=["namedt", "nameendt"], errors="ignore")

    return panel.reset_index(drop=True), mismatches


# ===========================================================================
# Step 6 — Attach December market cap
# ===========================================================================

def attach_market_cap(
    panel: pd.DataFrame,
    msf: pd.DataFrame,
    tracker: AttritionTracker,
) -> pd.DataFrame:
    """
    Attach end-of-fiscal-year market cap from CRSP monthly stock file.

    Uses the December observation of the calendar year of the fiscal
    year-end. Market cap is never imputed — obs with missing market cap
    are dropped (Pre-Specified Empirical Design §6.4).

    Parameters
    ----------
    panel : pd.DataFrame
    msf : pd.DataFrame
    tracker : AttritionTracker

    Returns
    -------
    pd.DataFrame
        With columns me_dec (market cap, $M) and shares_dec.
    """
    print("\nAttaching fiscal year-end market capitalisation ...")

    # Extract year and month from msf date; compute me = prc × shrout / 1000
    msf = msf.copy()
    msf["me_year"]  = msf["date"].dt.year
    msf["me_month"] = msf["date"].dt.month

    # Market cap per PERMNO × calendar month-end (in millions of USD)
    # StkMthSecurityData CIZ format provides MthCap (→ me) directly — no shrout needed.
    if "me" not in msf.columns:
        if "prc" in msf.columns and "shrout" in msf.columns:
            msf["me"] = msf["prc"].abs() * msf["shrout"] / 1_000
        else:
            msf["me"] = np.nan

    # Keep one observation per permno × year × month (last by date if duplicates)
    msf_keep = ["permno", "me_year", "me_month", "me"]
    if "prc" in msf.columns:
        msf_keep.append("prc")
    if "shrout" in msf.columns:
        msf_keep.append("shrout")
    rename_map = {"me": "me_fyend", "prc": "prc_fyend", "shrout": "shrout_fyend"}

    msf_monthly = (
        msf.sort_values("date")
           .drop_duplicates(subset=["permno", "me_year", "me_month"], keep="last")
        [msf_keep]
           .rename(columns=rename_map)
    )

    # Match to fiscal year-end month: use the year and month of each firm's datadate
    panel = panel.copy()
    panel["me_year"]  = panel["datadate"].dt.year
    panel["me_month"] = panel["datadate"].dt.month

    panel = panel.merge(
        msf_monthly,
        on=["permno", "me_year", "me_month"],
        how="left",
    )
    panel = panel.drop(columns=["me_year", "me_month"])

    # Market cap is never imputed — drop obs with missing me_fyend
    n_before = len(panel)
    panel = panel[panel["me_fyend"].notna() & (panel["me_fyend"] > 0)].copy()
    n_dropped = n_before - len(panel)
    print(f"  Dropped {n_dropped:,} obs with missing or zero fiscal year-end market cap")
    tracker.record(panel, "10  Excl. missing market cap")

    return panel.reset_index(drop=True)


# ===========================================================================
# Step 7 — Construct distress label
# ===========================================================================

def build_distress_label(
    panel: pd.DataFrame,
    delist: pd.DataFrame,
    codes: list[int] | None = None,
    horizon_days: int = DISTRESS_HORIZON_DAYS,
    anchor: str = "fdate",
    reach_days: int | None = None,
    event_label_col: str | None = None,
    unique_event_assignment: bool = False,
) -> pd.DataFrame:
    """
    Construct the binary distress label D_{i,t}.

    D_{i,t} = 1 iff any DLSTCD in `codes` is observed for firm i in the
    window [A_{i,t}, A_{i,t} + reach], where A_{i,t} is the window anchor.

    Frozen / primary design (anchor="fdate"): A_{i,t} = F_{i,t}, the actual
    10-K filing date, and reach = horizon_days. This is the core identification
    assumption of the leakage-free design: using fiscal year-end instead would
    let the model observe the filing date's information content before it is
    publicly available.

    Alternative, fix-ready anchor (anchor="datadate"): A_{i,t} = fiscal
    year-end (datadate) and reach = reach_days (default horizon_days). This
    relaxes the 10-K-filing requirement (documented robustness concern /
    Status §10): it recovers "went-dark" failures whose for-cause delisting
    falls more than a year after the firm's last 10-K, at the cost of replacing
    the realised filing date with an *assumed* availability date. It is the
    operationalisation of the proposed fiscal-year-end-anchored robustness check
    and is NOT used in the frozen specification. The default arguments reproduce
    the frozen 10-K-anchored label byte-for-byte.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged panel with fdate column (and datadate if anchor="datadate").
    delist : pd.DataFrame
        CRSP delisting file.
    codes : list[int] or None
        Delisting codes to flag as distress.
        None → use primary definition (DISTRESS_CODES_PRIMARY = 400–499).
    horizon_days : int
        Forward-looking window in days for the frozen anchor (default 365).
    anchor : {"fdate", "datadate"}
        Window anchor column. "fdate" (default) = frozen 10-K-filing-date
        design; "datadate" = alternative fiscal-year-end design.
    reach_days : int or None
        Window length in days for the datadate anchor. None → horizon_days.
        Ignored when anchor="fdate" (which always uses horizon_days).

    Returns
    -------
    pd.DataFrame
        With column distress (int, 0/1) and delist_date (datetime or NaT).
    """
    if codes is None and event_label_col is None:
        from src.config import DISTRESS_CODES_PRIMARY as codes
    if anchor not in ("fdate", "datadate"):
        raise ValueError(f"anchor must be 'fdate' or 'datadate', got {anchor!r}")
    # Frozen path always uses horizon_days; only the datadate anchor honours reach_days.
    reach = horizon_days if anchor == "fdate" else (reach_days or horizon_days)
    definition = (f"event flag={event_label_col}"
                  if event_label_col is not None else f"{len(codes)} codes")
    print(f"\nConstructing distress label (anchor={anchor}, reach={reach}d, "
          f"{definition}) ...")

    # Prefer a direct CIZ event flag for the current academic specification.
    # The numeric-code path remains available for historical reproducibility.
    if event_label_col is not None:
        if event_label_col not in delist.columns:
            raise KeyError(
                f"Delisting file lacks required event flag {event_label_col!r}. "
                "Rebuild the delisting extract with mapping='academic'."
            )
        mask = pd.to_numeric(
            delist[event_label_col], errors="coerce"
        ).fillna(0).astype(bool)
        detail_col = ("dlstcd_reconstructed"
                      if "dlstcd_reconstructed" in delist.columns else "dlstcd")
        delist_distress = delist.loc[mask, [
            "permno", "dlstdt", detail_col
        ]].copy().rename(columns={detail_col: "dlstcd"})
    else:
        delist_distress = delist[delist["dlstcd"].isin(codes)][
            ["permno", "dlstdt", "dlstcd"]
        ].copy()

    # Merge all matching delisting events onto panel (one-to-many)
    panel = panel.copy()
    merged = panel[["gvkey", "fyear", "permno", anchor]].merge(
        delist_distress,
        on="permno",
        how="left",
    )

    # The event counts as distress if it falls within the prediction window
    window_start = merged[anchor]
    window_end = window_start + pd.Timedelta(days=reach)
    in_window = (
        merged["dlstdt"].notna()
        & merged["dlstdt"].ge(window_start)
        & merged["dlstdt"].le(window_end)
    )
    if unique_event_assignment and in_window.any():
        # Discrete-time failure design: one corporate delisting event creates
        # one positive firm-year.  When annual filing windows overlap, assign
        # the event to the most recent eligible filing before the event.
        candidates = merged.loc[in_window].copy()
        candidates = candidates.sort_values(
            ["permno", "dlstdt", anchor, "fyear", "gvkey"]
        )
        chosen = candidates.drop_duplicates(
            subset=["permno", "dlstdt"], keep="last"
        ).index
        n_removed = int(in_window.sum() - len(chosen))
        in_window = pd.Series(False, index=merged.index)
        in_window.loc[chosen] = True
        if n_removed:
            print(f"  Unique-event assignment removed {n_removed:,} "
                  "overlapping positive window(s).")
    merged["_distress_event"] = in_window.astype(int)

    # Aggregate to one row per firm-year: distress = 1 if ANY event in window
    distress_agg = (
        merged.groupby(["gvkey", "fyear"])
              .agg(
                  distress=("_distress_event", "max"),
                  delist_date=("dlstdt", lambda s: s[in_window.loc[s.index]].min()
                               if in_window.loc[s.index].any() else pd.NaT),
                  delist_code=("dlstcd", lambda s: (
                      int(s[in_window.loc[s.index]].dropna().iloc[0])
                      if (in_window.loc[s.index].any()
                          and not s[in_window.loc[s.index]].dropna().empty)
                      else pd.NA
                  )),
              )
              .reset_index()
    )

    # Merge distress label back to panel
    panel = panel.merge(distress_agg, on=["gvkey", "fyear"], how="left")
    panel["distress"] = panel["distress"].fillna(0).astype(int)

    n_distressed  = panel["distress"].sum()
    n_total       = len(panel)
    prevalence    = n_distressed / n_total * 100
    print(f"  Distressed firm-years : {n_distressed:,} / {n_total:,} ({prevalence:.2f}%)")

    return panel


# ===========================================================================
# Step 8 — Consecutive years filter
# ===========================================================================

def apply_consecutive_years_filter(
    panel: pd.DataFrame,
    tracker: AttritionTracker,
    min_years: int = MIN_CONSECUTIVE_YEARS,
) -> pd.DataFrame:
    """
    Drop firms with fewer than min_years consecutive fiscal years.

    Required for CHIN (change in net income) and NITA_LAG (lagged ROA)
    construction, which need at least 2 consecutive observations.

    Parameters
    ----------
    panel : pd.DataFrame
    tracker : AttritionTracker
    min_years : int

    Returns
    -------
    pd.DataFrame
    """
    print(f"\nApplying ≥{min_years} consecutive fiscal years filter ...")

    panel = panel.copy().sort_values(["gvkey", "fyear"])

    # Within each firm, compute the gap to the preceding fiscal year
    panel["_fyear_lag"] = panel.groupby("gvkey")["fyear"].shift(1)
    panel["_gap"] = panel["fyear"] - panel["_fyear_lag"]

    # A firm-year is part of a consecutive run if gap == 1
    # Use fillna(0) first to handle nullable Int64 without NA-to-int errors
    panel["_consec"] = (panel["_gap"].fillna(0) == 1).astype(int)

    # Identify firms that have AT LEAST ONE pair of consecutive years
    # (i.e., at least one gap == 1), which satisfies min_years = 2
    # For min_years > 2, we need runs of length >= min_years.
    if min_years == 2:
        valid_firms = (
            panel.groupby("gvkey")["_consec"]
                 .max()
                 .loc[lambda s: s >= 1]
                 .index
        )
    else:
        # Compute maximum run length of consecutive years per firm
        def _max_run(group: pd.Series) -> int:
            run = 0
            max_run = 0
            for v in group:
                run = run + 1 if v == 1 else 1
                max_run = max(max_run, run)
            return max_run

        run_lengths = panel.groupby("gvkey")["_consec"].apply(_max_run)
        valid_firms = run_lengths[run_lengths >= min_years].index

    panel = panel[panel["gvkey"].isin(valid_firms)].copy()
    panel = panel.drop(columns=["_fyear_lag", "_gap", "_consec"])
    tracker.record(panel, f"11  ≥{min_years} consec. years filter")

    return panel.reset_index(drop=True)


# ===========================================================================
# Step 9 — Missingness filter
# ===========================================================================

def apply_missingness_filter(
    panel: pd.DataFrame,
    tracker: AttritionTracker,
    min_nonmissing: int = 6,
) -> pd.DataFrame:
    """
    Lenient pre-screen for the ≥8-of-11 accounting items filter.

    F4 fix (Pre-Specified Empirical Design §6.4) — the design states the strict ≥8-of-11
    filter is applied to the constructed accounting *features* (NITA,
    TLTA, ...) **after winsorisation**. That filter now lives in
    `src/features/build_features.py` (post-winsor, pre-imputation).

    The merge-layer filter is retained as a lenient sanity screen
    (default ≥6-of-11 raw Compustat items) so firms with almost no
    accounting coverage never enter the feature-engineering stage and
    waste compute. The substantive ≥8-of-11 features filter happens later.

    Parameters
    ----------
    panel : pd.DataFrame
    tracker : AttritionTracker
    min_nonmissing : int
        Default 6 (lenient pre-screen). The strict ≥8 filter is applied
        in build_features.py post-winsorisation per design.

    Returns
    -------
    pd.DataFrame
    """
    print(f"\nApplying ≥{min_nonmissing}-of-{len(_ACCOUNTING_ITEMS_RAW)} accounting items filter ...")

    # Count how many of the core accounting items are non-missing for each row
    available_items = [c for c in _ACCOUNTING_ITEMS_RAW if c in panel.columns]
    if len(available_items) < len(_ACCOUNTING_ITEMS_RAW):
        missing_cols = set(_ACCOUNTING_ITEMS_RAW) - set(available_items)
        print(f"  ⚠  Columns not found (treated as missing): {missing_cols}")

    panel = panel.copy()
    panel["_n_nonmissing"] = panel[available_items].notna().sum(axis=1)

    n_before = len(panel)
    panel = panel[panel["_n_nonmissing"] >= min_nonmissing].copy()
    n_dropped = n_before - len(panel)
    print(f"  Dropped {n_dropped:,} obs with < {min_nonmissing} non-missing accounting items")

    panel = panel.drop(columns=["_n_nonmissing"])
    tracker.record(panel, f"12  ≥{min_nonmissing}-of-{len(_ACCOUNTING_ITEMS_RAW)} acctg items")

    return panel.reset_index(drop=True)


# ===========================================================================
# Main
# ===========================================================================

def main_with_overrides(
    excluded_sic_ranges: list[tuple[int, int]] | None = None,
    output_panel_path=None,
    output_attrition_path=None,
    output_mismatches_path=None,
    comp_path=None,
    delist_path=None,
    secnames_path=None,
    sample_end_year: int | None = None,
    universe_policy: str = "frozen",
    universe_date_col: str | None = None,
    distress_event_col: str | None = None,
    raw_min_nonmissing: int | None = 6,
    unique_event_assignment: bool = False,
) -> None:
    """
    Run the full merge pipeline with optional overrides.

    Used by RC5 and the v2 rebuild (and any future robustness check that
    needs an alternative universe). Defaults reproduce the primary
    specification exactly.

    Parameters
    ----------
    excluded_sic_ranges : list of (low, high) inclusive ranges or None
        SIC ranges to exclude. None = design default (financials + utilities).
        RC5 passes [(6000, 6999)] to keep utilities in the sample.
    output_panel_path : Path or None
        Where to save the merged panel. None = default DATA_MERGED/panel_raw.parquet.
    output_attrition_path, output_mismatches_path : Path or None
        Where to save attrition table and CUSIP mismatches. None = defaults.
    comp_path, delist_path, secnames_path : Path or None
        Raw-input overrides (v2 rebuild passes the raw_v2/ extracts).
        None = frozen files.
    sample_end_year : int or None
        None = frozen SAMPLE_END_YEAR (2024); v2 passes 2023 (§18c).
    universe_policy : str
        "frozen" (default) applies no share-code universe filter,
        reproducing the pipeline as run. "v2" restricts the panel to
        PERMNOs eligible under the CIZ letter-field equivalent of
        SHRCD 10/11 (src/data/universe.py; §18e) — requires a
        secnames file that carries the letter fields.
    universe_date_col : str or None
        None (frozen/PERMNO-level) keeps the any-time-eligible rule.
        The v2 rebuild passes "datadate" so eligibility is evaluated
        as-of each firm-year's fiscal year-end against the security-
        info validity ranges (2026-07-12 second-audit fix).
    """
    tracker = AttritionTracker()

    comp, filing, msf, delist, ccm, secnames = load_raw(
        comp_path=comp_path, delist_path=delist_path,
        secnames_path=secnames_path,
    )

    comp   = apply_compustat_filters(comp, tracker,
                                     excluded_sic_ranges=excluded_sic_ranges,
                                     sample_end_year=sample_end_year)
    comp   = attach_filing_dates(comp, filing,
                                 drop_missing_fdates=_DROP_MISSING_FDATES)
    tracker.record(comp, "F1: Drop missing/fallback filing dates")
    panel  = merge_ccm_primary(comp, ccm, tracker)

    # Universe filter (§18e) — after the CCM merge assigns PERMNOs, before
    # market-cap attachment. "frozen" is a no-op.
    if universe_policy != "frozen":
        from src.data.universe import apply_universe_filter
        panel = apply_universe_filter(panel, secnames, policy=universe_policy,
                                      date_col=universe_date_col)
        tracker.record(panel, "U1: Universe filter (US common shares)")

    panel, mismatches = validate_cusip(panel, secnames)
    panel  = attach_market_cap(panel, msf, tracker)
    panel  = build_distress_label(
        panel, delist, event_label_col=distress_event_col,
        unique_event_assignment=unique_event_assignment,
    )
    panel  = apply_consecutive_years_filter(panel, tracker)
    if raw_min_nonmissing is not None:
        panel = apply_missingness_filter(
            panel, tracker, min_nonmissing=raw_min_nonmissing
        )

    tracker.record(panel, "FINAL analysis sample")

    # ── Save ──────────────────────────────────────────────────────────────
    DATA_MERGED.mkdir(parents=True, exist_ok=True)

    out_panel = output_panel_path if output_panel_path is not None \
        else DATA_MERGED / "panel_raw.parquet"
    panel.to_parquet(out_panel, index=False)
    print(f"\nSaved panel         → {out_panel}")

    out_attrition = output_attrition_path if output_attrition_path is not None \
        else DATA_MERGED / "attrition_table.csv"
    tracker.to_dataframe().to_csv(out_attrition, index=False)
    print(f"Saved attrition     → {out_attrition}")

    out_mismatches_p = output_mismatches_path if output_mismatches_path is not None \
        else DATA_MERGED / "cusip_mismatches.csv"
    mismatches.to_csv(out_mismatches_p, index=False)
    print(f"Saved CUSIP flags   → {out_mismatches_p}  ({len(mismatches):,} rows for review)")

    print("\nAttrition table:")
    print(tracker.to_dataframe().to_string(index=False))

    return panel


def main() -> None:
    """Primary merge pipeline entry point — design defaults."""
    tracker = AttritionTracker()

    comp, filing, msf, delist, ccm, secnames = load_raw()

    comp   = apply_compustat_filters(comp, tracker)
    comp   = attach_filing_dates(comp, filing,
                                 drop_missing_fdates=_DROP_MISSING_FDATES)
    tracker.record(comp, "F1: Drop missing/fallback filing dates")
    panel  = merge_ccm_primary(comp, ccm, tracker)
    panel, mismatches = validate_cusip(panel, secnames)
    panel  = attach_market_cap(panel, msf, tracker)
    panel  = build_distress_label(panel, delist)
    panel  = apply_consecutive_years_filter(panel, tracker)
    panel  = apply_missingness_filter(panel, tracker)

    tracker.record(panel, "FINAL analysis sample")

    # ── Save ──────────────────────────────────────────────────────────────
    DATA_MERGED.mkdir(parents=True, exist_ok=True)

    out_panel = DATA_MERGED / "panel_raw.parquet"
    panel.to_parquet(out_panel, index=False)
    print(f"\nSaved panel         → {out_panel}")

    out_attrition = DATA_MERGED / "attrition_table.csv"
    tracker.to_dataframe().to_csv(out_attrition, index=False)
    print(f"Saved attrition     → {out_attrition}")

    out_mismatches = DATA_MERGED / "cusip_mismatches.csv"
    mismatches.to_csv(out_mismatches, index=False)
    print(f"Saved CUSIP flags   → {out_mismatches}  ({len(mismatches):,} rows for review)")

    print("\nAttrition table:")
    print(tracker.to_dataframe().to_string(index=False))

    print("\nNext step: python -m src.features.build_features")


if __name__ == "__main__":
    main()
