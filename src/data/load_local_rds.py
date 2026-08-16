"""
Local RDS → Parquet loader  (replaces WRDS extraction entirely).
================================================================
Reads the Seafile .rds files downloaded by download_data.py and
produces the six parquet files that merge_crsp_compustat.py expects.

Confirmed column structure (from explore_rds.py):
  perioddescriptorannual  → KYGVKEY, KEYSET, FYYYY, FYR, lpermno,
                             lpermco, FDATE, SICH, CURCD  (37 cols)
  balancesheetindustrialannual → same key cols + AT, LT, ACT, LCT,
                             CHE, WCAP, CEQ, RE, DLTT, DLC, ... (735 cols)
  incomestatementindustrialannual → same keys + NI, IB, SALE, ...
  cashflowannual          → same keys + OANCF, ...
  StkMthSecurityData      → CRSP CIZ monthly (column names confirmed
                             at runtime; candidates listed below)
  StkDelists              → CRSP delisting file
  StkSecurityInfoHdr/Hist → CRSP security names / CUSIP / SHRCD

Run
---
    python -m src.data.load_local_rds
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_RAW_COMPUSTAT,
    DATA_RAW_CRSP,
    SAMPLE_START_YEAR,
    SAMPLE_END_YEAR,
    FILING_DATE_SOURCE,
    FILING_DATE_UNMATCHED_POLICY,
)

COMP = DATA_RAW_COMPUSTAT
CRSP = DATA_RAW_CRSP

# ── Join keys shared across all Compustat annual files ──────────────────────
_JOIN_KEYS = ["KYGVKEY", "KEYSET", "FYYYY"]


# ===========================================================================
# Low-level read helpers
# ===========================================================================

def _read_full(path: Path) -> pd.DataFrame:
    """Read a .rds file using pyreadr (normal path)."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[MISSING] {path}\nRun: python download_data.py"
        )
    print(f"  Reading {path.name} ...", end="", flush=True)
    result = pyreadr.read_r(str(path))
    df = result[None]
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def _read_rds_cols(path: Path, needed: list[str]) -> pd.DataFrame:
    """
    Read a .rds file and return ONLY the columns in `needed`.

    For files that are too wide to consolidate into a 2-D numpy block
    (e.g. incomestatementindustrialannual with 600+ float64 cols),
    we temporarily disable pandas block-consolidation so pyreadr can
    finish constructing the DataFrame, then immediately slim it down.

    Fallback order:
      1. Normal pyreadr read → select cols
      2. Patch pandas consolidation → read again → select cols → restore
      3. rdata library (if installed)
      4. Raise informative error
    """
    if not path.exists():
        raise FileNotFoundError(f"\n[MISSING] {path}")

    print(f"  Reading {path.name} (selecting {len(needed)} cols) ...", end="", flush=True)

    # ── Attempt 1: normal read ───────────────────────────────────────────
    try:
        result = pyreadr.read_r(str(path))
        df = result[None]
        available = [c for c in needed if c in df.columns]
        out = df[available].copy()
        del df; gc.collect()
        print(f"  {out.shape[0]:,} rows, kept {len(available)} cols")
        return out
    except MemoryError:
        print(f"\n  [MEMORY] Normal read OOM — trying consolidation patch ...")

    # ── Attempt 2: patch pandas consolidation ────────────────────────────
    try:
        _mgr = pd.core.internals.managers.BlockManager
        _orig = _mgr._consolidate_inplace
        _mgr._consolidate_inplace = lambda self: None          # disable

        gc.collect()
        result = pyreadr.read_r(str(path))
        df = result[None]
        available = [c for c in needed if c in df.columns]
        out = df[available].copy()                             # small copy
        del df; gc.collect()
        _mgr._consolidate_inplace = _orig                      # restore

        print(f"  {out.shape[0]:,} rows, kept {len(available)} cols (patch)")
        return out

    except Exception as e2:
        try:
            _mgr._consolidate_inplace = _orig
        except Exception:
            pass
        print(f"\n  [PATCH FAILED] {e2}")

    # ── Attempt 3: rdata library ─────────────────────────────────────────
    try:
        import rdata as _rdata
        print("  Trying rdata library ...")
        parsed = _rdata.parser.parse_file(str(path))
        converted = _rdata.conversion.convert(parsed)
        if isinstance(converted, dict):
            for k, v in converted.items():
                if hasattr(v, "columns"):
                    converted = v; break
        available = [c for c in needed if c in converted.columns]
        out = converted[available].copy()
        del converted; gc.collect()
        print(f"  {out.shape[0]:,} rows, kept {len(available)} cols (rdata)")
        return out
    except ImportError:
        print("  rdata not installed — run: pip install rdata")
    except Exception as e3:
        print(f"  rdata failed: {e3}")

    raise RuntimeError(
        f"Could not read {path.name}. "
        "Install rdata (pip install rdata) or add more RAM."
    )


def _month_end(year: pd.Series, month: pd.Series) -> pd.Series:
    """Return the last calendar day of (year, month) as Timestamp."""
    from pandas.tseries.offsets import MonthEnd
    ym = pd.to_datetime(
        dict(year=year.astype(int), month=month.astype(int), day=1),
        errors="coerce",
    )
    return ym + MonthEnd(0)


def compustat_datadate(
    fyear: pd.Series,
    fyr: pd.Series,
    convention: str = "frozen",
) -> pd.Series:
    """
    Reconstruct the fiscal year-end date (datadate) from (FYEAR, FYR).

    The local CIZ period descriptor carries no DATADATE column, so the
    pipeline reconstructs it from the fiscal-year label and the fiscal
    year-end month. The reconstruction must respect the standard Compustat
    fiscal-year convention (the "June rule"): a fiscal year ending in
    January--May belongs to FYEAR = calendar year of the period end MINUS 1.
    Equivalently, for FYR in {1..5} the period actually ends in calendar
    year FYEAR + 1 (e.g. Walmart FYEAR 2019, FYR 1 -> period end 2020-01-31).

    Parameters
    ----------
    fyear : pd.Series
        Fiscal-year label (Compustat FYEAR).
    fyr : pd.Series
        Fiscal year-end month, 1--12 (Compustat FYR). NaN treated as 12.
    convention : {"frozen", "standard"}
        "frozen"   -- reproduces the original (buggy) construction
                      month_end(FYEAR, FYR), which mis-dates Jan--May
                      fiscal year-ends by exactly 12 months (Implementation
                      Status paragraph 17). Retained as the default so the
                      frozen pipeline stays byte-for-byte reproducible.
        "standard" -- the correct Compustat convention:
                      month_end(FYEAR + 1{FYR<=5}, FYR).

    Returns
    -------
    pd.Series
        Month-end fiscal year-end dates, indexed like the inputs.
    """
    if convention not in ("frozen", "standard"):
        raise ValueError(
            f"convention must be 'frozen' or 'standard', got {convention!r}"
        )
    fyr_int = pd.to_numeric(fyr, errors="coerce").fillna(12).astype(int)
    year = pd.to_numeric(fyear, errors="coerce").astype(int)
    if convention == "standard":
        year = year + (fyr_int <= 5).astype(int)
    return _month_end(year, fyr_int)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ===========================================================================
# COMPUSTAT — annual fundamentals
# ===========================================================================

# Columns we need from each file (all uppercase, as confirmed)
_PERIOD_NEEDED = _JOIN_KEYS + [
    "fyra", "lpermno", "lpermco",
    "FDATE", "FYEAR", "FYR", "SICH", "CURCD",
    "ACCTCHG", "ACCTSTD", "POPSRC", "CONSOL",
]

_BS_NEEDED = _JOIN_KEYS + [
    "AT", "LT", "ACT", "LCT", "CHE", "WCAP",
    "CEQ", "RE", "DLTT", "DLC", "PPENT",
    "RECT", "INVT", "SEQ", "CSHO",
]

_IS_NEEDED = _JOIN_KEYS + [
    "NI", "IB", "SALE", "OIBDP", "DP", "TXT", "XINT", "REVT",
]

_CF_NEEDED = _JOIN_KEYS + [
    "OANCF", "IVNCF", "FINCF", "CAPX", "DPC",
]

_CO_NEEDED = ["KYGVKEY", "CUSIP", "CONM", "CIK", "STATE", "LOC", "FIC"]


def _dedup_keyset(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Keep one row per gvkey x fyear using the academic standard KEYSET filter.

    Compustat CIZ stores multiple data vintages per firm-year via KEYSET.
    For academic finance (datafmt=STD, consol=C, indfmt=INDL, popsrc=D)
    only KEYSET 1 and 2 are the standard consolidated annual formats.
    All other KEYSETs (4, 2100, 2120, etc.) are segments, quarterly, or
    non-standard formats that must be excluded.

    Priority within standard KEYSETs:
      1. KEYSET = 1  (primary standard consolidated)
      2. KEYSET = 2  (restated standard consolidated)
    Within the same KEYSET, keep the row with the most non-null values.

    Academic justification: matches comp.funda filters used by Campbell,
    Hilscher & Szilagyi (2008), Shumway (2001), and Ohlson (1980).
    """
    STANDARD_KEYSETS = [1, 2]

    if "KEYSET" in df.columns:
        df_std = df[df["KEYSET"].isin(STANDARD_KEYSETS)].copy()
        n_dropped = len(df) - len(df_std)
        if len(df_std) == 0:
            print(f"  [{label}] WARNING: No KEYSET 1/2 rows — keeping all")
            df_std = df.copy()
        else:
            print(f"  [{label}] Dropped {n_dropped:,} non-standard KEYSET rows")
    else:
        df_std = df.copy()

    key_cols = {"gvkey", "fyear", "KEYSET"}
    data_cols = [c for c in df_std.columns if c not in key_cols]
    df_std = df_std.copy()
    df_std["_nn"] = df_std[data_cols].notna().sum(axis=1)
    df_std["_keyset_rank"] = df_std["KEYSET"].map({1: 0, 2: 1}).fillna(99)

    df_std = (
        df_std
        .sort_values(
            ["gvkey", "fyear", "_keyset_rank", "_nn"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(subset=["gvkey", "fyear"], keep="first")
        .drop(columns=["_nn", "_keyset_rank"])
        .reset_index(drop=True)
    )
    print(f"  [{label}] After dedup: {len(df_std):,} unique firm-years")
    return df_std


def build_compustat_annual(datadate_convention: str = "frozen",
                           out_path=None) -> None:
    """
    Merge all Compustat annual files and save compustat_annual_raw.parquet.

    Parameters
    ----------
    out_path : Path or None
        Output parquet path. None (default) = the frozen
        compustat_annual_raw.parquet. The v2 rebuild passes a raw_v2/
        path so the frozen extract is never overwritten.
    datadate_convention : {"frozen", "standard"}
        How the fiscal year-end date is reconstructed from (FYEAR, FYR);
        see ``compustat_datadate``. "frozen" (default) reproduces the
        original construction exactly; "standard" applies the correct
        Compustat June-rule convention (fix-ready, Implementation Status
        paragraph 17 -- adopt only together with a full rebuild).
    """
    print("\n" + "="*60)
    print("  COMPUSTAT — building compustat_annual_raw.parquet")
    print(f"  (datadate convention: {datadate_convention})")
    print("="*60)

    # ── 1. Period descriptor ─────────────────────────────────────────────
    period = _read_rds_cols(COMP / "perioddescriptorannual.rds", _PERIOD_NEEDED)

    # Rename identifiers
    period = period.rename(columns={
        "KYGVKEY": "gvkey",
        "FYYYY":   "fyear_raw",
        "FYEAR":   "fyear",
        "FYR":     "fyr",
        "fyra":    "fyra",
        "lpermno": "permno",
        "lpermco": "permco",
        "FDATE":   "fdate",
        "SICH":    "sich",
        "CURCD":   "curcd",
    })

    period["gvkey"] = period["gvkey"].astype(str).str.strip()

    # Use FYEAR (fiscal year label) or fyear_raw as fyear
    if "fyear" not in period.columns or period["fyear"].isna().all():
        period["fyear"] = period["fyear_raw"]
    period["fyear"] = pd.to_numeric(period["fyear"], errors="coerce").astype("Int64")
    period["fdate"] = pd.to_datetime(period["fdate"], errors="coerce")
    period["fdate_source"] = "actual"

    # Restrict to sample period (include one year before for lagged features)
    period = period[
        period["fyear"].between(SAMPLE_START_YEAR - 1, SAMPLE_END_YEAR)
    ].copy()

    # Construct fiscal year-end date (datadate) from fyear + fyr (end month).
    # See compustat_datadate: "frozen" reproduces the original construction,
    # "standard" applies the Compustat June rule (FYR 1-5 end in FYEAR + 1).
    fyr_col = period["fyr"] if "fyr" in period.columns \
        else pd.Series(12, index=period.index)
    period["datadate"] = compustat_datadate(
        period["fyear"], fyr_col, convention=datadate_convention
    )

    print(f"  Period descriptor after filter: {len(period):,} rows")

    # Deduplicate: keep one row per gvkey × fyear
    period = (
        period.sort_values("datadate")
              .drop_duplicates(subset=["gvkey", "fyear"], keep="last")
              .reset_index(drop=True)
    )

    # ── 2. Balance sheet ─────────────────────────────────────────────────
    bs = _read_rds_cols(COMP / "balancesheetindustrialannual.rds", _BS_NEEDED)
    bs = bs.rename(columns={"KYGVKEY": "gvkey", "FYYYY": "fyear_raw"})
    bs["gvkey"] = bs["gvkey"].astype(str).str.strip()
    bs["fyear_raw"] = pd.to_numeric(bs["fyear_raw"], errors="coerce").astype("Int64")
    bs.columns = [c if c in ["gvkey", "KEYSET", "fyear_raw", "fyra",
                              "lpermno", "lpermco", "LinkRangeTypeCd"]
                  else c.lower() for c in bs.columns]
    bs = bs.rename(columns={"fyear_raw": "fyear"})
    bs = _dedup_keyset(bs, "balance sheet")

    # ── 3. Income statement ──────────────────────────────────────────────
    is_ = _read_rds_cols(COMP / "incomestatementindustrialannual.rds", _IS_NEEDED)
    is_ = is_.rename(columns={"KYGVKEY": "gvkey", "FYYYY": "fyear_raw"})
    is_["gvkey"] = is_["gvkey"].astype(str).str.strip()
    is_["fyear_raw"] = pd.to_numeric(is_["fyear_raw"], errors="coerce").astype("Int64")
    is_.columns = [c if c in ["gvkey", "KEYSET", "fyear_raw", "fyra",
                               "lpermno", "lpermco", "LinkRangeTypeCd"]
                   else c.lower() for c in is_.columns]
    is_ = is_.rename(columns={"fyear_raw": "fyear"})
    is_ = _dedup_keyset(is_, "income statement")

    # ── 4. Cash flow ─────────────────────────────────────────────────────
    cf = _read_rds_cols(COMP / "cashflowannual.rds", _CF_NEEDED)
    cf = cf.rename(columns={"KYGVKEY": "gvkey", "FYYYY": "fyear_raw"})
    cf["gvkey"] = cf["gvkey"].astype(str).str.strip()
    cf["fyear_raw"] = pd.to_numeric(cf["fyear_raw"], errors="coerce").astype("Int64")
    cf.columns = [c if c in ["gvkey", "KEYSET", "fyear_raw", "fyra",
                              "lpermno", "lpermco", "LinkRangeTypeCd"]
                  else c.lower() for c in cf.columns]
    cf = cf.rename(columns={"fyear_raw": "fyear"})
    cf = _dedup_keyset(cf, "cash flow")

    # ── 5. Company description (CUSIP, current SIC as fallback) ──────────
    try:
        co = _read_full(COMP / "companydescription.rds")
        co.columns = [c.upper() for c in co.columns]
        co = co.rename(columns={"KYGVKEY": "gvkey"})
        co["gvkey"] = co["gvkey"].astype(str).str.strip()
        co_cols = ["gvkey"] + [c for c in ["CUSIP","CONM","CIK","STATE","LOC","SIC"]
                                if c in co.columns]
        co = co[co_cols].rename(columns=str.lower)
        if "cusip" in co.columns:
            co["cusip"] = co["cusip"].astype(str).str[:8].str.upper()
        if "sic" in co.columns:
            co["sic"] = pd.to_numeric(co["sic"], errors="coerce").astype("Int64")
    except Exception as e:
        print(f"  companydescription skipped: {e}")
        co = pd.DataFrame(columns=["gvkey", "cusip"])

    # ── 5b. Company history (time-consistent historical SIC) ─────────────
    # HSIC is the Compustat CIZ equivalent of funda.sich — historical SIC
    # with date-range validity. This is the time-consistent source the
    # frozen design specifies; companydescription.SIC above is retained as
    # a fallback for firm-years with no companyhistory interval match.
    try:
        ch = _read_full(COMP / "companyhistory.rds")
        ch.columns = [c.upper() for c in ch.columns]
        ch = ch.rename(columns={"KYGVKEY": "gvkey"})
        ch["gvkey"]     = ch["gvkey"].astype(str).str.strip()
        ch["HCHGDT"]    = pd.to_datetime(ch["HCHGDT"],    errors="coerce")
        ch["HCHGENDDT"] = pd.to_datetime(ch["HCHGENDDT"], errors="coerce")
        ch["HSIC"]      = pd.to_numeric(ch["HSIC"], errors="coerce").astype("Int64")
        ch = ch[["gvkey", "HCHGDT", "HCHGENDDT", "HSIC"]].dropna(subset=["HCHGDT"])
    except Exception as e:
        print(f"  companyhistory skipped: {e}")
        ch = pd.DataFrame(columns=["gvkey", "HCHGDT", "HCHGENDDT", "HSIC"])

    # ── 5c. Security history (time-consistent Compustat CUSIP) ──────────
    # The company-description table in the local CIZ delivery does not carry
    # CUSIP.  securityheaderhistory does: HSCUSIP is the historical security
    # identifier and HSCHGDT/HSCHGENDDT delimit its validity interval.  Use
    # the primary issue (HIID="01", then lowest issue sequence) so the CUSIP
    # comparison validates the same ordinary share linked through CCM.
    try:
        sh = _read_rds_cols(
            COMP / "securityheaderhistory.rds",
            ["KYGVKEY", "HSCHGDT", "HSCHGENDDT", "HSCUSIP",
             "HIID", "HIID_SEQ_NUM"],
        )
        sh = sh.rename(columns={"KYGVKEY": "gvkey"})
        sh["gvkey"] = sh["gvkey"].astype(str).str.strip()
        sh["HSCHGDT"] = pd.to_datetime(sh["HSCHGDT"], errors="coerce")
        sh["HSCHGENDDT"] = pd.to_datetime(sh["HSCHGENDDT"], errors="coerce")
        sh["HSCUSIP"] = (
            sh["HSCUSIP"].astype("string").str.strip().str.upper().str[:8]
        )
        sh["_issue_rank"] = np.where(
            sh.get("HIID", pd.Series(index=sh.index, dtype="string"))
              .astype("string").str.strip().eq("01"),
            0,
            1,
        )
        sh["_issue_seq"] = pd.to_numeric(
            sh.get("HIID_SEQ_NUM"), errors="coerce"
        ).fillna(9999)
        sh = sh.dropna(subset=["HSCHGDT", "HSCUSIP"])
    except Exception as e:
        print(f"  securityheaderhistory skipped: {e}")
        sh = pd.DataFrame(columns=[
            "gvkey", "HSCHGDT", "HSCHGENDDT", "HSCUSIP",
            "_issue_rank", "_issue_seq",
        ])

    # ── Merge all on gvkey × fyear ────────────────────────────────────────
    bs_cols = ["gvkey", "fyear"] + [
        c for c in ["at","lt","act","lct","che","wcap","ceq","re",
                    "dltt","dlc","ppent","rect","invt","seq","csho"]
        if c in bs.columns
    ]
    comp = period.merge(bs[bs_cols], on=["gvkey","fyear"], how="left")
    # Drop duplicated cols from second merge
    comp = comp.loc[:, ~comp.columns.duplicated()]

    is_cols = ["gvkey","fyear"] + [c for c in ["ni","ib","sale","oibdp","dp","txt","xint","revt"]
                                    if c in is_.columns]
    comp = comp.merge(is_[is_cols], on=["gvkey","fyear"], how="left")
    comp = comp.loc[:, ~comp.columns.duplicated()]

    cf_cols = ["gvkey","fyear"] + [c for c in ["oancf","ivncf","fincf","capx","dpc"]
                                    if c in cf.columns]
    comp = comp.merge(cf[cf_cols], on=["gvkey","fyear"], how="left")
    comp = comp.loc[:, ~comp.columns.duplicated()]

    comp = comp.merge(co, on="gvkey", how="left")
    comp = comp.loc[:, ~comp.columns.duplicated()]

    # ── Historical SIC via date-ranged join on companyhistory ─────────────
    # For each (gvkey, datadate), find the companyhistory row whose interval
    # [HCHGDT, HCHGENDDT] contains datadate. If multiple intervals overlap,
    # keep the most recent (latest HCHGDT). Where no interval matches, fall
    # back to companydescription.SIC (already merged via `co`).
    if not ch.empty and "datadate" in comp.columns:
        n_before = len(comp)
        comp_keys = comp[["gvkey", "fyear", "datadate"]].copy()
        joined = comp_keys.merge(ch, on="gvkey", how="left")
        match = (
            joined["datadate"].ge(joined["HCHGDT"])
            & (joined["HCHGENDDT"].isna() | joined["datadate"].le(joined["HCHGENDDT"]))
        )
        joined = joined[match]
        joined = (
            joined.sort_values(["gvkey", "fyear", "HCHGDT"])
                  .drop_duplicates(subset=["gvkey", "fyear"], keep="last")
                  [["gvkey", "fyear", "HSIC"]]
        )
        comp = comp.merge(joined, on=["gvkey", "fyear"], how="left")
        # Prefer HSIC (historical) over current SIC from companydescription
        if "sic" in comp.columns:
            comp["sic"] = comp["HSIC"].fillna(comp["sic"])
        else:
            comp["sic"] = comp["HSIC"]
        comp = comp.drop(columns=["HSIC"])
        assert len(comp) == n_before, "companyhistory merge changed row count"
        n_pop = comp["sic"].notna().sum()
        print(f"  Historical SIC populated: {n_pop:,}/{len(comp):,} "
              f"firm-years ({100*n_pop/len(comp):.1f}%)")

    # Historical Compustat CUSIP via the same interval-validity principle.
    # Retain a current company-description CUSIP only as a fallback when it
    # actually exists; the local delivery normally relies entirely on HSCUSIP.
    if not sh.empty and "datadate" in comp.columns:
        n_before = len(comp)
        comp_keys = comp[["gvkey", "fyear", "datadate"]].copy()
        joined = comp_keys.merge(sh, on="gvkey", how="left")
        match = (
            joined["datadate"].ge(joined["HSCHGDT"])
            & (
                joined["HSCHGENDDT"].isna()
                | joined["datadate"].le(joined["HSCHGENDDT"])
            )
        )
        joined = joined[match]
        joined = (
            joined.sort_values(
                ["gvkey", "fyear", "_issue_rank", "_issue_seq", "HSCHGDT"],
                ascending=[True, True, True, True, False],
            )
            .drop_duplicates(subset=["gvkey", "fyear"], keep="first")
            [["gvkey", "fyear", "HSCUSIP"]]
            .rename(columns={"HSCUSIP": "cusip_historical"})
        )
        comp = comp.merge(joined, on=["gvkey", "fyear"], how="left")
        if "cusip" in comp.columns:
            comp["cusip"] = comp["cusip_historical"].fillna(comp["cusip"])
        else:
            comp["cusip"] = comp["cusip_historical"]
        comp = comp.drop(columns=["cusip_historical"])
        assert len(comp) == n_before, "security-history merge changed row count"
        n_cusip = int(comp["cusip"].notna().sum())
        print(f"  Historical CUSIP populated: {n_cusip:,}/{len(comp):,} "
              f"firm-years ({100*n_cusip/len(comp):.1f}%)")

    # ── Derived: wcap if not directly available ───────────────────────────
    if "wcap" not in comp.columns or comp["wcap"].isna().all():
        if "act" in comp.columns and "lct" in comp.columns:
            comp["wcap"] = comp["act"] - comp["lct"]

    # ── Final safety dedup (catches any residual merge artefacts) ─────────
    n_before = len(comp)
    comp = (
        comp
        .sort_values(["gvkey", "fyear"])
        .drop_duplicates(subset=["gvkey", "fyear"], keep="first")
        .reset_index(drop=True)
    )
    n_after = len(comp)
    if n_before != n_after:
        print(f"  Final dedup removed {n_before - n_after:,} residual duplicates")

    # ── Sanity checks ─────────────────────────────────────────────────────
    dups = comp.duplicated(subset=["gvkey", "fyear"]).sum()
    assert dups == 0, f"Duplicates remain after dedup: {dups}"
    print(f"  Sanity check passed: 0 duplicate gvkey×fyear rows")
    print(f"  Unique firms      : {comp['gvkey'].nunique():,}")
    print(f"  Fiscal years      : {int(comp['fyear'].min())}–{int(comp['fyear'].max())}")
    print(f"  Merged Compustat panel: {comp.shape}")

    out = Path(out_path) if out_path is not None else COMP / "compustat_annual_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    comp.to_parquet(out, index=False)
    print(f"  Saved → {out}")


# ===========================================================================
# COMPUSTAT — 10-K filing dates
# ===========================================================================
# F_{i,t} (the filing date that anchors the distress window) is sourced from
# the dedicated filingdates.rds file — real SEC submission dates, SRCTYPE
# '10K', covering 1966–2025. The legacy perioddescriptorannual.FDATE source
# is retained behind a config switch for reproducibility only: it is empty
# pre-2006 and, where present, disagrees with the true 10-K date ~75% of the
# time (it is a Compustat record/update date, not the filing date).
# Controlled by config.FILING_DATE_SOURCE / FILING_DATE_UNMATCHED_POLICY.

# SRCTYPE → preference rank when a firm-period has several annual-report
# records. Lower = preferred. Only consulted under the "broaden" policy;
# under the default the frame is already 10-K-only.
_FILING_SRCTYPE_RANK = {"10K": 0, "20F": 1, "AR": 2}


def _build_filing_dates_from_dedicated(policy: str) -> pd.DataFrame:
    """
    Build the filing-dates frame from the dedicated filingdates.rds file.

    Uses the real SEC submission date (FILEDATE) of the original 10-K
    (SRCTYPE='10K'). Under policy='broaden', non-10-K filers additionally
    fall back to 20-F (foreign annual report) and AR (annual report) dates,
    tagged with their own source.

    Returns columns: gvkey, datadate, fdate, fdate_source.
    """
    from pandas.tseries.offsets import MonthEnd

    fd = _read_full(COMP / "filingdates.rds")

    keep_types = ["10K", "20F", "AR"] if policy == "broaden" else ["10K"]
    fd = fd[fd["SRCTYPE"].isin(keep_types)].copy()

    fd["gvkey"] = fd["KYGVKEY"].astype(str).str.strip()
    # FDATADATE is the fiscal period-end; normalise to month-end so it aligns
    # with the panel's constructed datadate (_month_end(fyear, fyr)).
    period = pd.to_datetime(fd["FDATADATE"], errors="coerce")
    fd["datadate"] = period + MonthEnd(0)
    fd["fdate"] = pd.to_datetime(fd["FILEDATE"], errors="coerce")
    fd["_rank"] = fd["SRCTYPE"].map(_FILING_SRCTYPE_RANK).fillna(99).astype(int)
    fd["fdate_source"] = "actual_" + fd["SRCTYPE"].astype(str).str.lower()

    fd = fd.dropna(subset=["gvkey", "datadate", "fdate"])

    # One row per (gvkey, datadate): prefer 10K > 20F > AR, then the earliest
    # FILEDATE (original filing, not a later amendment / restatement).
    fd = (
        fd.sort_values(["gvkey", "datadate", "_rank", "fdate"])
          .drop_duplicates(subset=["gvkey", "datadate"], keep="first")
          [["gvkey", "datadate", "fdate", "fdate_source"]]
          .reset_index(drop=True)
    )

    print(f"  Source: filingdates.rds ({', '.join(keep_types)})")
    print(f"  Filing dates by type: {fd['fdate_source'].value_counts().to_dict()}")
    return fd


def _build_filing_dates_from_perioddescriptor() -> pd.DataFrame:
    """
    Legacy: derive filing dates from perioddescriptorannual.FDATE.

    DEPRECATED — FDATE is empty for FY1990–2005 and is not the SEC filing
    date where present. Retained only for reproducibility / QC comparison
    (see src/analysis/filing_date_qc.py). Selected via
    config.FILING_DATE_SOURCE = "perioddescriptor".
    """
    needed = ["KYGVKEY", "KEYSET", "FYYYY", "FYEAR", "FYR", "FDATE"]
    fd = _read_rds_cols(COMP / "perioddescriptorannual.rds", needed)

    fd = fd.rename(columns={
        "KYGVKEY": "gvkey",
        "FYYYY":   "fyear_raw",
        "FYEAR":   "fyear",
        "FYR":     "fyr",
        "FDATE":   "fdate",
    })
    fd["gvkey"] = fd["gvkey"].astype(str).str.strip()

    if "fyear" not in fd.columns or fd["fyear"].isna().all():
        fd["fyear"] = fd["fyear_raw"]
    fd["fyear"] = pd.to_numeric(fd["fyear"], errors="coerce").astype("Int64")
    fd["fdate"] = pd.to_datetime(fd["fdate"], errors="coerce")
    fd["fdate_source"] = fd["fdate"].notna().map({True: "actual", False: "missing"})

    # Construct datadate
    fyr = pd.to_numeric(fd.get("fyr", 12), errors="coerce").fillna(12).astype(int)
    fd["datadate"] = _month_end(fd["fyear"], fyr)

    fd = fd.dropna(subset=["gvkey", "fyear"])
    fd = (
        fd.sort_values("datadate")
          .drop_duplicates(subset=["gvkey", "datadate"], keep="last")
    [["gvkey", "datadate", "fdate", "fdate_source"]]
    )
    return fd


def build_filing_dates(out_path: "Path | None" = None) -> None:
    """
    Build compustat_filing_dates.parquet from the configured source.

    Dispatches on config.FILING_DATE_SOURCE:
      "filingdates_10k"  → dedicated filingdates.rds (real 10-K dates)
      "perioddescriptor" → legacy perioddescriptorannual.FDATE (deprecated)

    Parameters
    ----------
    out_path : Path or None
        Output parquet path. None → canonical
        data/raw/compustat/compustat_filing_dates.parquet. A non-default
        path lets callers verify the build without overwriting production.
    """
    print("\n" + "="*60)
    print("  COMPUSTAT — building compustat_filing_dates.parquet")
    print("="*60)

    if FILING_DATE_SOURCE == "filingdates_10k":
        fd = _build_filing_dates_from_dedicated(FILING_DATE_UNMATCHED_POLICY)
    elif FILING_DATE_SOURCE == "perioddescriptor":
        fd = _build_filing_dates_from_perioddescriptor()
    else:
        raise ValueError(
            f"Unknown FILING_DATE_SOURCE: {FILING_DATE_SOURCE!r} "
            "(expected 'filingdates_10k' or 'perioddescriptor')"
        )

    out = out_path if out_path is not None else (COMP / "compustat_filing_dates.parquet")
    fd.to_parquet(out, index=False)
    print(f"  Saved → {out}  ({len(fd):,} rows)")


# ===========================================================================
# CRSP — monthly stock data  (StkMthSecurityData)
# ===========================================================================

# Candidate column names for the new CRSP CIZ v2 format
# (actual names confirmed at runtime by _resolve)
_MSF_PERMNO   = ["PERMNO",    "permno",    "Permno",    "HdrPermno"]
_MSF_DATE     = ["MthCalDt",  "mthcaldt",  "caldt",     "date",      "MthDt",
                  "CalDt",    "mthdt"]
_MSF_PRC      = ["MthPrc",    "mthprc",    "prc",       "Prc",       "MthEndPrc"]
_MSF_RET      = ["MthRet",    "mthret",    "ret",       "MthTotRet", "TotRet"]
_MSF_RETX     = ["MthRetx",   "mthretx",   "retx"]
# MthCap = market cap in $thousands (CIZ); rescaled to $M after load
_MSF_ME       = ["MthCap",    "me",        "MthMktCap", "mktcap",    "MthPrevCap"]
_MSF_VOL      = ["MthVol",    "mthvol",    "vol",       "MthTradedVol","TradedVol"]
_MSF_PERMCO   = ["permco",    "Permco",    "PERMCO"]


def build_crsp_monthly() -> None:
    """
    Build crsp_monthly_raw.parquet from StkMthSecurityData.rds.
    """
    print("\n" + "="*60)
    print("  CRSP — building crsp_monthly_raw.parquet")
    print("="*60)

    msf = _read_full(CRSP / "StkMthSecurityData.rds")
    print(f"  All cols: {list(msf.columns)}")

    # Resolve column names
    def _col(candidates):
        return _pick(msf, candidates)

    renames = {}
    for target, cands in [
        ("permno",  _MSF_PERMNO),
        ("permco",  _MSF_PERMCO),
        ("date",    _MSF_DATE),
        ("prc",     _MSF_PRC),
        ("ret",     _MSF_RET),
        ("retx",    _MSF_RETX),
        ("me",      _MSF_ME),      # MthCap → me ($M after /1000 rescale)
        ("vol",     _MSF_VOL),
    ]:
        found = _col(cands)
        if found and found != target:
            renames[found] = target

    msf = msf.rename(columns=renames)

    for col, dtype in [("permno", "Int64"), ("permco", "Int64")]:
        if col in msf.columns:
            msf[col] = pd.to_numeric(msf[col], errors="coerce").astype(dtype)

    if "date" in msf.columns:
        msf["date"] = pd.to_datetime(msf["date"], errors="coerce")
    for col in ["prc", "ret", "retx", "me", "vol"]:
        if col in msf.columns:
            msf[col] = pd.to_numeric(msf[col], errors="coerce")

    # CRSP CIZ MthCap is in $thousands. Convert to $millions for consistency
    # with Compustat fundamentals (ceq, at, etc.) before downstream features.
    if "me" in msf.columns:
        msf["me"] = msf["me"] / 1000.0

    keep = [c for c in ["permno","permco","date","prc","ret","retx","me","vol"]
            if c in msf.columns]
    msf = msf[keep]
    print(f"  CRSP monthly: {msf.shape}   cols: {list(msf.columns)}")

    CRSP.mkdir(parents=True, exist_ok=True)
    out = CRSP / "crsp_monthly_raw.parquet"
    msf.to_parquet(out, index=False)
    print(f"  Saved → {out}")


# ===========================================================================
# CRSP — delisting file  (StkDelists)
# ===========================================================================

_DL_PERMNO = ["permno",    "Permno",    "PERMNO"]
_DL_DATE   = ["DelistingDt","DlstDt",   "dlstdt",    "delistdt",   "DlstCalDt",
               "DlstEvtDt", "EventDt"]
_DL_CODE   = ["DlstCd",    "dlstcd",    "delistcd",  "delist_code","DlstTypCd",
               "DlstEvtCd", "EventCd",   "DlstType"]
_DL_AMT    = ["DelDivAmt", "DlAmt",     "dlamt",     "delistamt",  "DlstAmt"]
_DL_RET    = ["DelRet",    "DlRetx",    "dlretx",    "delistret",  "DlstRetx"]


def _to_int(series) -> pd.Series:
    """Convert series to nullable Int64 safely (handles numpy and pandas)."""
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.array(numeric, dtype=pd.Int64Dtype())


def synthesize_dlstcd(dl: pd.DataFrame, mapping: str = "frozen") -> pd.Series:
    """
    Synthesize a numeric legacy-style DLSTCD from CIZ string delisting codes.

    The CRSP CIZ v2 delisting file carries no numeric DLSTCD; the delisting
    action and reason are string columns (DelActionType / DelReasonType).
    The frozen design defines the primary distress label as "any DLSTCD
    400--499", so a synthetic numeric code must be constructed.

    mapping="frozen" (default) reproduces the original construction exactly:

        default                       -> 500  (merger / other exit)
        DelActionType == 'GDR'        -> 450  (dropped by exchange)
        DelReasonType == 'BKPY'       -> 572  (bankruptcy; OVERWRITES 450)

    This mapping has two defects (Implementation Status paragraph 17):
    (a) bankruptcies end up at 572, OUTSIDE the primary 400--499 range, so
        the primary label excludes every bankruptcy delisting, and RC1
        (bankruptcy-only, codes 572/574/584) is DISJOINT from the primary
        label instead of the nested subset the frozen design describes; and
    (b) liquidations (DelActionType == 'GLI') fall through to the 500
        default and are excluded, although the design's own justification
        names liquidation as the core of the 400--499 range.

    mapping="corrected" repairs both defects while leaving every other
    assignment untouched:

        default                       -> 500  (merger / other exit)
        DelActionType == 'GLI'        -> 400  (liquidation)
        DelActionType == 'GDR'        -> 450  (dropped by exchange)
        DelReasonType == 'BKPY'       -> 470  (bankruptcy; OVERWRITES 450)

    All three distress codes now lie inside 400--499, so the primary label
    captures liquidations, exchange drops, and bankruptcies, and the
    bankruptcy subset (470, config.DISTRESS_CODES_RC1_CORRECTED) is nested
    in the primary label as the design intends. The numeric values are
    synthetic under both mappings (the CIZ source has no numeric codes);
    what changes is only which event classes the 400--499 range contains.

    Parameters
    ----------
    dl : pd.DataFrame
        Delisting frame with DelActionType / DelReasonType string columns.
    mapping : {"frozen", "corrected"}

    Returns
    -------
    pd.Series
        Synthetic numeric dlstcd, indexed like ``dl``.
    """
    if mapping not in ("frozen", "corrected", "academic"):
        raise ValueError(
            "mapping must be 'frozen', 'corrected', or 'academic', "
            f"got {mapping!r}"
        )

    if mapping == "academic":
        # Do not manufacture a generic numeric code from DelActionType alone.
        # CRSP CIZ unpacks legacy DLSTCD into four fields.  The dedicated
        # reconstruction keeps unknown/non-relevant combinations missing and
        # maps GDR by its reason (e.g. BKPY -> 574, FING -> 584).
        from src.data.distress_definition import reconstruct_legacy_dlstcd
        return reconstruct_legacy_dlstcd(dl)

    dlstcd = pd.Series(500, index=dl.index)  # default: merger or other exit

    if mapping == "corrected" and "DelActionType" in dl.columns:
        is_gli = dl["DelActionType"].astype(str).str.strip() == "GLI"
        dlstcd[is_gli] = 400  # liquidation

    if "DelActionType" in dl.columns:
        is_gdr = dl["DelActionType"].astype(str).str.strip() == "GDR"
        dlstcd[is_gdr] = 450  # dropped by exchange

    if "DelReasonType" in dl.columns:
        is_bkpy = dl["DelReasonType"].astype(str).str.strip() == "BKPY"
        dlstcd[is_bkpy] = 572 if mapping == "frozen" else 470  # bankruptcy

    return dlstcd


def read_stkdelists(mapping: str = "frozen") -> pd.DataFrame:
    """
    Read StkDelists.rds and return the cleaned delisting frame in memory.

    Same transformation as ``build_crsp_delisting`` (which persists the
    frozen-mapping result to crsp_delisting_raw.parquet), exposed so the
    correction-sensitivity harness can obtain the corrected-mapping frame
    without touching the frozen parquet.

    Parameters
    ----------
    mapping : {"frozen", "corrected"}
        Passed to ``synthesize_dlstcd`` when the file carries CIZ string
        codes instead of a numeric dlstcd.

    Returns
    -------
    pd.DataFrame
        Columns: permno, dlstdt, dlstcd (+ dlamt/dlretx where present) and,
        for CIZ files, the DelActionType/DelReasonType strings for audit.
    """
    dl = _read_full(CRSP / "StkDelists.rds")

    renames = {}
    for target, cands in [
        ("permno", _DL_PERMNO),
        ("dlstdt", _DL_DATE),
        ("dlstcd", _DL_CODE),
        ("dlamt",  _DL_AMT),
        ("dlretx", _DL_RET),
    ]:
        found = _pick(dl, cands)
        if found and found != target:
            renames[found] = target
    dl = dl.rename(columns=renames)

    has_ciz_codes = "dlstcd" not in dl.columns
    if has_ciz_codes:
        dl["dlstcd"] = synthesize_dlstcd(dl, mapping=mapping)
        counts = dl["dlstcd"].value_counts().sort_index()
        print(f"  dlstcd built from CIZ codes (mapping={mapping}): "
              + "  ".join(f"{c}={n:,}" for c, n in counts.items()))

    if "permno" in dl.columns:
        dl["permno"] = _to_int(dl["permno"])
    if "dlstdt" in dl.columns:
        dl["dlstdt"] = pd.to_datetime(dl["dlstdt"], errors="coerce")
    if "dlstcd" in dl.columns:
        dl["dlstcd"] = _to_int(dl["dlstcd"])

    if has_ciz_codes:
        from src.data.distress_definition import classify_delisting_events
        dl = classify_delisting_events(dl)

    keep = [c for c in ["permno", "dlstdt", "dlstcd", "dlamt", "dlretx"]
            if c in dl.columns]
    if has_ciz_codes:
        keep += [c for c in [
            "DelActionType", "DelReasonType", "DelStatusType",
            "DelPaymentType", "dlstcd_reconstructed",
            "is_distress_performance", "is_distress_performance_core",
            "is_distress_narrow_financial",
            "is_bankruptcy_proxy",
            "is_broad_for_cause",
        ] if c in dl.columns and c not in keep]
    return dl[keep]


def build_crsp_delisting(mapping: str = "frozen", out_path=None) -> None:
    """
    Build crsp_delisting_raw.parquet from StkDelists.rds.

    out_path=None writes the frozen crsp_delisting_raw.parquet; the v2
    rebuild passes a raw_v2/ path (with mapping="corrected") so the
    frozen extract is never overwritten.

    Handles both old CRSP format (numeric dlstcd) and new CIZ format
    (string DelActionType / DelReasonType). The CIZ-to-numeric mapping is
    documented in ``synthesize_dlstcd``; the default mapping="frozen"
    reproduces the original file (bankruptcies at 572, liquidations at 500),
    mapping="corrected" places liquidations (400) and bankruptcies (470)
    inside the primary 400--499 range (fix-ready, Implementation Status
    paragraph 17 -- adopt only together with a full rebuild).
    """
    print("\n" + "="*60)
    print("  CRSP -- building crsp_delisting_raw.parquet")
    print(f"  (dlstcd mapping: {mapping})")
    print("="*60)

    dl = read_stkdelists(mapping=mapping)
    # Frozen/corrected modes retain their historical narrow schema.  The
    # academic mode persists the raw CIZ fields and direct label flags so the
    # outcome can be independently audited and reproduced.
    keep = [c for c in ["permno", "dlstdt", "dlstcd", "dlamt", "dlretx"]
            if c in dl.columns]
    if mapping == "academic":
        keep += [c for c in [
            "DelActionType", "DelReasonType", "DelStatusType",
            "DelPaymentType", "dlstcd_reconstructed",
            "is_distress_performance", "is_distress_performance_core",
            "is_distress_narrow_financial",
            "is_bankruptcy_proxy",
            "is_broad_for_cause",
        ] if c in dl.columns and c not in keep]
    dl = dl[keep]
    print(f"  Delisting: {dl.shape}   cols: {list(dl.columns)}")

    out = Path(out_path) if out_path is not None else CRSP / "crsp_delisting_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    dl.to_parquet(out, index=False)
    print(f"  Saved -- {out}")


# ===========================================================================
# CRSP — security names  (StkSecurityInfoHdr + StkSecurityInfoHist)
# ===========================================================================

_SN_PERMNO   = ["PERMNO",   "permno",   "Permno",   "HdrPermno"]
# Confirmed CIZ cols: CUSIP/CUSIP9 are point-in-time values on each history
# row; HdrCUSIP/HdrCUSIP9 are current header identifiers and must only be a
# fallback. Preferring HdrCUSIP back-casts successor identifiers across
# earlier intervals and creates false CCM/CUSIP mismatches.
_SN_NCUSIP   = ["CUSIP", "CUSIP9", "ncusip", "Ncusip", "NCUSIP",
                 "Cusip8", "cusip8", "HdrCUSIP", "HdrCUSIP9"]
# CIZ has ShareClass, IssuerType, SecurityType — no direct SHRCD equivalent
_SN_SHRCD    = ["shrcd",    "Shrcd",    "SHRCD",    "ShareCd",  "ShareClass",
                 "share_code","ShareTypeCd"]
# CIZ has PrimaryExch — maps to exchcd
_SN_EXCHCD   = ["PrimaryExch","exchcd", "Exchcd",   "EXCHCD",   "ExchCd",
                 "exchange_code"]
# CIZ date columns confirmed: SecInfoStartDt, SecurityBegDt
_SN_NAMEDT   = ["SecInfoStartDt","SecurityBegDt","namedt","NameDt","NameBegDt",
                 "SecurityBeginDt","begdt","st_date"]
# CIZ date columns confirmed: SecInfoEndDt, SecurityEndDt
_SN_NAMEENDT = ["SecInfoEndDt","SecurityEndDt","nameendt","NameEndDt",
                 "enddt","end_date"]
_SN_COMNAM   = ["SecurityNm","comnam","ComNam","COMNAM","CompanyName","compname"]
_SN_TICKER   = ["ticker",   "Ticker",   "TICKER"]


def build_crsp_security_names(out_path=None) -> None:
    """
    Build crsp_security_names.parquet from StkSecurityInfoHdr/Hist.

    out_path=None writes the frozen crsp_security_names.parquet; the v2
    rebuild passes a raw_v2/ path. Since the F5 fix this output also
    carries the CIZ letter fields needed by the universe filter.
    """
    print("\n" + "="*60)
    print("  CRSP — building crsp_security_names.parquet")
    print("="*60)

    hdr  = _read_full(CRSP / "StkSecurityInfoHdr.rds")
    hist = _read_full(CRSP / "StkSecurityInfoHist.rds")

    print(f"  HDR  cols: {list(hdr.columns[:20])} ...")
    print(f"  HIST cols: {list(hist.columns[:20])} ...")

    # Prefer hist (has date ranges); fall back to hdr
    has_dates_in_hist = (
        _pick(hist, _SN_NAMEDT) is not None or
        _pick(hist, _SN_NAMEENDT) is not None
    )
    base = hist if has_dates_in_hist else hdr

    renames = {}
    for target, cands in [
        ("permno",   _SN_PERMNO),
        ("ncusip",   _SN_NCUSIP),
        ("shrcd",    _SN_SHRCD),
        ("exchcd",   _SN_EXCHCD),
        ("namedt",   _SN_NAMEDT),
        ("nameendt", _SN_NAMEENDT),
        ("comnam",   _SN_COMNAM),
        ("ticker",   _SN_TICKER),
    ]:
        found = _pick(base, cands)
        if found and found != target:
            renames[found] = target
    base = base.rename(columns=renames)

    if "namedt" not in base.columns:
        base["namedt"]   = pd.Timestamp("1925-01-01")
    if "nameendt" not in base.columns:
        base["nameendt"] = pd.Timestamp("2099-12-31")

    base["permno"]   = pd.to_numeric(base.get("permno"),   errors="coerce").astype("Int64")
    base["namedt"]   = pd.to_datetime(base.get("namedt"),   errors="coerce")
    base["nameendt"] = pd.to_datetime(base.get("nameendt"), errors="coerce")
    base["nameendt"] = base["nameendt"].fillna(pd.Timestamp("2099-12-31"))

    # F5 fix: the CIZ files encode share type and
    # incorporation in LETTER fields (ShareType, USIncFlg, SecurityType,
    # SecuritySubType, IssuerType, ShareClass). The numeric coercion below
    # nulls any letter code that was mapped into shrcd/exchcd — which is
    # why the frozen extract has shrcd 99.99% null and the SHRCD 10/11
    # universe filter never bound. Preserve the raw letter fields as
    # string columns so a corrected rebuild can apply the universe filter
    # (src/data/universe.py). Additive: existing consumers are unaffected.
    _CIZ_LETTER_FIELDS = {
        "ShareType": "sharetype", "USIncFlg": "usincflg",
        "SecurityType": "securitytype", "SecuritySubType": "securitysubtype",
        "IssuerType": "issuertype", "ShareClass": "shareclass",
        "PrimaryExch": "primaryexch",
    }
    for src_col, dst_col in _CIZ_LETTER_FIELDS.items():
        if src_col in base.columns and dst_col not in base.columns:
            base[dst_col] = base[src_col].astype("string")
    # If a letter field was renamed into shrcd/exchcd, stash the raw string
    # before it is destroyed by to_numeric.
    if "shrcd" in base.columns and "shrcd_src" not in base.columns:
        base["shrcd_src"] = base["shrcd"].astype("string")
    if "exchcd" in base.columns and "exchcd_src" not in base.columns:
        base["exchcd_src"] = base["exchcd"].astype("string")

    if "shrcd" in base.columns:
        base["shrcd"] = pd.to_numeric(base["shrcd"], errors="coerce").astype("Int64")
    if "exchcd" in base.columns:
        base["exchcd"] = pd.to_numeric(base["exchcd"], errors="coerce").astype("Int64")

    keep = [c for c in ["permno","ncusip","namedt","nameendt","shrcd",
                          "exchcd","comnam","ticker",
                          "sharetype","usincflg","securitytype",
                          "securitysubtype","issuertype","shareclass",
                          "primaryexch","shrcd_src","exchcd_src"] if c in base.columns]
    secnames = base[keep].copy()
    print(f"  Security names: {secnames.shape}")

    out = Path(out_path) if out_path is not None else CRSP / "crsp_security_names.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    secnames.to_parquet(out, index=False)
    print(f"  Saved → {out}")


# ===========================================================================
# CCM link table — built from linkhistory.rds (Compustat CIZ)
# ===========================================================================

def build_ccm_linktable() -> None:
    """
    Build ccm_linktable_raw.parquet from the local linkhistory.rds file.

    linkhistory.rds is the Compustat-CIZ-side CCM link history with the
    modern WRDS link-type vocabulary (LC, LU, LS, LX, LN, LD, NP, NR, NU)
    and the linkprim codes P/C/N/J. We apply the Pre-Specified Empirical Design §6.2
    link-quality filters (linktype IN {LC,LU} AND linkprim IN {P,C}) at
    extraction time so merge_crsp_compustat.py receives a clean,
    design-faithful CCM file.

    This replaces an earlier builder that reconstructed CCM from
    perioddescriptorannual.lpermno (one row per gvkey-fyear, all rows
    labelled LC/P), which produced a non-standard linktable with
    pathological row blow-up at merge time.

    Saves → data/raw/crsp/ccm_linktable_raw.parquet
    """
    print("\n" + "="*60)
    print("  CCM — building ccm_linktable_raw.parquet from linkhistory.rds")
    print("="*60)

    lh = _read_full(COMP / "linkhistory.rds")
    lh.columns = [c.upper() for c in lh.columns]
    lh = lh.rename(columns={
        "KYGVKEY":   "gvkey",
        "LPERMNO":   "permno",
        "LPERMCO":   "permco",
        "LINKDT":    "linkdt",
        "LINKENDDT": "linkenddt",
        "LINKTYPE":  "linktype",
        "LINKPRIM":  "linkprim",
        "LIID":      "liid",
    })
    lh["gvkey"]     = lh["gvkey"].astype(str).str.strip()
    lh["permno"]    = pd.to_numeric(lh["permno"], errors="coerce").astype("Int64")
    lh["permco"]    = pd.to_numeric(lh["permco"], errors="coerce").astype("Int64")
    lh["linkdt"]    = pd.to_datetime(lh["linkdt"],    errors="coerce")
    lh["linkenddt"] = pd.to_datetime(lh["linkenddt"], errors="coerce")

    print(f"  Raw linkhistory rows   : {len(lh):,}")

    # Pre-Specified Empirical Design §6.2 link-quality filters
    keep_mask = lh["linktype"].isin(["LC", "LU"]) & lh["linkprim"].isin(["P", "C"])
    lh = lh[keep_mask].copy()

    # Drop no-link rows (defensive; LU/LC normally have a valid permno)
    lh = lh[lh["permno"].notna() & (lh["permno"] > 0)].copy()

    # Active links: linkenddt NaT → sentinel 2099-12-31
    sentinel = pd.Timestamp("2099-12-31")
    lh["linkenddt"] = lh["linkenddt"].fillna(sentinel)

    keep_cols = ["gvkey", "permno", "permco", "linktype", "linkprim",
                 "linkdt", "linkenddt"]
    lh = lh[keep_cols].sort_values(["gvkey", "linkdt"]).reset_index(drop=True)

    print(f"  Records after filter   : {len(lh):,}")
    print(f"  Unique GVKEYs          : {lh['gvkey'].nunique():,}")
    print(f"  Unique PERMNOs         : {lh['permno'].nunique():,}")
    print(f"  Linktype × linkprim    :")
    print(lh.groupby(["linktype", "linkprim"]).size().to_string())

    CRSP.mkdir(parents=True, exist_ok=True)
    out = CRSP / "ccm_linktable_raw.parquet"
    lh.to_parquet(out, index=False)
    print(f"  Saved → {out}")


# ===========================================================================
# Master runner
# ===========================================================================

def main() -> None:
    print("\n" + "="*60)
    print("  LOCAL RDS → PARQUET LOADER")
    print("="*60)

    steps = [
        ("Compustat annual",        build_compustat_annual),
        ("Compustat filing dates",  build_filing_dates),
        ("CRSP monthly",            build_crsp_monthly),
        ("CRSP delisting",          build_crsp_delisting),
        ("CRSP security names",     build_crsp_security_names),
        ("CCM link table",          build_ccm_linktable),
    ]
    errors = []
    for name, fn in steps:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"\n  [SKIP — MISSING] {name}: {e}")
            errors.append(name)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {name}: {e}")
            traceback.print_exc()
            errors.append(name)

    print("\n" + "="*60)
    if errors:
        print(f"  COMPLETED WITH {len(errors)} ERROR(S):")
        for e in errors:
            print(f"    × {e}")
        print("\n  Fix the errors above, then re-run this step.")
    else:
        print("  ALL 6 PARQUET FILES PRODUCED")
        print("\nNext step:")
        print("  python -m src.data.merge_crsp_compustat")
    print("="*60)


if __name__ == "__main__":
    main()
