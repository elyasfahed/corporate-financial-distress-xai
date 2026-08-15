"""
Legacy WRDS extraction helper.
==============================
The final thesis does not use this module for filing dates. It uses the
dedicated local `filingdates.rds` source restricted to `SRCTYPE='10K'` and
drops unmatched firm-years. This older helper is retained for provenance and
must not be used to regenerate the frozen final artifacts.

Historically this module pulled two files:

1. comp.funda — Compustat Annual Fundamentals (accounting variables)
   Filters: indfmt=INDL, datafmt=STD, popsrc=D, consol=C
   (standard filters for consolidated industrial-format US firms)

2. comp.fundq (Q4) — Actual 10-K filing dates (SEC submission dates)
   These are the cornerstone of the leakage-free timing convention.
   A predictor computed from FY-t data is only observable after F_{i,t}.
   Using fiscal year-end instead of filing date introduces look-ahead bias.

Blueprint v4 reference: §4.2, §5.2

Saves
-----
    data/raw/compustat/compustat_annual_raw.parquet
    data/raw/compustat/compustat_filing_dates.parquet

Run
---
    python -m src.data.extract_compustat
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    DATA_RAW_COMPUSTAT,
    SAMPLE_START_YEAR,
    SAMPLE_END_YEAR,
    FALLBACK_FILING_LAG_DAYS,
)
from src.data.wrds_connection import get_connection

# ---------------------------------------------------------------------------
# Variable lists
# ---------------------------------------------------------------------------

# All Compustat items required to construct the 11 accounting predictors
# plus auxiliary items needed for the merge and for variable construction.
# Reference: Campbell, Hilscher & Szilagyi (2008) Appendix A.
COMPUSTAT_VARS: list[str] = [
    # ── Identifiers ──────────────────────────────────────────────────────
    "gvkey",      # Compustat permanent firm identifier
    "datadate",   # Fiscal year-end date
    "fyear",      # Fiscal year (integer)
    "sich",       # Historical SIC code  ← preferred over sic (more accurate)
    "cik",        # SEC Central Index Key (for EDGAR cross-reference)
    "cusip",      # CUSIP (first 8 chars used in CCM cross-validation merge)
    "conm",       # Company name

    # ── Income statement ─────────────────────────────────────────────────
    "sale",       # Net sales / revenues                       ← denominator checks
    "revt",       # Total revenues
    "ib",         # Income before extraordinary items          ← NITA numerator
    "ni",         # Net income (after extraordinaries)         ← CHIN, INTWO
    "oiadp",      # Operating income after D&A
    "oibdp",      # Operating income before D&A (≈ EBITDA)
    "xint",       # Interest expense
    "dp",         # Depreciation & amortisation
    "txt",        # Total income taxes
    "pi",         # Pretax income

    # ── Balance sheet — assets ───────────────────────────────────────────
    "at",         # Total assets         ← NITA/TLTA/WCTA/CASHTA/OCF_TA denominator; LNTA
    "act",        # Current assets       ← WCTA (act - lct), CLCA numerator check
    "che",        # Cash & short-term investments              ← CASHTA numerator
    "rect",       # Total receivables
    "invt",       # Inventories
    "ppent",      # Net PP&E (tangible assets)

    # ── Balance sheet — liabilities & equity ────────────────────────────
    "lt",         # Total liabilities    ← TLTA numerator; OENEG (lt > at)
    "lct",        # Current liabilities  ← WCTA denominator check; CLCA denominator
    "dlc",        # Short-term debt (debt in current liabilities)
    "dltt",       # Long-term debt
    "ceq",        # Common equity        ← MB book-value denominator
    "seq",        # Total stockholders' equity
    "re",         # Retained earnings

    # ── Cash flow ────────────────────────────────────────────────────────
    "oancf",      # Operating cash flow  ← OCF_TA numerator

    # ── Shares (auxiliary — market data from CRSP is primary) ────────────
    "csho",       # Common shares outstanding
    "prcc_f",     # Price at fiscal year-end (Compustat — for sanity checks only)
    "ajex",       # Cumulative adjustment factor (shares)
]

# Standard Compustat filter string (industrial, consolidated, US domestic)
_COMPUSTAT_WHERE = """
    indfmt  = 'INDL'
    AND datafmt = 'STD'
    AND popsrc  = 'D'
    AND consol  = 'C'
    AND fyear BETWEEN {start} AND {end}
""".format(start=SAMPLE_START_YEAR, end=SAMPLE_END_YEAR)


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_fundamentals(db) -> pd.DataFrame:
    """
    Pull Compustat Fundamentals Annual from WRDS.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.

    Returns
    -------
    pd.DataFrame
        Raw annual fundamentals panel. Not yet filtered for SIC exclusions
        or missing values — that happens in the merge step.
    """
    # TODO: Execute query and return DataFrame
    cols = ", ".join(COMPUSTAT_VARS)
    query = f"""
        SELECT {cols}
        FROM comp.funda
        WHERE {_COMPUSTAT_WHERE}
        ORDER BY gvkey, datadate
    """
    print("Querying Compustat Fundamentals Annual ...")
    df = db.raw_sql(query, date_cols=["datadate"])
    print(f"  Rows returned : {len(df):,}")
    print(f"  Unique GVKEYs : {df['gvkey'].nunique():,}")
    print(f"  Fiscal years  : {int(df['fyear'].min())}–{int(df['fyear'].max())}")
    return df


def extract_filing_dates(db) -> pd.DataFrame:
    """
    DEPRECATED: pull filing-date fields from Compustat fundq Q4.

    The final thesis uses dedicated `filingdates.rds` 10-K submissions instead.
    Do not use this helper to regenerate the frozen final panel.

    The filing date F_{i,t} is used to anchor the distress label window
    [F_{i,t}, F_{i,t} + 365]. Using fiscal year-end instead of the actual
    filing date introduces look-ahead bias (Blueprint v4 §5.2).

    Where fdate is missing (typically pre-1995 observations), a fall-back
    lag of FALLBACK_FILING_LAG_DAYS days from fiscal year-end is applied.
    This deviation is documented transparently in the thesis (Chapter 3).

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gvkey, datadate, fdate, fdate_source.
        fdate_source is 'actual' or 'fallback_Xd_lag'.
    """
    # TODO: Execute query; handle missing dates; return DataFrame
    query = f"""
        SELECT gvkey, datadate, fdate, rdq
        FROM comp.fundq
        WHERE fqtr = 4
          AND fyearq BETWEEN {SAMPLE_START_YEAR} AND {SAMPLE_END_YEAR}
          AND indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
        ORDER BY gvkey, datadate
    """
    print("Querying Compustat 10-K filing dates (fundq Q4) ...")
    df = db.raw_sql(query, date_cols=["datadate", "fdate", "rdq"])
    df = df.drop_duplicates(subset=["gvkey", "datadate"])

    # Tag source of filing date
    df["fdate_source"] = "actual"
    missing_mask = df["fdate"].isna()
    df.loc[missing_mask, "fdate_source"] = f"fallback_{FALLBACK_FILING_LAG_DAYS}d_lag"
    df.loc[missing_mask, "fdate"] = (
        df.loc[missing_mask, "datadate"] + pd.Timedelta(days=FALLBACK_FILING_LAG_DAYS)
    )

    n_actual   = (~missing_mask).sum()
    n_fallback = missing_mask.sum()
    print(f"  Filing dates — actual: {n_actual:,}  |  fallback: {n_fallback:,}")
    return df[["gvkey", "datadate", "fdate", "fdate_source"]]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to parquet in the raw Compustat directory."""
    DATA_RAW_COMPUSTAT.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW_COMPUSTAT / filename
    df.to_parquet(out, index=False)
    print(f"  Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    db = get_connection()

    fundamentals = extract_fundamentals(db)
    save(fundamentals, "compustat_annual_raw.parquet")

    filing_dates = extract_filing_dates(db)
    save(filing_dates, "compustat_filing_dates.parquet")

    db.close()
    print("\nCompustat extraction complete.")
    print("Next: python -m src.data.extract_crsp")


if __name__ == "__main__":
    main()
