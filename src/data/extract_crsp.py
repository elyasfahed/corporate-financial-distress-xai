"""
Extract CRSP Monthly Stock File, Delisting File, and Security History from WRDS.
==================================================================================
Pulls three files:

1. crsp.msf — Monthly Stock Security File
   Provides equity returns, prices, and shares outstanding for all
   market-based predictor construction.
   Filter: SHRCD IN (10, 11) — ordinary common shares of domestic US firms only.

2. crsp.msedelist — Monthly Delisting File
   Provides DLSTCD codes and event dates for constructing the distress label.
   Primary definition: DLSTCD 400–499 (Pre-Specified Empirical Design §5.1).

3. crsp.stocknames — Security Information History
   Provides historical NCUSIP codes for the CCM CUSIP cross-validation
   step of the merge (Pre-Specified Empirical Design §6.2).

Design-specification reference: §4.2, §5.1, §6.2

Saves
-----
    data/raw/crsp/crsp_monthly_raw.parquet
    data/raw/crsp/crsp_delisting_raw.parquet
    data/raw/crsp/crsp_security_names.parquet

Run
---
    python -m src.data.extract_crsp
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_CRSP,
    SAMPLE_START_YEAR,
    SAMPLE_END_YEAR,
    VALID_SHRCDS,
    DISTRESS_CODES_PRIMARY,
)
from src.data.wrds_connection import get_connection

_START_DATE = f"{SAMPLE_START_YEAR}-01-01"
_END_DATE   = f"{SAMPLE_END_YEAR}-12-31"


# ---------------------------------------------------------------------------
# Monthly Stock File
# ---------------------------------------------------------------------------

def extract_monthly_stock(db) -> pd.DataFrame:
    """
    Pull CRSP Monthly Stock Security File.

    Restricted to SHRCD 10 and 11 (domestic ordinary common shares).
    This excludes ADRs, REITs, closed-end funds, preferred shares, and
    foreign private issuers (Pre-Specified Empirical Design §6.1).

    Key variables computed here:
    - prc : price (absolute value — negative means bid/ask midpoint)
    - me  : market capitalisation in millions (prc × shrout / 1000)

    Parameters
    ----------
    db : wrds.Connection

    Returns
    -------
    pd.DataFrame
        Monthly observations: permno, date, ret, retx, prc, shrout, vol,
        exchcd, shrcd, siccd, me.
    """
    # TODO: Execute query and compute market cap
    query = f"""
        SELECT
            a.permno,
            a.permco,
            a.date,
            a.ret,
            a.retx,
            ABS(a.prc)   AS prc,
            a.shrout,
            a.vol,
            a.cfacshr,
            a.cfacpr,
            b.exchcd,
            b.shrcd,
            b.siccd,
            b.ncusip
        FROM crsp.msf  AS a
        LEFT JOIN crsp.msenames AS b
            ON  a.permno = b.permno
            AND b.namedt  <= a.date
            AND b.nameendt >= a.date
        WHERE a.date BETWEEN '{_START_DATE}' AND '{_END_DATE}'
          AND b.shrcd IN ({', '.join(str(c) for c in VALID_SHRCDS)})
        ORDER BY a.permno, a.date
    """
    print("Querying CRSP Monthly Stock File ...")
    df = db.raw_sql(query, date_cols=["date"])

    # Market capitalisation in millions (shrout is in thousands of shares)
    df["me"] = df["prc"] * df["shrout"] / 1_000

    print(f"  Rows returned  : {len(df):,}")
    print(f"  Unique PERMNOs : {df['permno'].nunique():,}")
    print(f"  Date range     : {df['date'].min().date()} – {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Delisting File
# ---------------------------------------------------------------------------

def extract_delisting(db) -> pd.DataFrame:
    """
    Pull CRSP Monthly Delisting File.

    Used to construct the primary distress label D_{i,t}.
    Distress events: DLSTCD codes 400–499 (Pre-Specified Empirical Design §5.1).
    Subset 572, 574, 584 identifies formal Chapter 7/11 bankruptcy filings
    and is used in robustness check RC1.

    Parameters
    ----------
    db : wrds.Connection

    Returns
    -------
    pd.DataFrame
        Columns: permno, dlstdt, dlstcd, dlret, dlprc,
                 is_distress (bool, primary definition).
    """
    # TODO: Execute query and flag distress events
    query = f"""
        SELECT
            permno,
            dlstdt,
            dlstcd,
            dlret,
            dlprc,
            nwperm
        FROM crsp.msedelist
        WHERE dlstdt BETWEEN '{_START_DATE}' AND '{_END_DATE}'
        ORDER BY permno, dlstdt
    """
    print("Querying CRSP Delisting File ...")
    df = db.raw_sql(query, date_cols=["dlstdt"])

    # Flag primary distress events (DLSTCD 400–499)
    df["is_distress"] = df["dlstcd"].isin(DISTRESS_CODES_PRIMARY)

    n_total    = len(df)
    n_distress = df["is_distress"].sum()
    print(f"  Total delistings   : {n_total:,}")
    print(f"  Distress (400–499) : {n_distress:,}")
    _print_top_codes(df)
    return df


def _print_top_codes(df: pd.DataFrame, n: int = 15) -> None:
    """Print the most frequent delisting codes for manual sanity check."""
    summary = (
        df.groupby("dlstcd")
          .agg(count=("permno", "size"), is_distress=("is_distress", "first"))
          .sort_values("count", ascending=False)
          .head(n)
          .reset_index()
    )
    print(f"\n  Top {n} delisting codes (sanity check):")
    print(summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Security Names / History  (for CCM CUSIP cross-validation)
# ---------------------------------------------------------------------------

def extract_security_names(db) -> pd.DataFrame:
    """
    Pull CRSP Security Information History (stocknames).

    Provides historical NCUSIP codes used in the second layer of the
    CCM merge cross-validation (Pre-Specified Empirical Design §6.2):
      Match NCUSIP (CRSP) against first 8 chars of CUSIP (Compustat).
    Firm-years that pass CCM but fail CUSIP cross-validation are flagged
    for manual investigation before inclusion.

    Parameters
    ----------
    db : wrds.Connection

    Returns
    -------
    pd.DataFrame
        Columns: permno, ncusip, namedt, nameendt, shrcd, exchcd.
    """
    # TODO: Execute query and return DataFrame
    query = """
        SELECT
            permno,
            ncusip,
            namedt,
            nameendt,
            shrcd,
            exchcd,
            comnam
        FROM crsp.stocknames
        ORDER BY permno, namedt
    """
    print("Querying CRSP Security Information History ...")
    df = db.raw_sql(query, date_cols=["namedt", "nameendt"])
    print(f"  Rows returned  : {len(df):,}")
    print(f"  Unique PERMNOs : {df['permno'].nunique():,}")
    return df


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, filename: str) -> None:
    """Save DataFrame to parquet in the raw CRSP directory."""
    DATA_RAW_CRSP.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW_CRSP / filename
    df.to_parquet(out, index=False)
    print(f"  Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    db = get_connection()

    msf = extract_monthly_stock(db)
    save(msf, "crsp_monthly_raw.parquet")

    delist = extract_delisting(db)
    save(delist, "crsp_delisting_raw.parquet")

    secnames = extract_security_names(db)
    save(secnames, "crsp_security_names.parquet")

    db.close()
    print("\nCRSP extraction complete.")
    print("Next: python -m src.data.extract_ccm")


if __name__ == "__main__":
    main()
