"""
Master WRDS extraction script — downloads all raw data in one run.
====================================================================
Runs all four extractors in sequence:
  1. Compustat Fundamentals Annual + Filing Dates
  2. CRSP Monthly Stock File + Delisting File + Security Names
  3. CCM Link Table

Each extractor skips re-download if the output file already exists,
unless --force is passed. This makes the script safe to re-run.

All files are saved to the Seafile synced folder (DATA_ROOT) and will
automatically sync to the university server after download completes.

Usage
-----
    python -m src.data.run_extraction
    python -m src.data.run_extraction --force          # re-download everything
    python -m src.data.run_extraction --skip-crsp      # skip CRSP only

Expected runtime: 20–60 minutes depending on WRDS server load.
Expected disk usage: ~3–4 GB (CRSP monthly is the largest file).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_RAW_CRSP, DATA_RAW_COMPUSTAT
from src.data.wrds_connection import get_connection
from src.data.extract_compustat import (
    extract_fundamentals,
    extract_filing_dates,
    save as save_compustat,
)
from src.data.extract_crsp import (
    extract_monthly_stock,
    extract_delisting,
    extract_security_names,
    save as save_crsp,
)
from src.data.extract_ccm import extract_ccm, save as save_ccm


def _exists(path: Path) -> bool:
    return path.is_file()


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all WRDS raw data files."
    )
    parser.add_argument("--skip-compustat", action="store_true",
                        help="Skip Compustat extraction.")
    parser.add_argument("--skip-crsp",      action="store_true",
                        help="Skip CRSP extraction.")
    parser.add_argument("--skip-ccm",       action="store_true",
                        help="Skip CCM link table extraction.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if output files already exist.")
    args = parser.parse_args()

    db = get_connection()

    # ------------------------------------------------------------------
    # 1. Compustat Annual Fundamentals + Filing Dates
    # ------------------------------------------------------------------
    _section("1 / 3   Compustat")
    out_funda   = DATA_RAW_COMPUSTAT / "compustat_annual_raw.parquet"
    out_filing  = DATA_RAW_COMPUSTAT / "compustat_filing_dates.parquet"

    if args.skip_compustat:
        print("[SKIP] --skip-compustat flag set.")
    elif _exists(out_funda) and _exists(out_filing) and not args.force:
        print(f"[SKIP] Compustat files already exist. Use --force to re-download.")
    else:
        fundamentals = extract_fundamentals(db)
        save_compustat(fundamentals, "compustat_annual_raw.parquet")

        filing_dates = extract_filing_dates(db)
        save_compustat(filing_dates, "compustat_filing_dates.parquet")

    # ------------------------------------------------------------------
    # 2. CRSP Monthly Stock + Delisting + Security Names
    # ------------------------------------------------------------------
    _section("2 / 3   CRSP")
    out_msf     = DATA_RAW_CRSP / "crsp_monthly_raw.parquet"
    out_delist  = DATA_RAW_CRSP / "crsp_delisting_raw.parquet"
    out_names   = DATA_RAW_CRSP / "crsp_security_names.parquet"

    if args.skip_crsp:
        print("[SKIP] --skip-crsp flag set.")
    elif _exists(out_msf) and _exists(out_delist) and _exists(out_names) and not args.force:
        print(f"[SKIP] CRSP files already exist. Use --force to re-download.")
    else:
        msf = extract_monthly_stock(db)
        save_crsp(msf, "crsp_monthly_raw.parquet")

        delist = extract_delisting(db)
        save_crsp(delist, "crsp_delisting_raw.parquet")

        secnames = extract_security_names(db)
        save_crsp(secnames, "crsp_security_names.parquet")

    # ------------------------------------------------------------------
    # 3. CCM Link Table
    # ------------------------------------------------------------------
    _section("3 / 3   CCM Link Table")
    out_ccm = DATA_RAW_CRSP / "ccm_linktable_raw.parquet"

    if args.skip_ccm:
        print("[SKIP] --skip-ccm flag set.")
    elif _exists(out_ccm) and not args.force:
        print(f"[SKIP] CCM link table already exists. Use --force to re-download.")
    else:
        ccm = extract_ccm(db)
        save_ccm(ccm)

    db.close()

    _section("ALL EXTRACTIONS COMPLETE")
    print("Raw data saved to:", DATA_RAW_CRSP.parent.parent)
    print("\nNext step:")
    print("  python -m src.data.merge_crsp_compustat")


if __name__ == "__main__":
    main()
