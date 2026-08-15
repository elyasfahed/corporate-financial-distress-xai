"""
Fetch the CRSP CIZ flag-value dictionary (MetaFlagInfo) from WRDS.
==================================================================
Companion to ``scripts/pin_ciz_dictionary.py``. The local data drop contains
``MetaItemInfo`` and ``MetaColumnInfo`` but not the flag-VALUE dictionary that
defines the delisting mnemonics (GDR, GLI, CORQ, MTMK, MVOT, BKPY, ...). This
script locates that table on WRDS and saves it as

    data/raw/crsp/MetaFlagInfo.csv

after which ``python -m scripts.pin_ciz_dictionary`` closes the loop.

It does NOT hard-code a table name: WRDS library/table naming for the CIZ
(flat-file format 2.0) metadata varies by subscription vintage, so the script
first *discovers* candidate metadata tables and prints them, then pulls the
best match. If the automatic pick is wrong, re-run with an explicit target:

    python -m scripts.fetch_ciz_flag_dictionary --table crsp.metaflaginfo

Prerequisites
-------------
A WRDS account with CRSP access (LUH library subscription) and saved
credentials. If you have never connected from this machine, run once:

    python -c "import wrds; wrds.Connection()"

and enter your username/password when prompted; it offers to store them in
~/.pgpass so later runs are non-interactive.

Run:
    PYTHONUTF8=1 ./.venv311/Scripts/python.exe -m scripts.fetch_ciz_flag_dictionary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

DEST = ROOT / "data" / "raw" / "crsp" / "MetaFlagInfo.csv"
# Delisting flag types we need (from MetaItemInfo): action, reason, status, payment
NEEDED_FLAG_TYPES = ["CD", "DR", "CS", "PT"]


def discover(db) -> list[str]:
    """Return candidate <library>.<table> names that look like CIZ metadata."""
    cands: list[str] = []
    try:
        libs = [l for l in db.list_libraries() if "crsp" in l.lower()]
    except Exception as exc:
        print(f"  could not list libraries ({exc}); trying 'crsp' only")
        libs = ["crsp"]
    print(f"  scanning {len(libs)} CRSP library/ies: {libs}")
    for lib in libs:
        try:
            tables = db.list_tables(library=lib)
        except Exception:
            continue
        for t in tables:
            tl = t.lower()
            if "meta" in tl and ("flag" in tl or "item" in tl or "column" in tl):
                cands.append(f"{lib}.{t}")
    return cands


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", help="explicit <library>.<table> to pull")
    ap.add_argument("--list-only", action="store_true",
                    help="only discover and print candidates")
    args = ap.parse_args()

    from src.data.wrds_connection import get_connection
    db = get_connection()

    try:
        target = args.table
        if not target:
            print("\nDiscovering CIZ metadata tables on WRDS ...")
            cands = discover(db)
            if not cands:
                raise SystemExit(
                    "No CIZ metadata tables found. Your subscription may expose them\n"
                    "under a different library. Try:\n"
                    "  python -m scripts.fetch_ciz_flag_dictionary --list-only\n"
                    "or ask WRDS support for the CIZ flag-value dictionary "
                    "(flag types CD and DR)."
                )
            print("  candidates:")
            for c in cands:
                print("   ", c)
            if args.list_only:
                return
            flag = [c for c in cands if "flag" in c.lower()]
            if not flag:
                raise SystemExit(
                    "\nFound CIZ metadata tables but none matching 'flag'. Pull the "
                    "right one explicitly with --table <library>.<table>."
                )
            target = flag[0]

        print(f"\nPulling {target} ...")
        df = db.raw_sql(f"select * from {target}")
        print(f"  {len(df):,} rows, columns: {list(df.columns)}")

        # Narrow to the delisting flag types when a type column is identifiable
        tcol = next((c for c in df.columns if "flagtype" in c.lower()
                     or c.lower().endswith("type")), None)
        if tcol is not None:
            sub = df[df[tcol].astype("string").str.strip().isin(NEEDED_FLAG_TYPES)]
            if len(sub):
                print(f"  delisting flag types ({tcol}): {len(sub):,} rows "
                      f"-> {sorted(sub[tcol].astype('string').str.strip().unique())}")
            else:
                print(f"  note: no rows matched flag types {NEEDED_FLAG_TYPES} "
                      f"on column '{tcol}'; saving the full table instead")

        DEST.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DEST, index=False)
        print(f"\nSaved -> {DEST}")
        print("Next: python -m scripts.pin_ciz_dictionary")
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
