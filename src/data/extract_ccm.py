"""
Extract the CRSP/Compustat Merged (CCM) Link Table from WRDS.
==============================================================
The CCM link table is the CRSP-curated, officially maintained mapping
between Compustat GVKEYs and CRSP PERMNOs, with explicit validity date
ranges and link-quality codes.

This is more reliable than a purely CUSIP-based match, which can fail
for firms that have changed their CUSIP without a corresponding CCM update
(Blueprint v4 §6.2).

Link-quality filters applied:
  - linktype IN ('LC', 'LU')  : researched (LC) or unresearched (LU) links
  - linkprim IN ('P', 'C')    : primary or combined (one PERMNO per GVKEY-date)

The extracted link table is used in merge_crsp_compustat.py together with
the CRSP security names file for the two-layer CCM + CUSIP cross-validation.

Blueprint v4 reference: §6.2

Fallback note
-------------
When `data/raw/compustat/linkhistory.rds` is available locally, the CCM
linktable is built by `load_local_rds.build_ccm_linktable()` directly
from that Compustat-CIZ source, applying the same link-quality filters.
This WRDS path is retained as a fallback for environments without the
local CIZ files.

Saves
-----
    data/raw/crsp/ccm_linktable_raw.parquet

Run
---
    python -m src.data.extract_ccm
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW_CRSP
from src.data.wrds_connection import get_connection

# CCM link-quality filters (Blueprint v4 §6.2)
VALID_LINKTYPES = ("LC", "LU")
VALID_LINKPRIMS = ("P", "C")

# Sentinel date used when linkenddt is NULL (link is still active)
_ACTIVE_LINK_SENTINEL = pd.Timestamp("2099-12-31")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_ccm(db) -> pd.DataFrame:
    """
    Pull the CCM link history table with high-quality link filters.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.

    Returns
    -------
    pd.DataFrame
        Columns: gvkey, permno, permco, linktype, linkprim,
                 linkdt, linkenddt.
        linkenddt is filled with 2099-12-31 where NULL (still active).

    Notes
    -----
    The date-range constraint (datadate ∈ [linkdt, linkenddt]) is applied
    in the merge step, not here. This file is extracted once and reused.
    """
    # TODO: Execute query and return DataFrame
    lt_list = ", ".join(f"'{t}'" for t in VALID_LINKTYPES)
    lp_list = ", ".join(f"'{p}'" for p in VALID_LINKPRIMS)

    query = f"""
        SELECT
            gvkey,
            lpermno  AS permno,
            lpermco  AS permco,
            linktype,
            linkprim,
            liid,
            linkdt,
            linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ({lt_list})
          AND linkprim IN ({lp_list})
        ORDER BY gvkey, linkdt
    """
    print("Querying CCM Link Table ...")
    df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])

    # NULL linkenddt means the link is still active
    df["linkenddt"] = df["linkenddt"].fillna(_ACTIVE_LINK_SENTINEL)

    print(f"  Link records   : {len(df):,}")
    print(f"  Unique GVKEYs  : {df['gvkey'].nunique():,}")
    print(f"  Unique PERMNOs : {df['permno'].nunique():,}")
    _print_linktype_summary(df)
    return df


def _print_linktype_summary(df: pd.DataFrame) -> None:
    """Print link type breakdown for sanity check."""
    summary = (
        df.groupby(["linktype", "linkprim"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    print("\n  Link type / link primary breakdown:")
    print(summary.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame) -> None:
    """Save CCM link table to parquet."""
    DATA_RAW_CRSP.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW_CRSP / "ccm_linktable_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    db = get_connection()
    df = extract_ccm(db)
    save(df)
    db.close()
    print("\nCCM extraction complete.")
    print("Next: python -m src.data.run_extraction  (or directly merge_crsp_compustat)")


if __name__ == "__main__":
    main()
