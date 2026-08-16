"""
Distress label sensitivity analysis — filing-date vs fiscal-year-end window.
============================================================================

Tests whether the timing of the 12-month distress window affects the
distribution of crisis-period delistings across fiscal years.

This is a DIAGNOSTIC analysis, not a robustness check. The fiscal-year-end
window contains forward-looking information (uses data not yet observable
at the moment of prediction) and CANNOT be used as the primary label
without violating the chronological design rule. The purpose is to confirm
that the apparently muted 2008-09 and trough-shaped 2020 distress rates
in the primary specification are timing artifacts, not measurement errors.

Two label specifications compared
---------------------------------
1. PRIMARY (filing-date window):
   Window = [filing_date, filing_date + 365 days]
   A FY2020 firm whose 10-K is filed April 2021 receives distress=1 only
   if it delists between April 2021 and April 2022.
   COVID delistings (March–Dec 2020) fall outside FY2020's window —
   they show up in FY2019 (filed early 2020).

2. SENSITIVITY (fiscal-year-end window):
   Window = [fiscal_year_end, fiscal_year_end + 365 days]
   A FY2020 firm receives distress=1 if it delists between
   2020-12-31 and 2021-12-31 (calendar 2021).
   This pulls more 2020-2021 COVID delistings into FY2020,
   at the cost of using information not observable on the
   fiscal year-end date.

Output
------
outputs/tables/descriptive/label_sensitivity_annual.csv
  Side-by-side annual distress rates under both label specifications,
  with the difference column making the timing-induced shift explicit.

Run
---
    python -m src.analysis.distress_label_sensitivity

Design-specification reference: §5.4 (label definition); §6.3 (design rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_FEATURES,
    DATA_RAW_CRSP,
    DISTRESS_CODES_PRIMARY,
    DISTRESS_HORIZON_DAYS,
    OUT_TABLES_DESCRIPTIVE,
)
from src.utils.tables import save_table


# ---------------------------------------------------------------------------
# Label builder — parameterised on the window-start column
# ---------------------------------------------------------------------------

def _build_label(
    panel: pd.DataFrame,
    delist: pd.DataFrame,
    window_start_col: str,
    codes: list[int],
    horizon_days: int = DISTRESS_HORIZON_DAYS,
) -> pd.Series:
    """
    Compute a distress label using a configurable window-start column.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: gvkey, fyear, permno, <window_start_col>.
    delist : pd.DataFrame
        CRSP delisting file (permno, dlstdt, dlstcd).
    window_start_col : str
        Column to anchor the distress window. Use "fdate" for the
        primary filing-date specification, "datadate" for the
        fiscal-year-end sensitivity specification.
    codes : list[int]
        Delisting codes to count as distress events.
    horizon_days : int
        Forward-looking window length in days.

    Returns
    -------
    pd.Series
        Distress label aligned to panel index (0/1, int).
    """
    delist_d = delist[delist["dlstcd"].isin(codes)][
        ["permno", "dlstdt"]
    ].copy()

    base = panel[["gvkey", "fyear", "permno", window_start_col]].copy()
    base = base.rename(columns={window_start_col: "_w_start"})

    merged = base.merge(delist_d, on="permno", how="left")
    window_end = merged["_w_start"] + pd.Timedelta(days=horizon_days)
    in_window = (
        merged["dlstdt"].notna()
        & merged["dlstdt"].ge(merged["_w_start"])
        & merged["dlstdt"].le(window_end)
    )
    merged["_event"] = in_window.astype(int)

    agg = (
        merged.groupby(["gvkey", "fyear"])["_event"]
              .max()
              .reset_index()
    )
    out = panel[["gvkey", "fyear"]].merge(agg, on=["gvkey", "fyear"], how="left")
    return out["_event"].fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_label_sensitivity() -> pd.DataFrame:
    """
    Build the side-by-side annual distress-rate comparison.

    Returns
    -------
    pd.DataFrame
        Columns: fyear, n_total,
                 n_distress_primary, rate_primary_pct,
                 n_distress_fyend,   rate_fyend_pct,
                 diff_pct (= fyend − primary).
    """
    print("=" * 60)
    print("  DISTRESS LABEL SENSITIVITY  —  filing-date vs fiscal-year-end window")
    print("=" * 60)

    panel = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
    print(f"  Panel : {len(panel):,} firm-years")
    print(f"  Delist: {len(delist):,} events")

    # Ensure required columns exist
    needed = {"gvkey", "fyear", "permno", "fdate", "datadate"}
    missing = needed - set(panel.columns)
    if missing:
        raise KeyError(
            f"features_all.parquet is missing required columns: {missing}. "
            "Rebuild features with `python -m src.features.build_features`."
        )

    # Primary specification (filing-date window)
    panel = panel.copy()
    panel["distress_primary"] = _build_label(
        panel, delist,
        window_start_col="fdate",
        codes=DISTRESS_CODES_PRIMARY,
        horizon_days=DISTRESS_HORIZON_DAYS,
    )

    # Sensitivity specification (fiscal-year-end window)
    panel["distress_fyend"] = _build_label(
        panel, delist,
        window_start_col="datadate",
        codes=DISTRESS_CODES_PRIMARY,
        horizon_days=DISTRESS_HORIZON_DAYS,
    )

    # Annual aggregation
    annual = (
        panel.groupby("fyear")
             .agg(
                 n_total=("gvkey", "size"),
                 n_distress_primary=("distress_primary", "sum"),
                 n_distress_fyend=("distress_fyend", "sum"),
             )
             .reset_index()
    )
    annual["rate_primary_pct"] = (
        100 * annual["n_distress_primary"] / annual["n_total"]
    ).round(3)
    annual["rate_fyend_pct"] = (
        100 * annual["n_distress_fyend"] / annual["n_total"]
    ).round(3)
    annual["diff_pct"] = (
        annual["rate_fyend_pct"] - annual["rate_primary_pct"]
    ).round(3)

    # Sanity-check crisis years
    crisis = annual[annual["fyear"].isin([2001, 2002, 2008, 2009, 2020, 2021])]
    print("\n  Crisis-year distress rates (primary vs FY-end window):")
    print(crisis[["fyear", "rate_primary_pct", "rate_fyend_pct", "diff_pct"]]
          .to_string(index=False))

    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)
    save_table(
        annual,
        OUT_TABLES_DESCRIPTIVE / "label_sensitivity_annual",
        caption=(
            "Distress label timing sensitivity. The primary specification "
            "anchors the 12-month distress window at the 10-K filing date; "
            "the sensitivity specification anchors it at fiscal year-end "
            "(forward-looking — diagnostic use only). Reported rates are "
            "percentages of firm-years per fiscal year."
        ),
        label="tab:label_sensitivity",
    )
    print(f"\n  Annual comparison saved -> "
          f"{OUT_TABLES_DESCRIPTIVE / 'label_sensitivity_annual.csv'}")
    return annual


if __name__ == "__main__":
    run_label_sensitivity()
