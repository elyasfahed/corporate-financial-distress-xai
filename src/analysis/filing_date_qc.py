"""
Filing-date QC — why perioddescriptorannual.FDATE was rejected.
===============================================================
Read-only diagnostic that documents the decision to source the 10-K filing
date F_{i,t} from the dedicated filingdates.rds file rather than from
perioddescriptorannual.FDATE. Produces the evidence tables for the thesis
"transparent deviation" disclosure (Chapter 3 / Data & Empirical Setting).

It does NOT touch the pipeline data state — it reconstructs both candidate
date series straight from the raw .rds files and writes tables only.

Three outputs (→ outputs/tables/descriptive/):
  1. filing_date_source_agreement   — where both dates exist, how often does
     FDATE equal the real 10-K FILEDATE? (spoiler: ~75% disagree)
  2. filing_date_coverage_by_year    — real-date coverage of the analysis
     panel, FDATE source vs filingdates.rds source, by fiscal year
  3. filing_date_lag_distribution    — distribution of the real 10-K filing
     lag (FILEDATE − fiscal year-end), incl. share beyond the 180-day cap

Run
---
    python -m src.analysis.filing_date_qc
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_MERGED,
    OUT_TABLES_DESCRIPTIVE,
    MAX_FILING_LAG_DAYS,
)
from src.data.load_local_rds import (
    _build_filing_dates_from_dedicated,
    _build_filing_dates_from_perioddescriptor,
)
from src.utils.tables import save_table


def _load_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct both candidate filing-date series from the raw files."""
    peri = _build_filing_dates_from_perioddescriptor()
    peri = peri[peri["fdate"].notna()][["gvkey", "datadate", "fdate"]]
    peri = peri.rename(columns={"fdate": "fdate_peri"})

    ded = _build_filing_dates_from_dedicated("drop")  # 10-K only
    ded = ded[["gvkey", "datadate", "fdate"]].rename(columns={"fdate": "fdate_10k"})

    for d in (peri, ded):
        d["gvkey"] = d["gvkey"].astype(str).str.strip()
        d["datadate"] = pd.to_datetime(d["datadate"])
    return peri, ded


def table_agreement(peri: pd.DataFrame, ded: pd.DataFrame) -> pd.DataFrame:
    """Where both dates exist, quantify (dis)agreement."""
    both = peri.merge(ded, on=["gvkey", "datadate"], how="inner")
    diff = (both["fdate_peri"] - both["fdate_10k"]).dt.days
    n = len(both)
    rows = [
        ("Firm-years with both dates",        n,                        100.0),
        ("Exact same day",                    int((diff == 0).sum()),   100 * (diff == 0).mean()),
        ("Within +/- 1 day",                  int((diff.abs() <= 1).sum()),  100 * (diff.abs() <= 1).mean()),
        ("Within +/- 7 days",                 int((diff.abs() <= 7).sum()),  100 * (diff.abs() <= 7).mean()),
        ("Differ > 7 days",                   int((diff.abs() > 7).sum()),   100 * (diff.abs() > 7).mean()),
        ("FDATE later than 10-K (diff > 0)",  int((diff > 0).sum()),    100 * (diff > 0).mean()),
    ]
    out = pd.DataFrame(rows, columns=["statistic", "n", "pct"])
    out["pct"] = out["pct"].round(1)
    # Append central-tendency summary of the signed difference
    summary = pd.DataFrame({
        "statistic": ["Median diff (days)", "Mean diff (days)", "Std diff (days)"],
        "n": [round(diff.median(), 1), round(diff.mean(), 1), round(diff.std(), 1)],
        "pct": [np.nan, np.nan, np.nan],
    })
    return pd.concat([out, summary], ignore_index=True)


def table_coverage_by_year(ded: pd.DataFrame, peri: pd.DataFrame) -> pd.DataFrame:
    """Real-date coverage of the analysis panel, by fiscal year, both sources."""
    panel = pd.read_parquet(
        DATA_MERGED / "panel_raw.parquet", columns=["gvkey", "fyear", "datadate"]
    )
    panel["gvkey"] = panel["gvkey"].astype(str).str.strip()
    panel["datadate"] = pd.to_datetime(panel["datadate"])

    m = panel.merge(ded.assign(has_10k=True)[["gvkey", "datadate", "has_10k"]],
                    on=["gvkey", "datadate"], how="left")
    m = m.merge(peri.assign(has_peri=True)[["gvkey", "datadate", "has_peri"]],
                on=["gvkey", "datadate"], how="left")
    # True where the merge found a date, False (NaN) otherwise
    m["has_10k"] = m["has_10k"].notna()
    m["has_peri"] = m["has_peri"].notna()

    g = m.groupby("fyear").agg(
        n=("gvkey", "size"),
        pct_fdate=("has_peri", lambda s: round(100 * s.mean(), 1)),
        pct_10k=("has_10k", lambda s: round(100 * s.mean(), 1)),
    ).reset_index()
    return g


def table_lag_distribution(ded: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the real 10-K filing lag on the analysis panel."""
    panel = pd.read_parquet(
        DATA_MERGED / "panel_raw.parquet", columns=["gvkey", "datadate"]
    )
    panel["gvkey"] = panel["gvkey"].astype(str).str.strip()
    panel["datadate"] = pd.to_datetime(panel["datadate"])
    m = panel.merge(ded, on=["gvkey", "datadate"], how="inner")
    lag = (m["fdate_10k"] - m["datadate"]).dt.days

    pcts = [5, 25, 50, 75, 90, 95, 99]
    rows = [("n (matched firm-years)", len(lag))]
    rows += [("mean", round(lag.mean(), 1)), ("std", round(lag.std(), 1))]
    rows += [(f"p{p}", round(np.percentile(lag, p), 1)) for p in pcts]
    rows += [
        (f"share lag > {MAX_FILING_LAG_DAYS}d (%)", round(100 * (lag > MAX_FILING_LAG_DAYS).mean(), 2)),
        ("share lag < 0 (%)", round(100 * (lag < 0).mean(), 2)),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def main() -> None:
    print("\n" + "=" * 60)
    print("  FILING-DATE QC  (perioddescriptor.FDATE vs filingdates.rds)")
    print("=" * 60)

    peri, ded = _load_candidates()

    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)

    agree = table_agreement(peri, ded)
    save_table(
        agree, OUT_TABLES_DESCRIPTIVE / "filing_date_source_agreement",
        caption="Agreement between perioddescriptorannual.FDATE and the real "
                "10-K filing date (filingdates.rds) where both are present.",
        label="tab:filing_date_source_agreement",
        float_format="%.1f",
    )
    print("\n" + agree.to_string(index=False))

    cov = table_coverage_by_year(ded, peri)
    save_table(
        cov, OUT_TABLES_DESCRIPTIVE / "filing_date_coverage_by_year",
        caption="Real filing-date coverage of the analysis panel by fiscal "
                "year: legacy FDATE source vs. dedicated filingdates.rds.",
        label="tab:filing_date_coverage_by_year",
        float_format="%.1f",
    )
    print("\n" + cov.to_string(index=False))

    lag = table_lag_distribution(ded)
    save_table(
        lag, OUT_TABLES_DESCRIPTIVE / "filing_date_lag_distribution",
        caption="Distribution of the real 10-K filing lag (FILEDATE minus "
                "fiscal year-end) on the analysis panel. The panel already "
                "applies the 180-day filing-lag exclusion, so the share above "
                "180 days is zero by construction.",
        label="tab:filing_date_lag_distribution",
        float_format="%.2f",
    )
    print("\n" + lag.to_string(index=False))

    print("\nFiling-date QC complete.")


if __name__ == "__main__":
    main()
