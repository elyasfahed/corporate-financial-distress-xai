"""
Supplementary robustness: does the 10-K-anchored label select against the most
severe (abrupt) failures?
==========================================================================
Purpose
-------
Research concern: the distress label may select against severe cases when firms
stop filing before a for-cause delisting. This module measures that selection
and evaluates an alternative anchor as a supplementary robustness check.

Mechanism. The primary label sets D_{i,t}=1 iff a CRSP for-cause delisting
(DLSTCD 400-499) falls in the forward window [F_{i,t}, F_{i,t}+365], where
F_{i,t} is the *actual* 10-K filing date (src/data/merge_crsp_compustat.py,
construct_distress_label). Anchoring on a filed 10-K is what makes the design
point-in-time. Its side effect is a selection: a firm that stops filing audited
financials and is delisted for cause more than a year after its last 10-K has no
filing date whose 365-day window reaches the delisting, so the event is never
labelled distressed. These "went-dark-before-death" firms are exactly the most
severe, protracted failures affected by this selection mechanism.

This module is READ-ONLY with respect to the frozen pipeline. It does NOT re-fit
any model and does NOT touch outputs/models/* or the headline tables. It writes
ONLY:
    outputs/tables/robustness/severe_case_selection_footprint.{csv,tex}
    outputs/tables/robustness/severe_case_selection_recovery.{csv,tex}
    outputs/tables/robustness/severe_case_selection_dedup.{csv,tex}

Dedup accounting (added 2026-07-03). The D (fiscal-year-end, 730d) label used in
the sensitivity check counts positive FIRM-YEARS, not unique delistings: adjacent
[datadate_T, datadate_T+730] windows overlap by ~365 days, so one delisting can
label two consecutive firm-years. The dedup table quantifies, on the frozen test
split, how much of the D regime's event-count increase (434 -> ~885 firm-years)
is that mechanical double-labelling versus genuinely recovered went-dark events.

Footprint (scope of the selection). Among the modelled firms (firms
with enough financials to enter the panel), it counts every for-cause delisting
in the observable window and splits it into:
    captured  -- the event landed in some firm-year's [F, F+365] window (D=1);
    orphaned  -- no firm-year window reached it,
and, for the orphans, the gap between the firm's LAST available 10-K filing date
and the delisting date (the "went-dark" lead time).

Recovery (alternative anchor). It counts
how many orphaned events an alternative, fiscal-year-end-anchored window
[datadate_T, datadate_T + H] would recover, for several H. This anchors on the
fiscal year-end plus a fixed reach instead of the actual 10-K date, in the spirit
of the fixed-reporting-lag convention of Campbell, Hilscher & Szilagyi (2008).
The count motivates the proposed (future) robustness check; it does NOT re-fit
models, consistent with the design freeze and the evaluate-test-once rule.

Run from the project root:
    python -m src.robustness.severe_case_selection_footprint
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import (
    DATA_MERGED, DATA_RAW_CRSP, OUT_TABLES_ROBUSTNESS,
    DISTRESS_CODES_PRIMARY, DISTRESS_HORIZON_DAYS,
)

# Reach (days from fiscal year-end) for the alternative fiscal-year-end anchor.
RECOVERY_HORIZONS = [365, 547, 730, 912]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(DATA_MERGED / "panel_raw.parquet")
    delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
    delist["dlstdt"] = pd.to_datetime(delist["dlstdt"])
    for col in ("fdate", "datadate", "delist_date"):
        panel[col] = pd.to_datetime(panel[col])
    return panel, delist


def for_cause_events(panel: pd.DataFrame, delist: pd.DataFrame) -> pd.DataFrame:
    """For-cause (400-499) delistings of modelled firms, within the observable
    window [min(F), max(F)+horizon]."""
    uni = set(panel["permno"].unique())
    lo, hi = int(min(DISTRESS_CODES_PRIMARY)), int(max(DISTRESS_CODES_PRIMARY))
    fc = delist[delist["dlstcd"].between(lo, hi) & delist["permno"].isin(uni)].copy()
    fmin = panel["fdate"].min()
    fmax = panel["fdate"].max() + pd.Timedelta(days=DISTRESS_HORIZON_DAYS)
    fc = fc[(fc["dlstdt"] >= fmin) & (fc["dlstdt"] <= fmax)].copy()
    return fc


def build_footprint(panel: pd.DataFrame, fc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    captured_permnos = set(panel.loc[panel["distress"] == 1, "permno"].unique())
    fc = fc.copy()
    fc["captured"] = fc["permno"].isin(captured_permnos)

    last_fdate = panel.groupby("permno")["fdate"].max()
    fc["gap_days"] = (fc["dlstdt"] - fc["permno"].map(last_fdate)).dt.days

    n_total = len(fc)
    orph = fc[~fc["captured"]]
    n_capt = int(fc["captured"].sum())
    n_orph = int((~fc["captured"]).sum())
    n_dark = int((orph["gap_days"] > DISTRESS_HORIZON_DAYS).sum())

    footprint = pd.DataFrame(
        {
            "quantity": [
                "For-cause delistings of modelled firms (observable window)",
                "Captured by 10-K window (D=1)",
                "Orphaned (no 10-K window reaches the event)",
                "  of which: delisting > 365d after last 10-K (went dark)",
                "  of which: delisting at/before last 10-K (gap <= 0)",
            ],
            "events": [n_total, n_capt, n_orph, n_dark, int((orph["gap_days"] <= 0).sum())],
            "pct_of_for_cause": [
                100.0,
                round(100 * n_capt / n_total, 1),
                round(100 * n_orph / n_total, 1),
                round(100 * n_dark / n_total, 1),
                round(100 * int((orph["gap_days"] <= 0).sum()) / n_total, 1),
            ],
        }
    )
    return footprint, orph


def build_recovery(panel: pd.DataFrame, orph: pd.DataFrame) -> pd.DataFrame:
    last_dd = panel.groupby("permno")["datadate"].max()
    o = orph.copy()
    o["last_dd"] = o["permno"].map(last_dd)
    n_fc_total = len(orph) + int(panel.loc[panel["distress"] == 1, "permno"].nunique())
    rows = []
    for H in RECOVERY_HORIZONS:
        rec = int(((o["dlstdt"] >= o["last_dd"]) &
                   (o["dlstdt"] <= o["last_dd"] + pd.Timedelta(days=H))).sum())
        rows.append(
            {
                "reach_days_from_fyend": H,
                "orphans_recovered": rec,
                "pct_of_orphans": round(100 * rec / len(orph), 1),
                "pct_of_for_cause": round(100 * rec / n_fc_total, 1),
            }
        )
    return pd.DataFrame(rows)


def build_dedup_stats(delist: pd.DataFrame) -> pd.DataFrame:
    """Firm-year vs unique-event accounting of the F and D labels on the frozen
    TEST split (read-only; same relabelling mechanism as the sensitivity check).
    One delisting can label two consecutive firm-years under D because adjacent
    730-day fiscal-year-end windows overlap by ~365 days."""
    from src.config import DATA_SAMPLES
    from src.data.merge_crsp_compustat import build_distress_label

    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    base = test.drop(columns=["distress", "delist_date", "delist_code"], errors="ignore")

    regimes = {
        "F_fdate": dict(anchor="fdate"),
        "D_datadate_730": dict(anchor="datadate", reach_days=730),
    }
    stats: dict[str, dict] = {}
    for rg, kw in regimes.items():
        lab = build_distress_label(base.copy(), delist, codes=DISTRESS_CODES_PRIMARY, **kw)
        pos = lab[lab["distress"] == 1]
        ev = pos.groupby(["permno", "delist_date"]).size()
        stats[rg] = dict(
            firm_years=int(len(pos)),
            unique_events=int(len(ev)),
            events_labelling_2plus_firm_years=int((ev >= 2).sum()),
            max_firm_years_per_event=int(ev.max()) if len(ev) else 0,
            _keys=set(ev.index),
            _pos=pos,
        )

    f, d = stats["F_fdate"], stats["D_datadate_730"]
    added_firm_years = d["firm_years"] - f["firm_years"]
    new_events = len(d["_keys"] - f["_keys"])
    # D firm-years whose event was already captured under F, in excess of the
    # F firm-years themselves = mechanical double-labelling of known events.
    d_fy_on_f_events = int(
        d["_pos"].set_index(["permno", "delist_date"]).index.isin(f["_keys"]).sum()
    )
    dup_added = d_fy_on_f_events - f["firm_years"]

    rows = [
        ("F: positive firm-years",                      f["firm_years"]),
        ("F: unique delisting events",                  f["unique_events"]),
        ("D: positive firm-years",                      d["firm_years"]),
        ("D: unique delisting events",                  d["unique_events"]),
        ("D: events labelling 2+ firm-years",           d["events_labelling_2plus_firm_years"]),
        ("D: max firm-years per event",                 d["max_firm_years_per_event"]),
        ("Added positive firm-years (D - F)",           added_firm_years),
        ("  of which: extra firm-years on F-captured events", dup_added),
        ("  of which: firm-years on newly recovered events",  added_firm_years - dup_added),
        ("Unique events recovered by D but not F",      new_events),
    ]
    return pd.DataFrame(rows, columns=["quantity", "count"])


def main() -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    panel, delist = load()
    fc = for_cause_events(panel, delist)
    footprint, orph = build_footprint(panel, fc)
    recovery = build_recovery(panel, orph)
    dedup = build_dedup_stats(delist)

    print("\n=== Severe-case selection footprint ===")
    print(footprint.to_string(index=False))
    print("\nOrphan gap (days from last 10-K to delisting):")
    print(orph["gap_days"].describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_string())
    print("\n=== Recovery under a fiscal-year-end-anchored window ===")
    print(recovery.to_string(index=False))

    footprint.to_csv(OUT_TABLES_ROBUSTNESS / "severe_case_selection_footprint.csv", index=False)
    with open(OUT_TABLES_ROBUSTNESS / "severe_case_selection_footprint.tex", "w", encoding="utf-8") as fh:
        fh.write(footprint.to_latex(index=False, escape=True,
                 caption="Footprint of the 10-K-anchored label's selection against abrupt "
                         "failures, among modelled firms.",
                 label="tab:severe_selection_footprint"))

    recovery.to_csv(OUT_TABLES_ROBUSTNESS / "severe_case_selection_recovery.csv", index=False)
    with open(OUT_TABLES_ROBUSTNESS / "severe_case_selection_recovery.tex", "w", encoding="utf-8") as fh:
        fh.write(recovery.to_latex(index=False, escape=True,
                 caption="Orphaned for-cause delistings recovered by an alternative "
                         "fiscal-year-end-anchored window.",
                 label="tab:severe_selection_recovery"))

    print("\n=== Firm-year vs unique-event accounting (frozen test split) ===")
    print(dedup.to_string(index=False))
    dedup.to_csv(OUT_TABLES_ROBUSTNESS / "severe_case_selection_dedup.csv", index=False)
    with open(OUT_TABLES_ROBUSTNESS / "severe_case_selection_dedup.tex", "w", encoding="utf-8") as fh:
        fh.write(dedup.to_latex(index=False, escape=True,
                 caption="Firm-year versus unique-event accounting of the frozen 10-K "
                         "label (F) and the alternative fiscal-year-end label (D, 730-day "
                         "reach) on the held-out test split. Adjacent 730-day windows "
                         "overlap by roughly one year, so a single delisting can label "
                         "two consecutive firm-years under D; the decomposition separates "
                         "this mechanical double-labelling from genuinely recovered "
                         "went-dark events.",
                 label="tab:severe_selection_dedup"))

    for name in ("severe_case_selection_footprint", "severe_case_selection_recovery",
                 "severe_case_selection_dedup"):
        print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / (name + '.csv')}")
        print(f"Saved -> {OUT_TABLES_ROBUSTNESS / (name + '.tex')}")


if __name__ == "__main__":
    main()
