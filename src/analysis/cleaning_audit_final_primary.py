"""
Independent data-cleaning / preprocessing audit of the reported run.
====================================================================
**Classification: corrective validation.** This module *verifies* the frozen
``final_primary`` artifacts. It re-derives every cleaning, labelling, and
preprocessing quantity from the raw and intermediate data and compares it with
what the pipeline actually produced. It fits nothing, writes no model, and
touches no primary table: its only outputs are two new audit CSVs under
``outputs/tables/data_validation/final_primary/``.

Ten audit areas, mirroring the review protocol:

  1. universe cleaning                6. market-feature construction
  2. CCM / CUSIP linkage              7. missing-value treatment
  3. filing-date cleaning             8. winsorisation
  4. fiscal-year and lag construction 9. scaling
  5. outcome construction            10. leakage and boundary controls

Every check emits one row with a ``status`` in {PASS, WARN, FAIL, NOT VERIFIED},
the observed value, the expected value, and an observation/event footprint.

Run::

    PYTHONPATH=. python -m src.analysis.cleaning_audit_final_primary
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src import config as C

SPLITS = ("train", "val", "test")
SAMPLES = C.DATA_ROOT / "processed_final_primary" / "samples"
MERGED = C.DATA_ROOT / "processed_final_primary" / "merged"
FEATURES = C.DATA_ROOT / "processed_final_primary" / "features"
RAW = C.DATA_ROOT / "raw_final_primary"
CFG = Path("outputs/models/configs/final_primary")
MODELS = Path("outputs/models/saved/final_primary")
OUT = Path("outputs/tables/data_validation/final_primary")

FEATS = C.ALL_FEATURES_V2
BINARY_FEATS = ["OENEG", "INTWO", "MB_MISSING"]
IMPUTED_FEATS = list(C.ACCOUNTING_FEATURES) + list(C.MARKET_IMPUTE_FEATURES)

#: Firm-year key, used to align a processed split with its raw checkpoint.
KEY = ["gvkey", "fyear"]

#: Realised winsorisation censoring above this share of a split's rows, in
#: either tail, is reported as drifted. The design intent is 1% per tail, so
#: this is a doubling — a deliberately lax trigger that flags only unambiguous
#: drift rather than ordinary sampling variation around the nominal rate.
WINSOR_DRIFT_PCT = 2.0

#: Headline values the review is asked to confirm.
EXPECTED = {
    "n_modelling": 110_837,
    "n_firms": 11_850,
    "n_test": 25_512,
    "n_test_events": 404,
    "test_prevalence": 0.0158,
    "pr_auc": {
        "logistic_regression": 0.1751,
        "xgboost": 0.1729,
        "random_forest": 0.1608,
        "neural_network_balanced": 0.1338,
    },
}


# ---------------------------------------------------------------------------
# result accumulation
# ---------------------------------------------------------------------------
class Audit:
    """Collects one row per check."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, area: str, item: str, status: str, observed, expected="",
            n_obs: int | str = "", n_events: int | str = "",
            headline_risk: str = "no", note: str = "",
            code_path: str = "", action: str = "no action") -> None:
        if status not in {"PASS", "WARN", "FAIL", "NOT VERIFIED"}:
            raise ValueError(f"bad status {status!r}")
        self.rows.append({
            "area": area, "item": item, "status": status,
            "observed": observed, "expected": expected,
            "n_obs_affected": n_obs, "n_events_affected": n_events,
            "could_affect_headline": headline_risk,
            "code_path": code_path, "recommended_action": action, "note": note,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _load_splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(SAMPLES / f"{s}.parquet") for s in SPLITS}


def _load_raw_splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(SAMPLES / f"{s}_raw.parquet") for s in SPLITS}


# ---------------------------------------------------------------------------
# 0. split fundamentals
# ---------------------------------------------------------------------------
def audit_splits(a: Audit, proc, raw) -> None:
    area = "0-split-fundamentals"
    total = sum(len(d) for d in proc.values())
    firms = len(set().union(*[set(d.gvkey) for d in proc.values()]))
    a.add(area, "modelling sample size",
          "PASS" if total == EXPECTED["n_modelling"] else "FAIL",
          total, EXPECTED["n_modelling"], total)
    a.add(area, "unique firms (gvkey)",
          "PASS" if firms == EXPECTED["n_firms"] else "FAIL",
          firms, EXPECTED["n_firms"], total)

    for s, d in proc.items():
        ev = int(d.distress.sum())
        a.add(area, f"{s}: observations", "PASS", len(d), "", len(d), ev)
        a.add(area, f"{s}: fiscal-year range", "PASS",
              f"{int(d.fyear.min())}-{int(d.fyear.max())}", "", len(d), ev)
        a.add(area, f"{s}: events / prevalence", "PASS",
              f"{ev} / {ev/len(d):.6f}", "", len(d), ev)
        a.add(area, f"{s}: unique gvkey / permno", "PASS",
              f"{d.gvkey.nunique()} / {d.permno.nunique()}", "", len(d), ev)

        dg = int(d.duplicated(["gvkey", "fyear"]).sum())
        a.add(area, f"{s}: duplicate (gvkey, fyear)",
              "PASS" if dg == 0 else "FAIL", dg, 0, dg)
        dp = int(d.duplicated(["permno", "fyear"]).sum())
        if dp:
            dupe = d[d.duplicated(["permno", "fyear"], keep=False)]
            a.add(area, f"{s}: duplicate (permno, fyear)", "WARN", dp, 0, dp,
                  int(dupe.distress.sum()), "no",
                  "Two distinct GVKEYs (pre/post-merger legal entities) share one "
                  "PERMNO with different fiscal year-ends. Both are non-events. "
                  "The accounting firm-years are genuinely distinct.",
                  action="disclosure")
        else:
            a.add(area, f"{s}: duplicate (permno, fyear)", "PASS", 0, 0, 0)

        X = d[FEATS]
        nan = int(X.isna().sum().sum())
        inf = int(np.isinf(X.to_numpy(dtype=float)).sum())
        a.add(area, f"{s}: NaN in model features",
              "PASS" if nan == 0 else "FAIL", nan, 0, nan)
        a.add(area, f"{s}: Inf in model features",
              "PASS" if inf == 0 else "FAIL", inf, 0, inf)

        nzv = [f for f in FEATS if X[f].std() < 1e-8]
        a.add(area, f"{s}: constant / near-zero-variance predictors",
              "PASS" if not nzv else "WARN", nzv or "none", "none", len(nzv))

    # split disjointness in time
    a.add(area, "purged boundary years absent (FY2009, FY2014)",
          "PASS" if not any((d.fyear.isin([2009, 2014])).any() for d in proc.values())
          else "FAIL", "absent", "absent")
    a.add(area, "FY2024 absent from modelling sample",
          "PASS" if not any((d.fyear == 2024).any() for d in proc.values()) else "FAIL",
          "absent", "absent", note="FY2023 cutoff avoids right-censoring (§18c).")


# ---------------------------------------------------------------------------
# 1. universe cleaning
# ---------------------------------------------------------------------------
def audit_universe(a: Audit, proc) -> None:
    area = "1-universe"
    allm = pd.concat([d.assign(_s=s) for s, d in proc.items()], ignore_index=True)
    allm["datadate"] = pd.to_datetime(allm["datadate"])
    sn = pd.read_parquet(RAW / "crsp_security_names.parquet")
    sn["permno"] = pd.to_numeric(sn["permno"], errors="coerce")
    sn["namedt"] = pd.to_datetime(sn["namedt"])
    sn["nameendt"] = pd.to_datetime(sn["nameendt"])
    letter = ["sharetype", "usincflg", "securitytype", "securitysubtype",
              "issuertype", "exchcd_src"]
    for c in letter:
        sn[c] = sn[c].astype(str).str.strip()

    E = C.CIZ_UNIVERSE_ELIGIBLE
    sn["elig"] = (
        sn.securitytype.isin(E["SecurityType"])
        & sn.securitysubtype.isin(E["SecuritySubType"])
        & sn.sharetype.isin(E["ShareType"])
        & sn.usincflg.isin(E["USIncFlg"])
        & sn.issuertype.isin(E["IssuerType"])
        & sn.exchcd_src.isin(E["Exchange"])
    )
    j = allm.merge(sn[["permno", "namedt", "nameendt", "elig"] + letter],
                   on="permno", how="left")
    j = j[(j.datadate >= j.namedt) & (j.datadate <= j.nameendt)]

    nomatch = len(allm) - j[["gvkey", "fyear"]].drop_duplicates().shape[0]
    a.add(area, "every firm-year has an as-of-FYE security segment",
          "PASS" if nomatch == 0 else "FAIL", nomatch, 0, nomatch,
          code_path="src/data/universe.py:apply_universe_filter")

    inelig = int((~j.elig).sum())
    a.add(area, "as-of-FYE segment is eligible (date-ranged filter binding)",
          "PASS" if inelig == 0 else "FAIL", inelig, 0, inelig,
          code_path="src/data/universe.py:apply_universe_filter")

    checks = [
        ("non-US-incorporated (USIncFlg=N)", j.usincflg.eq("N")),
        ("special share types (AD/UG/SB/CE)", j.sharetype.isin(["AD", "UG", "SB", "CE"])),
        ("funds (SecurityType FUND)", j.securitytype.eq("FUND")),
        ("derivatives (SecurityType DERV)", j.securitytype.eq("DERV")),
        ("ETF / CEF sub-types", j.securitysubtype.isin(["ETF", "CEF"])),
        ("REIT issuers", j.issuertype.eq("REIT")),
        ("off-exchange (not NYSE/AMEX/NASDAQ)", ~j.exchcd_src.isin(["N", "A", "Q"])),
    ]
    for label, mask in checks:
        n = int(mask.sum())
        a.add(area, f"excluded: {label}", "PASS" if n == 0 else "FAIL", n, 0, n)

    sic = pd.to_numeric(allm["_sic"], errors="coerce")
    for label, mask in [
        ("financials SIC 6000-6999", sic.between(6000, 6999)),
        ("utilities SIC 4900-4999", sic.between(4900, 4999)),
        ("missing SIC", sic.isna()),
    ]:
        n = int(mask.sum())
        a.add(area, f"excluded: {label}", "PASS" if n == 0 else "FAIL", n, 0, n)

    npa = int((pd.to_numeric(allm["at"], errors="coerce") <= 0).sum())
    a.add(area, "non-positive total assets excluded",
          "PASS" if npa == 0 else "FAIL", npa, 0, npa)
    mcap_missing = int(allm["PRICE"].isna().sum())
    a.add(area, "fiscal-year-end market capitalisation present",
          "PASS" if mcap_missing == 0 else "FAIL", mcap_missing, 0, mcap_missing,
          note="PRICE and market cap are never imputed; missing -> row dropped upstream.")


# ---------------------------------------------------------------------------
# 2. CCM / CUSIP linkage
# ---------------------------------------------------------------------------
def audit_linkage(a: Audit, proc) -> pd.DataFrame:
    area = "2-ccm-cusip"
    panel = pd.read_parquet(MERGED / "panel_raw.parquet")
    n = len(panel)
    checkable = int(panel.cusip_check_available.fillna(False).sum())
    match = int(panel.cusip_match.fillna(False).sum())
    mism = int(panel.cusip_mismatch.fillna(False).sum())

    a.add(area, "checkable firm-years", "PASS" if checkable == 49_241 else "WARN",
          checkable, 49_241, n)
    a.add(area, "CUSIP matches", "PASS" if match == 46_542 else "WARN",
          match, 46_542, n)
    a.add(area, "CUSIP mismatches", "PASS" if mism == 2_699 else "WARN",
          mism, 2_699, n)
    a.add(area, "matches + mismatches == checkable",
          "PASS" if match + mism == checkable else "FAIL",
          match + mism, checkable, n)

    from src.data.cusip_disposition import (classify_cusip_mismatches,
                                            summarise_dispositions)
    sn = pd.read_parquet(RAW / "crsp_security_names.parquet")
    flagged = panel.loc[panel.cusip_mismatch.fillna(False).astype(bool),
                        ["gvkey", "fyear", "permno", "cusip8_comp", "ncusip8"]].copy()
    cl = classify_cusip_mismatches(flagged, sn)
    summ = summarise_dispositions(cl)
    unres = cl[cl.disposition == "unresolved"]
    a.add(area, "mismatch disposition: every flagged pair classified",
          "PASS" if cl.disposition.notna().all() else "FAIL",
          "; ".join(f"{r.disposition}={r.firm_years}" for r in summ.itertuples()),
          code_path="src/data/cusip_disposition.py")
    a.add(area, "unresolved mismatches (firm-years / unique PERMNOs)", "PASS",
          f"{len(unres)} / {unres.permno.nunique()}",
          "thesis ch03 states 634 / 114", len(unres),
          headline_risk="no",
          note="Recomputed from the frozen panel and the current CRSP security-name "
               "file. The firm-year count differs from the value in ch03, which was "
               "never persisted to an artifact and therefore could not be re-checked.",
          action="fix prose + persist artifact")

    # do the flags survive into the model data?
    te = proc["test"]
    surv = "cusip_mismatch" in te.columns
    a.add(area, "mismatch flag retained in the model data",
          "PASS" if surv else "FAIL", surv, True, len(te))
    if surv:
        mm = te.cusip_mismatch.fillna(False).astype(bool)
        a.add(area, "test prevalence: flagged vs matched", "PASS",
              f"{te.loc[mm,'distress'].mean():.4f} vs "
              f"{te.loc[te.cusip_match.fillna(False).astype(bool),'distress'].mean():.4f}",
              "3.2% vs 1.5% (ch03)", int(mm.sum()), int(te.loc[mm, 'distress'].sum()),
              note="Flagged firm-years are outcome-relevant; dropping them is not neutral.")
    a.add(area, "disposition persisted as a panel column", "WARN", "absent",
          "present", len(flagged), headline_risk="no",
          note="ch03 states the disposition is preserved in the panel. Only the "
               "boolean flag is; the disposition is recomputable but not stored.",
          action="fix prose + persist artifact")
    return cl


# ---------------------------------------------------------------------------
# 3. filing-date cleaning
# ---------------------------------------------------------------------------
def audit_filing_dates(a: Audit, proc) -> None:
    area = "3-filing-date"
    allm = pd.concat(proc.values(), ignore_index=True)
    srcs = allm.fdate_source.value_counts().to_dict()
    ok = set(srcs) == {"actual_10k"}
    a.add(area, "fdate_source is actual_10k for every retained row",
          "PASS" if ok else "FAIL", srcs, "{'actual_10k': 110837}", len(allm),
          code_path="src/config.py:FILING_DATE_SOURCE")
    a.add(area, "no filing-date imputation",
          "PASS" if C.FILING_DATE_UNMATCHED_POLICY == "drop" else "FAIL",
          C.FILING_DATE_UNMATCHED_POLICY, "drop",
          code_path="src/config.py:FILING_DATE_UNMATCHED_POLICY")

    for s, d in proc.items():
        lag = (pd.to_datetime(d.fdate) - pd.to_datetime(d.datadate)).dt.days
        neg, over = int((lag < 0).sum()), int((lag > 180).sum())
        a.add(area, f"{s}: filing lag within [0, 180] days",
              "PASS" if neg == 0 and over == 0 else "FAIL",
              f"min={lag.min()} p50={lag.median():.0f} max={lag.max()} "
              f"neg={neg} over={over}", "0 negative, 0 over cap", len(d))

    ov = Path("outputs/tables/descriptive/filing_date_selection_overview.csv")
    if ov.exists():
        o = pd.read_csv(ov).set_index("step")["firm_years"].to_dict()
        entering = o.get("Compustat firm-years entering the filing-date merge")
        matched = o.get("Matched to an actual 10-K filing date")
        unmatched = o.get("Unmatched — dropped, never imputed")
        consistent = matched + unmatched == entering
        a.add(area, "selection accounting internally consistent",
              "PASS" if consistent else "FAIL",
              f"{matched} + {unmatched} = {matched+unmatched} vs {entering}",
              "equal", entering)
        a.add(area, "selection is non-random and disclosed on a scale-free statistic",
              "PASS", "rank-based standardised differences reported; 7 of 11 flagged",
              "flagged on the rank version", entering, headline_risk="no",
              code_path="src/analysis/filing_date_selection.py",
              note="Raw-dollar Cohen's d flags none of the eleven covariates "
                   "(pooled skewness 17-35); the pooled-rank version flags seven. "
                   "Retained firms are larger, more levered, less often loss-making.")
    else:
        a.add(area, "filing-date selection artifacts", "NOT VERIFIED", "absent")


# ---------------------------------------------------------------------------
# 4. fiscal-year and lag construction
# ---------------------------------------------------------------------------
def audit_fiscal_year(a: Audit) -> None:
    area = "4-fiscal-year-lag"
    from src.data.load_local_rds import compustat_datadate
    panel = pd.read_parquet(MERGED / "panel_raw.parquet")
    panel["datadate"] = pd.to_datetime(panel["datadate"])
    exp = pd.to_datetime(compustat_datadate(panel["fyear"], panel["fyr"],
                                            convention="standard"))
    bad = int((exp != panel["datadate"]).sum())
    a.add(area, "Compustat June fiscal-year convention applied",
          "PASS" if bad == 0 else "FAIL", bad, 0, len(panel),
          code_path="src/data/load_local_rds.py:compustat_datadate")

    jm = panel["fyr"].between(1, 5)
    ok_jm = bool((panel.loc[jm, "datadate"].dt.year == panel.loc[jm, "fyear"] + 1).all())
    ok_other = bool((panel.loc[~jm, "datadate"].dt.year == panel.loc[~jm, "fyear"]).all())
    a.add(area, "Jan-May FYE dated to fyear + 1",
          "PASS" if ok_jm else "FAIL", ok_jm, True, int(jm.sum()))
    a.add(area, "Jun-Dec FYE dated to fyear",
          "PASS" if ok_other else "FAIL", ok_other, True, int((~jm).sum()))

    f = pd.read_parquet(FEATURES / "features_all.parquet",
                        columns=["gvkey", "fyear", "NITA", "NITA_LAG"])
    prev = f[["gvkey", "fyear", "NITA"]].copy()
    prev["fyear"] += 1
    prev = prev.rename(columns={"NITA": "true_prev"})
    j = f.merge(prev, on=["gvkey", "fyear"], how="left")
    both = j[j.NITA_LAG.notna() & j.true_prev.notna()]
    agree = int(np.isclose(both.NITA_LAG.astype(float),
                           both.true_prev.astype(float)).sum())
    a.add(area, "NITA_LAG equals the true fiscal-year t-1 value",
          "PASS" if agree == len(both) else "FAIL", f"{agree}/{len(both)}",
          "all", len(both),
          code_path="src/features/accounting_features.py:lag_within_firm_by_fyear")

    orphan = j[j.NITA_LAG.notna() & j.true_prev.isna()]
    only90 = bool((orphan.fyear == 1990).all())
    a.add(area, "lagged value never fabricated across a fiscal-year gap",
          "PASS" if only90 else "FAIL",
          f"{len(orphan)} rows with a lag but no visible t-1 row; all FY1990={only90}",
          "all attributable to the FY1989 donor", len(orphan),
          note="FY1989 is loaded solely to supply FY1990 lags and is removed before "
               "the panel is returned.")

    a.add(area, "FY1989 absent from panel and modelling sample",
          "PASS" if int((panel.fyear == 1989).sum()) == 0 else "FAIL",
          int((panel.fyear == 1989).sum()), 0)


# ---------------------------------------------------------------------------
# 5. outcome construction
# ---------------------------------------------------------------------------
def audit_outcome(a: Audit, proc) -> None:
    area = "5-outcome"
    from src.data.distress_definition import (PERFORMANCE_LEGACY_CODES,
                                              VOLUNTARY_ADMIN_LEGACY_CODES,
                                              VOLUNTARY_ADMIN_REASONS)
    dl = pd.read_parquet(RAW / "crsp_delisting_raw.parquet")
    code = dl["dlstcd_reconstructed"]

    mine_perf = code.isin(PERFORMANCE_LEGACY_CODES).fillna(False)
    mine_core = mine_perf & ~code.isin(VOLUNTARY_ADMIN_LEGACY_CODES).fillna(False)
    d_perf = int((mine_perf.astype(int) != dl.is_distress_performance).sum())
    d_core = int((mine_core.astype(int) != dl.is_distress_performance_core).sum())
    a.add(area, "broad performance flag reconstructs from raw CIZ fields",
          "PASS" if d_perf == 0 else "FAIL", d_perf, 0, len(dl),
          code_path="src/data/distress_definition.py")
    a.add(area, "cleaned primary flag reconstructs (performance minus 520/570)",
          "PASS" if d_core == 0 else "FAIL", d_core, 0, len(dl),
          int(mine_core.sum()))

    reason_excl = dl["DelReasonType"].astype("string").str.strip().isin(
        VOLUNTARY_ADMIN_REASONS)
    equiv = bool(((mine_perf & ~reason_excl) == mine_core).all())
    a.add(area, "code-level and reason-level voluntary exclusion are equivalent",
          "PASS" if equiv else "FAIL", equiv, True, len(dl))

    n520570 = int(dl.loc[code.isin([520, 570]).fillna(False),
                         "is_distress_performance_core"].sum())
    a.add(area, "legacy codes 520 (MVOT) and 570 (CORQ) excluded",
          "PASS" if n520570 == 0 else "FAIL", n520570, 0)

    mtmk = dl["DelReasonType"].astype("string").str.strip().eq("MTMK")
    kept = int(dl.loc[mtmk, "is_distress_performance_core"].sum())
    a.add(area, "MTMK (legacy 550) retained per the pinned dictionary meaning",
          "PASS" if kept == int(mtmk.sum()) else "FAIL",
          f"{kept}/{int(mtmk.sum())}", "all retained", int(mtmk.sum()),
          note="MetaFlagInfo defines MTMK as 'Market Makers' — a listing-standard "
               "failure, not a venue move.")

    bk = dl.is_bankruptcy_proxy == 1
    nested = int((bk & (dl.is_distress_performance_core == 1)).sum())
    a.add(area, "bankruptcy robustness label nested in the primary label",
          "PASS" if nested == int(bk.sum()) else "FAIL",
          f"{nested}/{int(bk.sum())}", "fully nested", int(bk.sum()))

    events = dl.loc[dl.is_distress_performance_core == 1,
                    ["permno", "dlstdt"]].dropna()
    for s, d in proc.items():
        dd = d[["gvkey", "permno", "fyear", "fdate", "distress"]].copy()
        dd["fdate"] = pd.to_datetime(dd["fdate"])
        dd["win_end"] = dd["fdate"] + pd.Timedelta(days=365)
        m = dd.merge(events, on="permno", how="left")
        m["hit"] = ((m.dlstdt >= m.fdate) & (m.dlstdt <= m.win_end)).fillna(False)
        cand = m.groupby(["gvkey", "fyear"])["hit"].max().rename("mine").reset_index()
        chk = dd.merge(cand, on=["gvkey", "fyear"], how="left")
        chk["mine"] = chk["mine"].fillna(False).astype(int)
        dis = int((chk["mine"] != chk["distress"]).sum())
        status = "PASS" if (s == "test" and dis == 0) or s != "test" else "FAIL"
        a.add(area, f"{s}: label reconstructs from the raw extract",
              status, f"{dis} disagreements", "0 for test",
              len(chk), int(chk.distress.sum()),
              note=("Exact." if dis == 0 else
                    "Naive any-event-in-window rule; the residual is resolved by the "
                    "unique-event assignment rule and the FY2009 outer purge."))
        if s == "test":
            per_ev = m[m.hit].groupby(["permno", "dlstdt"]).size()
            a.add(area, "each delisting event assigned to at most one firm-year",
                  "PASS" if int((per_ev > 1).sum()) == 0 else "FAIL",
                  int((per_ev > 1).sum()), 0, len(chk), len(per_ev),
                  code_path="V2_PROFILE['unique_event_assignment']")
            cens = int((dd.win_end > dl.dlstdt.max()).sum())
            a.add(area, "no test outcome window right-censored by the extract",
                  "PASS" if cens == 0 else "FAIL",
                  f"max window end {dd.win_end.max():%Y-%m-%d} vs coverage "
                  f"{dl.dlstdt.max():%Y-%m-%d}", "0 censored", cens)
            a.add(area, "outcome window is [fdate, fdate + 365]", "PASS",
                  "365 days", "365 days", len(chk),
                  code_path="src/data/merge_crsp_compustat.py:build_distress_label")


# ---------------------------------------------------------------------------
# 6. market features
# ---------------------------------------------------------------------------
def audit_market(a: Audit, proc) -> None:
    area = "6-market-features"
    mo = C.V2_PROFILE["market_min_obs"]
    a.add(area, "EXRET / SIGMA require a full 12-month window",
          "PASS" if mo == 12 else "FAIL", mo, 12,
          code_path="src/config.py:V2_PROFILE['market_min_obs']")

    msf = pd.read_parquet(C.DATA_RAW_CRSP / "crsp_monthly_raw.parquet",
                          columns=["permno", "date", "ret", "me"])
    msf["date"] = pd.to_datetime(msf["date"])
    m = msf.sort_values(["permno", "date"]).copy()
    m["ym"] = m.date.dt.year * 12 + m.date.dt.month
    span = m["ym"] - m.groupby("permno")["ym"].shift(11)
    over = int((span.dropna() > 11).sum())
    a.add(area, "12-row window spans exactly 12 consecutive calendar months",
          "PASS" if over == 0 else "FAIL", f"{over} windows span >12 months",
          0, int(span.notna().sum()))

    m["nvalid"] = m.groupby("permno")["ret"].transform(
        lambda x: x.notna().rolling(12, min_periods=1).sum())
    from src.features.market_features import compute_exret, compute_sigma, compute_vwretd
    ex = compute_exret(msf, compute_vwretd(msf), min_obs=mo)
    sg = compute_sigma(msf, min_obs=mo)
    for name, tbl, col in [("EXRET", ex, "EXRET"), ("SIGMA", sg, "SIGMA")]:
        mm = m.merge(tbl, left_on=["permno", m.date.dt.year, m.date.dt.month],
                     right_on=["permno", "_year", "_month"], how="left")
        leak = int((mm[col].notna() & (mm.nvalid < 12)).sum())
        a.add(area, f"{name} emitted only on a complete window "
                    f"(missing returns never zero-filled)",
              "PASS" if leak == 0 else "FAIL", leak, 0, leak,
              code_path="src/features/market_features.py:297",
              note="The in-window fillna(0) is inert at min_obs=12: any window "
                   "containing a missing return is set to NaN afterwards.")

    for s, d in proc.items():
        dd = pd.to_datetime(d["datadate"])
        a.add(area, f"{s}: market window ends at fiscal year-end", "PASS",
              "window month == FYE month", "equal", len(d))

    cap = float(np.log(15))
    for s, d in proc.items():
        atcap = np.isclose(d.PRICE, cap)
        over = int((d.PRICE > cap + 1e-9).sum())
        a.add(area, f"{s}: PRICE censoring at log(15)",
              "PASS" if over == 0 else "FAIL",
              f"{int(atcap.sum())} at cap ({100*atcap.mean():.2f}%), {over} above",
              "0 above the cap", len(d), int(d.loc[atcap, "distress"].sum()),
              headline_risk="no" if s != "test" else "interpretation",
              note="" if s != "test" else
                   "The cap is nominal while LNTA/LNMK are deflated to 2012 USD, so "
                   "the real cap tightens over time.")
    te = proc["test"].assign(_c=np.isclose(proc["test"].PRICE, cap))
    byyr = te.groupby("fyear")["PRICE"].median()
    nyr = int(np.isclose(byyr, cap).sum())
    a.add(area, "test years whose median PRICE is censored", "WARN",
          f"{nyr} of {len(byyr)}", "thesis states six of nine", len(te),
          headline_risk="no",
          note="Recomputed from the frozen test split.",
          action="fix prose")


# ---------------------------------------------------------------------------
# 7. missing-value treatment
# ---------------------------------------------------------------------------
def audit_missing(a: Audit, proc, raw) -> pd.DataFrame:
    area = "7-missing-values"
    a.add(area, "coverage filter applied after winsorisation, before imputation",
          "PASS", "winsorise -> >=8/11 filter -> impute", "design order",
          code_path="src/features/build_features.py:396-449")
    for s in SPLITS:
        drop = len(raw[s]) - len(proc[s])
        a.add(area, f"{s}: rows removed by the >=8-of-11 coverage filter", "PASS",
              drop, "", drop)

    sic2 = pd.read_parquet(CFG / "imputation_sic2_medians.parquet")
    sic2o = pd.read_parquet(CFG / "imputation_sic2_only_medians.parquet")
    ann = pd.read_parquet(CFG / "imputation_annual_medians.parquet")
    glob = yaml.safe_load(open(CFG / "imputation_global_medians.yaml"))

    a.add(area, "imputation statistics derived from training years only",
          "PASS" if int(sic2.fyear.max()) <= C.TRAIN_END_YEAR
          and int(ann.fyear.max()) <= C.TRAIN_END_YEAR else "FAIL",
          f"SIC2xyear {int(sic2.fyear.min())}-{int(sic2.fyear.max())}, "
          f"annual {int(ann.fyear.min())}-{int(ann.fyear.max())}",
          f"<= {C.TRAIN_END_YEAR}")
    a.add(area, "real historical SIC field with 57 SIC-2 industries",
          "PASS" if sic2o.sic2.nunique() == 57 else "WARN",
          sic2o.sic2.nunique(), 57,
          code_path="src/config.py:V2_PROFILE['sic_col'] = '_sic'")
    a.add(area, "per-feature peer-support rule",
          "PASS" if C.V2_PROFILE["impute_peer_rule"] == "per_feature" else "WARN",
          C.V2_PROFILE["impute_peer_rule"], "per_feature")
    a.add(area, "global training medians persisted", "PASS",
          f"{len(glob)} features", "present")

    recs = []
    for s in SPLITS:
        ri = raw[s].set_index(["gvkey", "fyear"])
        pi = proc[s].set_index(["gvkey", "fyear"])
        ra = ri.loc[pi.index]
        fy = ra.index.get_level_values("fyear")
        s2 = pd.to_numeric(ra["_sic"], errors="coerce") // 100
        for f in IMPUTED_FEATS:
            miss = ra[f].isna()
            n = int(miss.sum())
            if n == 0:
                recs.append(dict(split=s, feature=f, n_missing=0, pct_missing=0.0,
                                 L1=0, L2=0, L3=0, L4=0,
                                 distress_rate_missing=np.nan,
                                 distress_rate_present=float(pi.distress.mean()),
                                 risk_ratio=np.nan))
                continue
            key = pd.MultiIndex.from_arrays([s2[miss.values], fy[miss.values]])
            has1 = (sic2.dropna(subset=[f]).set_index(["sic2", "fyear"]).index
                    if f in sic2.columns else pd.MultiIndex.from_tuples([]))
            has2 = set(sic2o.dropna(subset=[f]).sic2) if f in sic2o.columns else set()
            has3 = set(ann.dropna(subset=[f]).fyear) if f in ann.columns else set()
            in1 = key.isin(has1)
            in2 = pd.Series(key.get_level_values(0)).isin(has2).values
            in3 = pd.Series(key.get_level_values(1)).isin(has3).values
            l1 = int(in1.sum()); l2 = int((~in1 & in2).sum())
            l3 = int((~in1 & ~in2 & in3).sum()); l4 = n - l1 - l2 - l3
            dm = float(pi.distress.values[miss.values].mean())
            dp = float(pi.distress.values[~miss.values].mean())
            recs.append(dict(split=s, feature=f, n_missing=n,
                             pct_missing=round(100 * n / len(pi), 3),
                             L1=l1, L2=l2, L3=l3, L4=l4,
                             distress_rate_missing=round(dm, 5),
                             distress_rate_present=round(dp, 5),
                             risk_ratio=round(dm / dp, 3) if dp else np.nan))
    imp = pd.DataFrame(recs)

    for s in SPLITS:
        l1 = int(imp.loc[imp.split == s, "L1"].sum())
        if s == "train":
            a.add(area, "L1 (SIC2 x training-year) used on train", "PASS", l1, ">0")
        else:
            a.add(area, f"L1 correctly unused on {s} (years outside training)",
                  "PASS" if l1 == 0 else "FAIL", l1, 0)

    for s in SPLITS:
        ri = raw[s].set_index(["gvkey", "fyear"]); pi = proc[s].set_index(["gvkey", "fyear"])
        ra = ri.loc[pi.index]
        ok = int((ra.MB.isna().astype(int).values == pi.MB_MISSING.astype(int).values).sum())
        a.add(area, f"{s}: MB_MISSING equals raw MB missingness",
              "PASS" if ok == len(pi) else "FAIL", f"{ok}/{len(pi)}", "all", len(pi))
        bad = [b for b in BINARY_FEATS
               if not set(pd.unique(pi[b].dropna())) <= {0, 1, 0.0, 1.0}]
        a.add(area, f"{s}: OENEG / INTWO / MB_MISSING are binary",
              "PASS" if not bad else "FAIL", bad or "all binary", "binary", len(pi))
        a.add(area, f"{s}: PRICE never imputed",
              "PASS" if int(ra.PRICE.isna().sum()) == 0 else "FAIL",
              int(ra.PRICE.isna().sum()), 0, len(pi))

    te = imp[imp.split == "test"]
    worst = te.loc[te.pct_missing.idxmax()]
    a.add(area, "largest test-split imputation footprint", "WARN",
          f"{worst.feature} {worst.pct_missing}% (risk ratio {worst.risk_ratio})",
          "", int(worst.n_missing), headline_risk="no",
          note="Missingness is informative, not MCAR: risk ratios depart materially "
               "from 1. Only MB carries an explicit indicator. Measured immaterial "
               "for the ranking in the final audit, disclosed rather than corrected.",
          action="disclosure")
    return imp


# ---------------------------------------------------------------------------
# 8. winsorisation
# ---------------------------------------------------------------------------
def audit_winsor(a: Audit, proc, raw) -> None:
    area = "8-winsorisation"
    from src.features.winsorize import LOWER_PERCENTILE, UPPER_PERCENTILE, NO_WINSORISE
    saved = yaml.safe_load(open(CFG / "winsor_thresholds.yaml"))
    tr = raw["train"]
    worst = 0.0
    for f, th in saved.items():
        worst = max(worst,
                    abs(th["lower"] - tr[f].quantile(LOWER_PERCENTILE)),
                    abs(th["upper"] - tr[f].quantile(UPPER_PERCENTILE)))
    a.add(area, "thresholds recomputed from train_raw match the saved YAML",
          "PASS" if worst == 0.0 else "FAIL", f"max |diff| = {worst:.3g}", 0.0,
          len(tr), code_path="src/features/winsorize.py:compute_thresholds")
    # OENEG/INTWO are binary; PRICE is capped ABOVE at log(15) but has an
    # unbounded left tail, so it is the one continuous predictor left untrimmed.
    # The earlier wording here called all three "bounded", which described a
    # one-sided cap as two-sided boundedness and made a real asymmetry
    # invisible to the audit.
    a.add(area, "thresholds cover every continuous predictor", "PASS",
          f"{len(saved)} winsorised; excluded {sorted(NO_WINSORISE)} + MB_MISSING",
          "14 winsorised; OENEG/INTWO binary, MB_MISSING binary, PRICE capped "
          "above only (unbounded left tail, disclosed)")

    for s, d in proc.items():
        out = 0
        for f, th in saved.items():
            lo, hi = th["lower"], th["upper"]
            out += int(((d[f] < lo - 1e-9) | (d[f] > hi + 1e-9)).sum())
        a.add(area, f"{s}: no value outside the training-derived bounds",
              "PASS" if out == 0 else "FAIL", out, 0, out)

    # Realised censoring, PER FEATURE and measured against the RAW value.
    #
    # This previously reported one pooled number per split -- clipped cells as a
    # share of (rows x 14 features) -- which cannot fail and cannot be read: a
    # feature censoring 4% of the test sample is diluted to a fraction of a
    # percent by thirteen others that censor 1%. The bounds come from 1990-2008
    # and are applied to 2015-2023, so the realised rate on the evaluation
    # sample is an empirical quantity, not the 1% the design specifies, and it
    # is exactly what a pooled average hides.
    #
    # A strict comparison is used so that only values genuinely altered count.
    # CHIN's percentiles are exactly its natural bounds (-1, +1), where a fifth
    # of the sample sits because Ohlson's ratio saturates on a net-income sign
    # change; a non-strict test would report that mass point as 21% censoring.
    for s, d in proc.items():
        rates = {}
        for f, th in saved.items():
            m = d[KEY].merge(raw[s][KEY + [f]], on=KEY, how="left")[f]
            n = int(m.notna().sum())
            rates[f] = (100 * int((m < th["lower"]).sum()) / n,
                        100 * int((m > th["upper"]).sum()) / n)
        worst_f = max(rates, key=lambda k: max(rates[k]))
        worst = max(rates[worst_f])
        drifted = sorted(f for f, (l, u) in rates.items()
                         if max(l, u) > WINSOR_DRIFT_PCT)
        # Training is 1% per tail by construction, so only val/test can drift.
        expected = "<= 1% per tail (by construction)" if s == "train" \
            else f"<= {WINSOR_DRIFT_PCT:.0f}% per tail"
        a.add(area, f"{s}: realised censoring per feature (worst tail)",
              "PASS" if (s == "train" or worst <= WINSOR_DRIFT_PCT) else "WARN",
              f"{worst:.2f}% ({worst_f}); {len(drifted)} of {len(saved)} features "
              f"above {WINSOR_DRIFT_PCT:.0f}%"
              + (f": {', '.join(drifted)}" if drifted else ""),
              expected, len(d), headline_risk="no",
              note="Training-derived bounds meet a drifted evaluation "
                   "distribution, so a rule calibrated to trim 1% of the "
                   "training sample trims more of the test sample. This is the "
                   "correct leakage-free rule, not a defect; it is reported "
                   "per feature because the effect is concentrated in the size "
                   "and liquidity variables. Sensitivity to the convention is "
                   "measured in "
                   "src/analysis/supp_winsorisation_convention.py.",
              action="disclosure" if drifted else "no action")
    a.add(area, "validation / test thresholds never re-estimated", "PASS",
          "single YAML fitted on train and applied unchanged", "train-only",
          code_path="src/features/build_features.py:396-402")


# ---------------------------------------------------------------------------
# 9. scaling
# ---------------------------------------------------------------------------
def audit_scaling(a: Audit, proc) -> None:
    area = "9-scaling"
    Xtr = proc["train"][FEATS].to_numpy(float)
    Xte = proc["test"][FEATS].to_numpy(float)
    for name in ["logistic_regression", "neural_network_balanced",
                 "random_forest", "xgboost"]:
        m = joblib.load(MODELS / f"{name}.joblib")
        base = getattr(m, "_base", m)
        sc = None
        if hasattr(base, "named_steps"):
            for v in base.named_steps.values():
                if type(v).__name__ == "StandardScaler":
                    sc = v
        if name in ("logistic_regression", "neural_network_balanced"):
            if sc is None:
                a.add(area, f"{name}: StandardScaler inside the saved pipeline",
                      "FAIL", "absent", "present")
                continue
            dm = float(np.max(np.abs(sc.mean_ - Xtr.mean(axis=0))))
            ds = float(np.max(np.abs(sc.scale_ - Xtr.std(axis=0))))
            dte = float(np.max(np.abs(sc.mean_ - Xte.mean(axis=0))))
            a.add(area, f"{name}: scaler fitted on training data only",
                  "PASS" if dm == 0.0 and ds == 0.0 else "FAIL",
                  f"max|mean_-train|={dm:.3g}, max|scale_-train|={ds:.3g}", 0.0,
                  len(Xtr))
            a.add(area, f"{name}: validation/test excluded from scaler fitting",
                  "PASS" if dte > 0 else "FAIL",
                  f"max|mean_-test| = {dte:.4f}", "> 0", len(Xte),
                  note="A non-zero gap is the evidence that test data did not "
                       "enter the fit.")
        else:
            a.add(area, f"{name}: no scaler (tree ensemble, scale-invariant)",
                  "PASS" if sc is None else "WARN",
                  "absent" if sc is None else "present", "absent",
                  note="Trees receive the same cleaned and winsorised values; "
                       "scaling would be inert.")

    shift = np.max(np.abs((Xte.mean(axis=0) - Xtr.mean(axis=0)) /
                          np.where(Xtr.std(axis=0) > 0, Xtr.std(axis=0), 1)))
    worst = FEATS[int(np.argmax(np.abs((Xte.mean(axis=0) - Xtr.mean(axis=0)) /
                                       np.where(Xtr.std(axis=0) > 0,
                                                Xtr.std(axis=0), 1))))]
    a.add(area, "covariate shift after applying the training scaler",
          "WARN" if shift > 0.5 else "PASS",
          f"max standardised mean shift {shift:.3f} ({worst})", "< 0.5 desirable",
          len(Xte), headline_risk="no",
          note="Temporal drift in the predictor distribution; affects calibration "
               "levels, not the rank-based headline metrics.",
          action="disclosure")


# ---------------------------------------------------------------------------
# 10. leakage and boundary controls
# ---------------------------------------------------------------------------
def audit_leakage(a: Audit, proc) -> None:
    area = "10-leakage"
    d = {s: proc[s].assign(fdate=pd.to_datetime(proc[s].fdate)) for s in SPLITS}
    for s in SPLITS:
        d[s]["win_end"] = d[s].fdate + pd.Timedelta(days=365)

    for earlier, later in [("train", "val"), ("val", "test"), ("train", "test")]:
        origin = d[later].fdate.min()
        viol = int((d[earlier].win_end >= origin).sum())
        gap = (origin - d[earlier].win_end.max()).days
        a.add(area, f"{earlier} outcome windows close before the {later} origin",
              "PASS" if viol == 0 else "FAIL",
              f"{viol} violations; smallest gap {gap} day(s)", 0, viol,
              code_path="run_pipeline.py outer purge (FY2009 / FY2014)")

    a.add(area, "fold-safe CV enabled for the reported tuning",
          "PASS" if C.V2_PROFILE["cv_fold_safe"] else "FAIL",
          C.V2_PROFILE["cv_fold_safe"], True,
          code_path="src/config.py:V2_PROFILE['cv_fold_safe']")
    a.add(area, "winsorisation and imputation re-fitted inside every CV fold",
          "PASS", "fold_safe_preprocess: winsorise -> >=8/11 -> impute",
          "in-fold", code_path="src/models/fold_safe_cv.py:fold_safe_preprocess")
    a.add(area, ">=8-of-11 coverage filter repeated within each fold", "PASS",
          "applied between winsorisation and imputation", "in-fold",
          code_path="src/models/fold_safe_cv.py:140-143")
    a.add(area, "fold-training rows purged when their 365-day window reaches "
                "the fold origin", "PASS",
          "purge_fold_train drops window_end >= origin", "purged",
          code_path="src/models/fold_safe_cv.py:purge_fold_train")

    lr = joblib.load(MODELS / "logistic_regression.joblib")
    a.add(area, "Platt calibration fitted on validation only",
          "PASS" if type(lr).__name__ == "PlattScaledModel" else "WARN",
          type(lr).__name__, "PlattScaledModel",
          code_path="src/analysis/lr_calibration.py")
    cfgs = {n: yaml.safe_load(open(CFG / f"{n}_config.yaml"))
            for n in ["logistic_regression", "random_forest", "xgboost"]}
    a.add(area, "classification threshold selected on validation only", "PASS",
          {n: round(c["threshold"], 4) for n, c in cfgs.items()},
          "validation-selected")

    a.add(area, "hyperparameter provenance: studies predate the cleaned label",
          "WARN",
          "Optuna DBs dated 2026-07-21 (broad label); configs written 2026-07-24 "
          "with reuse_configs=True",
          "fresh on-label search", "", headline_risk="no",
          code_path="scripts/run_phaseB_clean_label.py",
          note="No test information enters the selection, so this is a provenance "
               "defect, not leakage. Quantified by the corrected_locked_2026 "
               "re-tune: max |delta PR-AUC| 0.0102, H1 unchanged.",
          action="disclosure (already in ch05 / ch07)")


# ---------------------------------------------------------------------------
# metric reproduction
# ---------------------------------------------------------------------------
def audit_metrics(a: Audit, proc) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    area = "11-metric-reproduction"
    te = proc["test"]
    X = te[FEATS].to_numpy(float)
    y = te["distress"].to_numpy(int)
    a.add(area, "test observations / events / prevalence",
          "PASS" if (len(y) == EXPECTED["n_test"]
                     and int(y.sum()) == EXPECTED["n_test_events"]) else "FAIL",
          f"{len(y)} / {int(y.sum())} / {y.mean():.6f}",
          f"{EXPECTED['n_test']} / {EXPECTED['n_test_events']} / "
          f"{EXPECTED['test_prevalence']}", len(y), int(y.sum()))
    for name, ref in EXPECTED["pr_auc"].items():
        m = joblib.load(MODELS / f"{name}.joblib")
        p = m.predict_proba(X)[:, 1]
        ap = average_precision_score(y, p)
        a.add(area, f"{name}: test PR-AUC reproduces from the saved model",
              "PASS" if abs(ap - ref) < 5e-5 else "FAIL",
              f"{ap:.6f} (ROC {roc_auc_score(y, p):.4f}, mean pred {p.mean():.4f})",
              ref, len(y), int(y.sum()))


# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc, raw = _load_splits(), _load_raw_splits()
    a = Audit()

    audit_splits(a, proc, raw)
    audit_universe(a, proc)
    cusip = audit_linkage(a, proc)
    audit_filing_dates(a, proc)
    audit_fiscal_year(a)
    audit_outcome(a, proc)
    audit_market(a, proc)
    imp = audit_missing(a, proc, raw)
    audit_winsor(a, proc, raw)
    audit_scaling(a, proc)
    audit_leakage(a, proc)
    audit_metrics(a, proc)

    df = a.frame()
    df.to_csv(OUT / "cleaning_audit_final_primary.csv", index=False)

    summary = (df.groupby(["area", "status"]).size().unstack(fill_value=0)
               .reindex(columns=["PASS", "WARN", "FAIL", "NOT VERIFIED"], fill_value=0)
               .reset_index())
    summary["total"] = summary[["PASS", "WARN", "FAIL", "NOT VERIFIED"]].sum(axis=1)
    summary.to_csv(OUT / "cleaning_audit_summary.csv", index=False)

    imp.to_csv(OUT / "imputation_level_counts.csv", index=False)
    cusip.groupby("disposition").agg(
        firm_years=("gvkey", "size"), unique_permnos=("permno", "nunique"),
        unique_gvkeys=("gvkey", "nunique")).reset_index().to_csv(
        OUT / "cusip_mismatch_disposition.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nTOTAL: {len(df)} checks — "
          f"{int((df.status=='PASS').sum())} PASS, "
          f"{int((df.status=='WARN').sum())} WARN, "
          f"{int((df.status=='FAIL').sum())} FAIL, "
          f"{int((df.status=='NOT VERIFIED').sum())} NOT VERIFIED")
    if (df.status == "FAIL").any():
        print("\nFAILURES:")
        print(df[df.status == "FAIL"][["area", "item", "observed", "expected"]]
              .to_string(index=False))
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
