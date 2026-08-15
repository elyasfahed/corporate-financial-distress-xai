"""
Supplementary robustness: fiscal-year alignment + delisting-code corrections.
=============================================================================
Purpose
-------
Two data-construction defects were found in the local-RDS extraction layer
(Implementation Status paragraph 17):

  F1  Fiscal-year misdating. datadate was reconstructed as
      month_end(FYEAR, FYR), ignoring the Compustat June rule (fiscal years
      ending Jan--May belong to FYEAR = period-end calendar year - 1). Every
      firm-year with FYR in {1..5} was therefore dated 12 months early:
      its accounting data comes from a fiscal year that ends ~10 months
      AFTER the 10-K filing date the row was matched to (look-ahead), its
      market features are matched 12 months before the true fiscal year-end,
      and its label window is anchored on the PREVIOUS year's filing date.
      Verified on Walmart (gvkey 11259): the FYEAR 2019 row carries the
      balance sheet of the period ending 2020-01-31 (AT = 236,495) paired
      with the 2019-03-28 filing date of the PRIOR year's 10-K.

  F2  Delisting-code mapping. The CIZ StkDelists file has no numeric DLSTCD;
      the synthetic mapping placed bankruptcies (DelReasonType 'BKPY') at
      572 -- OUTSIDE the primary 400--499 label range -- and let liquidations
      (DelActionType 'GLI') fall through to the 500 default. The primary
      distress label therefore excluded the two least ambiguous distress
      classes, and RC1 (bankruptcy-only) was disjoint from the primary label
      instead of the nested subset the frozen design describes.

This script quantifies the materiality of both corrections by re-fitting the
four headline models (LR, RF, XGBoost, balanced NN) with their FROZEN
hyperparameters (loaded from outputs/models/configs/*.yaml, no re-tune)
under four scenarios, varying one dimension at a time:

  S0_frozen     frozen train/val/test parquets, refit as-is
                (reproduction control -- must match the published headline)
  A_fye_aligned full in-memory rebuild of the panel with the corrected
                datadate convention (compustat_datadate 'standard'),
                frozen delisting mapping
  B_label_corr  frozen splits, label rebuilt under the corrected delisting
                mapping (liquidation 400 / dropped 450 / bankruptcy 470,
                all inside the primary 400--499 range)
  AB_both       corrected datadate AND corrected delisting mapping
                (the full correction)

Winsorisation thresholds and imputation medians are recomputed on each
scenario's training sample only (exactly what an adopted rebuild would do),
so every scenario remains leakage-free by construction.

Read-only guarantees
--------------------
  * does NOT overwrite outputs/models/saved/*.joblib
  * does NOT write to data/processed/** (all rebuilds are in-memory)
  * does NOT touch outputs/tables/model_results/*
  * writes ONLY:
       outputs/tables/robustness/fye_misdating_footprint.{csv,tex}
       outputs/tables/robustness/delisting_mapping_footprint.{csv,tex}
       outputs/tables/robustness/fye_delisting_correction_sensitivity.{csv,tex}

Run from the project root (about 30--60 min, NN fits dominate):
    python -m src.robustness.fye_delisting_correction_sensitivity
Smoke test (LR + XGBoost only, writes NOTHING):
    python -m src.robustness.fye_delisting_correction_sensitivity --quick
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)

from src.config import (
    ALL_FEATURES,
    DATA_MERGED,
    DATA_RAW_COMPUSTAT,
    DATA_RAW_CRSP,
    DATA_SAMPLES,
    DISTRESS_CODES_PRIMARY,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
)
from src.data.load_local_rds import compustat_datadate, read_stkdelists
from src.data.merge_crsp_compustat import (
    AttritionTracker,
    apply_compustat_filters,
    apply_consecutive_years_filter,
    apply_missingness_filter,
    attach_filing_dates,
    attach_market_cap,
    build_distress_label,
    merge_ccm_primary,
    validate_cusip,
)
from src.features.accounting_features import build_accounting_features
from src.features.build_features import (
    apply_post_winsor_missingness_filter,
    load_gdp_deflator,
    split_panel,
)
from src.features.impute import apply_imputation, compute_imputation_medians
from src.features.market_features import build_market_features
from src.features.winsorize import apply_thresholds, compute_thresholds
from src.models.evaluate import select_threshold
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn

LABEL = "distress"
SPLITS = ["train", "val", "test"]
SCENARIOS = ["S0_frozen", "A_fye_aligned", "B_label_corr", "AB_both"]

# Reproduction-control tolerances vs the published headline. LR/RF/NN are
# deterministic under a fixed seed on identical data (hard-fail); XGBoost
# carries documented run-to-run threading variation (warn only).
_REPRO_TOL_STRICT = 0.006
_REPRO_TOL_XGB = 0.015


# ---------------------------------------------------------------------------
# Frozen hyperparameters -- loaded from the SAVED configs (the exact
# parameters the frozen .joblib models were fitted with), never hard-coded.
# ---------------------------------------------------------------------------

def load_frozen_params() -> dict:
    params = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        with open(OUT_MODELS_CONFIGS / f"{name}_config.yaml") as fh:
            params[name] = yaml.safe_load(fh)["best_params"]
    try:
        with open(OUT_MODELS_CONFIGS / "neural_network_balanced_config.yaml") as fh:
            params["neural_network_balanced"] = yaml.safe_load(fh)["best_params"]
    except FileNotFoundError:
        pass
    return params


def _fit_eval(name: str, params: dict,
              Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    if name == "logistic_regression":
        mdl = build_logistic_regression(**params)
    elif name == "random_forest":
        mdl = build_random_forest(**params)
    elif name == "xgboost":
        mdl = build_xgboost(**params,
                            scale_pos_weight=compute_scale_pos_weight(ytr))
    elif name == "neural_network_balanced":
        mdl = build_balanced_nn(**params)
    else:
        raise ValueError(f"unknown model {name}")
    mdl.fit(Xtr, ytr)
    th = select_threshold(yva, mdl.predict_proba(Xva)[:, 1])
    pt = mdl.predict_proba(Xte)[:, 1]
    yp = (pt >= th).astype(int)
    return dict(
        pr_auc=average_precision_score(yte, pt),
        roc_auc=roc_auc_score(yte, pt),
        f1=f1_score(yte, yp, zero_division=0),
        precision=precision_score(yte, yp, zero_division=0),
        recall=recall_score(yte, yp, zero_division=0),
        threshold=th,
    )


def _matrices(df: pd.DataFrame):
    X = df[ALL_FEATURES].astype(float).fillna(0).values
    y = df[LABEL].astype(int).values
    return X, y


# ---------------------------------------------------------------------------
# Footprint 1 -- fiscal-year misdating (F1) on the frozen splits
# ---------------------------------------------------------------------------

def footprint_fye(frozen: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        df = frozen[s]
        aff = df["fyr"].between(1, 5)
        n, d = len(df), int(df[LABEL].sum())
        na, da = int(aff.sum()), int(df.loc[aff, LABEL].sum())
        rows.append(dict(
            split=s, firm_years=n, distress_events=d,
            misdated_firm_years=na,
            misdated_pct=round(100 * na / n, 2),
            misdated_events=da,
            misdated_events_pct=round(100 * da / max(d, 1), 2),
        ))
    tot = pd.DataFrame(rows)
    total_row = dict(
        split="total",
        firm_years=int(tot["firm_years"].sum()),
        distress_events=int(tot["distress_events"].sum()),
        misdated_firm_years=int(tot["misdated_firm_years"].sum()),
        misdated_pct=round(100 * tot["misdated_firm_years"].sum()
                           / tot["firm_years"].sum(), 2),
        misdated_events=int(tot["misdated_events"].sum()),
        misdated_events_pct=round(100 * tot["misdated_events"].sum()
                                  / max(tot["distress_events"].sum(), 1), 2),
    )
    out = pd.concat([tot, pd.DataFrame([total_row])], ignore_index=True)
    print("\nF1 footprint -- firm-years with FYR in {1..5} (mis-dated 12 months):")
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Footprint 2 -- delisting-mapping exclusions (F2) on the frozen panel
# ---------------------------------------------------------------------------

def footprint_delisting(panel: pd.DataFrame,
                        delist_raw: pd.DataFrame,
                        frozen: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count in-window BKPY / GLI events the frozen primary label excludes."""
    dl = delist_raw.copy()
    dl["action"] = dl["DelActionType"].astype(str).str.strip()
    dl["reason"] = dl["DelReasonType"].astype(str).str.strip()

    m = panel[["gvkey", "fyear", "permno", "fdate", LABEL]].merge(
        dl[["permno", "dlstdt", "action", "reason"]], on="permno", how="inner"
    )
    inwin = (
        m["dlstdt"].notna()
        & (m["dlstdt"] >= m["fdate"])
        & (m["dlstdt"] <= m["fdate"] + pd.Timedelta(days=365))
    )
    hits = m[inwin]

    # split assignment by fyear (matches split_panel boundaries)
    def _split_of(fy):
        if fy <= 2009:
            return "train"
        if fy <= 2014:
            return "val"
        return "test"

    rows = []
    for cls, mask in [
        ("bankruptcy_BKPY_excluded", hits["reason"] == "BKPY"),
        ("liquidation_GLI_excluded", hits["action"] == "GLI"),
        ("currently_labelled_GDR", (hits["action"] == "GDR")
                                   & (hits["reason"] != "BKPY")),
    ]:
        sub = hits[mask]
        by = sub["fyear"].map(_split_of).value_counts()
        rows.append(dict(
            event_class=cls,
            firm_year_windows=len(sub),
            unique_delistings=int(sub["permno"].nunique()),
            train=int(by.get("train", 0)),
            val=int(by.get("val", 0)),
            test=int(by.get("test", 0)),
        ))
    out = pd.DataFrame(rows)
    print("\nF2 footprint -- in-window delisting events by class "
          "(frozen 10-K windows):")
    print(out.to_string(index=False))

    cov = dl["dlstdt"].max()
    n_cens = int((frozen["test"]["fdate"] + pd.Timedelta(days=365) > cov).sum())
    print(f"  Delisting-file coverage ends {cov.date()}; frozen test rows with "
          f"windows extending past coverage: {n_cens:,}")
    return out


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

def build_corrected_scenarios(delist_frozen: pd.DataFrame,
                              delist_corrected: pd.DataFrame,
                              ) -> tuple[dict, dict]:
    """
    Full in-memory rebuild of the panel under the corrected datadate
    convention, labelled under BOTH delisting mappings.

    Returns (scenario_A_splits, scenario_AB_splits); each is a dict with
    'train'/'val'/'test' DataFrames. Nothing is written to disk.

    Note: correcting datadate on the saved compustat_annual_raw.parquet via
    compustat_datadate(..., 'standard') is exactly equivalent to re-running
    build_compustat_annual(datadate_convention='standard'), because datadate
    is a pure function of (FYEAR, FYR); the expensive .rds reads are skipped.
    """
    print("\n" + "=" * 60)
    print("  REBUILD -- corrected datadate convention (in-memory)")
    print("=" * 60)

    comp = pd.read_parquet(DATA_RAW_COMPUSTAT / "compustat_annual_raw.parquet")
    n_shift = int(pd.to_numeric(comp["fyr"], errors="coerce").between(1, 5).sum())
    comp["datadate"] = compustat_datadate(comp["fyear"], comp["fyr"],
                                          convention="standard")
    print(f"  datadate corrected for {n_shift:,} of {len(comp):,} raw firm-years "
          f"(FYR 1-5 -> period ends in FYEAR+1)")

    filing   = pd.read_parquet(DATA_RAW_COMPUSTAT / "compustat_filing_dates.parquet")
    msf      = pd.read_parquet(DATA_RAW_CRSP / "crsp_monthly_raw.parquet")
    ccm      = pd.read_parquet(DATA_RAW_CRSP / "ccm_linktable_raw.parquet")
    secnames = pd.read_parquet(DATA_RAW_CRSP / "crsp_security_names.parquet")

    tracker = AttritionTracker()
    comp = apply_compustat_filters(comp, tracker)
    comp = attach_filing_dates(comp, filing, drop_missing_fdates=True)
    tracker.record(comp, "F1: Drop missing/fallback filing dates")
    panel = merge_ccm_primary(comp, ccm, tracker)
    panel, _ = validate_cusip(panel, secnames)
    panel = attach_market_cap(panel, msf, tracker)

    # Label under both mappings on the SAME corrected panel (codes 400-499
    # for both; what differs is which event classes carry 400-499 codes).
    slim = panel[["gvkey", "fyear", "permno", "fdate"]].copy()
    lab_frozen = build_distress_label(slim.copy(), delist_frozen)[
        ["gvkey", "fyear", "distress"]].rename(columns={"distress": "_dA"})
    lab_corr = build_distress_label(slim.copy(), delist_corrected)[
        ["gvkey", "fyear", "distress"]].rename(columns={"distress": "_dAB"})
    panel = panel.merge(lab_frozen, on=["gvkey", "fyear"], how="left")
    panel = panel.merge(lab_corr, on=["gvkey", "fyear"], how="left")
    panel["_dA"] = panel["_dA"].fillna(0).astype(int)
    panel["_dAB"] = panel["_dAB"].fillna(0).astype(int)
    panel[LABEL] = panel["_dA"]

    panel = apply_consecutive_years_filter(panel, tracker)
    panel = apply_missingness_filter(panel, tracker)

    # Features (label-independent -> computed once for both scenarios)
    gdp = load_gdp_deflator()
    panel = build_accounting_features(panel, gdp)
    panel = build_market_features(panel, msf, gdp)

    print("\nSplitting corrected panel ...")
    tr, va, te = split_panel(panel)

    print("\nWinsorisation thresholds (corrected train only) ...")
    thresholds = compute_thresholds(tr)
    tr, va, te = (apply_thresholds(d, thresholds) for d in (tr, va, te))

    tr = apply_post_winsor_missingness_filter(tr, "train")
    va = apply_post_winsor_missingness_filter(va, "val")
    te = apply_post_winsor_missingness_filter(te, "test")

    print("\nImputation medians (corrected train only) ...")
    s1, s2, an, gl = compute_imputation_medians(tr)
    tr = apply_imputation(tr, s1, s2, an, gl)
    va = apply_imputation(va, s1, s2, an, gl)
    te = apply_imputation(te, s1, s2, an, gl)

    def _with_label(col):
        return {n: d.assign(**{LABEL: d[col].astype(int)})
                for n, d in [("train", tr), ("val", va), ("test", te)]}

    return _with_label("_dA"), _with_label("_dAB")


def build_scenario_B(frozen: dict[str, pd.DataFrame],
                     delist_corrected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Frozen splits, label rebuilt under the corrected delisting mapping.
    Isolates the F2 correction: identical firm-years, features, winsorisation
    and imputation; only the event classes inside the 400-499 range change.
    """
    print("\n" + "=" * 60)
    print("  SCENARIO B -- frozen splits, corrected delisting mapping")
    print("=" * 60)
    panel = pd.read_parquet(DATA_MERGED / "panel_raw.parquet",
                            columns=["gvkey", "fyear", "permno", "fdate"])
    lab = build_distress_label(panel, delist_corrected)[
        ["gvkey", "fyear", "distress"]].rename(columns={"distress": "_dB"})

    out = {}
    for s in SPLITS:
        df = frozen[s].merge(lab, on=["gvkey", "fyear"], how="left")
        df["_dB"] = df["_dB"].fillna(0).astype(int)
        added = int(((df["_dB"] == 1) & (df[LABEL] == 0)).sum())
        removed = int(((df["_dB"] == 0) & (df[LABEL] == 1)).sum())
        print(f"  {s:5s}: events {int(df[LABEL].sum()):4d} -> "
              f"{int(df['_dB'].sum()):4d}  (+{added} recovered, -{removed})")
        out[s] = df.assign(**{LABEL: df["_dB"].astype(int)}).drop(columns="_dB")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(quick: bool = False) -> None:
    params = load_frozen_params()
    models = ["logistic_regression", "xgboost"] if quick else list(params)
    print(f"Models: {models}  (hyperparameters from frozen configs; no re-tune)")

    # ---- frozen splits + reference headline --------------------------------
    frozen = {s: pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in SPLITS}
    ref = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test_4models.csv")
    # The 4-model table carries display names; map back to internal keys.
    _display_to_key = {
        "Logistic Regression (ridge)": "logistic_regression",
        "Logistic Regression": "logistic_regression",
        "LR (Platt-scaled)": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "Neural Network (MLP)": "neural_network_balanced",
        "Neural Network (MLP, balanced)": "neural_network_balanced",
    }
    ref_pr = {
        _display_to_key.get(m, m): p
        for m, p in zip(ref["model"], ref["pr_auc"])
    }
    missing_ref = [m for m in params if m not in ref_pr]
    if missing_ref:
        raise KeyError(
            f"Could not map published headline rows to model keys for "
            f"{missing_ref}; reproduction control would be vacuous. "
            f"CSV rows: {list(ref['model'])}"
        )
    print("Published headline PR-AUC:",
          {k: round(v, 4) for k, v in ref_pr.items()})

    # ---- delisting frames ---------------------------------------------------
    delist_frozen = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
    delist_corrected = read_stkdelists(mapping="corrected")

    # ---- footprints ---------------------------------------------------------
    fp_fye = footprint_fye(frozen)
    panel_raw = pd.read_parquet(
        DATA_MERGED / "panel_raw.parquet",
        columns=["gvkey", "fyear", "permno", "fdate", LABEL])
    fp_dl = footprint_delisting(panel_raw, delist_corrected, frozen)

    # ---- scenario datasets ---------------------------------------------------
    scen_data = {"S0_frozen": frozen}
    scen_data["B_label_corr"] = build_scenario_B(frozen, delist_corrected)
    scen_A, scen_AB = build_corrected_scenarios(delist_frozen, delist_corrected)
    scen_data["A_fye_aligned"] = scen_A
    scen_data["AB_both"] = scen_AB

    # ---- refit + evaluate ----------------------------------------------------
    rows = []
    for scen in SCENARIOS:
        d = scen_data[scen]
        Xtr, ytr = _matrices(d["train"])
        Xva, yva = _matrices(d["val"])
        Xte, yte = _matrices(d["test"])
        print(f"\n[{scen}] train={len(ytr):,} ({ytr.sum()} ev)  "
              f"val={len(yva):,} ({yva.sum()} ev)  "
              f"test={len(yte):,} ({yte.sum()} ev, prev {100*yte.mean():.2f}%)")
        for name in models:
            r = _fit_eval(name, params[name], Xtr, ytr, Xva, yva, Xte, yte)
            r.update(
                scenario=scen, model=name,
                n_train=len(ytr), n_test=len(yte),
                events_test=int(yte.sum()),
                prevalence_test=round(float(yte.mean()), 4),
            )
            rows.append(r)
            print(f"  {name:24s} PR-AUC={r['pr_auc']:.4f}  "
                  f"ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

    df = pd.DataFrame(rows)

    # ---- reproduction control ------------------------------------------------
    print("\nReproduction control (S0_frozen refit vs published headline):")
    s0 = df[df["scenario"] == "S0_frozen"].set_index("model")["pr_auc"]
    failures = []
    for name in models:
        if name not in ref_pr:
            continue
        diff = abs(s0[name] - ref_pr[name])
        tol = _REPRO_TOL_XGB if name == "xgboost" else _REPRO_TOL_STRICT
        status = "OK" if diff <= tol else "FAIL"
        print(f"  {name:24s} refit={s0[name]:.4f}  frozen={ref_pr[name]:.4f}  "
              f"|diff|={diff:.4f}  (tol {tol})  {status}")
        if diff > tol:
            if name == "xgboost":
                print("    (XGBoost run-to-run threading variation is documented; "
                      "treated as WARNING)")
            else:
                failures.append(name)
    if failures:
        raise AssertionError(
            f"S0_frozen refit does not reproduce the published headline for "
            f"{failures}; hyperparameter configs and frozen splits are out of "
            f"sync -- results below would not be attributable to the corrections."
        )

    # ---- deltas vs S0 ---------------------------------------------------------
    wide = df.pivot(index="model", columns="scenario", values="pr_auc")
    wide = wide[[c for c in SCENARIOS if c in wide.columns]]
    for c in SCENARIOS[1:]:
        if c in wide.columns:
            wide[f"delta_{c}"] = (wide[c] - wide["S0_frozen"]).round(4)
    print("\nPR-AUC by scenario (deltas vs S0_frozen):")
    print(wide.round(4).to_string())

    if quick:
        print("\n--quick smoke run: NOT writing any outputs.")
        return

    # ---- save (new files only) ------------------------------------------------
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)

    fp1_csv = OUT_TABLES_ROBUSTNESS / "fye_misdating_footprint.csv"
    fp_fye.to_csv(fp1_csv, index=False)
    with open(OUT_TABLES_ROBUSTNESS / "fye_misdating_footprint.tex",
              "w", encoding="utf-8") as fh:
        fh.write(fp_fye.to_latex(
            index=False,
            caption=("Footprint of the fiscal year-end misdating. Firm-years with "
                     "fiscal year-end months January--May (FYR 1--5) were dated "
                     "twelve months early by the frozen datadate reconstruction, "
                     "pairing each such firm-year's accounting data with the "
                     "previous fiscal year's 10-K filing date (look-ahead) and "
                     "with market data measured twelve months before the true "
                     "fiscal year-end."),
            label="tab:fye_misdating_footprint",
        ))

    fp2_csv = OUT_TABLES_ROBUSTNESS / "delisting_mapping_footprint.csv"
    fp_dl.to_csv(fp2_csv, index=False)
    with open(OUT_TABLES_ROBUSTNESS / "delisting_mapping_footprint.tex",
              "w", encoding="utf-8") as fh:
        fh.write(fp_dl.to_latex(
            index=False,
            caption=("Footprint of the delisting-code mapping correction. "
                     "In-window delisting events among modelled firm-years "
                     "(window $[F_{i,t}, F_{i,t}+365]$, frozen panel) by event "
                     "class. Under the frozen mapping, bankruptcies "
                     "(DelReasonType BKPY, synthetic code 572) and liquidations "
                     "(DelActionType GLI, code 500) fell outside the primary "
                     "400--499 label range and were excluded from the distress "
                     "label; the corrected mapping codes them 470 and 400 "
                     "respectively, inside the primary range, making the "
                     "bankruptcy robustness subset (RC1) nested in the primary "
                     "label as the design intends."),
            label="tab:delisting_mapping_footprint",
        ))

    sens_csv = OUT_TABLES_ROBUSTNESS / "fye_delisting_correction_sensitivity.csv"
    df_out = df[[
        "scenario", "model", "n_train", "n_test", "events_test",
        "prevalence_test", "pr_auc", "roc_auc", "f1", "precision", "recall",
        "threshold",
    ]].round(4)
    df_out.to_csv(sens_csv, index=False)
    max_abs = float(wide[[c for c in wide.columns
                          if str(c).startswith("delta_")]].abs().max().max())
    with open(OUT_TABLES_ROBUSTNESS / "fye_delisting_correction_sensitivity.tex",
              "w", encoding="utf-8") as fh:
        fh.write(wide.round(4).to_latex(
            float_format="%.4f",
            caption=("Sensitivity of test-set PR-AUC to the two data-construction "
                     "corrections. All models are re-fitted with their frozen "
                     "hyperparameters (no re-tune); the frozen saved models are "
                     "not modified. S0: frozen data (reproduction control). "
                     "A: fiscal-year-aligned datadate (Compustat June rule). "
                     "B: corrected delisting-code mapping (liquidations and "
                     "bankruptcies inside the primary 400--499 label). AB: both "
                     "corrections. Winsorisation and imputation are recomputed "
                     "on each scenario's training sample only. Maximum "
                     f"$|\\Delta$PR-AUC$|$ vs S0 = {max_abs:.4f}."),
            label="tab:fye_delisting_correction_sensitivity",
        ))
    print(f"\nSaved -> {fp1_csv}")
    print(f"Saved -> {fp2_csv}")
    print(f"Saved -> {sens_csv}  (+ .tex companions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="LR + XGBoost only; writes nothing (smoke test).")
    args = ap.parse_args()
    main(quick=args.quick)
