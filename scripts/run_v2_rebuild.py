"""
v2 corrected rebuild — one coherent run of every adopted data-layer fix.
========================================================================
Phase 4 of the 2026-07-11 audit remediation (V2_PROFILE).
Everything writes to v2-specific paths; the frozen v1 raw parquets,
processed data, and model artifacts are NEVER touched.

Stages (resumable — a stage is skipped if its outputs exist, unless
--force):

  a   Extraction: rebuild the three corrected raw parquets into
      DATA_ROOT/raw_v2/ (standard Jan–May FYE dating, corrected
      delisting mapping, letter-field security names).
  a2  Universe pinning gate: dump the CIZ letter-field value counts and
      the eligibility match rates; ABORTS if the configured
      CIZ_UNIVERSE_ELIGIBLE values do not plausibly match the data.
      Review the report before running stage b.
  b   Merge: corrected panel into processed_v2/merged/ (date-ranged
      universe filter incl. NYSE/AMEX/NASDAQ restriction, FY2023 cutoff,
      corrected label codes are inherent in the corrected delisting
      extract).
  c   Features + splits into processed_v2/ (min_obs=12, eligible-universe
      index/RSIZE denominator, _sic per-feature imputation, FY1989 lag
      donors, strict OENEG/INTWO missingness, MB_MISSING indicator,
      EXRET/SIGMA/MB imputation — 18-feature v2 set), plus RAW
      pre-preprocessing splits for fold-safe tuning. Winsor/imputation
      artifacts (incl. global fall-back medians) go to
      outputs/models/configs/v2/.
  d   Fresh 100-trial tuning + training + single test evaluation for
      LR/RF/XGB under spec="v2" (fold-safe CV with in-fold >=8/11
      coverage filter and label-maturity purging, tuned on the raw
      training frame). Run manifest hashes the V2 splits and is
      refreshed after LR Platt calibration.

  NN (v2) is intentionally NOT part of this script — run it after the
  three-model stage completes (adapt rc7b machinery; multi-hour Optuna).

  IMPORTANT: run with the project .venv interpreter
  (./.venv/Scripts/python.exe), NOT the system Python — the 2026-07-12
  audit found a stage-d run under Python 3.13 while the pinned project
  environment is 3.11 (requirements-lock.txt).

Usage:
  python -m scripts.run_v2_rebuild --stages a
  python -m scripts.run_v2_rebuild --stages a2
  python -m scripts.run_v2_rebuild --stages bcd
  python -m scripts.run_v2_rebuild --stages all [--force] [--n-trials N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    DATA_ROOT,
    DATA_RAW_V2,
    DATA_MERGED_V2,
    DATA_FEATURES_V2,
    DATA_SAMPLES_V2,
    DISTRESS_HORIZON_DAYS,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    ACCOUNTING_FEATURES,
    MARKET_IMPUTE_FEATURES,
    SAMPLE_START_YEAR,
    V2_PROFILE,
)

RAW_V2 = DATA_RAW_V2
COMP_V2_PATH     = RAW_V2 / "compustat_annual_raw.parquet"
DELIST_V2_PATH   = RAW_V2 / "crsp_delisting_raw.parquet"
SECNAMES_V2_PATH = RAW_V2 / "crsp_security_names.parquet"
PANEL_V2_PATH    = DATA_MERGED_V2 / "panel_raw.parquet"

# Universe-eligibility sanity floor/ceiling for the a2 gate: the share of
# PERMNOs passing the filter must be plausible for "US ordinary common
# shares within all CRSP securities". Outside this band the configured
# letter values are probably wrong for this data drop.
ELIGIBLE_SHARE_BOUNDS = (0.30, 0.95)


def _section(tag: str, title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  V2 REBUILD — stage {tag}: {title}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Stage A — corrected extraction
# ---------------------------------------------------------------------------

def stage_a(force: bool = False) -> None:
    _section("a", "corrected raw extraction -> raw_v2/")
    from src.data.load_local_rds import (
        build_compustat_annual,
        build_crsp_delisting,
        build_crsp_security_names,
    )

    RAW_V2.mkdir(parents=True, exist_ok=True)

    if COMP_V2_PATH.exists() and not force:
        print(f"  [skip] {COMP_V2_PATH.name} exists")
    else:
        build_compustat_annual(
            datadate_convention=V2_PROFILE["datadate_convention"],
            out_path=COMP_V2_PATH,
        )

    if DELIST_V2_PATH.exists() and not force:
        print(f"  [skip] {DELIST_V2_PATH.name} exists")
    else:
        build_crsp_delisting(
            mapping=V2_PROFILE["delisting_mapping"],
            out_path=DELIST_V2_PATH,
        )

    if SECNAMES_V2_PATH.exists() and not force:
        print(f"  [skip] {SECNAMES_V2_PATH.name} exists")
    else:
        build_crsp_security_names(out_path=SECNAMES_V2_PATH)


# ---------------------------------------------------------------------------
# Stage A2 — universe-constant pinning gate
# ---------------------------------------------------------------------------

def eligible_permnos_v2() -> set:
    """Compute the eligible PERMNO set from the v2 security-names file."""
    from src.data.universe import universe_eligibility
    sec = pd.read_parquet(SECNAMES_V2_PATH)
    mask = universe_eligibility(sec, policy=V2_PROFILE["universe_policy"])
    return set(
        pd.to_numeric(sec.loc[mask, "permno"], errors="coerce")
        .dropna().astype("int64")
    )


def stage_a2() -> None:
    _section("a2", "CIZ letter-field value counts + eligibility gate")
    from src.config import CIZ_UNIVERSE_ELIGIBLE
    from src.data.universe import universe_eligibility, _FIELD_MAP

    sec = pd.read_parquet(SECNAMES_V2_PATH)
    print(f"  Security-info rows: {len(sec):,}  "
          f"PERMNOs: {sec['permno'].nunique():,}")

    # Dump raw value counts for every letter field (audit artifact)
    rows = []
    for col in ["securitytype", "securitysubtype", "sharetype", "usincflg",
                "issuertype", "shareclass", "primaryexch",
                "shrcd_src", "exchcd_src"]:
        if col not in sec.columns:
            rows.append({"field": col, "value": "<COLUMN MISSING>", "n": 0})
            continue
        vc = sec[col].astype("string").fillna("<NA>").value_counts()
        for val, n in vc.head(25).items():
            rows.append({"field": col, "value": val, "n": int(n)})
    counts = pd.DataFrame(rows)
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    counts_path = OUT_TABLES_ROBUSTNESS / "v2_universe_field_value_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"  Field value counts -> {counts_path}")
    with pd.option_context("display.max_rows", 250):
        print(counts.to_string(index=False))

    # Per-field match rates against the configured eligible values
    print("\n  Per-field eligibility match rates:")
    for col, key in _FIELD_MAP.items():
        if col not in sec.columns:
            print(f"    {col:18s}: COLUMN MISSING")
            continue
        allowed = [v.upper() for v in CIZ_UNIVERSE_ELIGIBLE.get(key, [])]
        vals = sec[col].astype("string").str.strip().str.upper()
        rate = vals.isin(allowed).mean()
        print(f"    {col:18s}: {rate:6.1%} of rows in {allowed}")

    mask = universe_eligibility(sec, policy="v2")
    n_perm = sec["permno"].nunique()
    elig = pd.to_numeric(sec.loc[mask, "permno"], errors="coerce").dropna()
    share = elig.nunique() / max(n_perm, 1)
    print(f"\n  Eligible PERMNOs: {elig.nunique():,} of {n_perm:,} "
          f"({share:.1%})")

    lo, hi = ELIGIBLE_SHARE_BOUNDS
    if not (lo <= share <= hi):
        raise SystemExit(
            f"\nABORT: eligible-PERMNO share {share:.1%} is outside the "
            f"plausibility band [{lo:.0%}, {hi:.0%}]. The configured "
            "CIZ_UNIVERSE_ELIGIBLE values likely do not match this data "
            f"drop — inspect {counts_path} and adjust src/config.py "
            "before running stage b."
        )
    print("  GATE PASSED — universe constants look plausible.")


# ---------------------------------------------------------------------------
# Stage B — corrected merge
# ---------------------------------------------------------------------------

def stage_b(force: bool = False) -> None:
    _section("b", "corrected merge -> processed_v2/merged/")
    if PANEL_V2_PATH.exists() and not force:
        print(f"  [skip] {PANEL_V2_PATH.name} exists")
        return
    from src.data.merge_crsp_compustat import main_with_overrides

    DATA_MERGED_V2.mkdir(parents=True, exist_ok=True)
    main_with_overrides(
        comp_path=COMP_V2_PATH,
        delist_path=DELIST_V2_PATH,
        secnames_path=SECNAMES_V2_PATH,
        output_panel_path=PANEL_V2_PATH,
        output_attrition_path=DATA_MERGED_V2 / "attrition_table.csv",
        output_mismatches_path=DATA_MERGED_V2 / "cusip_mismatches.csv",
        sample_end_year=V2_PROFILE["sample_end_year"],
        universe_policy=V2_PROFILE["universe_policy"],
        universe_date_col=V2_PROFILE["universe_date_col"],
        distress_event_col=V2_PROFILE["distress_event_col"],
        raw_min_nonmissing=None,  # binding >=8/11 constructed-feature rule only
        unique_event_assignment=V2_PROFILE["unique_event_assignment"],
    )


# ---------------------------------------------------------------------------
# Stage C — features + splits
# ---------------------------------------------------------------------------

def _build_lag_donor() -> pd.DataFrame:
    """
    FY(SAMPLE_START_YEAR-1) Compustat rows for lagged-feature construction.

    The corrected extraction keeps SAMPLE_START_YEAR-1 (FY1989) exactly for
    this purpose, but the merge drops it before features are built, leaving
    every FY1990 lag needlessly missing (2026-07-12 second-audit fix). The
    donor rows are restricted to gvkeys present in the merged panel and are
    removed again inside build_accounting_features — they can never enter
    the modelling sample.
    """
    comp = pd.read_parquet(COMP_V2_PATH)
    donor = comp[
        pd.to_numeric(comp["fyear"], errors="coerce") == SAMPLE_START_YEAR - 1
    ].copy()
    panel_gvkeys = pd.read_parquet(PANEL_V2_PATH, columns=["gvkey"])["gvkey"].unique()
    donor = donor[donor["gvkey"].isin(panel_gvkeys)]
    print(f"  Lag donors: {len(donor):,} FY{SAMPLE_START_YEAR - 1} rows "
          f"matching panel gvkeys")
    return donor


def _splits_are_current() -> bool:
    """
    Trust existing v2 splits only if they postdate the merged panel AND
    carry the v2 feature signature (MB_MISSING). Guards against stale
    artifacts resurrected by cloud-sync (observed 2026-07-13: Google
    Drive restored the quarantined pre-blockerfix splits next to their
    quarantine copy, and the plain exists() check skipped stage c).
    """
    test_p = DATA_SAMPLES_V2 / "test.parquet"
    if not test_p.exists():
        return False
    if PANEL_V2_PATH.exists() and test_p.stat().st_mtime < PANEL_V2_PATH.stat().st_mtime:
        print("  Existing v2 splits are OLDER than the merged panel — rebuilding.")
        return False
    import pyarrow.parquet as pq
    cols = pq.read_schema(test_p).names
    if "MB_MISSING" not in cols:
        print("  Existing v2 splits lack MB_MISSING (pre-blockerfix vintage) — rebuilding.")
        return False
    return True


def stage_c(force: bool = False) -> None:
    _section("c", "features + splits -> processed_v2/")
    if _splits_are_current() and not force:
        print("  [skip] v2 splits exist and are current")
        return
    from src.features.build_features import run_feature_pipeline

    run_feature_pipeline(
        panel_path=PANEL_V2_PATH,
        features_path=DATA_FEATURES_V2 / "features_all.parquet",
        samples_dir=DATA_SAMPLES_V2,
        save_winsor_imputation_configs=True,
        configs_out_dir=OUT_MODELS_CONFIGS / V2_PROFILE["spec"],
        market_min_obs=V2_PROFILE["market_min_obs"],
        eligible_permnos=eligible_permnos_v2(),
        sic_col=V2_PROFILE["sic_col"],
        peer_rule=V2_PROFILE["impute_peer_rule"],
        save_raw_splits=True,
        missing_policy=V2_PROFILE["missing_policy"],
        lag_donor=_build_lag_donor() if V2_PROFILE["lag_buffer"] else None,
        impute_market=V2_PROFILE["impute_market"],
        add_mb_missing=True,   # feature_set = ALL_FEATURES_V2 (18)
        outer_purge_horizon_days=DISTRESS_HORIZON_DAYS,
    )


# ---------------------------------------------------------------------------
# Stage D — tuning + training + single evaluation (spec="v2")
# ---------------------------------------------------------------------------

def _stage_d_complete(saved_dir, configs_dir) -> bool:
    """
    A resumed stage d counts as finished only when the COMPLETE artifact
    set exists: all three models + configs, the results table, and the
    run manifest (2026-07-12 second-audit fix — previously the presence
    of xgboost.joblib alone skipped the stage, so an interrupted run
    could be treated as done without results or provenance).
    """
    from src.utils.run_manifest import MANIFEST_NAME
    needed = (
        [saved_dir / f"{m}.joblib" for m in
         ("logistic_regression", "random_forest", "xgboost")]
        + [configs_dir / f"{m}_config.yaml" for m in
           ("logistic_regression", "random_forest", "xgboost")]
        + [configs_dir / MANIFEST_NAME,
           OUT_TABLES_MODEL / V2_PROFILE["spec"] / "model_performance_test_3models.csv"]
    )
    missing = [p for p in needed if not p.exists()]
    if missing:
        print("  Stage d incomplete — missing: "
              + ", ".join(p.name for p in missing))
    return not missing


def _assert_no_feature_nans(splits: dict, features: list[str]) -> None:
    """
    Fail fast if any model feature still carries NaN after imputation.

    Under v1 the model-input fillna(0) silently absorbed missing market
    features (2026-07-12 audit P1: zero-filled MB was an implicit
    distress signal). v2 imputes everything it feeds the models, so a
    residual NaN here means the preprocessing is broken — abort rather
    than let fillna(0) reintroduce the defect.
    """
    problems = []
    for name, df in splits.items():
        counts = df[features].isna().sum()
        bad = counts[counts > 0]
        problems += [f"{name}.{feat}={int(n)}" for feat, n in bad.items()]
    if problems:
        raise AssertionError(
            "v2 splits still contain NaN in model features (fillna(0) "
            "would silently absorb them): " + ", ".join(problems)
        )
    print(f"  NaN gate passed: {len(features)} features complete in all splits.")


def stage_d(force: bool = False, n_trials: int | None = None) -> None:
    _section("d", "fresh tuning + training + test evaluation (spec=v2)")
    import sys as _sys
    if ".venv" not in _sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv "
              f"(sys.prefix={_sys.prefix}) — package versions are not the "
              "pinned ones (requirements-lock.txt).")

    from src.models.train import (
        apply_lr_platt_calibration,
        evaluate_on_test,
        resolve_artifact_dirs,
        train_all_models,
    )
    from src.utils.run_manifest import write_run_manifest

    features = list(V2_PROFILE["feature_set"])            # 18 incl. MB_MISSING
    saved_dir, configs_dir = resolve_artifact_dirs(V2_PROFILE["spec"])
    if _stage_d_complete(saved_dir, configs_dir) and not force:
        print(f"  [skip] complete v2 model set exists in {saved_dir}")
        return

    print("  Loading v2 splits ...")
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    train_raw = pd.read_parquet(DATA_SAMPLES_V2 / "train_raw.parquet")
    print(f"  Train {len(train):,} | Val {len(val):,} | Test {len(test):,} "
          f"| events(test) {int(test['distress'].sum()):,} "
          f"(prev {test['distress'].mean():.2%})")
    _assert_no_feature_nans(
        {"train": train, "val": val, "test": test}, features
    )

    trained = train_all_models(
        train, val,
        features=features,
        n_trials=n_trials,                       # None = config default (100)
        spec=V2_PROFILE["spec"],
        tune_raw_train=train_raw,
        tune_fold_safe=V2_PROFILE["cv_fold_safe"],
        tune_purge_horizon_days=DISTRESS_HORIZON_DAYS,
        tune_sic_col=V2_PROFILE["sic_col"],
        tune_peer_rule=V2_PROFILE["impute_peer_rule"],
        tune_impute_features=ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES,
        samples_dir=DATA_SAMPLES_V2,             # manifest hashes V2 data
        # Interruption-safe: finished models are loaded, not re-tuned, and
        # in-progress Optuna studies persist to sqlite and resume.
        skip_if_saved=True,
        tune_storage_dir=OUT_MODELS_CONFIGS / V2_PROFILE["spec"],
    )
    trained = apply_lr_platt_calibration(trained, val, features=features,
                                         spec=V2_PROFILE["spec"])

    results = evaluate_on_test(
        trained, test,
        features=features,
        output_stem=(OUT_TABLES_MODEL / V2_PROFILE["spec"] /
                     "model_performance_test_3models"),
        run_label=V2_PROFILE["spec"],
    )

    # Refresh the manifest AFTER Platt calibration re-saved the LR joblib,
    # so the recorded model checksums match what is on disk (2026-07-12
    # second-audit fix).
    write_run_manifest(
        configs_dir=configs_dir,
        saved_dir=saved_dir,
        spec=V2_PROFILE["spec"],
        features=features,
        samples_dir=DATA_SAMPLES_V2,
        input_files=[
            COMP_V2_PATH,
            DELIST_V2_PATH,
            SECNAMES_V2_PATH,
            PANEL_V2_PATH,
            DATA_FEATURES_V2 / "features_all.parquet",
            DATA_ROOT / "raw" / "crsp" / "crsp_monthly_raw.parquet",
            DATA_ROOT / "raw" / "compustat" / "compustat_filing_dates.parquet",
            DATA_ROOT / "raw" / "crsp" / "ccm_linktable_raw.parquet",
            DATA_ROOT / "external" / "macro" / "gdp_deflator.csv",
            OUT_MODELS_CONFIGS / V2_PROFILE["spec"] /
            "outcome_definition_approval.yaml",
        ],
        extra={"n_trials": n_trials, "post_calibration": True},
    )

    print(f"\n  {V2_PROFILE['display_name']} results:")
    print(results.to_string(index=False))
    print("\n  NOTE: run the balanced neural-network step before generating "
          "the four-model headline results.")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="all",
                    help="Any of a, a2, b, c, d concatenated (e.g. 'bcd'), "
                         "or 'all'.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if outputs exist.")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Override Optuna trials for stage d (default: "
                         "config OPTUNA_TRIALS=100). Use small values only "
                         "for smoke tests.")
    args = ap.parse_args()

    stages = args.stages.lower()
    run_all = stages == "all"

    if run_all or "a" in stages.replace("a2", ""):
        stage_a(force=args.force)
    if run_all or "a2" in stages:
        stage_a2()
    if run_all or "b" in stages:
        stage_b(force=args.force)
    if run_all or "c" in stages:
        stage_c(force=args.force)
    if run_all or "d" in stages:
        stage_d(force=args.force, n_trials=args.n_trials)

    print("\nV2 rebuild stages complete.")


if __name__ == "__main__":
    main()
