"""
v2 robustness re-runs — RC1-corrected, RC3-SMOTENC, H2 (corrected estimand).
=============================================================================
The last v2 evidence gap. Runs the three audit-flagged
robustness/secondary analyses on the corrected v2 data, using the Phase-5
repaired machinery. Sequential (one CPU-heavy stage at a time), resumable
(a stage is skipped when its results CSV exists), all outputs v2_-prefixed
— no v1 artifact is touched.

  RC1  Nested bankruptcy-only label (DISTRESS_CODES_RC1_CORRECTED = [470],
       genuinely NESTED in the corrected 400-499 primary label, unlike the
       frozen v1 RC1 which was disjoint). Relabels the v2 splits from the
       corrected raw_v2 delisting extract and RE-TUNES LR/RF/XGB
       (100-trial Optuna, spec="v2_rc1" artifact namespace) — the
       per-check re-tune convention of the frozen RC set. NOTE: RC tuning
       uses the standard (non-fold-safe) CV on the preprocessed training
       frame, as in v1's RC1-RC5; the primary v2 spec alone uses
       fold-safe tuning — same per-check asymmetry already disclosed in
       ch07's introduction.
       -> outputs/tables/robustness/v2_rc1_results.{csv,tex}

  RC3  SMOTE imbalance treatment in the corrected mode (SMOTENC keeps the
       binary indicators {0,1}; continuous features standardised on train
       statistics; v2 tuned hyperparameters loaded from configs/v2).
       No re-tune — cheap.
       -> outputs/tables/robustness/v2_rc3_results.{csv,tex}

  H2   Design-optimism experiment with the corrected estimand: the change
       in the ML-over-LR ADVANTAGE under leaky split designs (not raw
       PR-AUC levels), 5 seeds for the random designs, preprocessing
       fitted within each design's training split, _sic industry
       imputation. Uses the full v2 feature panel.
       -> outputs/tables/robustness/v2_h2_results.{csv,tex}
          outputs/tables/robustness/v2_h2_design_optimism.{csv,tex}

Run with the project .venv:
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_robustness
    ... --stages 13h --force   (any of 1, 3, h)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    ACCOUNTING_FEATURES,
    DATA_FEATURES_V2,
    DATA_MERGED_V2,
    DATA_RAW_V2,
    DATA_ROOT,
    DATA_SAMPLES_V2,
    DISTRESS_CODES_RC1_CORRECTED,
    DISTRESS_HORIZON_DAYS,
    DISTRESS_HORIZON_RC2,
    MARKET_IMPUTE_FEATURES,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    V2_PROFILE,
)
from src.utils.tables import save_table

DELIST_V2_PATH = DATA_RAW_V2 / "crsp_delisting_raw.parquet"
COMP_V2_PATH = DATA_RAW_V2 / "compustat_annual_raw.parquet"
SECNAMES_V2_PATH = DATA_RAW_V2 / "crsp_security_names.parquet"
MONTHLY_PATH = DATA_ROOT / "raw" / "crsp" / "crsp_monthly_raw.parquet"
LABEL_APPROVAL_PATH = (OUT_MODELS_CONFIGS / V2_PROFILE["spec"] /
                       "outcome_definition_approval.yaml")
RC5_PANEL_PATH = DATA_MERGED_V2 / "panel_raw_rc5.parquet"
RC5_SAMPLES_DIR = DATA_ROOT / "processed_final_primary" / "samples_rc5"
FEATURES = list(V2_PROFILE["feature_set"])          # 18 incl. MB_MISSING
ROBUSTNESS_DIR = OUT_TABLES_ROBUSTNESS / V2_PROFILE["spec"]


def _rc_spec(number: int) -> str:
    """Return an artifact namespace isolated from every historical run."""
    return f"{V2_PROFILE['spec']}_rc{number}"


def _section(tag: str, title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  V2 ROBUSTNESS — {tag}: {title}")
    print("=" * 68)


def _load_splits():
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    print(f"  v2 splits: train {len(train):,} | val {len(val):,} | "
          f"test {len(test):,}")
    return train, val, test


def _load_raw_train(samples_dir=DATA_SAMPLES_V2) -> pd.DataFrame:
    path = Path(samples_dir) / "train_raw.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Fold-safe robustness tuning requires {path}. Rebuild the "
            "feature splits before running robustness checks."
        )
    return pd.read_parquet(path)


def stage_rc1(force: bool = False) -> None:
    _section("RC1", "bankruptcy-reason proxy (fold-safe re-tune)")
    out = ROBUSTNESS_DIR / "rc1_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] v2_rc1_results.csv exists")
        return
    from src.robustness.rc1_bankruptcy_codes import run_rc1

    train, val, test = _load_splits()
    res = run_rc1(
        train, val, test,
        features=FEATURES,
        codes=DISTRESS_CODES_RC1_CORRECTED,
        delist_path=DELIST_V2_PATH,
        spec=_rc_spec(1),
        run_label="RC1 bankruptcy-reason proxy",
        skip_if_saved=True,
        tune_storage_dir=OUT_MODELS_CONFIGS / _rc_spec(1),
        event_label_col=V2_PROFILE["bankruptcy_event_col"],
        unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        samples_dir=DATA_SAMPLES_V2,
        manifest_input_files=[DELIST_V2_PATH, LABEL_APPROVAL_PATH],
        manifest_extra={
            "outcome_variant": "GDR DelReasonType=BKPY bankruptcy-reason proxy",
            "horizon_days": DISTRESS_HORIZON_DAYS,
            "n_trials_effective": 100,
        },
        train_raw=_load_raw_train(),
        tune_fold_safe=True,
        tune_purge_horizon_days=DISTRESS_HORIZON_DAYS,
        tune_sic_col=V2_PROFILE["sic_col"],
        tune_peer_rule=V2_PROFILE["impute_peer_rule"],
        tune_impute_features=ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES,
    )
    save_table(
        res, out,
        caption=(
            "RC1 under the primary specification: distress restricted to "
            "the CRSP bankruptcy-reason proxy (DelReasonType=BKPY), nested "
            "inside the primary financial-distress label. This is a CRSP "
            "reason proxy, not a verified Chapter 7/11 court-filing measure. "
            "All other design dimensions "
            "unchanged; LR/RF/XGBoost re-tuned per the robustness-check "
            "convention (100-trial Optuna, spec-specific artifacts)."),
        label="tab:v2_rc1",
    )
    print(res.to_string(index=False))


def stage_narrow_label(force: bool = False) -> None:
    """Sensitivity check using the narrower five-reason CIZ definition."""
    _section("N", "narrow financial-reason label (fold-safe re-tune)")
    out = ROBUSTNESS_DIR / "narrow_label_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] narrow_label_results.csv exists")
        return
    from src.robustness.rc1_bankruptcy_codes import run_rc1

    train, val, test = _load_splits()
    spec = f"{V2_PROFILE['spec']}_narrow_label"
    result = run_rc1(
        train, val, test,
        features=FEATURES,
        delist_path=DELIST_V2_PATH,
        spec=spec,
        run_label="Narrow financial-reason label sensitivity",
        skip_if_saved=True,
        tune_storage_dir=OUT_MODELS_CONFIGS / spec,
        event_label_col=V2_PROFILE["narrow_distress_event_col"],
        unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        samples_dir=DATA_SAMPLES_V2,
        manifest_input_files=[DELIST_V2_PATH, LABEL_APPROVAL_PATH],
        manifest_extra={
            "outcome_variant": (
                "GLI plus GDR reasons BKPY/FING/INSC/EQRQ/LP"
            ),
            "horizon_days": DISTRESS_HORIZON_DAYS,
            "n_trials_effective": 100,
            "role": "supplementary label sensitivity",
        },
        train_raw=_load_raw_train(),
        tune_fold_safe=True,
        tune_purge_horizon_days=DISTRESS_HORIZON_DAYS,
        tune_sic_col=V2_PROFILE["sic_col"],
        tune_peer_rule=V2_PROFILE["impute_peer_rule"],
        tune_impute_features=ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES,
    )
    save_table(
        result, out,
        caption=(
            "Supplementary outcome-definition sensitivity using the narrower "
            "CIZ rule (GLI plus GDR reasons BKPY, FING, INSC, EQRQ, and LP) "
            "instead of the literature-aligned legacy CRSP performance-"
            "delisting family. LR, RF, and XGBoost are independently re-tuned "
            "with 100 completed Optuna trials and fold-safe purged CV."
        ),
        label="tab:narrow_label_sensitivity",
    )
    print(result.to_string(index=False))


def stage_rc3(force: bool = False) -> None:
    _section("RC3", "SMOTENC imbalance treatment (corrected mode, no re-tune)")
    out = ROBUSTNESS_DIR / "rc3_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] v2_rc3_results.csv exists")
        return
    from src.robustness.rc3_smote import run_rc3

    train, val, test = _load_splits()
    res = run_rc3(
        train, val, test,
        features=FEATURES,
        corrected=True,
        params_dir=OUT_MODELS_CONFIGS / V2_PROFILE["spec"],
    )
    save_table(
        res, out,
        caption=(
            "RC3 under the primary specification: SMOTE replaces class "
            "weighting, in the corrected mode — SMOTENC treats the binary "
            "indicators (OENEG, INTWO, MB\\_MISSING) as categorical so no "
            "fractional indicator values are synthesised, continuous "
            "features are standardised on training statistics before "
            "resampling, and the primary tuned hyperparameters are reused."),
        label="tab:v2_rc3",
    )
    print(res.to_string(index=False))


def stage_h2(force: bool = False, n_seeds: int = 5) -> None:
    _section("H2", f"design-optimism experiment ({n_seeds} seeds, corrected estimand)")
    out_res = ROBUSTNESS_DIR / "h2_results"
    if Path(f"{out_res}.csv").exists() and not force:
        print("  [skip] v2_h2_results.csv exists")
        return
    from src.analysis.h2_leakage_sensitivity import (
        compute_advantage_differential,
        run_leakage_comparison,
    )

    panel = pd.read_parquet(DATA_FEATURES_V2 / "features_all.parquet")
    print(f"  v2 feature panel: {len(panel):,} firm-years")
    results, optimism = run_leakage_comparison(
        panel,
        features=FEATURES,
        n_seeds=n_seeds,
        within_design_preprocessing=True,
        sic_col=V2_PROFILE["sic_col"],
        write_tables=False,   # v2 saves its own prefixed tables below —
                              # never overwrite the frozen v1 h2_* files
    )
    advantage = compute_advantage_differential(results)

    save_table(
        results, out_res,
        caption=(
            "H2 design-optimism experiment under the primary "
            "specification: PR-AUC and ROC-AUC of the primary models across "
            "four train/test split designs (chronological; firm-level "
            "random; observation-level random; prevalence-controlled "
            "contaminated-train). Random designs replicated over "
            f"{n_seeds} seeds (mean $\\pm$ sd); preprocessing fitted within "
            "each design's training split; default hyperparameters "
            "(disclosed)."),
        label="tab:v2_h2_results",
    )
    save_table(
        advantage, ROBUSTNESS_DIR / "h2_advantage_differential",
        caption=(
            "H2 estimand: the change in the ML-over-LR "
            "advantage (PR-AUC units and prevalence-controlled lift) when "
            "moving from the chronological design to each leaky design. "
            "A positive differential means the leaky design inflates the "
            "apparent ML advantage — the design-optimism effect H2 asserts."),
        label="tab:v2_h2_advantage",
    )
    save_table(
        optimism, ROBUSTNESS_DIR / "h2_design_optimism",
        caption=(
            "H2 design-optimism pivot under the primary specification "
            "(seed-averaged)."),
        label="tab:v2_h2_optimism",
    )
    print(advantage.to_string(index=False))


def _retune_and_eval(train, val, test, features, spec, run_label,
                     out_stem: Path, caption: str, label: str,
                     train_raw: pd.DataFrame,
                     samples_dir=None, manifest_input_files=None,
                     manifest_extra: dict | None = None) -> None:
    """Shared RC re-tune harness: 100-trial Optuna per model (resumable via
    persistent storage + skip_if_saved), validation-locked threshold,
    single test evaluation, v2_-prefixed table. Mirrors the v1 RC
    convention (non-fold-safe tuning on the preprocessed frame — the
    per-check asymmetry disclosed in ch07's introduction)."""
    from src.models.train import evaluate_on_test, train_all_models

    trained = train_all_models(
        train, val, features=features, spec=spec,
        skip_if_saved=True, tune_storage_dir=OUT_MODELS_CONFIGS / spec,
        tune_raw_train=train_raw,
        tune_fold_safe=True,
        tune_purge_horizon_days=DISTRESS_HORIZON_DAYS,
        tune_sic_col=V2_PROFILE["sic_col"],
        tune_peer_rule=V2_PROFILE["impute_peer_rule"],
        tune_impute_features=[
            feature for feature in ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES
            if feature in features
        ],
        samples_dir=samples_dir,
        manifest_input_files=manifest_input_files,
        manifest_extra=manifest_extra,
    )
    res = evaluate_on_test(trained, test, features=features,
                           output_stem=None, run_label=run_label)
    save_table(res, out_stem, caption=caption, label=label)
    print(res.to_string(index=False))


def stage_rc2(force: bool = False) -> None:
    _section("RC2", "6-month horizon (re-tune, ~hours)")
    out = ROBUSTNESS_DIR / "rc2_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] v2_rc2_results.csv exists")
        return
    from src.data.merge_crsp_compustat import build_distress_label

    train, val, test = _load_splits()
    delist = pd.read_parquet(DELIST_V2_PATH)

    def relabel(split):
        split = split.drop(columns=["distress"], errors="ignore")
        return build_distress_label(
            split, delist,
            horizon_days=DISTRESS_HORIZON_RC2,
            event_label_col=V2_PROFILE["distress_event_col"],
            unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        )

    _retune_and_eval(
        relabel(train), relabel(val), relabel(test), FEATURES,
        spec=_rc_spec(2), run_label="RC2 six-month horizon", out_stem=out,
        train_raw=relabel(_load_raw_train()),
        caption=(
            "RC2 under the primary specification: 6-month (182-day) "
            "distress horizon instead of the primary 12 months; corrected "
            "delisting label, all other design dimensions unchanged; "
            "LR/RF/XGBoost re-tuned per the robustness-check convention."),
        label="tab:v2_rc2",
        samples_dir=DATA_SAMPLES_V2,
        manifest_input_files=[DELIST_V2_PATH, LABEL_APPROVAL_PATH],
        manifest_extra={
            "outcome_variant": "primary financial-distress label",
            "horizon_days": DISTRESS_HORIZON_RC2,
            "n_trials_effective": 100,
        },
    )


def stage_rc4(force: bool = False) -> None:
    _section("RC4", "accounting-only predictors (re-tune, ~hours)")
    out = ROBUSTNESS_DIR / "rc4_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] v2_rc4_results.csv exists")
        return
    train, val, test = _load_splits()
    _retune_and_eval(
        train, val, test, list(ACCOUNTING_FEATURES),
        spec=_rc_spec(4), run_label="RC4 accounting-only", out_stem=out,
        train_raw=_load_raw_train(),
        caption=(
            "RC4 under the primary specification: accounting-only "
            "predictor set (the 11 accounting variables; the 6 market "
            "predictors AND the market-derived MB\\_MISSING indicator "
            "removed), quantifying the marginal value of market "
            "information; LR/RF/XGBoost re-tuned."),
        label="tab:v2_rc4",
        samples_dir=DATA_SAMPLES_V2,
        manifest_input_files=[LABEL_APPROVAL_PATH],
        manifest_extra={
            "predictor_variant": "11 accounting predictors only",
            "horizon_days": DISTRESS_HORIZON_DAYS,
            "n_trials_effective": 100,
        },
    )


def stage_rc5(force: bool = False) -> None:
    _section("RC5", "include utilities (dataset rebuild + re-tune, ~hours)")
    out = ROBUSTNESS_DIR / "rc5_results"
    if Path(f"{out}.csv").exists() and not force:
        print("  [skip] v2_rc5_results.csv exists")
        return

    # 5a. RC5 dataset under the full v2 policy, utilities included
    if not (RC5_SAMPLES_DIR / "test.parquet").exists() or force:
        from src.data.merge_crsp_compustat import main_with_overrides
        from src.features.build_features import run_feature_pipeline
        from scripts.run_v2_rebuild import eligible_permnos_v2

        print("  Building v2 RC5 dataset (utilities included) ...")
        main_with_overrides(
            excluded_sic_ranges=[(6000, 6999)],       # financials only
            comp_path=COMP_V2_PATH,
            delist_path=DELIST_V2_PATH,
            secnames_path=SECNAMES_V2_PATH,
            output_panel_path=RC5_PANEL_PATH,
            output_attrition_path=DATA_MERGED_V2 / "attrition_table_rc5.csv",
            output_mismatches_path=DATA_MERGED_V2 / "cusip_mismatches_rc5.csv",
            sample_end_year=V2_PROFILE["sample_end_year"],
            universe_policy=V2_PROFILE["universe_policy"],
            universe_date_col=V2_PROFILE["universe_date_col"],
            distress_event_col=V2_PROFILE["distress_event_col"],
            raw_min_nonmissing=None,
            unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        )
        comp = pd.read_parquet(COMP_V2_PATH)
        donor = comp[pd.to_numeric(comp["fyear"], errors="coerce") == 1989]
        gv = pd.read_parquet(RC5_PANEL_PATH, columns=["gvkey"])["gvkey"].unique()
        run_feature_pipeline(
            panel_path=RC5_PANEL_PATH,
            features_path=DATA_FEATURES_V2 / "features_all_rc5.parquet",
            samples_dir=RC5_SAMPLES_DIR,
            save_winsor_imputation_configs=False,     # RC5 fits its own, unsaved
            save_raw_splits=True,                     # required by fold-safe CV
            market_min_obs=V2_PROFILE["market_min_obs"],
            eligible_permnos=eligible_permnos_v2(),
            sic_col=V2_PROFILE["sic_col"],
            peer_rule=V2_PROFILE["impute_peer_rule"],
            missing_policy=V2_PROFILE["missing_policy"],
            lag_donor=donor[donor["gvkey"].isin(gv)],
            impute_market=V2_PROFILE["impute_market"],
            add_mb_missing=True,
            outer_purge_horizon_days=DISTRESS_HORIZON_DAYS,
        )
    train = pd.read_parquet(RC5_SAMPLES_DIR / "train.parquet")
    val   = pd.read_parquet(RC5_SAMPLES_DIR / "val.parquet")
    test  = pd.read_parquet(RC5_SAMPLES_DIR / "test.parquet")
    print(f"  RC5 splits: train {len(train):,} | val {len(val):,} | "
          f"test {len(test):,} ({int(test['distress'].sum())} events)")
    _retune_and_eval(
        train, val, test, FEATURES,
        spec=_rc_spec(5), run_label="RC5 include utilities", out_stem=out,
        train_raw=_load_raw_train(RC5_SAMPLES_DIR),
        caption=(
            "RC5 under the primary specification: regulated utilities "
            "(SIC 4900--4999) included in the sample (financials still "
            "excluded); dedicated dataset rebuilt under the full corrected "
            "policy (date-ranged universe, FY2023 cutoff, 18 predictors, "
            "market imputation); LR/RF/XGBoost re-tuned."),
        label="tab:v2_rc5",
        samples_dir=RC5_SAMPLES_DIR,
        manifest_input_files=[
            COMP_V2_PATH,
            DELIST_V2_PATH,
            SECNAMES_V2_PATH,
            MONTHLY_PATH,
            LABEL_APPROVAL_PATH,
            RC5_PANEL_PATH,
            DATA_FEATURES_V2 / "features_all_rc5.parquet",
        ],
        manifest_extra={
            "universe_variant": "utilities included; financials excluded",
            "horizon_days": DISTRESS_HORIZON_DAYS,
            "n_trials_effective": 100,
        },
    )


def stage_consolidated(force: bool = False) -> None:
    _section("C", "consolidated v2 robustness table (RC1-RC5)")
    out = ROBUSTNESS_DIR / "robustness_consolidated"
    pieces = [
        ("Primary specification", OUT_TABLES_MODEL / V2_PROFILE["spec"] /
         "model_performance_test_3models.csv"),
        ("RC1 bankruptcy-reason proxy", ROBUSTNESS_DIR / "rc1_results.csv"),
        ("RC2 6-month horizon", ROBUSTNESS_DIR / "rc2_results.csv"),
        ("RC3 SMOTENC", ROBUSTNESS_DIR / "rc3_results.csv"),
        ("RC4 accounting-only", ROBUSTNESS_DIR / "rc4_results.csv"),
        ("RC5 incl. utilities", ROBUSTNESS_DIR / "rc5_results.csv"),
    ]
    rows = []
    for check, p in pieces:
        if not p.exists():
            print(f"  [warn] missing {p.name} — consolidated table incomplete")
            continue
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            rows.append({"check": check, "model": r["model"],
                         "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                         "baseline_pr_auc": r.get("prevalence_baseline_pr_auc"),
                         "pr_auc_ci_lower": r.get("pr_auc_ci_lower"),
                         "pr_auc_ci_upper": r.get("pr_auc_ci_upper")})
    cons = pd.DataFrame(rows)
    save_table(
        cons, out,
        caption=(
            "Consolidated robustness results under the primary "
            "specification (Pre-Specified Empirical Design §11, RC1--RC5). Each check varies "
            "exactly one design dimension; prevalence (and hence the "
            "PR-AUC baseline) differs across label/sample variants, so "
            "compare within checks. LR/RF/XGBoost re-tuned per check; the "
            "co-primary NN is reported for the primary specification only "
            "(extension disclosed)."),
        label="tab:v2_robustness_consolidated",
    )
    print(cons.to_string(index=False))


def main() -> None:
    if ".venv" not in sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv ({sys.prefix})")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="3hn1245c",
                    help="Any of 1-5 (RC1..RC5), n (narrow-label sensitivity), "
                         "h (H2), c (consolidated). "
                         "Default: everything pending.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--h2-seeds", type=int, default=5)
    args = ap.parse_args()
    st = args.stages.lower()

    if "3" in st:                          # cheap first — fail fast
        stage_rc3(force=args.force)
    if "h" in st:
        stage_h2(force=args.force, n_seeds=args.h2_seeds)
    if "n" in st:
        stage_narrow_label(force=args.force)
    if "1" in st:
        stage_rc1(force=args.force)
    if "2" in st:
        stage_rc2(force=args.force)
    if "4" in st:
        stage_rc4(force=args.force)
    if "5" in st:
        stage_rc5(force=args.force)
    if "c" in st:
        stage_consolidated(force=args.force)

    print("\nV2 robustness stages complete.")


if __name__ == "__main__":
    main()
