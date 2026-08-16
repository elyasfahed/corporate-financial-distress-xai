"""
Master end-to-end pipeline — runs all thesis steps in sequence.
===============================================================
Orchestrates every stage from feature engineering to final SHAP analysis:

  Stage 0  Prerequisites check (paths, GDP deflator)
  Stage 1  Feature engineering + chronological split
  Stage 2  Descriptive statistics (§8)
  Stage 3  Model training + test-set evaluation (§9) — ONCE
  Stage 4  H2 Leakage sensitivity analysis
  Stage 5  Global SHAP analysis (§10.2)
  Stage 6  Non-linear SHAP effects (§10.3)
  Stage 7  Local SHAP explanations (§10.4)
  Stage 8  Theory-consistency validation table (§10.5)
  Stage 9  Robustness checks RC1–RC5 (§11)
  Stage 12 Sensitivity analysis — No-RSIZE model (economist challenge)

Prerequisites
-------------
Before running this script:
  1. Run:  python -m src.data.run_extraction
           (downloads CRSP, Compustat, CCM from WRDS — ~45 min)
  2. Run:  python -m src.data.merge_crsp_compustat
           (builds the merged panel — ~5 min)
  3. Ensure data/external/macro/gdp_deflator.csv is present
           (download FRED series GDPDEF)

Usage
-----
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --from-stage 3  # resume from stage 3
    python run_pipeline.py --stages 2 3    # run only stages 2 and 3
    python run_pipeline.py --skip-shap     # skip SHAP (for quick test)

Expected runtime (after data extraction):
  Stage 1: ~5  min   (feature engineering)
  Stage 2: ~2  min   (descriptive stats)
  Stage 3: ~60–120 min (Optuna tuning × 3 models × 100 trials)
  Stage 4: ~20 min   (H2 leakage comparison)
  Stage 5: ~10 min   (SHAP TreeExplainer on full test set)
  Stage 6: ~5  min   (dependence plots + heatmap)
  Stage 7: ~2  min   (3 waterfall plots)
  Stage 8: <1 min    (theory-consistency table)
  Stage 9: ~180 min  (5 × full retraining)
  Stage 12: ~90 min  (No-RSIZE sensitivity — 3 models × 100 Optuna trials)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# ── Make sure the project root is on sys.path when run as a script ─────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd

from src.config import (
    DATA_FEATURES,
    DATA_SAMPLES,
    DATA_MACRO,
    OUT_TABLES_MODEL,
    OUT_TABLES_SHAP,
    OUT_TABLES_ROBUSTNESS,
    OUT_FIGURES_MODEL,
    ALL_FEATURES,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(n: int, title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  STAGE {n}  —  {title}")
    print(f"{bar}")


def _check_prerequisites() -> None:
    """Verify that required input files exist before starting."""
    required = [
        DATA_FEATURES.parent.parent / "raw" / "crsp"       / "crsp_monthly_raw.parquet",
        DATA_FEATURES.parent.parent / "raw" / "compustat"  / "compustat_annual_raw.parquet",
        DATA_FEATURES.parent.parent / "processed" / "merged" / "panel_raw.parquet",
        DATA_MACRO / "gdp_deflator.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nERROR: The following required files are missing:\n")
        for p in missing:
            print(f"  {p}")
        print(
            "\nSteps to resolve (Seafile local data path):"
            "\n  0. python download_data.py                 # download .rds files from Seafile"
            "\n  1. python -m src.data.load_local_rds       # convert .rds → parquet"
            "\n  2. python -m src.data.merge_crsp_compustat # build merged panel"
            "\n  3. Place gdp_deflator.csv in data/external/macro/"
            "\n     (FRED series GDPDEF: https://fred.stlouisfed.org/series/GDPDEF)"
        )
        sys.exit(1)
    print("  Prerequisites OK.")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_1_features() -> None:
    """Feature engineering + chronological split."""
    _section(1, "Feature engineering + chronological split")
    import src.features.build_features as bf
    bf.main()


def stage_2_descriptive() -> None:
    """Descriptive statistics — 4 required outputs."""
    _section(2, "Descriptive statistics (design §8)")
    from src.analysis.descriptive import run_all_descriptive
    # Use the FINAL modeling sample (train+val+test) so all descriptive
    # tables reconcile exactly with the model samples (124,285 firm-years
    # after the post-winsorisation >=8/11 filter; the old 124,443 figure
    # was a stale pre-F4-fix count — corrected 2026-07-12).
    panel = pd.concat(
        [pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    run_all_descriptive(panel)


def stage_3_train(reuse_configs: bool = False) -> dict:
    """Train all models + final test-set evaluation (ONCE)."""
    _section(3, "Model training + test-set evaluation (ONCE)")
    from src.models.train import (
        load_splits, train_all_models, evaluate_on_test,
        apply_lr_platt_calibration,
    )
    train, val, test = load_splits()
    if reuse_configs:
        print("  --reuse-configs: skipping Optuna; reusing saved best_params.")
    # Stage 3 is the designated owner of the primary artifacts — the only
    # caller allowed to overwrite them (guard in train_all_models).
    trained_models   = train_all_models(train, val, reuse_configs=reuse_configs,
                                        allow_primary_overwrite=True)
    # D8 — apply Platt scaling to LR before headline evaluation.
    # Corrects the inflated Brier score from class_weight='balanced'
    # without changing rank-based metrics. Also re-selects the LR
    # threshold on the calibrated scale.
    trained_models   = apply_lr_platt_calibration(trained_models, val)
    # Explicit output_stem ensures the primary CSV is written only here,
    # never by a downstream robustness check (resolves review issues B/C).
    results_df       = evaluate_on_test(
        trained_models, test,
        output_stem=OUT_TABLES_MODEL / "model_performance_test",
        run_label="primary",
    )

    # fillna(0) is a documented final fallback after three-level imputation.
    # Residual NaN can only arise for market features of firms with very thin
    # CRSP coverage. Zero is conservative (neutral signal) for these cases.
    # Disclosed in the Design Deviations appendix.
    n_residual_nan = test[ALL_FEATURES].isna().sum().sum()
    if n_residual_nan > 0:
        print(f"  WARNING: {n_residual_nan} residual NaN values in test features — filling with 0.")
    X_test = test[ALL_FEATURES].astype(float).fillna(0).values
    for name, info in trained_models.items():
        info["y_prob"] = info["model"].predict_proba(X_test)[:, 1]
        info["X_test"] = X_test

    print(f"\nStage 3 complete. Results in: {OUT_TABLES_MODEL}")
    return {"trained_models": trained_models, "test": test, "results_df": results_df}


def stage_4_h2(panel: pd.DataFrame | None = None) -> None:
    """H2 Leakage sensitivity experiment."""
    _section(4, "H2 Leakage sensitivity — chronological vs random split")
    from src.analysis.h2_leakage_sensitivity import run_leakage_comparison

    if panel is None:
        panel = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    run_leakage_comparison(panel)


def stage_5_global_shap(trained_models: dict, test: pd.DataFrame) -> dict:
    """Global SHAP analysis."""
    _section(5, "Global SHAP analysis (design §10.2)")
    from src.explainability.shap_global import (
        compute_shap_values,
        plot_beeswarm,
        plot_mean_abs_shap,
        run_global_shap,
    )

    shap_artifacts: dict = {}

    xgb_info   = trained_models["xgboost"]
    logit_info = trained_models["logistic_regression"]

    # Run global SHAP for the best ML model (XGBoost)
    # Kendall's tau compares XGBoost SHAP ranking vs LR standardised coefficients
    run_global_shap(
        model      = xgb_info["model"],
        logit_model= logit_info["model"],
        test       = test,
        features   = ALL_FEATURES,
        model_name = "xgboost",
    )

    # Run global SHAP for the classical benchmark (Logistic Regression)
    # Runs XAI for the classical benchmark alongside the non-linear model.
    # Kendall's tau here measures LinearExplainer internal consistency with LR coefficients
    run_global_shap(
        model      = logit_info["model"],
        logit_model= logit_info["model"],
        test       = test,
        features   = ALL_FEATURES,
        model_name = "logistic_regression",
    )

    # Run global SHAP for Random Forest — its beeswarm (fig:shap_rf) and
    # ranking table are used in ch06's cross-model attribution comparison.
    run_global_shap(
        model      = trained_models["random_forest"]["model"],
        logit_model= logit_info["model"],
        test       = test,
        features   = ALL_FEATURES,
        model_name = "random_forest",
    )

    # Compute SHAP values for all 3 models (needed for theory-consistency cross-check)
    for name in ["xgboost", "random_forest", "logistic_regression"]:
        info = trained_models[name]
        sv   = compute_shap_values(info["model"], info["X_test"], name)
        shap_artifacts[name] = sv

    return shap_artifacts


def stage_6_nonlinear_shap(shap_artifacts: dict, test: pd.DataFrame) -> None:
    """Non-linear SHAP effects: dependence plots + LEV×ROA heatmap."""
    _section(6, "Non-linear SHAP effects (design §10.3)")
    from src.explainability.shap_nonlinear import (
        plot_all_dependence_plots,
        plot_lev_roa_heatmap,
    )
    from src.models.train import load_splits

    _, _, test_data = load_splits()
    X_test_df = test_data[ALL_FEATURES]

    # XGBoost SHAP values
    sv_xgb = shap_artifacts["xgboost"]

    plot_all_dependence_plots(sv_xgb, X_test_df, model_name="xgboost")

    # LEV × ROA interaction heatmap
    from src.utils.io import load_model
    xgb_model = load_model("xgboost")
    plot_lev_roa_heatmap(xgb_model, test_data, features=ALL_FEATURES, model_name="xgboost")


def stage_7_local_shap(trained_models: dict, test: pd.DataFrame) -> None:
    """Local SHAP waterfall plots for 3 selected firm-years."""
    _section(7, "Local SHAP explanations (design §10.4)")
    from src.explainability.shap_local import run_local_shap

    xgb_info = trained_models["xgboost"]
    run_local_shap(
        model      = xgb_info["model"],
        test       = test,
        y_prob     = xgb_info["y_prob"],
        threshold  = xgb_info["threshold"],
        features   = ALL_FEATURES,
        model_name = "xgboost",
    )


def stage_8_theory(shap_artifacts: dict, test: pd.DataFrame) -> None:
    """Theory-consistency validation table (H3 and H4)."""
    _section(8, "Theory-consistency validation table (design §10.5)")
    from src.explainability.theory_consistency import build_theory_consistency_table

    X_test_df = test[ALL_FEATURES]
    sv_xgb    = shap_artifacts["xgboost"]
    table     = build_theory_consistency_table(sv_xgb, X_test_df, top_n=10)

    print("\n  Theory-consistency results (top 10 features):")
    print(table[["rank", "feature", "theoretical_sign",
                 "observed_sign", "verdict"]].to_string(index=False))


def stage_9_robustness(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    primary_results: pd.DataFrame,
) -> pd.DataFrame:
    """All 5 pre-specified robustness checks."""
    _section(9, "Robustness checks RC1–RC5 (design §11)")
    from src.robustness.run_all_robustness import run_all_robustness
    return run_all_robustness(train, val, test, primary_results)


def stage_10_extended(
    trained_models: dict,
    test: pd.DataFrame,
    val: pd.DataFrame,
    shap_artifacts: dict,
) -> None:
    """
    Extended academic analyses for publication quality.

    Produces:
      (a) ROC and PR curve figures
      (b) Calibration plots (reliability diagrams)
      (c) Formal pairwise significance tests (DeLong + bootstrap PR-AUC)
      (d) Ohlson (1980) classical benchmark
      (e) SHAP feature ranking stability (100 bootstrap samples)
    """
    _section(10, "Extended analyses — significance, benchmarks, SHAP stability")
    import numpy as np
    from src.utils.plots import plot_roc_curves, plot_pr_curves, plot_calibration
    from src.analysis.significance import run_significance_tests
    from src.analysis.classical_benchmarks import run_classical_benchmarks
    from src.explainability.shap_stability import run_shap_stability

    y_test = test["distress"].astype(int).values

    # ── (a) ROC curves ────────────────────────────────────────────────────
    print("\n  Generating ROC curves ...")
    plot_roc_curves(trained_models, y_test,
                    OUT_FIGURES_MODEL / "roc_curves")

    # ── (b) PR curves ─────────────────────────────────────────────────────
    print("  Generating PR curves ...")
    plot_pr_curves(trained_models, y_test,
                   OUT_FIGURES_MODEL / "pr_curves")

    # ── (c) Calibration plots ─────────────────────────────────────────────
    print("  Generating calibration plots ...")
    plot_calibration(trained_models, y_test,
                     OUT_FIGURES_MODEL / "calibration_plots")

    # ── (d) Significance tests ────────────────────────────────────────────
    print("\n  Running pairwise significance tests ...")
    run_significance_tests(trained_models, test, ALL_FEATURES)

    # ── (e) Ohlson (1980) classical benchmark ────────────────────────────
    print("\n  Computing Ohlson (1980) O-score baseline ...")
    run_classical_benchmarks(val, test)

    # ── (f) SHAP stability ────────────────────────────────────────────────
    if "xgboost" in shap_artifacts and "xgboost" in trained_models:
        print("\n  Running SHAP stability bootstrap ...")
        run_shap_stability(
            model      = trained_models["xgboost"]["model"],
            test       = test,
            features   = ALL_FEATURES,
            model_name = "xgboost",
        )
    else:
        print("  Skipping SHAP stability — XGBoost model not loaded.")


def stage_11_regime_and_calibration(
    trained_models: dict,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Additional diagnostics for regime stability and calibration:

      (a) Sub-period stability — PR-AUC for 2015–2019 vs. 2020–2024.
          A good model should not collapse under COVID or rising-rate regime.

      (b) LR post-hoc calibration via Platt scaling.
          LR is the best discriminator (PR-AUC 0.1103) but has the worst
          Brier score (0.2427). Platt scaling fixes the calibration without
          changing the ranking ability.
    """
    _section(11, "Regime stability + LR calibration (academic extensions)")

    from src.analysis.subperiod import run_subperiod_analysis
    from src.analysis.lr_calibration import run_lr_calibration
    from src.config import OUT_MODELS_CONFIGS

    # Load validation-locked thresholds from saved configs
    thresholds: dict[str, float] = {}
    for name in ["logistic_regression", "random_forest", "xgboost"]:
        cfg_path = OUT_MODELS_CONFIGS / f"{name}_config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            thresholds[name] = float(cfg.get("threshold", 0.5))

    # ── (a) Sub-period stability ──────────────────────────────────────────
    print("\n  Running sub-period stability analysis ...")
    run_subperiod_analysis(trained_models, test, thresholds, ALL_FEATURES)

    # ── (b) LR calibration ────────────────────────────────────────────────
    print("\n  Applying Platt scaling to LR ...")
    if "logistic_regression" in trained_models:
        run_lr_calibration(trained_models, val, test, ALL_FEATURES)
    else:
        print("  Skipping LR calibration — LR model not loaded.")


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    stages_to_run = set(args.stages) if args.stages else set(range(1, 12))
    if args.from_stage:
        stages_to_run = set(range(args.from_stage, 13))
    if args.skip_shap:
        stages_to_run -= {5, 6, 7, 8}

    print("\n" + "="*60)
    print("  THESIS PIPELINE — Explainable AI for Corporate Financial Distress Prediction")
    print(f"  Stages to run: {sorted(stages_to_run)}")
    print("="*60)

    # Prerequisites check (always)
    if 1 in stages_to_run:
        _check_prerequisites()

    # Shared objects (populated lazily as stages run)
    trained_models: dict       = {}
    test_data:      pd.DataFrame | None = None
    results_df:     pd.DataFrame | None = None
    shap_artifacts: dict       = {}

    # ── Stage 1 ────────────────────────────────────────────────────────────
    if 1 in stages_to_run:
        try:
            stage_1_features()
        except Exception:
            print("ERROR in Stage 1:\n"); traceback.print_exc(); sys.exit(1)

    # ── Stage 2 ────────────────────────────────────────────────────────────
    if 2 in stages_to_run:
        try:
            stage_2_descriptive()
        except Exception:
            print("ERROR in Stage 2:\n"); traceback.print_exc(); sys.exit(1)

    # ── Stage 3 ────────────────────────────────────────────────────────────
    if 3 in stages_to_run:
        try:
            out = stage_3_train(reuse_configs=getattr(args, "reuse_configs", False))
            trained_models = out["trained_models"]
            test_data      = out["test"]
            results_df     = out["results_df"]
        except Exception:
            print("ERROR in Stage 3:\n"); traceback.print_exc(); sys.exit(1)
    else:
        # Load pre-saved artifacts if skipping stage 3
        _try_load_stage3_artifacts(trained_models)

    # Load test set if not already in memory
    if test_data is None:
        try:
            test_data = pd.read_parquet(DATA_SAMPLES / "test.parquet")
        except FileNotFoundError:
            pass

    # ── Stage 4 ────────────────────────────────────────────────────────────
    if 4 in stages_to_run:
        try:
            stage_4_h2()
        except Exception:
            print("ERROR in Stage 4:\n"); traceback.print_exc()
            print("  Continuing — H2 failure does not block downstream stages.")

    # ── Stage 5 ────────────────────────────────────────────────────────────
    if 5 in stages_to_run and trained_models and test_data is not None:
        try:
            shap_artifacts = stage_5_global_shap(trained_models, test_data)
        except Exception:
            print("ERROR in Stage 5:\n"); traceback.print_exc()

    # If stages 6 or 8 are needed but SHAP was never computed, recompute now
    if not shap_artifacts and ({6, 8} & stages_to_run) and trained_models:
        print("\n  Stage 5 was skipped — recomputing SHAP values from saved models...")
        _try_compute_shap_artifacts(trained_models, shap_artifacts)

    # ── Stage 6 ────────────────────────────────────────────────────────────
    if 6 in stages_to_run and shap_artifacts and test_data is not None:
        try:
            stage_6_nonlinear_shap(shap_artifacts, test_data)
        except Exception:
            print("ERROR in Stage 6:\n"); traceback.print_exc()

    # ── Stage 7 ────────────────────────────────────────────────────────────
    if 7 in stages_to_run and trained_models and test_data is not None:
        try:
            stage_7_local_shap(trained_models, test_data)
        except Exception:
            print("ERROR in Stage 7:\n"); traceback.print_exc()

    # ── Stage 8 ────────────────────────────────────────────────────────────
    if 8 in stages_to_run and shap_artifacts and test_data is not None:
        try:
            stage_8_theory(shap_artifacts, test_data)
        except Exception:
            print("ERROR in Stage 8:\n"); traceback.print_exc()

    # ── Stage 9 ────────────────────────────────────────────────────────────
    if 9 in stages_to_run:
        try:
            train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
            val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
            test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")
            if results_df is None:
                results_df = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test.csv")
            stage_9_robustness(train, val, test, results_df)
        except Exception:
            print("ERROR in Stage 9:\n"); traceback.print_exc()

    # ── Stage 10 ───────────────────────────────────────────────────────────
    if 10 in stages_to_run and trained_models and test_data is not None:
        try:
            val_data = pd.read_parquet(DATA_SAMPLES / "val.parquet")
            if not shap_artifacts and trained_models:
                _try_compute_shap_artifacts(trained_models, shap_artifacts)
            stage_10_extended(trained_models, test_data, val_data, shap_artifacts)
        except Exception:
            print("ERROR in Stage 10:\n"); traceback.print_exc()

    # ── Stage 11 ───────────────────────────────────────────────────────────
    if 11 in stages_to_run and trained_models and test_data is not None:
        try:
            val_data = pd.read_parquet(DATA_SAMPLES / "val.parquet")
            stage_11_regime_and_calibration(trained_models, val_data, test_data)
        except Exception:
            print("ERROR in Stage 11:\n"); traceback.print_exc()

    # ── Stage 12 — RSIZE sensitivity analysis ──────────────────────────────
    if 12 in stages_to_run:
        try:
            from src.robustness.rc6_no_rsize import run_rc6
            print("\n" + "="*60)
            print("  STAGE 12  —  Sensitivity: No-RSIZE model")
            print("="*60)
            train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
            val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
            test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")

            # Read true primary results from model_performance_test.csv.
            # After the B/C fix, this file is written exactly once by Stage 3
            # and is never overwritten by any robustness check.
            _prim_csv = OUT_TABLES_MODEL / "model_performance_test.csv"
            primary = pd.read_csv(_prim_csv)

            # run_rc6 retrains a 16-feature model (no RSIZE) and returns
            # the evaluation metrics. The corrected evaluate_on_test()
            # writes nothing to model_performance_test.csv (output_stem=None).
            rc6_results = run_rc6(train, val, test)

            rc6_results.insert(0, "check", "no_rsize")

            # Compare with true primary
            print("\n  PRIMARY vs NO-RSIZE comparison (PR-AUC):")
            for _, row in primary.iterrows():
                m = row.get("model")
                if not m:
                    continue
                rc6_row = rc6_results[rc6_results["model"] == m]
                if not rc6_row.empty:
                    primary_auc = row["pr_auc"]
                    rc6_auc     = rc6_row["pr_auc"].values[0]
                    delta       = rc6_auc - primary_auc
                    print(f"  {m:<25} Primary={primary_auc:.4f}  No-RSIZE={rc6_auc:.4f}  Δ={delta:+.4f}")

            out_path = OUT_TABLES_ROBUSTNESS / "rc6_no_rsize_results.csv"
            rc6_results.to_csv(out_path, index=False)
            print(f"\n  Table saved -> {out_path}")
            print("  Stage 12 complete.")
        except Exception:
            print("ERROR in Stage 12:\n"); traceback.print_exc()

    # ── Stage 13 — Neural network (RC7, model-class sensitivity) ────────────
    # Transparent post-specification extension (not a pre-specified robustness
    # check). Opt-in
    # only — never runs in the default pipeline, never overwrites primary or
    # RC1–RC5 results. Writes its own rc7_neural_network_results.csv.
    if 13 in stages_to_run:
        try:
            from src.robustness.rc7_neural_network import run_rc7
            print("\n" + "="*60)
            print("  STAGE 13  —  RC7: Neural network (MLP) model-class sensitivity")
            print("="*60)
            train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
            val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
            test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")

            primary = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test.csv")

            rc7_results = run_rc7(
                train, val, test,
                reuse_config=getattr(args, "reuse_configs", False),
            )
            rc7_results.insert(0, "check", "RC7_neural_network")

            # Compare the NN against the primary models (especially XGBoost).
            print("\n  PRIMARY vs NEURAL-NETWORK comparison (PR-AUC):")
            nn_auc = float(rc7_results["pr_auc"].values[0])
            for _, row in primary.iterrows():
                m = row.get("model")
                if not m:
                    continue
                print(f"  {m:<25} PR-AUC={row['pr_auc']:.4f}   "
                      f"(NN={nn_auc:.4f}, Δ={nn_auc - row['pr_auc']:+.4f})")

            OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
            out_path = OUT_TABLES_ROBUSTNESS / "rc7_neural_network_results.csv"
            rc7_results.to_csv(out_path, index=False)
            print(f"\n  Table saved -> {out_path}")
            print("  Stage 13 complete.")
        except Exception:
            print("ERROR in Stage 13:\n"); traceback.print_exc()

    # ── Stage 14 — LIME supplementary XAI (SHAP cross-check) ────────────────
    # Opt-in only. Explains the SAME three firm-years as the SHAP waterfalls,
    # on the XGBoost model, and writes a SHAP-vs-LIME agreement table. SHAP
    # remains the primary XAI framework; nothing existing is overwritten.
    if 14 in stages_to_run:
        try:
            from src.explainability.lime_analysis import run_lime_analysis
            print("\n" + "="*60)
            print("  STAGE 14  —  LIME supplementary XAI (SHAP cross-check)")
            print("="*60)
            train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
            test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")

            if not trained_models:
                _try_load_stage3_artifacts(trained_models)
            xgb_info = trained_models.get("xgboost")
            if xgb_info is None:
                print("  ERROR: XGBoost model not available — run Stage 3 first.")
            else:
                run_lime_analysis(
                    model      = xgb_info["model"],
                    train      = train,
                    test       = test,
                    y_prob     = xgb_info["y_prob"],
                    threshold  = xgb_info["threshold"],
                    features   = ALL_FEATURES,
                    model_name = "xgboost",
                )
            print("  Stage 14 complete.")
        except Exception:
            print("ERROR in Stage 14:\n"); traceback.print_exc()

    # ── Stage 15 — RC7 comparison figures (NN overlaid on primary models) ────
    # Opt-in only. Builds ROC, PR and calibration figures that show the neural
    # network (RC7 extension) alongside the three frozen primary models, written
    # to outputs/figures/robustness/. The primary outputs/figures/model/ figures
    # (frozen 3-model portfolio) are never touched. Requires the primary
    # 17-feature models on disk: if a robustness run last overwrote them, the
    # module raises a clear instruction to re-run Stage 3 first.
    if 15 in stages_to_run:
        try:
            from src.robustness.rc7_figures import run_rc7_figures
            print("\n" + "="*60)
            print("  STAGE 15  —  RC7 comparison figures (NN + primary models)")
            print("="*60)
            run_rc7_figures()
            print("  Stage 15 complete.")
        except Exception:
            print("ERROR in Stage 15:\n"); traceback.print_exc()

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print("\nOutput locations:")
    print(f"  Figures   -> outputs/figures/")
    print(f"  Tables    -> outputs/tables/")
    print(f"  Models    -> outputs/models/saved/")
    print(f"  Configs   -> outputs/models/configs/")
    print(
        "\nNext: open writing/main.tex in your LaTeX editor and fill "
        "in the result sections using the saved tables and figures."
    )


def _try_load_stage3_artifacts(trained_models: dict) -> None:
    """Attempt to reload saved model artifacts if stage 3 was skipped."""
    from src.utils.io import load_model, load_config
    from src.config import ALL_FEATURES
    import numpy as np

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        try:
            model  = load_model(name)
            config = load_config(f"{name}_config")
            test   = pd.read_parquet(DATA_SAMPLES / "test.parquet")
            X_test = test[ALL_FEATURES].astype(float).fillna(0).values
            y_prob = model.predict_proba(X_test)[:, 1]
            trained_models[name] = {
                "model":     model,
                "threshold": config["threshold"],
                "y_prob":    y_prob,
                "X_test":    X_test,
            }
        except FileNotFoundError:
            pass
        except ValueError:
            # Saved model was trained on a different feature set (e.g. RC4
            # accounting-only overwrote the primary model file). Skip quietly —
            # Stage 9 does not need these artifacts; re-run Stage 3 afterward
            # to restore the correct primary models.
            print(f"  WARNING: saved {name} model has wrong feature count "
                  f"(robustness run overwrote it) — skipping load. "
                  f"Re-run Stage 3 after Stage 9 to restore primary models.")


def _try_compute_shap_artifacts(trained_models: dict, shap_artifacts: dict) -> None:
    """Recompute SHAP values from saved models when stage 5 was skipped."""
    from src.explainability.shap_global import compute_shap_values

    for name in ["xgboost", "random_forest", "logistic_regression"]:
        info = trained_models.get(name)
        if info is None:
            print(f"  WARNING: no loaded model for {name} — skipping SHAP recomputation.")
            continue
        try:
            print(f"  Recomputing SHAP values for {name}...")
            shap_artifacts[name] = compute_shap_values(info["model"], info["X_test"], name)
        except Exception:
            print(f"  WARNING: SHAP recomputation failed for {name}.")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.superseded import abort_if_superseded
    abort_if_superseded(__file__, "python scripts/run_thesis_pipeline.py")
    parser = argparse.ArgumentParser(
        description="Master pipeline for the thesis empirical analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from-stage", type=int, metavar="N",
        help="Start from stage N (1–10). Assumes earlier stages are already done.",
    )
    parser.add_argument(
        "--stages", type=int, nargs="+", metavar="N",
        help="Run only specific stages (e.g., --stages 2 3 4).",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP stages 5–8 (useful for quick end-to-end test).",
    )
    parser.add_argument(
        "--reuse-configs", action="store_true",
        help=(
            "Skip Optuna tuning and reuse best_params from saved YAML "
            "configs in outputs/models/configs/. Used after data-layer "
            "fixes (A, F1, F2, F4) to re-fit the same hyperparameter "
            "configuration on a corrected sample without paying the "
            "~2-hour Optuna re-tune cost."
        ),
    )
    args = parser.parse_args()
    main(args)
