"""
v2 neural network (imbalance-matched) — co-primary model under the
corrected data policy.
=====================================================================
Companion to scripts/run_v2_rebuild.py stage d: the NN is
co-primary (Implementation Status §6), so it must be re-tuned under the
SAME corrected policy as the v2 LR/RF/XGB:

  * v2 splits (data/processed_v2/samples), FY1990-2023, corrected label
  * 18 features (frozen 17 + MB_MISSING), no model-time zero-fill
    (NaN gate aborts if any feature carries NaN)
  * fold-safe rolling-origin CV: in-fold winsorisation -> >=8/11
    coverage filter -> imputation (_sic, per_feature), plus 365d
    label-maturity purging — identical to the LR/RF/XGB tuning folds
  * 100-trial Optuna TPE, PR-AUC objective, persistent per-model
    storage (interruption-safe resume)
  * headline variant = imbalance-matched ("balanced") NN — random
    minority oversampling to parity (RC7b design; the raw RC7 NN is a
    sensitivity, regenerated later with the other secondary evidence)
  * artifacts under the v2 namespace only; run manifest refreshed so it
    records the NN checksum next to the three primary models

Run with the project .venv (see requirements-lock.txt):
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_nn
    ... --trials 2   # smoke test only (does not overwrite a full run)
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

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import (
    ACCOUNTING_FEATURES,
    DATA_FEATURES_V2,
    DATA_ROOT,
    DATA_SAMPLES_V2,
    DISTRESS_HORIZON_DAYS,
    MARKET_IMPUTE_FEATURES,
    OPTUNA_TRIALS,
    OUT_TABLES_ROBUSTNESS,
    OUT_MODELS_CONFIGS,
    RANDOM_SEED,
    V2_PROFILE,
)
from src.robustness.rc7b_neural_network_balanced import (
    build_balanced_nn,
    _search_space,
)
from src.models.evaluate import (
    bootstrap_pr_auc_ci,
    compute_all_metrics,
    select_threshold,
)
from src.models.train import _append_evaluation_manifest, resolve_artifact_dirs
from src.models.tune import prepare_fold_matrices
from src.analysis.lr_calibration import apply_platt_scaling
from src.utils.run_manifest import write_run_manifest
from src.utils.tables import save_table
from scripts.run_v2_rebuild import (
    COMP_V2_PATH,
    DELIST_V2_PATH,
    PANEL_V2_PATH,
    SECNAMES_V2_PATH,
    _assert_no_feature_nans,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_COL = "distress"
MODEL_NAME = "neural_network_balanced"
RESULTS_STEM = "neural_network_balanced_results"
RESULTS_DIR = OUT_TABLES_ROBUSTNESS / V2_PROFILE["spec"]


def _complete(saved_dir: Path, configs_dir: Path) -> bool:
    needed = [
        saved_dir / f"{MODEL_NAME}.joblib",
        configs_dir / f"{MODEL_NAME}_config.yaml",
        RESULTS_DIR / f"{RESULTS_STEM}.csv",
    ]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        print("  v2 NN incomplete — missing: " + ", ".join(missing))
    return not missing


def main(n_trials: int, force: bool = False) -> None:
    if ".venv" not in sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv "
              f"(sys.prefix={sys.prefix}) — package versions are not the "
              "pinned ones (requirements-lock.txt).")

    features = list(V2_PROFILE["feature_set"])            # 18 incl. MB_MISSING
    saved_dir, configs_dir = resolve_artifact_dirs(V2_PROFILE["spec"])
    if _complete(saved_dir, configs_dir) and not force:
        print(f"  [skip] complete v2 NN artifact set exists in {saved_dir}")
        return

    print("Loading v2 splits ...")
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    train_raw = pd.read_parquet(DATA_SAMPLES_V2 / "train_raw.parquet")
    print(f"  Train {len(train):,} | Val {len(val):,} | Test {len(test):,} "
          f"| events(test) {int(test[LABEL_COL].sum()):,} "
          f"(prev {test[LABEL_COL].mean():.2%})")
    _assert_no_feature_nans({"train": train, "val": val, "test": test}, features)

    # ── 1. Tune on fold-safe rolling-origin CV (identical folds to LR/RF/XGB)
    print("\nPreparing fold-safe CV folds (winsor -> >=8/11 -> impute + purge) ...")
    folds = prepare_fold_matrices(
        train_raw, features,
        fold_safe=V2_PROFILE["cv_fold_safe"],
        purge_horizon_days=DISTRESS_HORIZON_DAYS,
        sic_col=V2_PROFILE["sic_col"],
        peer_rule=V2_PROFILE["impute_peer_rule"],
        impute_features=ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES,
    )
    print(f"  CV folds prepared: {len(folds)}")

    def objective(trial) -> float:
        params = _search_space(trial)
        pr_aucs = []
        for X_tr, y_tr, X_va, y_va in folds:
            if y_tr.sum() == 0 or y_va.sum() == 0:
                continue
            model = build_balanced_nn(**params)
            model.fit(X_tr, y_tr)                      # oversamples tr-fold only
            pr_aucs.append(
                average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))
        return float(np.mean(pr_aucs)) if pr_aucs else 0.0

    db = configs_dir / f"optuna_{MODEL_NAME}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{db.as_posix()}",
        study_name=MODEL_NAME,
        load_if_exists=True,
    )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    if done:
        print(f"  Resuming persisted study: {done} finished trials; "
              f"{remaining} to go (target {n_trials}).")
    while remaining:
        before = done
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
        done = sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        )
        if done <= before:
            raise RuntimeError(
                f"NN Optuna study made no progress toward {n_trials} "
                "completed trials"
            )
        remaining = max(0, n_trials - done)
    best_params = study.best_params
    print(f"  Best CV PR-AUC: {study.best_value:.4f} | {best_params}")

    # ── 2. Refit on the full (outer-preprocessed) v2 training set ──────────
    X_train = train[features].astype(float).values
    y_train = train[LABEL_COL].astype(int).values
    model = build_balanced_nn(**best_params)
    model.fit(X_train, y_train)

    # ── 3. Threshold on validation (F1), locked before test ────────────────
    X_val = val[features].astype(float).values
    y_val = val[LABEL_COL].astype(int).values
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  Optimal threshold (val F1): {threshold:.4f}")

    # ── 4. Evaluate on test (once) ──────────────────────────────────────────
    X_test = test[features].astype(float).values
    y_test = test[LABEL_COL].astype(int).values
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_all_metrics(y_test, y_prob, threshold, model_name=MODEL_NAME)
    ci_lo, ci_hi = bootstrap_pr_auc_ci(y_test, y_prob, test["gvkey"].values)
    metrics["pr_auc_ci_lower"] = round(ci_lo, 4)
    metrics["pr_auc_ci_upper"] = round(ci_hi, 4)
    metrics["check"] = "Primary specification"

    platt = apply_platt_scaling(model, val, features=features)
    metrics["brier_calibrated"] = round(
        float(brier_score_loss(y_test, platt.predict_proba(X_test)[:, 1])), 4)

    print(f"\n  {MODEL_NAME} ({V2_PROFILE['spec']}): "
          f"PR-AUC={metrics['pr_auc']:.4f} "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"F1={metrics['f1']:.4f}  Brier(Platt)={metrics['brier_calibrated']:.4f}")

    # ── 5. Persist under the v2 namespace only ──────────────────────────────
    saved_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, saved_dir / f"{MODEL_NAME}.joblib")
    with open(configs_dir / f"{MODEL_NAME}_config.yaml", "w") as f:
        yaml.dump({"model": MODEL_NAME, "best_params": best_params,
                   "threshold": float(threshold),
                   "imbalance_treatment": "random_oversampling_to_parity",
                   "optuna_trials_target": n_trials,
                   "optuna_trials_complete": done,
                   "optuna_trials_total": len(study.trials),
                   "best_cv_pr_auc": float(study.best_value)}, f)

    cols = ["check", "model", "prevalence_baseline_pr_auc", "pr_auc", "roc_auc",
            "precision", "recall", "f1", "ks_stat", "brier_score",
            "brier_calibrated", "threshold", "pr_auc_ci_lower", "pr_auc_ci_upper"]
    save_table(
        pd.DataFrame([metrics])[cols],
        RESULTS_DIR / RESULTS_STEM,
        caption=(
            "Imbalance-matched neural network (co-primary model) under the "
            "final primary data policy: standard FYE dating and the literature-"
            "aligned CRSP performance-delisting definition, "
            "date-ranged universe, FY2023 cutoff, 18 predictors incl. "
            "MB\\_MISSING, market-feature imputation, fold-safe purged CV. "
            "Random minority oversampling to parity matches the class "
            "weighting of LR/RF/XGBoost (RC7b design). Brier(Platt) is the "
            "validation-fit Platt-scaled Brier."),
        label="tab:v2_nn_balanced",
    )

    _append_evaluation_manifest(
        run_label="primary_neural_network_balanced", n_test=len(test),
        n_distress=int(y_test.sum()), features=features,
        output_stem=RESULTS_DIR / RESULTS_STEM,
    )

    # Refresh the v2 run manifest so the NN checksum is recorded next to
    # the three primary models (globs *.joblib in saved_dir).
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
            OUT_MODELS_CONFIGS / V2_PROFILE["spec"] /
            "outcome_definition_approval.yaml",
        ],
        extra={"n_trials": n_trials, "post_calibration": True, "post_nn": True},
    )

    print(f"\nSaved -> {saved_dir / MODEL_NAME}.joblib, "
          f"{RESULTS_STEM}.{{csv,tex}}, run manifest refreshed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                    help="Optuna trials (default: config OPTUNA_TRIALS=100).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if the artifact set exists.")
    args = ap.parse_args()
    main(args.trials, force=args.force)
