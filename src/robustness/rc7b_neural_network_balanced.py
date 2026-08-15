"""
RC7b — Imbalance-matched neural network (clean model-family comparison).
========================================================================
Motivation
----------
The RC7 neural network (`src/models/neural_network.py`) is the ONLY model
trained without an imbalance correction: sklearn's `MLPClassifier` accepts
neither `class_weight` nor `sample_weight`, whereas LR uses
`class_weight='balanced'`, RF uses `class_weight`, and XGBoost uses
`scale_pos_weight`. That makes the RC7 "model-family" comparison confounded —
"NN is weakest" mixes the *family* effect (MLP vs. trees) with the *imbalance
treatment* effect (none vs. class weighting). One cannot attribute the gap.

This script removes that confound. It gives the MLP the resampling-equivalent
of balanced class weights — **random minority oversampling to parity** — so
that every model now receives the same imbalance correction. The comparison
becomes apples-to-apples: any remaining gap is attributable to the model
family alone.

Why random oversampling (and why it is NOT RC3 SMOTE)
-----------------------------------------------------
`class_weight='balanced'` multiplies each minority example's loss by
n/(2·n_min). Duplicating a minority row k times multiplies its loss
contribution by k. Random oversampling to parity is therefore the
resampling implementation of balanced class weights *in expectation* — it
DUPLICATES existing rows. RC3 (frozen) uses SMOTE, which SYNTHESISES new
interpolated rows. These are different mechanisms, so this extension does
NOT collide with, or amend, the frozen RC3 deviation.

Leakage control
---------------
The oversampler sits AFTER the scaler in an imblearn Pipeline, so:
  * the StandardScaler is fit on the ORIGINAL (unbalanced) training
    distribution — identical to the LR/NN pipelines; and
  * resampling happens ONLY during `.fit()` on the training fold; the
    validation/test folds are scored on their true, untouched distribution.
In rolling-origin CV, folds are split by fiscal year and only the training
years are resampled, so duplicated rows never cross into a validation fold.
Early stopping is disabled for the resampled variant so that duplicated
minority rows cannot appear on both sides of the MLP's internal
early-stopping split; regularisation is supplied by the tuned L2 penalty
(alpha). This also brings the network closer to the no-early-stopping
protocol of the other three models.

Design consistency (everything else held identical to RC7 / the primary models)
-------------------------------------------------------------------------------
  * same 17 predictors, same chronological train/val/test splits
  * same random seed (42)
  * same Optuna rolling-origin CV, PR-AUC objective, same search space
  * same validation-set F1 threshold, locked before test
  * same metrics + firm-block bootstrap PR-AUC CI
  * Platt-scaled Brier reported so calibration is comparable across models

Read-only w.r.t. the frozen pipeline — writes ONLY new files:
  outputs/models/saved/neural_network_balanced.joblib
  outputs/models/configs/neural_network_balanced_config.yaml
  outputs/tables/robustness/rc7b_neural_network_balanced_results.{csv,tex}
  outputs/tables/robustness/model_family_comparison.{csv,tex}
Appends one audit row (run_label="RC7b_NN_balanced") to evaluation_manifest.csv.

Run
---
    python -m src.robustness.rc7b_neural_network_balanced            # full 100-trial re-tune
    python -m src.robustness.rc7b_neural_network_balanced --trials 5 # quick smoke test
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_FEATURES,
    DATA_SAMPLES,
    OPTUNA_TRIALS,
    OUT_MODELS_SAVED,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_ROBUSTNESS,
    OUT_TABLES_MODEL,
    RANDOM_SEED,
)
from src.models.neural_network import _ARCHITECTURES, _architecture_to_tuple
from src.models.evaluate import compute_all_metrics, select_threshold, bootstrap_pr_auc_ci
from src.models.tune import rolling_origin_cv_splits
from src.models.train import _append_evaluation_manifest
from src.analysis.lr_calibration import apply_platt_scaling
from src.utils.tables import save_table

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_COL = "distress"
MODEL_NAME = "neural_network_balanced"


# ---------------------------------------------------------------------------
# Model builder — identical MLP to RC7, plus minority oversampling to parity
# ---------------------------------------------------------------------------

def build_balanced_nn(
    architecture: str = "64_32",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    activation: str = "relu",
    max_iter: int = 300,
) -> ImbPipeline:
    """
    Pipeline(StandardScaler -> RandomOverSampler -> MLPClassifier).

    The scaler precedes the sampler, so it is fit on the original (unbalanced)
    distribution; the sampler balances the scaled training data ONLY during
    fit; the MLP then trains on the balanced set. At predict time the sampler
    is a no-op (imblearn semantics), so the network scores the true
    distribution. MLP settings are identical to `build_neural_network` except
    `early_stopping=False` (see module docstring).
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", RandomOverSampler(random_state=RANDOM_SEED)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=_architecture_to_tuple(architecture),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            solver="adam",
            max_iter=max_iter,
            early_stopping=False,       # see docstring — avoids duplicate leak
            random_state=RANDOM_SEED,
        )),
    ])


def _search_space(trial) -> dict:
    """Identical search space to the RC7 neural network."""
    return {
        "architecture":       trial.suggest_categorical(
            "architecture", list(_ARCHITECTURES.keys())),
        "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
    }


# ---------------------------------------------------------------------------
# Tuning — same rolling-origin CV / PR-AUC objective as the primary models
# ---------------------------------------------------------------------------

def _objective(trial, train_df: pd.DataFrame, features: list[str]) -> float:
    params = _search_space(trial)
    splits = rolling_origin_cv_splits(train_df)
    pr_aucs = []
    for tr_idx, val_idx in splits:
        tr  = train_df.loc[tr_idx]
        va  = train_df.loc[val_idx]
        X_tr = tr[features].astype(float).fillna(0).values
        y_tr = tr[LABEL_COL].astype(int).values
        X_va = va[features].astype(float).fillna(0).values
        y_va = va[LABEL_COL].astype(int).values
        if y_va.sum() == 0 or y_tr.sum() == 0:
            continue
        model = build_balanced_nn(**params)
        model.fit(X_tr, y_tr)                    # resamples tr-fold internally
        y_prob = model.predict_proba(X_va)[:, 1]  # scores untouched val-fold
        pr_aucs.append(average_precision_score(y_va, y_prob))
    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def tune(train_df: pd.DataFrame, features: list[str], n_trials: int) -> dict:
    """Tune with crash-safe SQLite storage.

    Re-running the SAME command resumes where it left off (until n_trials
    total complete), so a long run survives an interruption (sleep, reboot,
    closed laptop). Falls back to an in-memory study if storage is unavailable
    — the run still completes, it just cannot resume. Delete
    outputs/models/configs/rc7b_optuna.db to force a fresh study.
    """
    print(f"  Tuning {MODEL_NAME} (target {n_trials} trials, rolling-origin CV) ...")
    db_path = OUT_MODELS_CONFIGS / "rc7b_optuna.db"
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            study_name="rc7b_nn_balanced",
            storage=f"sqlite:///{db_path.as_posix()}",
            load_if_exists=True,
        )
    except Exception as exc:                       # storage unavailable -> no resume
        print(f"  [warn] SQLite storage unavailable ({exc}); in-memory study, no resume.")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    if done:
        print(f"  Resuming - {done} trial(s) already complete; running {remaining} "
              f"more (target {n_trials}).")
    if remaining:
        study.optimize(lambda t: _objective(t, train_df, features),
                       n_trials=remaining, show_progress_bar=True)
    print(f"  Best CV PR-AUC: {study.best_value:.4f} | {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_existing_family() -> pd.DataFrame:
    """Pull the frozen LR/RF/XGB and RC7 raw-NN rows for the comparison table."""
    rows = []
    prim = OUT_TABLES_MODEL / "model_performance_test.csv"
    if prim.exists():
        df = pd.read_csv(prim)
        for _, r in df.iterrows():
            rows.append({
                "model": r["model"], "imbalance_treatment": "class weights",
                "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                "pr_auc_ci_lower": r["pr_auc_ci_lower"],
                "pr_auc_ci_upper": r["pr_auc_ci_upper"],
            })
    rc7 = OUT_TABLES_ROBUSTNESS / "rc7_neural_network_results.csv"
    if rc7.exists():
        r = pd.read_csv(rc7).iloc[0]
        rows.append({
            "model": "neural_network (RC7, raw)",
            "imbalance_treatment": "none (threshold only)",
            "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
            "pr_auc_ci_lower": r["pr_auc_ci_lower"],
            "pr_auc_ci_upper": r["pr_auc_ci_upper"],
        })
    return pd.DataFrame(rows)


def main(n_trials: int) -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    OUT_MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    OUT_MODELS_CONFIGS.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    features = ALL_FEATURES

    print(f"\nRC7b: imbalance-matched neural network - {len(features)} predictors")
    print(f"  train={len(train):,} ({int(train[LABEL_COL].sum())} distress)  "
          f"val={len(val):,}  test={len(test):,}")

    # 1) tune
    best_params = tune(train, features, n_trials)

    # 2) refit on full training set (resampled internally)
    X_train = train[features].astype(float).fillna(0).values
    y_train = train[LABEL_COL].astype(int).values
    model = build_balanced_nn(**best_params)
    model.fit(X_train, y_train)

    # 3) threshold on validation (F1), locked before test
    X_val = val[features].astype(float).fillna(0).values
    y_val = val[LABEL_COL].astype(int).values
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  Optimal threshold (val F1): {threshold:.4f}")

    # 4) evaluate on test (once)
    X_test = test[features].astype(float).fillna(0).values
    y_test = test[LABEL_COL].astype(int).values
    firm_ids = test["gvkey"].values
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_all_metrics(y_test, y_prob, threshold, model_name=MODEL_NAME)
    ci_lo, ci_hi = bootstrap_pr_auc_ci(y_test, y_prob, firm_ids)
    metrics["pr_auc_ci_lower"] = round(ci_lo, 4)
    metrics["pr_auc_ci_upper"] = round(ci_hi, 4)
    metrics["check"] = "RC7b_neural_network_balanced"

    # Platt-scaled Brier for cross-model-comparable calibration
    platt = apply_platt_scaling(model, val, features=features)
    brier_calib = brier_score_loss(y_test, platt.predict_proba(X_test)[:, 1])
    metrics["brier_calibrated"] = round(float(brier_calib), 4)

    print(f"\n  {MODEL_NAME}: PR-AUC={metrics['pr_auc']:.4f} "
          f"[{ci_lo:.4f}, {ci_hi:.4f}]  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"F1={metrics['f1']:.4f}  Brier(raw)={metrics['brier_score']:.4f}  "
          f"Brier(Platt)={metrics['brier_calibrated']:.4f}")

    # 5) persist (new files only)
    joblib.dump(model, OUT_MODELS_SAVED / f"{MODEL_NAME}.joblib")
    with open(OUT_MODELS_CONFIGS / f"{MODEL_NAME}_config.yaml", "w") as f:
        yaml.dump({"model": MODEL_NAME, "best_params": best_params,
                   "threshold": float(threshold),
                   "imbalance_treatment": "random_oversampling_to_parity",
                   "n_trials": n_trials}, f)

    cols = ["check", "model", "prevalence_baseline_pr_auc", "pr_auc", "roc_auc",
            "precision", "recall", "f1", "ks_stat", "brier_score",
            "brier_calibrated", "threshold", "pr_auc_ci_lower", "pr_auc_ci_upper"]
    res_df = pd.DataFrame([metrics])[cols]
    save_table(
        res_df, OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results",
        caption=(
            "RC7b --- Imbalance-matched neural network. Identical to the RC7 MLP "
            "in every design dimension except that the training set is rebalanced "
            "by random minority oversampling to parity (the resampling-equivalent "
            "of the balanced class weights used by the other models; distinct from "
            "the SMOTE used in RC3). Removes the imbalance-treatment confound from "
            "the model-family comparison. Brier(Platt) is the validation-fit "
            "Platt-scaled Brier, comparable across models."),
        label="tab:rc7b_nn_balanced",
    )

    # 6) consolidated model-family comparison table
    fam = _load_existing_family()
    fam = pd.concat([fam, pd.DataFrame([{
        "model": "neural_network (RC7b, balanced)",
        "imbalance_treatment": "oversampling to parity",
        "pr_auc": metrics["pr_auc"], "roc_auc": metrics["roc_auc"],
        "pr_auc_ci_lower": metrics["pr_auc_ci_lower"],
        "pr_auc_ci_upper": metrics["pr_auc_ci_upper"],
    }])], ignore_index=True)
    save_table(
        fam, OUT_TABLES_ROBUSTNESS / "model_family_comparison",
        caption=(
            "Model-family comparison on the held-out test sample (2015--2024). "
            "All models share the same 17 predictors, chronological splits, seed, "
            "tuning protocol, and validation-locked F1 threshold. The raw RC7 MLP "
            "receives no imbalance treatment; the RC7b MLP receives random minority "
            "oversampling to parity, matching the class weighting of LR/RF/XGBoost. "
            "PR-AUC baseline (prevalence) = 0.0142."),
        label="tab:model_family_comparison",
    )

    _append_evaluation_manifest(
        run_label="RC7b_NN_balanced", n_test=len(test),
        n_distress=int(y_test.sum()), features=features,
        output_stem=OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results",
    )

    print("\n  Model-family comparison:")
    print(fam.to_string(index=False))
    print("\nSaved -> rc7b_neural_network_balanced_results.{csv,tex}, "
          "model_family_comparison.{csv,tex}, neural_network_balanced.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                    help="Optuna trials (default: config OPTUNA_TRIALS=100).")
    args = ap.parse_args()
    main(args.trials)
