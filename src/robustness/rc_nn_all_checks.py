"""
Neural network across ALL robustness checks (RC1–RC6) — full re-tune.
=====================================================================
Promotes the neural network to a full, equal model in the robustness analysis:
for each check it RE-TUNES the NN with Optuna (rolling-origin CV, PR-AUC
objective, same search space as RC7/RC7b), refits, selects the validation F1
threshold, and evaluates on test ONCE — exactly the protocol LR/RF/XGB receive
inside each robustness check.

Imbalance treatment per check (matches how that check treats the OTHER models):
  RC1, RC2, RC4, RC5, RC6 : random minority oversampling to parity
                            (the resampling-equivalent of the balanced class
                            weights LR/RF/XGB use; identical to the headline
                            balanced NN, RC7b)
  RC3                      : SMOTE  (RC3 *is* the SMOTE imbalance check)

Check definitions (one design dimension varied each; all else fixed):
  RC1 : bankruptcy-only DLSTCD codes (572/574/584)
  RC2 : 6-month prediction horizon
  RC3 : SMOTE instead of oversampling
  RC4 : accounting-only predictors (11; drop the 6 market features)
  RC5 : utilities (SIC 4900–4999) included      [needs samples_rc5/ — see below]
  RC6 : drop RSIZE (collinearity check)

⚠ RUNTIME: ~one NN fit ≈ 105 s on this data; a full 100-trial re-tune is
   ~hours PER CHECK, ~1–3 DAYS for all six. Run it in PyCharm / a terminal and
   leave it. It is SAFE TO INTERRUPT: each finished check is checkpointed to
   rc_nn_<check>_results.csv and skipped on the next run, and Optuna resumes
   each check's study from its SQLite file. Re-running continues where it left
   off. SMOKE-TEST FIRST with a tiny trial count (see bottom of file).

Read-only w.r.t. the frozen pipeline. Writes ONLY new files:
  outputs/tables/robustness/rc_nn_<check>_results.csv     (per-check checkpoint)
  outputs/tables/robustness/rc_nn_all_checks_results.{csv,tex}  (consolidated)
  outputs/models/configs/rc_nn_<check>_optuna.db          (Optuna resume store)
Appends audit rows (run_label="RC_NN_<check>") to evaluation_manifest.csv.
Does NOT touch the existing LR/RF/XGB robustness results or any saved model.

After it finishes, re-run  python -m src.analysis.results_figures  — the full
robustness figure auto-detects this file and draws the NN bar in every check.

Run
---
    python -m src.robustness.rc_nn_all_checks --trials 2 --checks RC4   # smoke test (~min)
    python -m src.robustness.rc_nn_all_checks                           # full job (~days)
    python -m src.robustness.rc_nn_all_checks --checks RC1,RC2          # subset
"""

from __future__ import annotations

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_FEATURES,
    ACCOUNTING_ONLY_FEATURES,
    DATA_SAMPLES,
    DATA_RAW_CRSP,
    DISTRESS_CODES_RC1,
    DISTRESS_CODES_PRIMARY,
    DISTRESS_HORIZON_RC2,
    OPTUNA_TRIALS,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_ROBUSTNESS,
    RANDOM_SEED,
)
from src.data.merge_crsp_compustat import build_distress_label
from src.models.neural_network import _ARCHITECTURES, _architecture_to_tuple
from src.models.evaluate import compute_all_metrics, select_threshold, bootstrap_pr_auc_ci
from src.models.tune import rolling_origin_cv_splits
from src.models.train import _append_evaluation_manifest
from src.utils.tables import save_table

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_COL = "distress"
MODEL_NAME = "neural_network"
ALL_CHECKS = ["RC1", "RC2", "RC3", "RC4", "RC5", "RC6"]


# ---------------------------------------------------------------------------
# Per-check data preparation — faithfully mirrors each RC's design dimension
# ---------------------------------------------------------------------------

def _relabel(split: pd.DataFrame, delist: pd.DataFrame, codes, horizon_days=None):
    split = split.drop(columns=[LABEL_COL], errors="ignore")
    kw = {"codes": codes}
    if horizon_days is not None:
        kw["horizon_days"] = horizon_days
    return build_distress_label(split, delist, **kw)


def get_check_data(check: str):
    """
    Return (train, val, test, features, sampler_kind) for one check,
    or None if the check cannot be run on this machine (RC5 without samples_rc5).
    """
    train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")

    if check == "RC1":
        delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
        return (_relabel(train, delist, DISTRESS_CODES_RC1),
                _relabel(val,   delist, DISTRESS_CODES_RC1),
                _relabel(test,  delist, DISTRESS_CODES_RC1),
                ALL_FEATURES, "oversample")

    if check == "RC2":
        delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
        return (_relabel(train, delist, DISTRESS_CODES_PRIMARY, DISTRESS_HORIZON_RC2),
                _relabel(val,   delist, DISTRESS_CODES_PRIMARY, DISTRESS_HORIZON_RC2),
                _relabel(test,  delist, DISTRESS_CODES_PRIMARY, DISTRESS_HORIZON_RC2),
                ALL_FEATURES, "oversample")

    if check == "RC3":
        return train, val, test, ALL_FEATURES, "smote"

    if check == "RC4":
        return train, val, test, ACCOUNTING_ONLY_FEATURES, "oversample"

    if check == "RC5":
        from src.robustness.rc5_include_utilities import _load_rc5_splits
        splits = _load_rc5_splits()
        if splits is None:
            return None
        tr, va, te = splits
        return tr, va, te, ALL_FEATURES, "oversample"

    if check == "RC6":
        feats = [f for f in ALL_FEATURES if f != "RSIZE"]
        return train, val, test, feats, "oversample"

    raise ValueError(f"Unknown check: {check!r}")


# ---------------------------------------------------------------------------
# Model + tuning — identical NN to RC7/RC7b, sampler chosen per check
# ---------------------------------------------------------------------------

def build_nn(sampler_kind: str, architecture="64_32", alpha=1e-4,
             learning_rate_init=1e-3, activation="relu", max_iter=300) -> ImbPipeline:
    sampler = (SMOTE(random_state=RANDOM_SEED) if sampler_kind == "smote"
               else RandomOverSampler(random_state=RANDOM_SEED))
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", sampler),
        ("clf", MLPClassifier(
            hidden_layer_sizes=_architecture_to_tuple(architecture),
            activation=activation, alpha=alpha,
            learning_rate_init=learning_rate_init, solver="adam",
            max_iter=max_iter, early_stopping=False, random_state=RANDOM_SEED,
        )),
    ])


def _search_space(trial) -> dict:
    return {
        "architecture":       trial.suggest_categorical("architecture", list(_ARCHITECTURES.keys())),
        "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
    }


def _objective(trial, train_df, features, sampler_kind) -> float:
    params = _search_space(trial)
    pr_aucs = []
    for tr_idx, va_idx in rolling_origin_cv_splits(train_df):
        tr, va = train_df.loc[tr_idx], train_df.loc[va_idx]
        X_tr = tr[features].astype(float).fillna(0).values
        y_tr = tr[LABEL_COL].astype(int).values
        X_va = va[features].astype(float).fillna(0).values
        y_va = va[LABEL_COL].astype(int).values
        if y_tr.sum() == 0 or y_va.sum() == 0:
            continue
        model = build_nn(sampler_kind, **params)
        model.fit(X_tr, y_tr)
        pr_aucs.append(average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))
    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def tune(check: str, train_df, features, sampler_kind, n_trials: int) -> dict:
    """Tune with crash-safe SQLite storage so each check resumes after interruption."""
    print(f"  Tuning NN for {check} (target {n_trials} trials, rolling-origin CV) ...")
    db_path = OUT_MODELS_CONFIGS / f"rc_nn_{check}_optuna.db"
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            study_name=f"rc_nn_{check}",
            storage=f"sqlite:///{db_path.as_posix()}",
            load_if_exists=True,
        )
    except Exception as exc:
        print(f"  [warn] SQLite storage unavailable ({exc}); in-memory study, no resume.")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - done)
    if done:
        print(f"  Resuming {check} — {done} trial(s) done; running {remaining} more.")
    if remaining:
        study.optimize(lambda t: _objective(t, train_df, features, sampler_kind),
                       n_trials=remaining, show_progress_bar=True)
    print(f"  {check} best CV PR-AUC: {study.best_value:.4f} | {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Per-check runner with checkpoint/resume
# ---------------------------------------------------------------------------

_RESULT_COLS = ["check", "model", "prevalence_baseline_pr_auc", "pr_auc", "roc_auc",
                "precision", "recall", "f1", "ks_stat", "brier_score", "threshold",
                "pr_auc_ci_lower", "pr_auc_ci_upper"]


def _checkpoint_path(check: str):
    return OUT_TABLES_ROBUSTNESS / f"rc_nn_{check}_results.csv"


def run_check(check: str, n_trials: int) -> pd.DataFrame | None:
    ckpt = _checkpoint_path(check)
    sample_file = DATA_SAMPLES / "test.parquet"
    if ckpt.exists() and ckpt.stat().st_mtime >= sample_file.stat().st_mtime:
        print(f"  Checkpoint found — skipping {check}, loading {ckpt.name}")
        return pd.read_csv(ckpt)

    data = get_check_data(check)
    if data is None:
        print(f"  *** {check} SKIPPED *** — RC5 needs the utilities sample. "
              "Build it first:  python -m scripts.build_rc5_dataset")
        return None

    train_c, val_c, test_c, features, sampler_kind = data
    print(f"\n=== {check}  (features={len(features)}, imbalance={sampler_kind}) ===")
    print(f"  train={len(train_c):,} ({int(train_c[LABEL_COL].sum())} distress)  "
          f"val={len(val_c):,}  test={len(test_c):,} ({int(test_c[LABEL_COL].sum())} distress)")

    t0 = time.time()
    best_params = tune(check, train_c, features, sampler_kind, n_trials)

    # Refit on full training set, threshold on validation, evaluate on test once.
    X_tr = train_c[features].astype(float).fillna(0).values
    y_tr = train_c[LABEL_COL].astype(int).values
    model = build_nn(sampler_kind, **best_params)
    model.fit(X_tr, y_tr)

    X_va = val_c[features].astype(float).fillna(0).values
    y_va = val_c[LABEL_COL].astype(int).values
    threshold = select_threshold(y_va, model.predict_proba(X_va)[:, 1])

    X_te = test_c[features].astype(float).fillna(0).values
    y_te = test_c[LABEL_COL].astype(int).values
    firm_ids = test_c["gvkey"].values
    y_prob = model.predict_proba(X_te)[:, 1]

    metrics = compute_all_metrics(y_te, y_prob, threshold, model_name=MODEL_NAME)
    ci_lo, ci_hi = bootstrap_pr_auc_ci(y_te, y_prob, firm_ids)
    metrics["pr_auc_ci_lower"] = round(ci_lo, 4)
    metrics["pr_auc_ci_upper"] = round(ci_hi, 4)
    metrics["check"] = check
    row = pd.DataFrame([metrics])[_RESULT_COLS]

    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    row.to_csv(ckpt, index=False)
    _append_evaluation_manifest(run_label=f"RC_NN_{check}", n_test=len(test_c),
                                n_distress=int(y_te.sum()), features=features,
                                output_stem=ckpt.with_suffix(""))
    dt = (time.time() - t0) / 60
    print(f"  {check} NN: PR-AUC={metrics['pr_auc']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
          f"ROC-AUC={metrics['roc_auc']:.4f}  ({dt:.1f} min)  -> {ckpt.name}")
    return row


def main(checks: list[str], n_trials: int) -> None:
    print("="*64)
    print(f"  NEURAL NETWORK ACROSS ROBUSTNESS CHECKS  (trials={n_trials})")
    print(f"  Checks: {', '.join(checks)}")
    print("  Safe to interrupt — finished checks are checkpointed and skipped.")
    print("="*64)

    rows = []
    for check in checks:
        result = run_check(check, n_trials)
        if result is not None:
            rows.append(result)

    if rows:
        consolidated = pd.concat(rows, ignore_index=True)
        save_table(
            consolidated, OUT_TABLES_ROBUSTNESS / "rc_nn_all_checks_results",
            caption=(
                "Neural network across robustness checks. For each check the MLP "
                "is re-tuned (Optuna, rolling-origin CV, PR-AUC objective), refit, "
                "thresholded on validation (F1) and evaluated once on test — the "
                "same protocol the other models receive. Imbalance is handled by "
                "random oversampling to parity (RC3: SMOTE), matching each check's "
                "treatment of LR/RF/XGBoost. PR-AUC CIs are 95\\% firm-block "
                "bootstrap (1{,}000 resamples)."),
            label="tab:rc_nn_all_checks",
        )
        print("\nConsolidated NN-across-checks results:")
        print(consolidated.to_string(index=False))
        print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'rc_nn_all_checks_results.csv'}")
        print("Now re-run:  python -m src.analysis.results_figures")
    else:
        print("\nNo checks completed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=OPTUNA_TRIALS,
                    help="Optuna trials per check (default: config OPTUNA_TRIALS=100).")
    ap.add_argument("--checks", type=str, default=",".join(ALL_CHECKS),
                    help="Comma-separated checks to run, e.g. RC1,RC2 (default: all).")
    args = ap.parse_args()
    selected = [c.strip().upper() for c in args.checks.split(",") if c.strip()]
    main(selected, args.trials)
