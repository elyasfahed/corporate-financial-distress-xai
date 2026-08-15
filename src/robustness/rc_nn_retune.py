"""
Per-check hyperparameter search for the balanced network (RC1-RC5).
===================================================================
**Classification: pre-specified robustness check, re-tuned.**

This removes the last protocol asymmetry in Chapter 7. ``rc_nn_final_primary``
runs the network through every check with the *primary* hyperparameters, and
bounds the concession that makes; this module removes the concession by giving
each check its own 100-trial search, exactly as the benchmark and the two tree
ensembles receive.

Cost, and why the design is built around it
-------------------------------------------
One balanced-network fit on this data takes minutes, not seconds, because
oversampling to parity roughly doubles the training sample. A 100-trial search
over five rolling-origin folds is therefore hours to tens of hours **per
check**, and the five checks together are a multi-day computation. Two
consequences are designed in rather than hoped for:

* **Per-check checkpointing.** Each finished check writes its own results CSV
  and is skipped on the next invocation.
* **Persistent Optuna storage.** Each check's study lives in its own SQLite
  file, so an interrupted search resumes at its completed-trial count instead
  of restarting.

Re-running the driver after an interruption of any kind --- a sleep, a reboot,
a deliberate stop --- continues where it left off. This is the same guarantee
the earlier multi-day NN battery was built with, and this repository has
already lost mid-check runs to laptop shutdowns twice.

Fold construction matches the other models exactly
--------------------------------------------------
Folds come from ``tune.prepare_fold_matrices`` under the ``V2_PROFILE``
settings --- fold-safe preprocessing with label-maturity purging --- which is
the same call the primary network's own search used and the same one behind
LR/RF/XGBoost. For RC1 and RC2 the raw training frame is re-labelled before the
folds are built, so the search optimises against the check's own outcome rather
than against the primary label.

RC3 is again the special case: it changes the imbalance treatment itself, so
inside each fold the network gets standardisation fitted on fold-training rows,
SMOTENC on fold-training rows only, and a bare MLP with no sampler. That
mirrors ``rc3_smote_retune`` for the other three models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.config import (ACCOUNTING_FEATURES, DISTRESS_HORIZON_DAYS,
                        MARKET_IMPUTE_FEATURES, OPTUNA_TRIALS,
                        OUT_MODELS_CONFIGS, RANDOM_SEED, V2_PROFILE)
from src.robustness.rc_nn_final_primary import (CHECK_LABEL, get_check_data,
                                                get_check_raw_train)

#: One artifact namespace per check, so no search can overwrite another's.
def spec_for(check: str) -> str:
    return f"final_primary_{check}_nn"


def configs_dir(check: str) -> Path:
    return Path(OUT_MODELS_CONFIGS) / spec_for(check)


def build_folds(check: str, features: list[str]):
    """Fold-safe rolling-origin folds for this check, matching the other models."""
    from src.models.tune import prepare_fold_matrices

    raw = get_check_raw_train(check)
    return prepare_fold_matrices(
        raw, features,
        fold_safe=V2_PROFILE["cv_fold_safe"],
        purge_horizon_days=DISTRESS_HORIZON_DAYS,
        sic_col=V2_PROFILE["sic_col"],
        peer_rule=V2_PROFILE["impute_peer_rule"],
        impute_features=[f for f in ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES
                         if f in features],
    )


def prepare_rc3_folds(folds, features):
    """
    Pre-scale and pre-resample each fold once, for the RC3 search.

    Hyperparameter-independent, so it is done outside the trial loop; without
    this a 100-trial search rebuilds identical SMOTENC output 500 times.
    """
    from src.robustness.rc3_smote import _scale_continuous, smote_resample

    out = []
    for X_tr, y_tr, X_va, y_va in folds:
        X_tr_s, X_va_s, _, _ = _scale_continuous(X_tr, X_va, X_va, features)
        X_res, y_res = smote_resample(X_tr_s, y_tr, features, corrected=True)
        out.append((X_res, y_res, X_va_s, y_va))
    return out


def _objective(trial, folds, check: str) -> float:
    """Mean PR-AUC across folds, under this check's imbalance treatment."""
    from src.robustness.rc7b_neural_network_balanced import (_search_space,
                                                             build_balanced_nn)

    params = _search_space(trial)
    scores = []
    for X_tr, y_tr, X_va, y_va in folds:
        if y_tr.sum() == 0 or y_va.sum() == 0:
            continue
        if check == "rc3":
            # Folds arrive already scaled and resampled; the sampler and scaler
            # steps of the pipeline would be wrong here, so the bare MLP is used.
            model = build_balanced_nn(**params).named_steps["clf"]
        else:
            model = build_balanced_nn(**params)
        model.fit(X_tr, y_tr)
        scores.append(
            average_precision_score(y_va, model.predict_proba(X_va)[:, 1]))
    return float(np.mean(scores)) if scores else 0.0


def tune_check(check: str, folds, n_trials: int = OPTUNA_TRIALS
               ) -> tuple[dict, float, int]:
    """Search one check, resuming a persisted study. Returns (params, cv, done)."""
    d = configs_dir(check)
    d.mkdir(parents=True, exist_ok=True)
    db = d / "optuna_neural_network_balanced.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{db.as_posix()}",
        study_name=f"nn_{check}",
        load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        print(f"    resuming persisted study: {done} complete trials")
    remaining = max(0, n_trials - done)
    if remaining:
        print(f"    running {remaining} trials ...")
        study.optimize(lambda t: _objective(t, folds, check),
                       n_trials=remaining, show_progress_bar=False)
    total = len([t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"    best CV PR-AUC = {study.best_value:.5f} "
          f"(trial {study.best_trial.number}, {total} complete)")
    return dict(study.best_params), float(study.best_value), total


def write_config(check: str, params: dict, cv: float, n_trials: int) -> Path:
    d = configs_dir(check)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "neural_network_balanced_config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump({
            "model": "neural_network_balanced",
            "check": check,
            "spec": spec_for(check),
            "best_params": params,
            "best_cv_pr_auc": cv,
            "optuna_trials_target": n_trials,
            "imbalance_treatment": ("smotenc_in_fold" if check == "rc3"
                                    else "random_oversampling_to_parity"),
        }, fh, default_flow_style=False, sort_keys=True)
    return path


def evaluate_check(check: str, params: dict) -> dict:
    """Refit on the full training split with the tuned params and score test."""
    from src.robustness.rc_nn_final_primary import run_check
    row = run_check(check, params)
    row["tuning"] = "re-tuned (100 trials)"
    return row
