"""
RC3 with an independent hyperparameter search.
==============================================
**Classification: pre-specified robustness check, re-tuned.**

Why this module exists
----------------------
``rc3_smote.run_rc3`` holds the *primary* tuned configurations fixed and varies
only the imbalance treatment. That is informative about the substitution a
practitioner makes when swapping SMOTE into an existing pipeline, but it is not
what the other pre-specified checks do: RC1, RC2, RC4 and RC5 each re-tune from
scratch, so each design variant gets its best shot. Chapter 7 originally claimed
the same of RC3, which was false; the claim was corrected, and this module
closes the gap properly.

The distinction matters here more than elsewhere. The primary configurations
were selected against a 74,172-row training sample carrying 3.01% positives.
SMOTE presents the learner with a resampled sample of roughly twice the size at
parity, and settings that regularise sensibly in the first regime need not do so
in the second --- XGBoost's selected ``min_child_weight`` of 48 and its 200
boosting rounds at a learning rate of 0.016 are calibrated to a rare-event
problem that resampling has removed. Holding them fixed therefore confounds the
imbalance method with hyperparameter compatibility.

Resampling inside the folds, never across them
----------------------------------------------
The one thing this search must not do is let synthetic minority points reach a
validation fold. SMOTE interpolates between minority neighbours, so a synthetic
point built from a validation observation's neighbours leaks that observation
into training and inflates the cross-validated score --- the classic way
SMOTE-plus-CV goes wrong. The order enforced per fold is therefore:

1. fold-safe preprocessing (winsorisation, the >=8-of-11 coverage filter and
   imputation medians all re-fitted on the fold's own training years, plus
   label-maturity purging) --- inherited unchanged from
   ``tune.prepare_fold_matrices``;
2. continuous-feature standardisation fitted on **fold-training rows only**;
3. SMOTENC applied to **fold-training rows only**;
4. the fold's validation rows are scaled with the fold-training scaler and are
   never resampled.

Steps 1--3 are hyperparameter-independent, so they are computed **once** and
reused across all trials, mirroring what ``prepare_fold_matrices`` already does
for the primary search. Without that, a 100-trial search would rebuild identical
resampled matrices 500 times.

Class weighting is disabled throughout (``class_weight=None`` for the benchmark
and the random forest, ``scale_pos_weight=1.0`` for XGBoost), so the imbalance
correction is not applied twice over. This matches ``run_rc3``.

Output
------
Tuned configurations are written to ``outputs/models/configs/final_primary_rc3/``
and the final evaluation is delegated to ``run_rc3(corrected=True,
params_dir=...)``, so the fitted pipeline is bit-for-bit the one already audited
rather than a second implementation of it.

Run
---
    PYTHONPATH=. python -m scripts.run_rc3_retune
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.config import OPTUNA_TRIALS, OUT_MODELS_CONFIGS, RANDOM_SEED
from src.robustness.rc3_smote import _scale_continuous, smote_resample

#: Artifact namespace for this check. Kept separate from the primary configs so
#: a re-tune can never overwrite them (the §18a defect).
RC3_SPEC = "final_primary_rc3"

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


# ---------------------------------------------------------------------------
# Fold preparation (hyperparameter-independent — computed once)
# ---------------------------------------------------------------------------

def prepare_resampled_folds(
    train_raw: pd.DataFrame,
    features: list[str],
    purge_horizon_days: int,
    sic_col: str,
    peer_rule: str,
    impute_features: list[str] | None,
    corrected: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Materialise the rolling-origin folds with in-fold scaling and resampling.

    Returns one ``(X_train_resampled, y_train_resampled, X_val_scaled, y_val)``
    tuple per usable fold. The validation arrays are scaled with the fold's own
    training scaler and are **not** resampled.
    """
    from src.models.tune import prepare_fold_matrices

    folds = prepare_fold_matrices(
        train_raw, features,
        fold_safe=True,
        purge_horizon_days=purge_horizon_days,
        sic_col=sic_col,
        peer_rule=peer_rule,
        impute_features=impute_features,
    )

    out = []
    for i, (X_tr, y_tr, X_val, y_val) in enumerate(folds, start=1):
        # _scale_continuous fits on its first argument and transforms all three;
        # the third slot is unused here, so the validation block is passed twice.
        X_tr_s, X_val_s, _, _ = _scale_continuous(X_tr, X_val, X_val, features)
        X_res, y_res = smote_resample(X_tr_s, y_tr, features, corrected=corrected)
        out.append((X_res, y_res, X_val_s, y_val))
        print(f"    fold {i}: train {len(y_tr):,} -> {len(y_res):,} resampled "
              f"({int(y_tr.sum())} positives -> {int(y_res.sum()):,}) | "
              f"val {len(y_val):,} ({int(y_val.sum())} events)")
    return out


# ---------------------------------------------------------------------------
# Model construction with class weighting OFF
# ---------------------------------------------------------------------------

def build_unweighted(model_name: str, params: dict):
    """Instantiate a model with its imbalance correction disabled."""
    from src.models.logistic_regression import build_logistic_regression
    from src.models.random_forest import build_random_forest
    from src.models.xgboost_model import build_xgboost

    if model_name == "logistic_regression":
        return build_logistic_regression(**params, class_weight=None)
    if model_name == "random_forest":
        return build_random_forest(**params, class_weight=None)
    if model_name == "xgboost":
        return build_xgboost(**params, scale_pos_weight=1.0)
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _search_space(model_name: str, trial) -> dict:
    from src.models.logistic_regression import get_optuna_search_space as lr_space
    from src.models.random_forest import get_optuna_search_space as rf_space
    from src.models.xgboost_model import get_optuna_search_space as xgb_space

    return {"logistic_regression": lr_space,
            "random_forest": rf_space,
            "xgboost": xgb_space}[model_name](trial)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def objective(trial, model_name: str, folds) -> float:
    """Mean PR-AUC across the resampled rolling-origin folds."""
    params = _search_space(model_name, trial)
    scores = []
    for X_res, y_res, X_val, y_val in folds:
        model = build_unweighted(model_name, params)
        model.fit(X_res, y_res)
        scores.append(
            average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
    return float(np.mean(scores)) if scores else 0.0


def tune_rc3_model(model_name: str, folds, n_trials: int = OPTUNA_TRIALS,
                   storage_dir: Path | None = None) -> tuple[dict, float]:
    """
    Run the search for one model, resuming a persisted study if present.

    Returns ``(best_params, best_cv_pr_auc)``.
    """
    storage_dir = Path(storage_dir or (OUT_MODELS_CONFIGS / RC3_SPEC))
    storage_dir.mkdir(parents=True, exist_ok=True)
    db = storage_dir / f"optuna_{model_name}.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{db}",
        study_name=f"rc3_{model_name}",
        load_if_exists=True,
    )
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    if done:
        print(f"  Resuming persisted study: {done} complete trials found")
    remaining = max(0, n_trials - done)
    print(f"  Tuning {model_name} under SMOTENC ({remaining} trials to run) ...")
    if remaining:
        study.optimize(lambda t: objective(t, model_name, folds),
                       n_trials=remaining, show_progress_bar=False)

    print(f"    best CV PR-AUC = {study.best_value:.5f}  "
          f"(trial {study.best_trial.number})")
    return dict(study.best_params), float(study.best_value)


def write_config(model_name: str, params: dict, cv_score: float,
                 n_trials: int, configs_dir: Path | None = None) -> Path:
    """Persist a tuned configuration in the format ``run_rc3`` reads."""
    configs_dir = Path(configs_dir or (OUT_MODELS_CONFIGS / RC3_SPEC))
    configs_dir.mkdir(parents=True, exist_ok=True)
    path = configs_dir / f"{model_name}_config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump({
            "model": model_name,
            "best_params": params,
            "best_cv_pr_auc": cv_score,
            "optuna_trials_target": n_trials,
            "imbalance_treatment": "smotenc_in_fold",
            "class_weighting": "disabled",
            "spec": RC3_SPEC,
        }, fh, default_flow_style=False, sort_keys=True)
    return path
