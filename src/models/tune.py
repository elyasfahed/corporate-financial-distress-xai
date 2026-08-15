"""
Bayesian hyperparameter optimisation using Optuna.
====================================================
Implements rolling-origin cross-validation WITHIN the training sample
for unbiased hyperparameter selection (Blueprint v4 §9.3).

Protocol (Blueprint v4 §9.3 — frozen):
  - Method    : Bayesian optimisation (Optuna TPE sampler)
  - Trials    : 100 per model
  - CV scheme : Rolling-origin CV within the retained training years
  - Objective : PR-AUC (most informative for severely imbalanced data)
  - After tuning: re-fit winning config on FULL training set

Rolling-origin CV:
  Each fold uses all data up to year T as training and year T+1 as
  validation. This preserves the chronological structure and prevents
  any future information from entering the cross-validation folds.

Blueprint v4 reference: §9.3
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import (
    RANDOM_SEED,
    OPTUNA_TRIALS,
    TUNE_METRIC,
    ALL_FEATURES,
    TRAIN_END_YEAR,
)

# Suppress Optuna INFO-level logging (too verbose)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Number of rolling-origin CV folds within the training set
_CV_FOLDS = 5


def rolling_origin_cv_splits(
    train_df: pd.DataFrame,
    n_folds: int = _CV_FOLDS,
) -> list[tuple[pd.Index, pd.Index]]:
    """
    Generate rolling-origin CV fold indices from the training panel.

    D7 fix — implementation now matches the methodology prose (Ch. 5 §5.5):
    five folds with an EXPANDING training window and a 1-YEAR validation
    fold per fold (the previous implementation used 3-year validation folds
    derived from `fold_size = n_years // (n_folds + 1)`).

    For the final-primary specification (training period 1990--2008),
    this yields five rolling folds with validation years 2004, 2005, 2006,
    2007, and 2008, each anchored on an expanding window beginning in 1990.
    No minimum training-window length is enforced; the window is simply every
    retained year strictly before the validation year, so the earliest fold
    trains on 14 years (1990--2003) and the latest on 18 (1990--2007).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training set. Must contain 'fyear'.
    n_folds : int
        Number of rolling-origin folds.

    Returns
    -------
    list of (train_idx, val_idx) index pairs.
    """
    years = sorted(train_df["fyear"].unique())
    n_years = len(years)

    # We need at least n_folds + 1 years (each of the last n_folds years is
    # a one-year validation fold, with at least one year of prior training).
    if n_years < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} fiscal years for {n_folds} rolling folds; "
            f"only {n_years} available."
        )

    # The validation years are the LAST n_folds years of the training period.
    # Each fold's training set is everything strictly before that validation year.
    splits = []
    val_year_block = years[-n_folds:]  # final-primary: [2004, ..., 2008]

    for val_year in val_year_block:
        train_years = [y for y in years if y < val_year]
        if not train_years:
            continue
        tr_idx  = train_df.index[train_df["fyear"].isin(train_years)]
        val_idx = train_df.index[train_df["fyear"] == val_year]
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((tr_idx, val_idx))

    return splits


def prepare_fold_matrices(
    train_df: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    fold_safe: bool = False,
    purge_horizon_days: int | None = None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
    impute_features: list[str] | None = None,
) -> list[tuple]:
    """
    Materialise the rolling-origin CV folds ONCE as numpy matrices.

    Fold preprocessing (purging, in-fold winsorisation/imputation) is
    hyperparameter-independent, so computing it inside every Optuna trial
    would repeat identical work n_trials times (~500 redundant
    preprocessing passes per model at 100 trials x 5 folds). tune_model
    calls this once and passes the result to objective().

    Returns
    -------
    list of (X_tr, y_tr, X_val, y_val) tuples, one per usable fold
    (folds with no positive validation cases are dropped, matching the
    frozen behaviour).
    """
    splits = rolling_origin_cv_splits(train_df)
    folds = []
    for tr_idx, val_idx in splits:
        if purge_horizon_days is not None:
            from src.models.fold_safe_cv import purge_fold_train
            tr_idx = purge_fold_train(train_df, tr_idx, val_idx,
                                      horizon_days=purge_horizon_days)
        tr  = train_df.loc[tr_idx]
        val = train_df.loc[val_idx]
        if fold_safe:
            from src.models.fold_safe_cv import fold_safe_preprocess
            tr, val = fold_safe_preprocess(tr, val, sic_col=sic_col,
                                           peer_rule=peer_rule,
                                           impute_features=impute_features)
        y_val = val["distress"].astype(int).values
        if y_val.sum() == 0:
            continue
        folds.append((
            tr[features].astype(float).fillna(0).values,
            tr["distress"].astype(int).values,
            val[features].astype(float).fillna(0).values,
            y_val,
        ))
    return folds


def objective(
    trial,
    model_name: str,
    train_df: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    fold_safe: bool = False,
    purge_horizon_days: int | None = None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
    precomputed_folds: list[tuple] | None = None,
    impute_features: list[str] | None = None,
) -> float:
    """
    Optuna objective function: cross-validated PR-AUC on training folds.

    Parameters
    ----------
    trial : optuna.Trial
    model_name : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.
    train_df : pd.DataFrame
        Full training set. With fold_safe=True this must be the RAW
        (pre-winsorisation, pre-imputation) training frame.
    features : list[str]
        Feature columns to use.
    fold_safe : bool, default False
        False reproduces the frozen behaviour (folds read the
        already-preprocessed frame). True re-fits winsorisation and
        imputation INSIDE each fold's training years
        (src/models/fold_safe_cv.py; v2 profile).
    purge_horizon_days : int or None, default None
        None (frozen) applies no purging. Set to the label horizon
        (e.g. DISTRESS_HORIZON_DAYS) to drop fold-train rows whose label
        window crosses the fold's earliest validation filing date.
    sic_col, peer_rule : str
        In-fold imputation options, used only when fold_safe=True.

    Returns
    -------
    float
        Mean PR-AUC across rolling-origin CV folds (to maximise).
    """
    # Import here to avoid circular imports at module load time
    from src.models.logistic_regression import build_logistic_regression
    from src.models.logistic_regression import get_optuna_search_space as lr_space
    from src.models.random_forest import build_random_forest
    from src.models.random_forest import get_optuna_search_space as rf_space
    from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
    from src.models.xgboost_model import get_optuna_search_space as xgb_space

    # Sample hyperparameters for the requested model
    if model_name == "logistic_regression":
        params = lr_space(trial)
        def build(y_tr): return build_logistic_regression(**params)
    elif model_name == "random_forest":
        params = rf_space(trial)
        def build(y_tr): return build_random_forest(**params)
    elif model_name == "xgboost":
        params = xgb_space(trial)
        def build(y_tr):
            spw = compute_scale_pos_weight(y_tr)
            return build_xgboost(**params, scale_pos_weight=spw)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    # Rolling-origin CV — folds are either supplied precomputed (fast
    # path used by tune_model) or materialised here (frozen behaviour).
    if precomputed_folds is not None:
        folds = precomputed_folds
    else:
        folds = prepare_fold_matrices(
            train_df, features, fold_safe=fold_safe,
            purge_horizon_days=purge_horizon_days,
            sic_col=sic_col, peer_rule=peer_rule,
            impute_features=impute_features,
        )

    pr_aucs = []
    for X_tr, y_tr, X_val, y_val in folds:
        model = build(y_tr)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, y_prob))

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def tune_model(
    model_name: str,
    train_df: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_trials: int = OPTUNA_TRIALS,
    fold_safe: bool = False,
    purge_horizon_days: int | None = None,
    sic_col: str = "sich",
    peer_rule: str = "rows",
    storage_dir=None,
    impute_features: list[str] | None = None,
) -> tuple[dict, optuna.Study]:
    """
    Run Bayesian hyperparameter optimisation for a given model.

    Parameters
    ----------
    model_name : str
        One of 'logistic_regression', 'random_forest', 'xgboost'.
    train_df : pd.DataFrame
        Full retained training set (1990--2008 in final-primary).
    features : list[str]
        Feature columns.
    n_trials : int
        Number of Optuna trials (default 100 per Blueprint v4 §9.3).
    fold_safe, purge_horizon_days, sic_col, peer_rule
        Fold-safe CV options, passed through to objective(). Defaults
        reproduce the frozen behaviour; the v2 profile sets
        fold_safe=True, purge_horizon_days=DISTRESS_HORIZON_DAYS and
        tunes on the RAW training frame (src/models/fold_safe_cv.py).

    Returns
    -------
    best_params : dict
        Hyperparameters of the best trial.
    study : optuna.Study
        Full Optuna study object (for diagnostics and logging).
    """
    print(f"  Tuning {model_name} ({n_trials} trials) ...")
    if storage_dir is not None:
        # Persistent study (audit recommendation): a multi-hour search
        # survives interruption and resumes at the completed-trial count.
        from pathlib import Path
        db = Path(storage_dir) / f"optuna_{model_name}.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            storage=f"sqlite:///{db}",
            study_name=model_name,
            load_if_exists=True,
        )
        done = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        if done:
            print(f"  Resuming persisted study: {done} complete trials found")
        n_trials = max(0, n_trials - done)
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
    # Materialise fold matrices ONCE (purging + in-fold preprocessing are
    # hyperparameter-independent; recomputing them per trial would waste
    # n_trials x n_folds preprocessing passes).
    folds = prepare_fold_matrices(
        train_df, features, fold_safe=fold_safe,
        purge_horizon_days=purge_horizon_days,
        sic_col=sic_col, peer_rule=peer_rule,
        impute_features=impute_features,
    )
    print(f"  CV folds prepared: {len(folds)} "
          f"(fold_safe={fold_safe}, purge={purge_horizon_days})")

    target_complete = done + n_trials if storage_dir is not None else n_trials
    while True:
        complete = sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        )
        remaining = target_complete - complete
        if remaining <= 0:
            break
        before = complete
        study.optimize(
            lambda trial: objective(trial, model_name, train_df, features,
                                    precomputed_folds=folds),
            n_trials=remaining,
            show_progress_bar=True,
        )
        complete = sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        )
        if complete <= before:
            raise RuntimeError(
                f"Optuna study {model_name!r} made no progress toward "
                f"{target_complete} completed trials"
            )
    best_params = study.best_params
    best_value  = study.best_value
    print(f"  Best PR-AUC (CV): {best_value:.4f} | Params: {best_params}")
    return best_params, study
