"""
Neural network (multi-layer perceptron) — extension model.
============================================================
This raw network was added after the three-model portfolio specified in the
frozen Pre-Specified Empirical Design.
Following the author's post-freeze decision, a neural network is presented as
a fourth co-primary model. The fair headline comparison uses the separate RC7b
imbalance-matched network; this raw RC7 model is retained as a sensitivity
because it received no training-stage imbalance correction.

Purpose
-------
Tests whether the ML-vs-benchmark finding (H1) is sensitive to the model
*family*: trees (RF, XGBoost) vs. a feed-forward neural network. A
consistent ordering across families strengthens the headline result;
a divergence is discussed substantively.

Design consistency (held identical to the primary models)
----------------------------------------------------------
  - Same 17 predictors, same train/val/test chronological splits
  - Same random seed (42)
  - Same Optuna PR-AUC tuning on rolling-origin CV within the training set
  - Same validation-set F1 threshold selection, locked before test
  - Same evaluation metrics and bootstrap PR-AUC CI

Imbalance treatment (documented deviation)
------------------------------------------
sklearn's MLPClassifier does NOT expose `class_weight` or `sample_weight`,
so the primary "balanced class weights" treatment cannot be applied to the
network during training. Imbalance is instead handled at the decision stage
via the validation-selected F1 threshold (identical procedure to the other
models). This difference is disclosed as a transparent deviation. Although
PR-AUC and ROC-AUC are threshold-free, training-stage reweighting can change
the fitted ranking, as the RC7b result demonstrates.

Feature scaling
---------------
Neural networks are sensitive to input scale, so the network is wrapped in
a Pipeline(StandardScaler + MLPClassifier). The scaler is fit on training
data only (inside the pipeline), preventing leakage — identical to the
logistic-regression pipeline.

Design-specification reference: extension to §9.1 (documented, not frozen)
"""

from __future__ import annotations

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_FEATURES,
    OPTUNA_TRIALS,
    OUT_MODELS_CONFIGS,
    OUT_MODELS_SAVED,
    RANDOM_SEED,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

LABEL_COL = "distress"
MODEL_NAME = "neural_network"

# Candidate hidden-layer architectures (string keys are Optuna-categorical-safe;
# mapped to tuples by _architecture_to_tuple). Kept deliberately small for a
# tabular problem of this size to avoid over-parameterisation.
_ARCHITECTURES = {
    "32":      (32,),
    "64":      (64,),
    "128":     (128,),
    "64_32":   (64, 32),
    "128_64":  (128, 64),
}


def _architecture_to_tuple(architecture: str) -> tuple[int, ...]:
    """Map an architecture string key (e.g. '64_32') to a layer-size tuple."""
    return _ARCHITECTURES[architecture]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_neural_network(
    architecture: str = "64_32",
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    activation: str = "relu",
    max_iter: int = 300,
) -> Pipeline:
    """
    Build a standardised multi-layer perceptron classifier pipeline.

    Parameters
    ----------
    architecture : str
        Key into `_ARCHITECTURES` selecting the hidden-layer sizes.
    alpha : float
        L2 regularisation strength on the network weights.
    learning_rate_init : float
        Initial learning rate for the Adam optimiser.
    activation : str
        Hidden-unit activation ('relu' or 'tanh').
    max_iter : int
        Maximum optimisation epochs. Early stopping (below) usually halts
        before this is reached.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline(StandardScaler + MLPClassifier). The scaler is fit on
        training data only, preventing leakage.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=_architecture_to_tuple(architecture),
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            solver="adam",
            max_iter=max_iter,
            early_stopping=True,        # internal val split from TRAIN only
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=RANDOM_SEED,
        )),
    ])


def get_optuna_search_space(trial) -> dict:
    """
    Define the Optuna hyperparameter search space for the neural network.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict
    """
    return {
        "architecture":       trial.suggest_categorical(
            "architecture", list(_ARCHITECTURES.keys())),
        "alpha":              trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "activation":         trial.suggest_categorical("activation", ["relu", "tanh"]),
    }


# ---------------------------------------------------------------------------
# Hyperparameter tuning (reuses the primary rolling-origin CV scheme)
# ---------------------------------------------------------------------------

def _objective(trial, train_df: pd.DataFrame, features: list[str]) -> float:
    """Optuna objective: mean PR-AUC across rolling-origin CV folds."""
    from src.models.tune import rolling_origin_cv_splits

    params = get_optuna_search_space(trial)
    splits = rolling_origin_cv_splits(train_df)
    pr_aucs = []

    for tr_idx, val_idx in splits:
        tr  = train_df.loc[tr_idx]
        val = train_df.loc[val_idx]
        X_tr  = tr[features].astype(float).fillna(0).values
        y_tr  = tr[LABEL_COL].astype(int).values
        X_val = val[features].astype(float).fillna(0).values
        y_val = val[LABEL_COL].astype(int).values

        if y_val.sum() == 0:        # PR-AUC undefined with no positives
            continue

        model = build_neural_network(**params)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        pr_aucs.append(average_precision_score(y_val, y_prob))

    return float(np.mean(pr_aucs)) if pr_aucs else 0.0


def tune_neural_network(
    train_df: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_trials: int = OPTUNA_TRIALS,
) -> tuple[dict, optuna.Study]:
    """
    Bayesian (TPE) hyperparameter optimisation for the neural network,
    using the SAME rolling-origin CV scheme and PR-AUC objective as the
    primary models (src.models.tune).

    Parameters
    ----------
    train_df : pd.DataFrame
        Full training set (1990–2009).
    features : list[str]
    n_trials : int
        Optuna trials (default 100, per §9.3 of the Pre-Specified Empirical Design).

    Returns
    -------
    best_params : dict
    study : optuna.Study
    """
    print(f"  Tuning {MODEL_NAME} ({n_trials} trials) ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda trial: _objective(trial, train_df, features),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"  Best PR-AUC (CV): {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params, study


# ---------------------------------------------------------------------------
# Train + threshold-select + persist (mirrors train_all_models, one model)
# ---------------------------------------------------------------------------

def train_neural_network(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_trials: int | None = None,
    reuse_config: bool = False,
) -> dict:
    """
    Tune, fit, and threshold-select the neural network, then persist it.

    Follows the identical four-step protocol used by train_all_models for
    the primary models:
      1. Hyperparameter selection (Optuna, or reuse saved config)
      2. Re-fit winning config on the FULL training set
      3. Select F1-maximising threshold on the validation set
      4. Save the fitted pipeline + config to outputs/models/

    Parameters
    ----------
    train, val : pd.DataFrame
    features : list[str]
    n_trials : int or None
        Override Optuna n_trials (None = config default of 100).
    reuse_config : bool
        If True, load best_params from neural_network_config.yaml and skip
        the Optuna search (re-fit + re-threshold still run).

    Returns
    -------
    dict
        {'model': fitted_pipeline, 'threshold': float, 'best_params': dict}
    """
    from src.models.evaluate import select_threshold

    X_train = train[features].astype(float).fillna(0).values
    y_train = train[LABEL_COL].astype(int).values
    X_val   = val[features].astype(float).fillna(0).values
    y_val   = val[LABEL_COL].astype(int).values

    print(f"\n{'-'*50}")
    print(f"  MODEL: {MODEL_NAME.upper()} (extension — RC7)")
    print(f"{'-'*50}")

    # Step 1 — hyperparameters
    cfg_path = OUT_MODELS_CONFIGS / f"{MODEL_NAME}_config.yaml"
    if reuse_config:
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"reuse_config=True but {cfg_path} does not exist. "
                "Run a full tuning pass first."
            )
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        best_params = cfg.get("best_params", {})
        print(f"  Reusing saved best_params: {best_params}")
    else:
        best_params, _ = tune_neural_network(
            train, features,
            **({} if n_trials is None else {"n_trials": n_trials}),
        )

    # Step 2 — re-fit on full training set
    model = build_neural_network(**best_params)
    model.fit(X_train, y_train)

    # Step 3 — threshold on validation (maximise F1)
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold  = select_threshold(y_val, y_prob_val)
    print(f"  Optimal threshold (val F1): {threshold:.4f}")

    # Step 4 — persist
    OUT_MODELS_SAVED.mkdir(parents=True, exist_ok=True)
    OUT_MODELS_CONFIGS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODELS_SAVED / f"{MODEL_NAME}.joblib")
    meta = {"model": MODEL_NAME, "best_params": best_params, "threshold": float(threshold)}
    with open(cfg_path, "w") as f:
        yaml.dump(meta, f)
    print(f"  Saved model -> {OUT_MODELS_SAVED / MODEL_NAME}.joblib")

    return {"model": model, "threshold": threshold, "best_params": best_params}
