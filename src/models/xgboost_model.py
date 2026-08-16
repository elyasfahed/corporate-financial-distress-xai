"""
XGBoost gradient boosting classifier — primary ML model.
==========================================================
XGBoost is the primary ML model because:
  1. It consistently achieves top tabular data performance.
  2. It provides the most stable SHAP values via TreeExplainer.
  3. It natively supports scale_pos_weight for class imbalance.

SHAP TreeExplainer is an exact, polynomial-time algorithm for tree models,
avoiding the approximation error of KernelExplainer. This makes XGBoost
the natural choice when SHAP explanations are a core deliverable.

Design reference: §9.1, §10 (Model Portfolio; XAI Strategy)
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from src.config import RANDOM_SEED


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute scale_pos_weight for XGBoost class imbalance handling.

    scale_pos_weight = n_negative / n_positive
    This is equivalent to the 'balanced' class weight in sklearn.

    Parameters
    ----------
    y_train : np.ndarray
        Binary distress labels from the training set.

    Returns
    -------
    float
        Ratio of non-distressed to distressed firm-years.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos == 0:
        raise ValueError("No positive (distressed) observations in training set.")
    return float(n_neg / n_pos)


def build_xgboost(
    n_estimators: int = 500,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 10,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    scale_pos_weight: float = 1.0,
) -> xgb.XGBClassifier:
    """
    Build an XGBoost classifier with the primary configuration.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth. Smaller = more regularisation.
    learning_rate : float
        Step size shrinkage (eta). Lower = more robust but slower.
    subsample : float
        Row subsampling ratio per tree. Regularisation + speed.
    colsample_bytree : float
        Feature subsampling ratio per tree.
    min_child_weight : int
        Minimum sum of instance weights in a child. Regularisation.
    reg_alpha : float
        L1 regularisation on leaf weights.
    reg_lambda : float
        L2 regularisation on leaf weights.
    scale_pos_weight : float
        Ratio of negative to positive class. Set via compute_scale_pos_weight().

    Returns
    -------
    xgb.XGBClassifier
    """
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        tree_method="hist",    # fast CPU histogram method
        n_jobs=-1,
    )


def get_optuna_search_space(trial) -> dict:
    """
    Define the Optuna hyperparameter search space for XGBoost.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict
    """
    # D7 fix — raise max_depth lower bound from 2 to 4 (depth-2 trees are
    # decision stumps; the previous run landed on depth-3 + 900 trees,
    # which is essentially a slow LR. Forcing depth ≥ 4 allows the
    # ensemble to capture multi-feature interactions properly.
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":         trial.suggest_int("max_depth", 4, 8),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }
