"""
Ridge-penalised Logistic Regression — classical benchmark model.
=================================================================
This is the primary benchmark against which Random Forest and XGBoost
are compared. It is a well-specified linear model estimated on the same
17-variable predictor set, making performance comparisons fair.

Regularisation: L2 (ridge) penalty to handle multicollinearity in the
predictor set (TLTA and LNMK are correlated; NITA and OCF_TA are
correlated). The regularisation strength C is tuned via Optuna.

Design-specification reference: §9.1 (Model Portfolio)
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def build_logistic_regression(
    C: float = 1.0,
    class_weight: str | dict = "balanced",
    max_iter: int = 1000,
) -> Pipeline:
    """
    Build a standardised ridge-penalised logistic regression pipeline.

    Standardisation is applied inside the pipeline to prevent leakage:
    the scaler is fit on training data only.

    Parameters
    ----------
    C : float
        Inverse regularisation strength (smaller = stronger ridge penalty).
        Tuned via Optuna; default 1.0.
    class_weight : str or dict
        'balanced' → inverse class frequency weighting (primary specification).
        Addresses class imbalance without SMOTE.
    max_iter : int
        Maximum iterations for the solver. Increase if convergence warnings.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with StandardScaler + LogisticRegression(penalty='l2').
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=C,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=RANDOM_SEED,
        )),
    ])


def get_optuna_search_space(trial) -> dict:
    """
    Define the Optuna hyperparameter search space for logistic regression.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict
        Hyperparameter names and sampled values.
    """
    # D7 fix — raise C lower bound from 1e-4 to 1e-2 to prevent
    # over-regularisation (previously Optuna landed at C≈0.005, shrinking
    # most coefficients to ~0). Range still spans 4 orders of magnitude.
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
    }
