"""
Random Forest classifier.
==========================
Tree-based ensemble that captures non-linear relationships and feature
interactions without requiring feature engineering beyond the 17 predictors.
Compared against logistic regression to quantify the non-linear gain.

Class imbalance is handled via class_weight='balanced', consistent
with the primary specification (Pre-Specified Empirical Design §9.2).

Design-specification reference: §9.1 (Model Portfolio)
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_SEED


def build_random_forest(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 20,
    max_features: str | float = "sqrt",
    class_weight: str | dict = "balanced",
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with the primary configuration.

    Parameters
    ----------
    n_estimators : int
        Number of trees. 500 is a reasonable default; tuned via Optuna.
    max_depth : int or None
        Maximum tree depth. None = grow until all leaves are pure.
        Constraining depth prevents overfitting.
    min_samples_leaf : int
        Minimum number of samples in a leaf node. Acts as regularisation.
    max_features : str or float
        Features to consider at each split. 'sqrt' is standard.
    class_weight : str or dict
        'balanced' → inverse class frequency. Primary imbalance treatment.

    Returns
    -------
    RandomForestClassifier
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def get_optuna_search_space(trial) -> dict:
    """
    Define the Optuna hyperparameter search space for Random Forest.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict
    """
    # D7 fix — raise max_depth lower bound from 3 to 5 to ensure trees
    # are deep enough to model meaningful interactions (depth-3 is near-stump
    # territory and effectively reduces RF to a shallow GBM).
    return {
        "n_estimators":    trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth":       trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }
