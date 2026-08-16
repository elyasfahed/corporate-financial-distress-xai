"""
RC7 — Model-class sensitivity: feed-forward neural network.
============================================================
Changes: Model family = multi-layer perceptron (MLP) instead of the
         tree-based ML models (RF, XGBoost).
Fixed  : All other design dimensions unchanged — same 17 predictors,
         same chronological splits, same seed, same Optuna PR-AUC tuning,
         same validation-F1 threshold, same evaluation metrics.

This is a transparent post-specification extension, not one of the five
robustness checks fixed in the original study design. It is framed
as a model-class sensitivity
analysis, analogous in spirit to the RC6 no-RSIZE sensitivity: it answers
a question raised by the results rather than varying a pre-specified design
dimension.

Interpretation
--------------
  - NN ordering consistent with trees (XGBoost ≳ RF ≳ LR, or similar)
        → the ML-vs-benchmark finding (H1) is robust across model families.
  - NN materially different
        → discussed substantively (architecture sensitivity, sample size,
          imbalance handling without class weights).

The network does NOT replace XGBoost as the primary SHAP model. The primary
explanation analysis remains on the frozen best-of-portfolio model.

Reference: extension to the pre-specified study design (§11; documented,
not frozen)
"""

from __future__ import annotations

import pandas as pd

from src.config import ALL_FEATURES


def run_rc7(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_trials: int | None = None,
    reuse_config: bool = False,
) -> pd.DataFrame:
    """
    Execute RC7: train and evaluate the neural network under the primary
    leakage-free protocol.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    features : list[str]
        Defaults to the full 17-predictor set (identical to primary).
    n_trials : int or None
        Override Optuna trials (None = config default of 100).
    reuse_config : bool
        Reuse a previously tuned neural_network_config.yaml if available.

    Returns
    -------
    pd.DataFrame
        Single-row performance table for the neural network, in the same
        schema as the primary model_performance_test.csv.
    """
    from src.models.neural_network import train_neural_network
    from src.models.train import evaluate_on_test

    print(f"\nRC7: Neural network (MLP) — {len(features)} predictors, primary protocol")
    nn_info = train_neural_network(
        train, val, features=features,
        n_trials=n_trials, reuse_config=reuse_config,
    )
    trained = {"neural_network": nn_info}

    # output_stem=None: never overwrites the primary results table; the
    # caller (Stage 13 in run_pipeline.py) persists RC7 to its own CSV.
    return evaluate_on_test(
        trained, test, features=features,
        output_stem=None, run_label="RC7",
    )
