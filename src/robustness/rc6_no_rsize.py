"""
Sensitivity Analysis — No-RSIZE model.
=======================================
Removes RSIZE from the 17-feature set to test whether relative size
dominates performance. RSIZE = log(firm market cap / total CRSP market cap)
is the top SHAP feature by a wide margin (mean |SHAP| = 1.21 vs 0.42 next).

A critical question: is RSIZE capturing genuine financial distress signal,
or is it collinear with other size proxies (LNMK, LNTA, PRICE)?

Interpretation:
  - Large PR-AUC drop  → RSIZE carries unique distress information
  - Small PR-AUC drop  → RSIZE is collinear; other size proxies substitute
  - Either way: discuss why relative size dominates SHAP attributions

This is framed as a sensitivity analysis, NOT a formal robustness check.
It does not vary a pre-specified design dimension — it answers an
econometric question raised by the SHAP results.
"""

from __future__ import annotations

import pandas as pd

from src.config import ALL_FEATURES


NO_RSIZE_FEATURES = [f for f in ALL_FEATURES if f != "RSIZE"]   # 16 features


def run_rc6(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sensitivity analysis: 16-feature model excluding RSIZE.

    Parameters
    ----------
    train, val, test : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Performance metrics for all three models without RSIZE.
    """
    from src.models.train import train_all_models, evaluate_on_test

    print(f"\nSensitivity: No-RSIZE model ({len(NO_RSIZE_FEATURES)} predictors — RSIZE excluded)")
    print(f"  Excluded: RSIZE")
    print(f"  Remaining features: {NO_RSIZE_FEATURES}")

    # spec="rc6": writes to outputs/models/{saved,configs}/rc6/ — this exact
    # call previously overwrote the shared primary artifacts.
    trained = train_all_models(train, val, features=NO_RSIZE_FEATURES, spec="rc6")
    # output_stem=None: Stage 12 in run_pipeline.py persists rc6 results
    # directly to rc6_no_rsize_results.csv after this returns.
    return evaluate_on_test(trained, test, features=NO_RSIZE_FEATURES,
                            output_stem=None, run_label="RC6")
