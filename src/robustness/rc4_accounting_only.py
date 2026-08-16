"""
RC4 — Accounting-only model: remove all 6 market-based predictors.
===================================================================
Changes: Feature set = 11 accounting predictors only (no EXRET, SIGMA,
         LNMK, RSIZE, MB, PRICE).
Fixed  : All other design dimensions unchanged.

Quantifies the marginal predictive value of incorporating market
information into the distress prediction model. If the accounting-only
model performs significantly worse, this confirms that market signals
provide incremental information beyond balance sheet and income statement
data (Campbell, Hilscher & Szilagyi 2008 finding).

Design-specification reference: §11, RC4
"""

from __future__ import annotations

import pandas as pd

from src.config import ACCOUNTING_ONLY_FEATURES


def run_rc4(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ACCOUNTING_ONLY_FEATURES,
) -> pd.DataFrame:
    """
    Execute RC4: accounting-only predictor set.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    features : list[str]
        Defaults to ACCOUNTING_ONLY_FEATURES (11 variables).

    Returns
    -------
    pd.DataFrame
        Performance metrics using only accounting predictors.
    """
    from src.models.train import train_all_models, evaluate_on_test

    print(f"\nRC4: Accounting-only features ({len(features)} predictors — no market variables)")
    trained = train_all_models(train, val, features=features, spec="rc4")
    # output_stem=None: results persisted via run_all_robustness checkpoint.
    return evaluate_on_test(trained, test, features=features,
                            output_stem=None, run_label="RC4")
