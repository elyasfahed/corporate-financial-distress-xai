"""
Regression tests — v2 18-feature set + market imputation (§8 fix, 2026-07-12)
==============================================================================
Under v1 the market features EXRET/SIGMA/MB escape imputation and reach
the models as fillna(0); on the v1 test split the 2,525 MB-missing rows
contain 168 of 434 distress events (6.65% vs 0.95%), making zero an
implicit distress signal. The v2 profile (a) routes EXRET/SIGMA/MB
through the standard train-fitted hierarchy and (b) adds an explicit
MB_MISSING indicator (17 -> 18 features).

  1. The v2 feature set is exactly the frozen 17 plus MB_MISSING; the
     frozen constants are untouched.
  2. V2_PROFILE wires every adopted data-layer correction.
  3. The imputation machinery genuinely fills market features when they
     are included in the features list (train-median hierarchy).
"""

import numpy as np
import pandas as pd

from src.config import (
    ACCOUNTING_FEATURES,
    ALL_FEATURES,
    ALL_FEATURES_V2,
    MARKET_FEATURES,
    MARKET_IMPUTE_FEATURES,
    MB_MISSING_COL,
    V2_PROFILE,
)
from src.features.impute import apply_imputation, compute_imputation_medians


def test_v2_feature_set_is_frozen_plus_mb_missing():
    assert ALL_FEATURES == ACCOUNTING_FEATURES + MARKET_FEATURES
    assert len(ALL_FEATURES) == 17                      # frozen, untouched
    assert ALL_FEATURES_V2 == ALL_FEATURES + [MB_MISSING_COL]
    assert len(ALL_FEATURES_V2) == 18
    assert set(MARKET_IMPUTE_FEATURES) == {"EXRET", "SIGMA", "MB"}
    # PRICE stays never-imputed by design
    assert "PRICE" not in MARKET_IMPUTE_FEATURES


def test_v2_profile_wires_all_blocker_fixes():
    assert V2_PROFILE["feature_set"] == ALL_FEATURES_V2
    assert V2_PROFILE["impute_market"] is True
    assert V2_PROFILE["missing_policy"] == "strict"
    assert V2_PROFILE["lag_buffer"] is True
    assert V2_PROFILE["universe_date_col"] == "datadate"
    assert V2_PROFILE["cv_fold_safe"] is True
    assert V2_PROFILE["retune"] is True
    # A corrected re-estimation must never overwrite the frozen top-level
    # primary models.  The public academic name remains "primary"; this is
    # only the artifact-storage namespace.
    assert V2_PROFILE["spec"] == "final_primary"
    # Primary outcome is the cleaned performance-core label (excludes the
    # voluntary/administrative GDR exits CORQ/MVOT); the broad performance
    # family is retained only as a documented sensitivity.
    assert V2_PROFILE["distress_event_col"] == "is_distress_performance_core"
    assert V2_PROFILE["broad_performance_event_col"] == "is_distress_performance"
    assert V2_PROFILE["narrow_distress_event_col"] == "is_distress_narrow_financial"


def test_market_features_impute_through_hierarchy():
    rng = np.random.default_rng(42)
    n = 12
    train = pd.DataFrame({
        "fyear": [2000] * n,
        "sich": [0] * n,
        "_sic": [3600] * n,
        "EXRET": rng.normal(0.0, 0.1, n),
        "SIGMA": rng.uniform(0.05, 0.3, n),
        "MB":    rng.uniform(0.5, 4.0, n),
    })
    target = train.copy()
    target.loc[0, "EXRET"] = np.nan
    target.loc[1, "SIGMA"] = np.nan
    target.loc[2, "MB"] = np.nan

    meds = compute_imputation_medians(
        train, features=MARKET_IMPUTE_FEATURES,
        sic_col="_sic", peer_rule="per_feature",
    )
    out = apply_imputation(
        target, *meds, features=MARKET_IMPUTE_FEATURES, sic_col="_sic",
    )
    for feat in MARKET_IMPUTE_FEATURES:
        assert out[feat].isna().sum() == 0, f"{feat} not imputed"
    # Filled from the train SIC-2 x fyear median, not with zero
    assert abs(out.loc[2, "MB"] - train["MB"].median()) < 1e-12
