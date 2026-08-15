"""
Regression tests — RC3 corrected SMOTE mode (Phase 5 audit remediation)
========================================================================
Frozen RC3 defects: plain SMOTE interpolates binary indicators into
fractional values, kNN runs in raw feature space dominated by size
variables, and models used library defaults. Tests pin the corrected
mode and the frozen default's unchanged behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from src.robustness.rc3_smote import (
    BINARY_FEATURES,
    _scale_continuous,
    smote_resample,
)

FEATURES = ["NITA", "LNTA", "OENEG", "INTWO"]


def make_training_arrays(n=300, seed=42):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(0.05, 0.1, n),          # NITA — small scale
        rng.normal(5.0, 2.0, n),           # LNTA — large scale
        rng.integers(0, 2, n).astype(float),   # OENEG binary
        rng.integers(0, 2, n).astype(float),   # INTWO binary
    ])
    y = (rng.random(n) < 0.08).astype(int)
    y[:5] = 1                              # ensure enough minority samples
    return X, y


def test_frozen_smote_produces_fractional_binaries():
    X, y = make_training_arrays()
    X_res, y_res = smote_resample(X, y, FEATURES, corrected=False)
    binary_cols = X_res[:, [2, 3]]
    fractional = ~np.isin(binary_cols, (0.0, 1.0))
    # The documented frozen defect: synthetic rows carry fractional
    # indicator values
    assert fractional.any()


def test_corrected_smotenc_keeps_binaries_binary():
    X, y = make_training_arrays()
    X_res, y_res = smote_resample(X, y, FEATURES, corrected=True)
    binary_cols = X_res[:, [2, 3]]
    assert np.isin(binary_cols, (0.0, 1.0)).all()
    # Classes balanced after resampling
    assert y_res.sum() == (y_res == 0).sum()


def test_scale_continuous_standardises_only_continuous():
    X, y = make_training_arrays()
    X_tr, X_val, X_te, binary_idx = _scale_continuous(X, X.copy(), X.copy(),
                                                      FEATURES)
    assert binary_idx == [2, 3]
    # Continuous columns standardised on train
    assert abs(X_tr[:, 0].mean()) < 1e-9 and abs(X_tr[:, 0].std() - 1) < 1e-6
    assert abs(X_tr[:, 1].mean()) < 1e-9
    # Binary columns untouched
    np.testing.assert_array_equal(X_tr[:, 2], X[:, 2])
    np.testing.assert_array_equal(X_tr[:, 3], X[:, 3])
    # val/test transformed with TRAIN statistics (same values here since
    # identical input)
    np.testing.assert_allclose(X_val[:, 0], X_tr[:, 0])


def test_corrected_resampling_distance_scale():
    """
    In corrected mode the caller scales before resampling, so the kNN
    interpolation happens in comparable units. Sanity: after scaling,
    the size variable no longer dominates total variance.
    """
    X, y = make_training_arrays()
    var_ratio_raw = X[:, 1].var() / X[:, 0].var()
    X_tr, _, _, _ = _scale_continuous(X, X.copy(), X.copy(), FEATURES)
    var_ratio_scaled = X_tr[:, 1].var() / X_tr[:, 0].var()
    assert var_ratio_raw > 100          # raw: LNTA dominates
    assert 0.5 < var_ratio_scaled < 2   # scaled: comparable
