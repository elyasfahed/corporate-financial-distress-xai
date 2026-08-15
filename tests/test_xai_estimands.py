"""
Regression tests — XAI estimand fixes (Phase 5 audit remediation)
==================================================================
1. select_dependence_features returns the model's EMPIRICAL top-k by
   mean |SHAP| plus the pre-specified H4 trio.
2. compute_interaction_contrast is exactly zero for a purely additive
   table and recovers an injected super-additive corner effect.
"""

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_nonlinear import (
    compute_interaction_contrast,
    select_dependence_features,
)


def test_select_dependence_features_empirical_plus_h4():
    feats = ["A", "B", "TLTA", "NITA", "SIGMA", "F", "G"]
    n = 100
    sv = np.zeros((n, len(feats)))
    # Importance order: G > F > A > B > TLTA > NITA > SIGMA
    for i, mag in enumerate([3.0, 2.0, 0.5, 0.4, 0.3, 6.0, 7.0]):
        sv[:, i] = mag
    out = select_dependence_features(sv, feats, k=2)
    assert out[:2] == ["G", "F"]                    # empirical top-2
    assert set(("TLTA", "NITA", "SIGMA")) <= set(out)   # H4 trio appended
    assert len(out) == 5


def test_interaction_contrast_zero_for_additive_table():
    rows = np.array([0.01, 0.02, 0.05])
    cols = np.array([0.00, 0.01, 0.03])
    additive = pd.DataFrame(rows[:, None] + cols[None, :],
                            index=["Low", "Mid", "High"],
                            columns=["Low", "Mid", "High"])
    contrast = compute_interaction_contrast(additive)
    assert np.allclose(contrast.to_numpy(), 0.0, atol=1e-12)


def test_interaction_contrast_recovers_superadditive_corner():
    rows = np.array([0.01, 0.02, 0.05])
    cols = np.array([0.00, 0.01, 0.03])
    table = pd.DataFrame(rows[:, None] + cols[None, :],
                         index=["Low", "Mid", "High"],
                         columns=["Low", "Mid", "High"])
    table.loc["High", "Low"] += 0.09        # injected interaction
    contrast = compute_interaction_contrast(table)
    # The injected corner carries the largest positive contrast
    assert contrast.loc["High", "Low"] == contrast.max().max()
    assert contrast.loc["High", "Low"] > 0.03