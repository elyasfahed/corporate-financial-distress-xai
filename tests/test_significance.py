"""Regression tests for corrected paired firm-block PR-AUC inference."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.significance import _bootstrap_pr_auc_diff
from scripts.run_v2_secondary import _holm


def test_identical_predictions_return_zero_difference_inference():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=int)
    p = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.05, 0.9])
    firms = np.arange(len(y))

    p_value, ci_lo, ci_hi = _bootstrap_pr_auc_diff(
        y, p, p, firms, n_reps=200
    )

    assert p_value == pytest.approx(1.0)
    assert ci_lo == pytest.approx(0.0)
    assert ci_hi == pytest.approx(0.0)


def test_clear_pr_auc_gap_excludes_zero():
    rng = np.random.default_rng(42)
    n = 800
    y = np.zeros(n, dtype=int)
    y[:80] = 1
    rng.shuffle(y)

    strong = np.clip(0.1 + 0.75 * y + rng.normal(0, 0.08, n), 0, 1)
    weak = rng.uniform(0, 1, n)
    firms = np.arange(n)

    p_value, ci_lo, ci_hi = _bootstrap_pr_auc_diff(
        y, strong, weak, firms, n_reps=400
    )

    assert ci_lo > 0
    assert ci_hi > ci_lo
    assert p_value < 0.05


def test_argument_order_reverses_interval_and_preserves_p_value():
    rng = np.random.default_rng(7)
    n = 500
    y = rng.binomial(1, 0.12, n)
    a = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.15, n), 0, 1)
    b = np.clip(0.25 + 0.35 * y + rng.normal(0, 0.20, n), 0, 1)
    firms = np.arange(n)

    p_ab, lo_ab, hi_ab = _bootstrap_pr_auc_diff(
        y, a, b, firms, n_reps=300
    )
    p_ba, lo_ba, hi_ba = _bootstrap_pr_auc_diff(
        y, b, a, firms, n_reps=300
    )

    assert p_ab == pytest.approx(p_ba)
    assert lo_ab == pytest.approx(-hi_ba)
    assert hi_ab == pytest.approx(-lo_ba)


def test_holm_uses_unrounded_p_values_at_significance_boundary():
    """A display-rounded 0.0100 must not replace the exact 10/1001 value."""
    pvals = [3 / 1001, 10 / 1001, 37 / 1001, 41 / 1001, 315 / 1001,
             860 / 1001]

    adjusted = _holm(pvals)

    assert adjusted[1] == pytest.approx(50 / 1001)
    assert adjusted[1] < 0.05
