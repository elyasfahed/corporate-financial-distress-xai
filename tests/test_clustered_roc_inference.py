"""
Tests for paired clustered ROC-AUC inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.analysis.clustered_roc_inference import (
    compare_models_clustered,
    firm_cluster_auc_ci,
    holm,
    paired_firm_cluster_roc_test,
    paired_twoway_roc_test,
)


@pytest.fixture
def panel():
    """200 firms x 5 years; model A is genuinely better than model B."""
    rng = np.random.default_rng(7)
    n_firms, n_years = 200, 5
    firms = np.repeat(np.arange(n_firms), n_years)
    years = np.tile(np.arange(2015, 2015 + n_years), n_firms)
    firm_effect = rng.normal(size=n_firms)[firms]
    y = (rng.uniform(size=len(firms)) < 0.08).astype(int)
    signal = y + 0.3 * firm_effect
    a = signal + rng.normal(scale=0.55, size=len(firms))
    b = signal + rng.normal(scale=1.10, size=len(firms))
    return y, a, b, firms, years


# ------------------------------------------------------------------ holm

def test_holm_matches_known_values():
    p = [0.01, 0.02, 0.03]
    # 3*.01=.03 ; 2*.02=.04 ; 1*.03=.03 -> monotone: .03, .04, .04
    assert holm(p) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_is_monotone_and_capped():
    p = [0.5, 0.001, 0.9]
    adj = holm(p)
    assert all(0.0 <= x <= 1.0 for x in adj)
    order = np.argsort(p)
    vals = [adj[i] for i in order]
    assert vals == sorted(vals)


def test_holm_preserves_input_order():
    adj = holm([0.9, 0.001])
    assert adj[1] < adj[0]


def test_holm_full_precision_changes_a_borderline_verdict():
    """Rounding before adjustment is exactly how a .0499 becomes a .05."""
    precise = 0.024999
    assert holm([precise, 0.9])[0] < 0.05
    assert holm([round(precise, 3), 0.9])[0] >= 0.05


# --------------------------------------------------------- point estimates

def test_delta_matches_direct_auc_difference(panel):
    y, a, b, firms, _ = panel
    res = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=50)
    assert res["delta_auc"] == pytest.approx(
        roc_auc_score(y, a) - roc_auc_score(y, b)
    )


def test_ci_brackets_the_observed_difference(panel):
    y, a, b, firms, _ = panel
    res = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=200)
    assert res["ci_lower"] <= res["delta_auc"] <= res["ci_upper"]


def test_single_model_ci_brackets_point(panel):
    y, a, _, firms, _ = panel
    res = firm_cluster_auc_ci(y, a, firms, n_reps=200)
    assert res["ci_lower"] <= res["roc_auc"] <= res["ci_upper"]


def test_better_model_is_detected(panel):
    y, a, b, firms, _ = panel
    res = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=300)
    assert res["delta_auc"] > 0
    assert res["p_normal"] < 0.05


def test_identical_models_give_zero_delta_and_null_result(panel):
    y, a, _, firms, _ = panel
    res = paired_firm_cluster_roc_test(y, a, a, firms, n_reps=200)
    assert res["delta_auc"] == pytest.approx(0.0)
    assert res["p_percentile"] == pytest.approx(1.0)


# ------------------------------------------------------------ reproducibility

def test_same_seed_reproduces(panel):
    y, a, b, firms, _ = panel
    r1 = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=100, seed=5)
    r2 = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=100, seed=5)
    assert r1["se_cluster"] == pytest.approx(r2["se_cluster"])


def test_different_seed_changes_se(panel):
    y, a, b, firms, _ = panel
    r1 = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=100, seed=5)
    r2 = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=100, seed=6)
    assert r1["se_cluster"] != r2["se_cluster"]


# ------------------------------------------------------- clustering structure

def test_firm_blocks_are_resampled_whole(panel):
    """A resample must contain each picked firm's observations together."""
    y, a, b, firms, _ = panel
    res = paired_firm_cluster_roc_test(y, a, b, firms, n_reps=10)
    assert res["n_firms"] == len(np.unique(firms))
    assert res["n_obs"] == len(y)


def test_twoway_reports_all_three_components(panel):
    y, a, b, firms, years = panel
    tw = paired_twoway_roc_test(y, a, b, firms, years, n_reps=150)
    for k in ("se_firm", "se_year", "se_iid", "se_twoway"):
        assert tw[k] >= 0
    assert tw["n_years"] == len(np.unique(years))


def test_twoway_variance_is_floored_at_zero(panel):
    y, a, b, firms, years = panel
    tw = paired_twoway_roc_test(y, a, b, firms, years, n_reps=100)
    assert np.isfinite(tw["se_twoway"]) and tw["se_twoway"] >= 0.0


def test_twoway_ci_is_symmetric_about_the_point(panel):
    y, a, b, firms, years = panel
    tw = paired_twoway_roc_test(y, a, b, firms, years, n_reps=100)
    mid = (tw["twoway_ci_lower"] + tw["twoway_ci_upper"]) / 2
    assert mid == pytest.approx(tw["delta_auc"])


# --------------------------------------------------------------- comparison

def test_compare_produces_all_pairs_and_holm_columns(panel):
    y, a, b, firms, years = panel
    probs = {"A": a, "B": b, "C": (a + b) / 2}
    out = compare_models_clustered(y, probs, firms, year_ids=years, n_reps=80)
    assert len(out) == 3                       # 3 choose 2
    for col in ("p_cluster_holm", "p_delong_holm", "p_twoway_holm"):
        assert col in out.columns
    assert (out["p_cluster_holm"] >= out["p_cluster_normal"] - 1e-12).all()


def test_compare_without_years_omits_twoway(panel):
    y, a, b, firms, _ = panel
    out = compare_models_clustered(y, {"A": a, "B": b}, firms, n_reps=50)
    assert "p_twoway_holm" not in out.columns
    assert "p_cluster_holm" in out.columns
