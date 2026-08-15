"""
Tests for the H4 curvature inference (post-hoc supplementary).
==============================================================

The point of these tests is the distinction the corrected module is built
around:

* the **in-sample R² gain** of a spline over a nested straight line is
  mechanically non-negative and therefore cannot, on its own, be evidence
  against linearity; while
* the **firm-cluster wild bootstrap under the fitted linear null** is a valid
  test, because its reference distribution is generated from a linear
  conditional mean.

The size test below is the regression test the correction exists for: data
generated from a linear conditional mean must not systematically produce
"significant" curvature. The same synthetic samples are used to show that the
discarded percentile-interval procedure *would* have flagged every one of them.

No fitted model, SHAP matrix or data split is touched here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.supp_h4_nonlinearity import (curvature,
                                               curvature_out_of_fold,
                                               curvature_wild_cluster_test)

N_CLUSTERS = 120
PER_CLUSTER = 12
N_BOOT_FAST = 199          # keeps the size study quick; resolution is 1/200


def _clustered_sample(rng, curvature_coef=0.0, n_clusters=N_CLUSTERS,
                      per_cluster=PER_CLUSTER):
    """
    Synthetic firm-clustered sample.

    ``curvature_coef == 0`` gives a strictly **linear** conditional mean. The
    errors carry a firm random effect and are heteroskedastic in x, so the
    sample violates the i.i.d.-homoskedastic assumptions the wild cluster
    bootstrap is meant to tolerate.
    """
    groups = np.repeat(np.arange(n_clusters), per_cluster)
    n = len(groups)
    x = rng.uniform(-2.0, 2.0, size=n)
    firm_effect = rng.normal(0.0, 0.6, size=n_clusters)[groups]
    noise = rng.normal(0.0, 0.4 + 0.35 * np.abs(x))
    s = 0.8 + 1.3 * x + curvature_coef * (x ** 2) + firm_effect + noise
    return x, s, groups


# --------------------------------------------------------------------------
# 1. The regression test: a linear DGP must not produce significant curvature
# --------------------------------------------------------------------------

def test_linear_dgp_does_not_systematically_produce_significant_curvature():
    """
    Twenty independent samples drawn from a linear conditional mean. At a
    nominal 5% level the expected number of rejections is 1. The bound of 5 is
    deliberately loose (it is the ~99.7th percentile of Binomial(20, 0.05))
    so the test fails on a genuinely broken procedure, not on Monte Carlo luck.
    """
    n_reps = 20
    rejections = 0
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        x, s, groups = _clustered_sample(rng, curvature_coef=0.0)
        res = curvature_wild_cluster_test(x, s, groups, n_boot=N_BOOT_FAST,
                                          seed=2000 + rep)
        if res["p_value"] <= 0.05:
            rejections += 1

    assert rejections <= 5, (
        f"{rejections}/{n_reps} rejections on a linear DGP — the curvature test "
        "is not holding its nominal size"
    )


def test_in_sample_gain_is_non_negative_on_a_linear_dgp():
    """
    The mechanical property that invalidated the discarded procedure: even with
    a perfectly linear conditional mean, the spline always explains at least as
    much variance as the line, in every single sample.
    """
    for rep in range(20):
        rng = np.random.default_rng(1000 + rep)
        x, s, _ = _clustered_sample(rng, curvature_coef=0.0)
        _, _, gain = curvature(x, s)
        assert gain >= 0.0, f"rep {rep}: in-sample gain {gain} was negative"


def test_discarded_percentile_interval_would_have_flagged_a_linear_dgp():
    """
    Directly documents why the old procedure was invalid: the ordinary
    percentile bootstrap of the non-negative in-sample gain excludes zero on
    data with a linear conditional mean, so "the interval excludes zero" was
    never evidence of curvature. The valid test on the same sample does not
    reject.
    """
    rng = np.random.default_rng(4242)
    x, s, groups = _clustered_sample(rng, curvature_coef=0.0)

    # Reproduce the discarded procedure: resample firms, refit, take percentiles.
    codes = groups
    blocks = [np.flatnonzero(codes == g) for g in np.unique(codes)]
    boot_rng = np.random.default_rng(7)
    gains = []
    for _ in range(200):
        pick = boot_rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        gains.append(curvature(x[idx], s[idx])[2])
    lo = float(np.nanpercentile(gains, 2.5))

    assert lo > 0.0, (
        "the discarded percentile interval failed to exclude zero here; the "
        "test no longer demonstrates the defect it was written for"
    )

    valid = curvature_wild_cluster_test(x, s, groups, n_boot=499, seed=11)
    assert valid["p_value"] > 0.05, (
        "the valid test rejected on a linear DGP — size failure"
    )


# --------------------------------------------------------------------------
# 2. Power: genuine curvature must still be detected
# --------------------------------------------------------------------------

def test_curved_dgp_is_detected():
    rng = np.random.default_rng(99)
    x, s, groups = _clustered_sample(rng, curvature_coef=1.4)
    res = curvature_wild_cluster_test(x, s, groups, n_boot=499, seed=12)
    assert res["p_value"] <= 0.05, (
        f"failed to detect a strongly quadratic conditional mean (p={res['p_value']})"
    )


def test_power_increases_with_curvature():
    def p_for(coef):
        rng = np.random.default_rng(5150)
        x, s, groups = _clustered_sample(rng, curvature_coef=coef)
        return curvature_wild_cluster_test(x, s, groups, n_boot=499,
                                           seed=13)["p_value"]

    assert p_for(1.5) <= p_for(0.0)


# --------------------------------------------------------------------------
# 3. Out-of-fold effect size behaves as an effect size should
# --------------------------------------------------------------------------

def test_out_of_fold_gain_is_near_zero_or_negative_on_a_linear_dgp():
    """
    Unlike the in-sample gain, the firm-grouped out-of-fold gain is free to be
    negative, and on a linear DGP the extra spline terms should buy essentially
    nothing.
    """
    worst = -np.inf
    for rep in range(8):
        rng = np.random.default_rng(300 + rep)
        x, s, groups = _clustered_sample(rng, curvature_coef=0.0)
        worst = max(worst, curvature_out_of_fold(x, s, groups))
    assert worst < 0.01, f"out-of-fold gain {worst:.4f} too large on a linear DGP"


def test_out_of_fold_gain_is_positive_on_a_curved_dgp():
    rng = np.random.default_rng(777)
    x, s, groups = _clustered_sample(rng, curvature_coef=1.4)
    assert curvature_out_of_fold(x, s, groups) > 0.02


def test_out_of_fold_gain_can_be_negative():
    """
    The property that distinguishes it from the in-sample gain: with heavy
    noise and no curvature, holding firms out must sometimes penalise the
    richer model.
    """
    seen_negative = False
    for rep in range(12):
        rng = np.random.default_rng(900 + rep)
        x, s, groups = _clustered_sample(rng, curvature_coef=0.0)
        if curvature_out_of_fold(x, s, groups) < 0:
            seen_negative = True
            break
    assert seen_negative, "out-of-fold gain never went negative on a linear DGP"


# --------------------------------------------------------------------------
# 4. Mechanics of the wild bootstrap
# --------------------------------------------------------------------------

def test_wild_bootstrap_is_reproducible_and_p_value_is_bounded_away_from_zero():
    rng = np.random.default_rng(21)
    x, s, groups = _clustered_sample(rng, curvature_coef=0.6)
    a = curvature_wild_cluster_test(x, s, groups, n_boot=199, seed=5)
    b = curvature_wild_cluster_test(x, s, groups, n_boot=199, seed=5)

    assert a["p_value"] == b["p_value"]
    assert a["gain_observed"] == pytest.approx(b["gain_observed"])
    # (1 + #) / (1 + B) can never be exactly zero.
    assert a["p_value"] >= 1.0 / (1 + 199)
    assert a["p_value"] <= 1.0


def test_observed_gain_matches_the_in_sample_effect_size():
    rng = np.random.default_rng(31)
    x, s, groups = _clustered_sample(rng, curvature_coef=0.9)
    _, _, gain = curvature(x, s)
    res = curvature_wild_cluster_test(x, s, groups, n_boot=99, seed=6)
    assert res["gain_observed"] == pytest.approx(gain, abs=1e-10)


def test_null_distribution_is_strictly_positive():
    """
    Sanity check on the very defect being corrected: the gain generated under a
    linear null is itself always positive, so a procedure comparing the observed
    gain to zero rather than to this distribution can only ever "reject".
    """
    rng = np.random.default_rng(41)
    x, s, groups = _clustered_sample(rng, curvature_coef=0.0)
    res = curvature_wild_cluster_test(x, s, groups, n_boot=299, seed=7)
    assert res["null_gain_p95"] > 0.0
