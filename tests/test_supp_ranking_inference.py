"""
Tests for the H1 ranking-inference helpers (post-hoc supplementary).
====================================================================

Two things are asserted here.

1. :func:`romano_wolf` implements the **conventional ordered** Romano--Wolf
   stepdown: descending order of the absolute studentised statistic, one
   hypothesis removed per step, the bootstrap maximum taken over the current
   hypothesis and every *smaller* one, and monotone adjusted p-values along the
   ordering. The tests use synthetic bootstrap arrays constructed so that the
   ordering and the expected adjusted p-values can be computed by hand.

2. :func:`bootstrap_range_model_set` is exercised on synthetic PR-AUC draws with
   a known separation structure. It is deliberately *not* asserted to satisfy
   any Hansen--Lunde--Nason coverage property: it is a range-based adaptation on
   a non-additive metric, and the tests only pin its elimination behaviour.

Nothing here loads a fitted model or touches a split.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.supp_ranking_inference import (bootstrap_range_model_set,
                                                 romano_wolf)


def _labels(k):
    return [f"H{i}" for i in range(k)]


def _reference_stepdown(obs, boot, alpha=0.05):
    """
    Independent by-hand reference implementation of the ordered stepdown.

    Written from the algorithm description rather than from the production
    code, so agreement is evidence about the algorithm and not a tautology.
    """
    obs = np.asarray(obs, float)
    boot = np.asarray(boot, float)
    se = np.nanstd(boot, axis=0, ddof=1)
    t = np.abs(obs) / se
    cen = np.abs(boot - obs[None, :]) / se[None, :]

    order = sorted(range(len(obs)), key=lambda i: -t[i])
    p = {}
    prev = 0.0
    for step, j in enumerate(order):
        active = order[step:]
        mx = np.nanmax(cen[:, active], axis=1)
        val = max(float(np.mean(mx >= t[j])), prev)
        p[j] = val
        prev = val
    return np.array([p[i] for i in range(len(obs))]), t


# --------------------------------------------------------------------------
# 1. Ordering
# --------------------------------------------------------------------------

def test_ordering_is_by_descending_absolute_studentised_statistic():
    """
    Three hypotheses with equal bootstrap spread but different observed effects.
    The adjusted p-values must be non-decreasing in the *descending |t|* order,
    i.e. the largest effect gets the smallest adjusted p-value.
    """
    rng = np.random.default_rng(0)
    boot = rng.normal(0.0, 1.0, size=(4000, 3))
    obs = np.array([3.0, 1.5, 0.2])          # already descending
    # Recentring uses obs, so the bootstrap spread is what studentises.
    boot = boot + obs[None, :]

    out = romano_wolf(obs, boot, _labels(3))
    p = out["p_romano_wolf"].to_numpy()
    t = out["t_statistic"].to_numpy()

    assert t[0] > t[1] > t[2]
    assert p[0] <= p[1] <= p[2], "adjusted p-values must follow the |t| ordering"


def test_ordering_is_recovered_when_input_order_is_shuffled():
    """The result must depend on the statistics, not on input row order."""
    rng = np.random.default_rng(1)
    obs = np.array([0.2, 3.0, 1.5])          # deliberately unsorted
    boot = rng.normal(0.0, 1.0, size=(4000, 3)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(3))
    p = out["p_romano_wolf"].to_numpy()

    # Descending |t| here is index 1, then 2, then 0.
    assert p[1] <= p[2] <= p[0]


# --------------------------------------------------------------------------
# 2. Monotonicity
# --------------------------------------------------------------------------

def test_adjusted_p_values_are_monotone_along_the_ordering():
    rng = np.random.default_rng(2)
    obs = np.array([2.6, 2.5, 2.4, 0.9, 0.1])
    boot = rng.normal(0.0, 1.0, size=(3000, 5)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(5))
    t = out["t_statistic"].to_numpy()
    p = out["p_romano_wolf"].to_numpy()

    order = np.argsort(-t)
    seq = p[order]
    assert np.all(np.diff(seq) >= -1e-12), f"not monotone along the ordering: {seq}"


def test_monotonicity_floor_binds_when_a_later_raw_p_is_smaller():
    """
    Construct a case where a later-ordered hypothesis has a *smaller* raw
    p-value than its predecessor, so the monotonicity step must lift it.

    Hypothesis 0 has a slightly larger |t| but a heavy-tailed bootstrap
    distribution; hypothesis 1 has a slightly smaller |t| on a tight
    distribution. Without the floor the second would come out below the first.
    """
    rng = np.random.default_rng(3)
    n = 20000
    heavy = rng.standard_t(df=3, size=n)
    heavy = heavy / heavy.std(ddof=1)
    tight = rng.normal(0.0, 1.0, size=n)

    obs = np.array([2.10, 2.05])
    boot = np.column_stack([heavy, tight]) + obs[None, :]

    ref_p, ref_t = _reference_stepdown(obs, boot)
    assert ref_t[0] > ref_t[1], "hypothesis 0 must sort first"

    out = romano_wolf(obs, boot, _labels(2))
    p = out["p_romano_wolf"].to_numpy()
    assert p[1] >= p[0] - 1e-12, "monotonicity floor did not bind"
    np.testing.assert_allclose(p, ref_p, atol=1e-12)


# --------------------------------------------------------------------------
# 3. Stepdown structure: one removal per step, max over the *remaining* set
# --------------------------------------------------------------------------

def test_matches_independent_reference_implementation():
    rng = np.random.default_rng(4)
    obs = np.array([2.9, 2.6, 2.2, 2.1, 1.1, 0.15])
    boot = rng.normal(0.0, 1.0, size=(5000, 6)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(6))
    ref_p, _ = _reference_stepdown(obs, boot)

    np.testing.assert_allclose(out["p_romano_wolf"].to_numpy(), ref_p, atol=1e-12)


def test_smallest_statistic_is_tested_against_itself_alone():
    """
    At the final step only one hypothesis remains, so its adjusted p-value is
    the unadjusted bootstrap p-value (subject to the monotonicity floor).
    """
    rng = np.random.default_rng(5)
    obs = np.array([3.0, 2.0, 0.10])
    boot = rng.normal(0.0, 1.0, size=(4000, 3)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(3))
    se = out["std_error"].to_numpy()
    t_last = abs(obs[2]) / se[2]
    marginal = float(np.mean(np.abs(boot[:, 2] - obs[2]) / se[2] >= t_last))

    got = out["p_romano_wolf"].to_numpy()[2]
    floor = out["p_romano_wolf"].to_numpy()[1]
    assert got == pytest.approx(max(marginal, floor), abs=1e-12)


def test_first_step_uses_the_maximum_over_all_hypotheses():
    rng = np.random.default_rng(6)
    obs = np.array([3.0, 2.0, 1.0, 0.4])
    boot = rng.normal(0.0, 1.0, size=(4000, 4)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(4))
    se = out["std_error"].to_numpy()
    cen = np.abs(boot - obs[None, :]) / se[None, :]
    expected = float(np.mean(cen.max(axis=1) >= out["t_statistic"].to_numpy()[0]))

    assert out["p_romano_wolf"].to_numpy()[0] == pytest.approx(expected, abs=1e-12)


def test_stepdown_is_no_more_conservative_than_simultaneous_removal():
    """
    The ordered stepdown takes its maximum over a shrinking set, so it can never
    return a *larger* adjusted p-value than the simultaneous-removal variant it
    replaced (which keeps larger-statistic hypotheses in the active set).
    """
    rng = np.random.default_rng(7)
    obs = np.array([2.8, 2.5, 2.0, 1.2, 0.3])
    boot = rng.normal(0.0, 1.0, size=(4000, 5)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(5))
    se = out["std_error"].to_numpy()
    t = out["t_statistic"].to_numpy()
    cen = np.abs(boot - obs[None, :]) / se[None, :]

    # Every-hypothesis-active p-value: the most conservative comparison.
    all_active = np.array([float(np.mean(cen.max(axis=1) >= t[j]))
                           for j in range(5)])
    assert np.all(out["p_romano_wolf"].to_numpy() <= all_active + 1e-12)


# --------------------------------------------------------------------------
# 4. Rejection rule
# --------------------------------------------------------------------------

def test_rejection_flag_matches_the_stepwise_rule():
    """
    With monotone adjusted p-values, `p_adj <= alpha` must coincide with the
    stepwise rule "reject until the first failure, then stop".
    """
    rng = np.random.default_rng(8)
    obs = np.array([4.5, 4.0, 0.8, 0.2])
    boot = rng.normal(0.0, 1.0, size=(4000, 4)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(4), alpha=0.05)
    t = out["t_statistic"].to_numpy()
    order = np.argsort(-t)
    rejected = out["reject_at_5pct"].to_numpy()[order]

    # Once False, never True again.
    seen_false = False
    for flag in rejected:
        if not flag:
            seen_false = True
        elif seen_false:
            pytest.fail(f"rejection resumed after a failure: {rejected}")


def test_no_rejection_when_nothing_separates():
    """Pure-noise differences must reject nothing at 5%."""
    rng = np.random.default_rng(9)
    obs = np.array([0.02, -0.01, 0.03])
    boot = rng.normal(0.0, 1.0, size=(3000, 3)) + obs[None, :]

    out = romano_wolf(obs, boot, _labels(3))
    assert not out["reject_at_5pct"].any()


# --------------------------------------------------------------------------
# 5. Range-based model set (explicitly not an HLN MCS)
# --------------------------------------------------------------------------

def test_range_model_set_keeps_indistinguishable_models():
    rng = np.random.default_rng(10)
    boot = rng.normal(0.0, 0.02, size=(2000, 3)) + np.array([0.17, 0.169, 0.171])
    alive, elim, _, _ = bootstrap_range_model_set(boot)
    assert sorted(alive) == [0, 1, 2]
    assert elim == []


def test_range_model_set_eliminates_a_clearly_worse_model():
    rng = np.random.default_rng(11)
    boot = rng.normal(0.0, 0.004, size=(2000, 3)) + np.array([0.175, 0.173, 0.060])
    alive, elim, elim_p, _ = bootstrap_range_model_set(boot)
    assert 2 in elim, "the far-worse model should be eliminated"
    assert 2 not in alive
    assert all(p <= 0.05 for p in elim_p)


def test_range_model_set_docstring_disclaims_the_hln_guarantee():
    """
    The terminology correction is part of the contract: the function must not
    be described as a Hansen--Lunde--Nason model confidence set.
    """
    doc = bootstrap_range_model_set.__doc__ or ""
    assert "not a Hansen--Lunde--Nason" in doc
    assert "loss differential" in doc
