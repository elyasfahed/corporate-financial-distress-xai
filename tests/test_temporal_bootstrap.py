"""
Regression tests for the temporal (two-way cluster) bootstrap primitives.
=========================================================================
Covers the pure resampling / variance-combination helpers used by
src/robustness/temporal_bootstrap.py:

  * group_index_map        -- unique group -> observation indices
  * resample_cluster_indices -- one block-bootstrap draw over groups
  * resample_iid_indices   -- one i.i.d. (observation-level) draw
  * twoway_variance        -- Cameron-Gelbach-Miller variance combination

These are deterministic given a seeded RNG, so the tests assert exact
structural properties rather than statistical ones.
"""
from __future__ import annotations

import numpy as np

from src.robustness.temporal_bootstrap import (
    group_index_map,
    resample_cluster_indices,
    resample_iid_indices,
    twoway_variance,
)


def test_group_index_map_partitions_observations():
    """Every observation index appears exactly once, grouped by its id."""
    groups = np.array([10, 10, 20, 30, 30, 30])
    m = group_index_map(groups)
    assert set(m.keys()) == {10, 20, 30}
    np.testing.assert_array_equal(m[10], [0, 1])
    np.testing.assert_array_equal(m[20], [2])
    np.testing.assert_array_equal(m[30], [3, 4, 5])
    # Partition: concatenated indices cover 0..n-1 with no gaps/overlaps.
    covered = np.sort(np.concatenate(list(m.values())))
    np.testing.assert_array_equal(covered, np.arange(len(groups)))


def test_cluster_resample_keeps_blocks_intact():
    """A drawn group contributes ALL its observations, and a group drawn k
    times contributes exactly k copies of the whole block (block structure
    preserved). With unequal block sizes the total resample length varies with
    the draw, so the invariant is on block multiplicity, not total length."""
    groups = np.array([1, 1, 1, 2, 3, 3])
    m = group_index_map(groups)
    rng = np.random.default_rng(0)
    idx = resample_cluster_indices(m, rng)
    total = 0
    for members in m.values():
        block = len(members)
        # Each group's observations appear a whole-block multiple of times.
        count_member_obs = int(np.isin(idx, members).sum())
        assert count_member_obs % block == 0
        total += count_member_obs
    # All resampled indices belong to some original block (nothing fabricated).
    assert total == len(idx)


def test_cluster_resample_draws_only_existing_groups():
    """No resampled index falls outside the original observation range."""
    groups = np.array([5, 5, 7, 9, 9])
    m = group_index_map(groups)
    rng = np.random.default_rng(42)
    idx = resample_cluster_indices(m, rng)
    assert idx.min() >= 0 and idx.max() < len(groups)


def test_iid_resample_shape_and_range():
    """The i.i.d. bootstrap returns n indices in [0, n)."""
    rng = np.random.default_rng(1)
    idx = resample_iid_indices(20, rng)
    assert len(idx) == 20
    assert idx.min() >= 0 and idx.max() < 20


def test_cluster_resample_is_reproducible():
    """Same seed -> same draw (needed for reproducible CIs)."""
    groups = np.array([1, 1, 2, 2, 3, 3])
    m = group_index_map(groups)
    a = resample_cluster_indices(m, np.random.default_rng(7))
    b = resample_cluster_indices(m, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_twoway_variance_cgm_formula():
    """Var_2way = Var_firm + Var_year - Var_iid when the sum is positive."""
    assert twoway_variance(0.02, 0.03, 0.01) == 0.04


def test_twoway_variance_floored_at_zero():
    """A negative CGM combination is truncated at zero (finite-sample fix)."""
    assert twoway_variance(0.01, 0.01, 0.05) == 0.0
