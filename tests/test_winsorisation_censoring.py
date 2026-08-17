"""
Tests for the realised-winsorisation-censoring measure.
=======================================================
The measure replaces a summary that reported clipped cells as a
single share of (rows x features) per split, which diluted a feature censoring
4% of the test sample below visibility and could not fail.

The one subtlety it has to get right is the comparison operator. A value sitting
exactly *at* a bound was not altered by the clip, and one feature makes that
distinction load-bearing: CHIN's 1st and 99th percentiles are exactly -1 and +1,
its natural bounds, because Ohlson's ratio saturates whenever net income changes
sign. Roughly a fifth of the sample sits at those bounds while nothing at all is
clipped, so a non-strict comparison would report a fifth of the panel as
censored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.supp_winsorisation_convention import (CENSORING_FLAG_PCT,
                                                        realised_censoring)


def _frames(raw_values: dict[str, list[float]], bounds: dict[str, tuple]):
    """Build matching processed/raw split dicts for the features supplied."""
    n = len(next(iter(raw_values.values())))
    key = {"gvkey": list(range(n)), "fyear": [2000] * n}
    raw = pd.DataFrame({**key, **raw_values})
    proc = raw.copy()
    for feat, (lo, hi) in bounds.items():
        proc[feat] = proc[feat].clip(lower=lo, upper=hi)
    return ({s: proc.copy() for s in ("train", "val", "test")},
            {s: raw.copy() for s in ("train", "val", "test")})


@pytest.fixture
def patched():
    """Run realised_censoring on constructed splits and explicit bounds."""
    def _run(raw_values, bounds):
        proc, raw = _frames(raw_values, bounds)
        thresholds = {f: {"lower": lo, "upper": hi} for f, (lo, hi) in bounds.items()}
        return realised_censoring(proc, raw, thresholds=thresholds)
    return _run


def test_mass_point_at_the_bound_is_not_censoring(patched):
    """CHIN's regression test: the bounds ARE the natural range, so nothing is
    clipped even though a large share of rows sit exactly on them."""
    values = [-1.0] * 20 + [0.0] * 60 + [1.0] * 20
    out = patched({"CHIN": values}, {"CHIN": (-1.0, 1.0)})
    row = out.set_index("feature").loc["CHIN"]
    assert row["test_lower_pct"] == 0.0
    assert row["test_upper_pct"] == 0.0
    assert bool(row["drifted"]) is False


def test_values_beyond_the_bound_are_counted(patched):
    """Five of a hundred rows lie strictly above the upper bound."""
    values = [0.0] * 95 + [9.0] * 5
    out = patched({"X": values}, {"X": (-1.0, 1.0)})
    row = out.set_index("feature").loc["X"]
    assert row["test_upper_pct"] == pytest.approx(5.0)
    assert row["test_lower_pct"] == 0.0


def test_both_tails_counted_independently(patched):
    values = [-9.0] * 3 + [0.0] * 90 + [9.0] * 7
    out = patched({"X": values}, {"X": (-1.0, 1.0)})
    row = out.set_index("feature").loc["X"]
    assert row["test_lower_pct"] == pytest.approx(3.0)
    assert row["test_upper_pct"] == pytest.approx(7.0)
    assert row["test_total_pct"] == pytest.approx(10.0)


def test_drift_flag_triggers_on_either_tail(patched):
    """The flag is a per-tail trigger, not a trigger on the total: a feature
    censoring just under the threshold in BOTH tails is not drifted."""
    below = int(CENSORING_FLAG_PCT) - 1          # 1% per tail, 2% total
    values = ([-9.0] * below + [9.0] * below
              + [0.0] * (100 - 2 * below))
    out = patched({"X": values}, {"X": (-1.0, 1.0)})
    assert bool(out.set_index("feature").loc["X", "drifted"]) is False

    above = int(CENSORING_FLAG_PCT) + 2          # 4% in one tail
    values = [9.0] * above + [0.0] * (100 - above)
    out = patched({"X": values}, {"X": (-1.0, 1.0)})
    assert bool(out.set_index("feature").loc["X", "drifted"]) is True


def test_missing_raw_values_excluded_from_the_denominator(patched):
    """Rows whose raw value is missing were imputed, not censored, and must not
    dilute the rate."""
    values = [9.0] * 5 + [0.0] * 45 + [np.nan] * 50
    out = patched({"X": values}, {"X": (-1.0, 1.0)})
    # 5 of the 50 observed rows, not 5 of 100.
    assert out.set_index("feature").loc["X", "test_upper_pct"] == pytest.approx(10.0)
