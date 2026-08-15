"""
Regression tests for the walk-forward (expanding-window) split logic.
=====================================================================
walk_forward_splits(panel, y, embargo) returns the expanding-window
(fit, val, test) frames for test origin year ``y``:

  * fit  = fyear <= y - embargo - 1   (expanding training window)
  * val  = fyear == y - 1             (rolling one-year threshold slice)
  * test = fyear == y

The embargo guarantees no forward-distress-window leakage: with embargo=1 the
fit window ends at y-2, so the most recent fitted outcome window cannot overlap
the scored year. See src/robustness/walk_forward_validation.py.
"""
from __future__ import annotations

import pandas as pd

from src.robustness.walk_forward_validation import walk_forward_splits


def _panel() -> pd.DataFrame:
    years = list(range(2008, 2018))
    return pd.DataFrame({"gvkey": 1, "fyear": years, "x": years})


def test_fit_window_is_embargoed():
    """With embargo=1, the fit window for origin y ends at y-2 (no y-1, no y)."""
    fit, val, test = walk_forward_splits(_panel(), y=2016, embargo=1)
    assert fit["fyear"].max() == 2014
    assert val["fyear"].tolist() == [2015]
    assert test["fyear"].tolist() == [2016]


def test_no_overlap_between_fit_val_test():
    """fit, val, and test fiscal years are mutually exclusive."""
    fit, val, test = walk_forward_splits(_panel(), y=2016, embargo=1)
    fy, vy, ty = set(fit["fyear"]), set(val["fyear"]), set(test["fyear"])
    assert fy.isdisjoint(vy) and fy.isdisjoint(ty) and vy.isdisjoint(ty)


def test_window_expands_with_origin():
    """A later origin year sees a strictly larger fit window (expanding)."""
    fit_early, _, _ = walk_forward_splits(_panel(), y=2014, embargo=1)
    fit_late, _, _ = walk_forward_splits(_panel(), y=2017, embargo=1)
    assert len(fit_late) > len(fit_early)


def test_embargo_zero_allows_prior_year_in_fit():
    """embargo=0 lets the fit window reach y-1 (no gap) -- contrast with default."""
    fit, _, _ = walk_forward_splits(_panel(), y=2016, embargo=0)
    assert fit["fyear"].max() == 2015
