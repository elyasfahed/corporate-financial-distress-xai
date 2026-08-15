"""
Regression tests for the CHS-style trailing-average feature construction.
=========================================================================
trailing_weighted_avg_by_fyear builds a geometrically weighted, fiscal-year-
aligned trailing average of a column over {t, t-1, ..., t-(window-1)}:

  * weights decay**k on the year k steps back (k=0 = current year);
  * a gap in the firm's fyear sequence skips the missing year (no back-fill);
  * fewer than min_obs observed years -> NaN;
  * uses only current/past fiscal years -> look-ahead-safe.

See src/robustness/chs_trailing_average.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.robustness.chs_trailing_average import trailing_weighted_avg_by_fyear


def _firm(gvkey, years, vals):
    return pd.DataFrame({"gvkey": gvkey, "fyear": years, "x": vals})


def test_geometric_weighting_full_window():
    """Three consecutive years -> weighted mean with weights 1, 0.5, 0.25."""
    df = _firm(1, [2010, 2011, 2012], [10.0, 20.0, 40.0])
    out = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=1)
    # Row at 2012: years {2012:40, 2011:20, 2010:10}, weights {1, .5, .25}
    expected = (1 * 40 + 0.5 * 20 + 0.25 * 10) / (1 + 0.5 + 0.25)
    assert np.isclose(out.iloc[2], expected)


def test_current_year_only_when_no_history():
    """First firm-year: only the current value is available."""
    df = _firm(1, [2010, 2011, 2012], [10.0, 20.0, 40.0])
    out = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=1)
    assert np.isclose(out.iloc[0], 10.0)  # 2010 alone


def test_min_obs_enforced():
    """min_obs=2 -> the first firm-year (one observation) is NaN."""
    df = _firm(1, [2010, 2011], [10.0, 20.0])
    out = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=2)
    assert np.isnan(out.iloc[0])
    assert not np.isnan(out.iloc[1])


def test_fyear_gap_is_skipped_not_backfilled():
    """A missing fiscal year is skipped; the average uses only observed years."""
    # Firm has 2010 and 2013 (2011, 2012 absent). At 2013, window {2013,2012,2011}
    # only 2013 is observed -> with min_obs=1 the result is just the 2013 value,
    # NOT contaminated by the far-away 2010 row.
    df = _firm(1, [2010, 2013], [10.0, 40.0])
    out = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=1)
    assert np.isclose(out.iloc[1], 40.0)


def test_no_lookahead_uses_only_past():
    """The trailing average at year t never depends on values at t+1 or later."""
    df = _firm(1, [2010, 2011, 2012], [10.0, 20.0, 40.0])
    out_full = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=1)
    # Drop the future year 2012; the 2010 and 2011 averages must be unchanged.
    out_trunc = trailing_weighted_avg_by_fyear(df.iloc[:2].copy(), "x",
                                               window=3, decay=0.5, min_obs=1)
    assert np.allclose(out_full.iloc[:2].values, out_trunc.values)


def test_two_firms_do_not_leak_across_gvkey():
    """Firm B's history must not contribute to firm A's average."""
    df = pd.concat([
        _firm(1, [2011, 2012], [100.0, 200.0]),
        _firm(2, [2011, 2012], [1.0, 2.0]),
    ], ignore_index=True)
    out = trailing_weighted_avg_by_fyear(df, "x", window=3, decay=0.5, min_obs=1)
    # Firm 1 @2012: (1*200 + .5*100)/1.5 = 166.67 ; firm 2 @2012: (1*2 + .5*1)/1.5
    assert np.isclose(out.iloc[1], (200 + 0.5 * 100) / 1.5)
    assert np.isclose(out.iloc[3], (2 + 0.5 * 1) / 1.5)
