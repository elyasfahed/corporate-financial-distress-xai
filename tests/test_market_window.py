"""
Regression tests for the EXRET / SIGMA trailing-window minimum-observation rule.
================================================================================
EXRET (12-month cumulative excess return) and SIGMA (12-month return volatility)
are built from a trailing 12-month window of CRSP monthly returns. The
``min_obs`` parameter controls how many valid monthly returns the window must
contain before a value is emitted (otherwise NaN).

  * ``min_obs=6``  reproduces the frozen pipeline. Firm-months with 6-11 valid
    returns receive a *partial-window* value: a cumulative excess return over
    fewer than twelve months (noisy / not comparable across firms).
  * ``min_obs=12`` is the principled full-window construction: only firm-months
    with a complete twelve-month history emit a value; partial windows -> NaN.

These tests pin both behaviours so the construction-noise fix cannot silently
regress. See src/features/market_features.py :: compute_exret / compute_sigma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.market_features import compute_exret, compute_sigma


def _msf_one_firm(n_months: int) -> pd.DataFrame:
    """One permno with `n_months` consecutive monthly returns ending 2010-12."""
    dates = pd.date_range("2010-12-31", periods=n_months, freq="ME")[::-1]
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "permno": 10001,
        "date": dates.sort_values(),
        "ret": rng.normal(0.01, 0.05, n_months),
    })


def _vwretd_for(msf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"date": msf["date"].unique(), "vwretd": 0.005})


def test_partial_window_is_nan_under_full_window_rule():
    """A firm with only 8 months of history: value under min_obs=6, NaN under 12."""
    msf = _msf_one_firm(8)
    vw = _vwretd_for(msf)
    last = msf["date"].max()
    ym = (last.year, last.month)

    ex6 = compute_exret(msf, vw, min_obs=6)
    ex12 = compute_exret(msf, vw, min_obs=12)
    sg6 = compute_sigma(msf, min_obs=6)
    sg12 = compute_sigma(msf, min_obs=12)

    def val(df, col):
        row = df[(df["_year"] == ym[0]) & (df["_month"] == ym[1])]
        return row[col].iloc[0]

    # Frozen rule (min_obs=6) emits a partial-window value ...
    assert np.isfinite(val(ex6, "EXRET"))
    assert np.isfinite(val(sg6, "SIGMA"))
    # ... the principled full-window rule (min_obs=12) marks it missing.
    assert np.isnan(val(ex12, "EXRET"))
    assert np.isnan(val(sg12, "SIGMA"))


def test_full_window_value_is_identical_under_both_rules():
    """For a firm-month with >=12 valid returns, min_obs does not change the value."""
    msf = _msf_one_firm(18)
    vw = _vwretd_for(msf)
    last = msf["date"].max()
    ym = (last.year, last.month)

    for fn, col in [(compute_exret, "EXRET"), (compute_sigma, "SIGMA")]:
        kw = dict(min_obs=6) if fn is compute_sigma else dict(vwretd=vw, min_obs=6)
        kw12 = dict(min_obs=12) if fn is compute_sigma else dict(vwretd=vw, min_obs=12)
        a = fn(msf, **kw)
        b = fn(msf, **kw12)
        ra = a[(a["_year"] == ym[0]) & (a["_month"] == ym[1])][col].iloc[0]
        rb = b[(b["_year"] == ym[0]) & (b["_month"] == ym[1])][col].iloc[0]
        assert np.isclose(ra, rb), f"{col}: full-window value differs across min_obs"


def test_below_minimum_is_nan():
    """Fewer than min_obs valid returns -> NaN under both rules."""
    msf = _msf_one_firm(4)
    vw = _vwretd_for(msf)
    last = msf["date"].max()
    ym = (last.year, last.month)
    ex = compute_exret(msf, vw, min_obs=6)
    row = ex[(ex["_year"] == ym[0]) & (ex["_month"] == ym[1])]
    assert np.isnan(row["EXRET"].iloc[0])
