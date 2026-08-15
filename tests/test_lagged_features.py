"""
Regression tests for the fiscal-year-aligned lagged accounting predictors.
===========================================================================
NITA_LAG, CHIN, and INTWO are lagged within firm. The lag must align on the
genuine prior *fiscal year* (fyear - 1), NOT on row position. A positional
``groupby('gvkey')[col].shift(1)`` silently imports a stale value across a gap
in a firm's fyear sequence (e.g. FY1993 used as the t-1 of FY1996), which is a
look-ahead/correctness bug. These tests pin the correct behaviour.

See src/features/accounting_features.py :: lag_within_firm_by_fyear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.accounting_features import (
    lag_within_firm_by_fyear,
    compute_nita,
    compute_nita_lag,
    compute_chin,
    compute_intwo,
)


def _panel_with_gap() -> pd.DataFrame:
    """
    Two firms:
      - firm 'A': consecutive FY 2000, 2001, 2002  (no gap)
      - firm 'B': FY 2005, then a gap, then FY 2008 (FY2006-07 missing)
    """
    rows = [
        # gvkey, fyear, ni,    at,  lt,  act, lct, che, ib,   oancf
        ("A", 2000, 10.0, 100, 40, 60, 30, 20, 10.0, 12.0),
        ("A", 2001, -5.0, 100, 40, 60, 30, 20, -5.0, -4.0),
        ("A", 2002, -3.0, 100, 40, 60, 30, 20, -3.0, -2.0),
        ("B", 2005, 20.0, 200, 80, 90, 40, 30, 20.0, 25.0),
        ("B", 2008, -8.0, 200, 80, 90, 40, 30, -8.0, -6.0),
    ]
    cols = ["gvkey", "fyear", "ni", "at", "lt", "act", "lct", "che", "ib", "oancf"]
    df = pd.DataFrame(rows, columns=cols)
    df["fyear"] = df["fyear"].astype("Int64")
    return df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)


def test_lag_helper_aligns_on_fiscal_year_not_position():
    df = _panel_with_gap()
    lag_ni = lag_within_firm_by_fyear(df, "ni")

    # firm A: first year NaN, then the genuine prior-year values
    a = df["gvkey"] == "A"
    assert lag_ni[a].tolist()[0] != lag_ni[a].tolist()[0] or np.isnan(lag_ni[a].iloc[0])
    assert np.isnan(lag_ni[a].iloc[0])
    assert lag_ni[a].iloc[1] == 10.0   # lag of FY2001 is FY2000 ni
    assert lag_ni[a].iloc[2] == -5.0   # lag of FY2002 is FY2001 ni

    # firm B: FY2005 NaN (first); FY2008 must be NaN because FY2007 is absent.
    # A positional shift would WRONGLY put 20.0 (the FY2005 value) here.
    b = df["gvkey"] == "B"
    assert np.isnan(lag_ni[b].iloc[0])
    assert np.isnan(lag_ni[b].iloc[1]), "lag crossed a fiscal-year gap (the bug)"


def test_nita_lag_is_nan_across_gap():
    df = _panel_with_gap()
    df["NITA"] = compute_nita(df)
    nita_lag = compute_nita_lag(df)

    b2008 = (df["gvkey"] == "B") & (df["fyear"] == 2008)
    assert nita_lag[b2008].isna().all(), "NITA_LAG must be NaN when FY t-1 is unobserved"

    a2001 = (df["gvkey"] == "A") & (df["fyear"] == 2001)
    assert np.isclose(nita_lag[a2001].iloc[0], 10.0 / 100.0)


def test_chin_is_nan_across_gap():
    df = _panel_with_gap()
    chin = compute_chin(df)
    b2008 = (df["gvkey"] == "B") & (df["fyear"] == 2008)
    assert chin[b2008].isna().all()

    # firm A FY2001: (-5 - 10) / (|-5| + |10|) = -15/15 = -1.0
    a2001 = (df["gvkey"] == "A") & (df["fyear"] == 2001)
    assert np.isclose(chin[a2001].iloc[0], -1.0)


def test_intwo_is_na_across_gap_and_correct_otherwise():
    df = _panel_with_gap()
    intwo = compute_intwo(df)

    # FY2008 of firm B: lag unobserved → NA (not 0/1)
    b2008 = (df["gvkey"] == "B") & (df["fyear"] == 2008)
    assert intwo[b2008].isna().all()

    # firm A FY2002: ni<0 in 2002 (-3) AND 2001 (-5) → 1
    a2002 = (df["gvkey"] == "A") & (df["fyear"] == 2002)
    assert intwo[a2002].iloc[0] == 1

    # firm A FY2001: ni<0 in 2001 (-5) but 2000 ni was +10 → 0
    a2001 = (df["gvkey"] == "A") & (df["fyear"] == 2001)
    assert intwo[a2001].iloc[0] == 0


def test_clean_rows_match_positional_shift():
    """Where there is NO gap, the fyear-aligned lag must equal the old
    positional shift exactly (so the fix changes only the affected rows)."""
    df = _panel_with_gap()
    df["NITA"] = compute_nita(df)
    fixed = compute_nita_lag(df)
    positional = df.groupby("gvkey")["NITA"].shift(1)

    prev = df.groupby("gvkey")["fyear"].shift(1)
    clean = (df["fyear"] - prev) == 1
    assert np.allclose(
        fixed[clean].astype(float), positional[clean].astype(float), equal_nan=True
    )
