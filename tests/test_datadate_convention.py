"""
Regression tests for the fiscal year-end (datadate) reconstruction.
====================================================================
The local CIZ period descriptor carries no DATADATE, so the pipeline
reconstructs it from (FYEAR, FYR). The standard Compustat convention (the
"June rule") assigns fiscal years ending January--May to FYEAR = calendar
year of the period end MINUS 1, i.e. the period actually ends in calendar
year FYEAR + 1 for FYR in {1..5}.

The frozen pipeline used month_end(FYEAR, FYR) unconditionally, which
mis-dates every Jan--May fiscal year-end by exactly 12 months (verified on
Walmart, gvkey 11259: the FYEAR 2019 row carries AT = 236,495 -- the balance
sheet of the period ending 2020-01-31 -- but received datadate 2019-01-31).
See Implementation Status paragraph 17.

These tests pin both behaviours:
  * convention="frozen"   reproduces the original construction exactly
  * convention="standard" applies the June rule correctly
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.load_local_rds import compustat_datadate


def _dd(fyear, fyr, convention):
    out = compustat_datadate(
        pd.Series([fyear]), pd.Series([fyr]), convention=convention
    )
    return out.iloc[0]


# ---------------------------------------------------------------------------
# Test 1 -- December fiscal year-end is identical under both conventions
# ---------------------------------------------------------------------------

def test_december_fye_unchanged():
    assert _dd(2019, 12, "frozen") == pd.Timestamp("2019-12-31")
    assert _dd(2019, 12, "standard") == pd.Timestamp("2019-12-31")


# ---------------------------------------------------------------------------
# Test 2 -- June--November fiscal year-ends are identical under both
# ---------------------------------------------------------------------------

def test_june_to_november_fye_unchanged():
    for fyr, month_end in [(6, "2019-06-30"), (9, "2019-09-30"), (11, "2019-11-30")]:
        assert _dd(2019, fyr, "frozen") == pd.Timestamp(month_end)
        assert _dd(2019, fyr, "standard") == pd.Timestamp(month_end)


# ---------------------------------------------------------------------------
# Test 3 -- January--May fiscal year-ends: the Walmart case (the bug)
# ---------------------------------------------------------------------------

def test_january_fye_walmart_case():
    # Walmart FYEAR 2019, FYR 1: the fiscal year ends 2020-01-31.
    assert _dd(2019, 1, "frozen") == pd.Timestamp("2019-01-31"), \
        "frozen convention must reproduce the original (mis-dated) construction"
    assert _dd(2019, 1, "standard") == pd.Timestamp("2020-01-31"), \
        "standard convention must apply the June rule (FYR<=5 ends in FYEAR+1)"


def test_all_jan_to_may_months_shift_forward():
    for fyr in range(1, 6):
        frozen = _dd(2010, fyr, "frozen")
        standard = _dd(2010, fyr, "standard")
        assert standard.year == frozen.year + 1
        assert standard.month == frozen.month == fyr


# ---------------------------------------------------------------------------
# Test 4 -- month-end day is correct (incl. February)
# ---------------------------------------------------------------------------

def test_month_end_day():
    assert _dd(2019, 2, "standard") == pd.Timestamp("2020-02-29")  # leap year
    assert _dd(2018, 2, "standard") == pd.Timestamp("2019-02-28")


# ---------------------------------------------------------------------------
# Test 5 -- missing FYR falls back to December under both conventions
# ---------------------------------------------------------------------------

def test_missing_fyr_defaults_to_december():
    out_f = compustat_datadate(
        pd.Series([2019]), pd.Series([float("nan")]), convention="frozen"
    )
    out_s = compustat_datadate(
        pd.Series([2019]), pd.Series([float("nan")]), convention="standard"
    )
    assert out_f.iloc[0] == pd.Timestamp("2019-12-31")
    assert out_s.iloc[0] == pd.Timestamp("2019-12-31")


# ---------------------------------------------------------------------------
# Test 6 -- invalid convention raises
# ---------------------------------------------------------------------------

def test_invalid_convention_raises():
    with pytest.raises(ValueError):
        compustat_datadate(pd.Series([2019]), pd.Series([12]), convention="bogus")
