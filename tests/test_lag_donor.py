"""
Regression tests for FY1989 lag donors
======================================
The loader keeps SAMPLE_START_YEAR-1 (FY1989) explicitly for lag
construction, but the merge dropped it before features were built, so
every FY1990 lagged predictor (NITA_LAG, CHIN, INTWO) was needlessly
missing and fell into imputation.

build_accounting_features(lag_donor=...) appends the pre-sample rows
for the lag computation only and removes them before returning:

  1. Without donors, FY1990 lags are all missing (frozen behaviour).
  2. With donors, FY1990 lags are the genuine FY1989-derived values.
  3. Donor rows never appear in the returned panel.
  4. Donors with in-sample fiscal years are rejected (guard against
     accidentally feeding modelling rows through the donor path).
  5. Non-donor rows are unchanged by the donor mechanism.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.accounting_features import build_accounting_features

LAGGED = ["NITA_LAG", "CHIN", "INTWO"]


def make_gdp():
    return pd.DataFrame({
        "year": list(range(1988, 1996)),
        "deflator": [1.0] * 8,
    })


def make_panel():
    """Two firms, FY1990–1991, all raw items present."""
    base = {
        "at": 100.0, "lt": 50.0, "act": 40.0, "lct": 20.0,
        "che": 10.0, "oancf": 8.0,
    }
    rows = [
        {"gvkey": "A", "fyear": 1990, "ni": -5.0, "ib": -5.0, **base},
        {"gvkey": "A", "fyear": 1991, "ni":  2.0, "ib":  2.0, **base},
        {"gvkey": "B", "fyear": 1990, "ni":  3.0, "ib":  3.0, **base},
        {"gvkey": "B", "fyear": 1991, "ni":  4.0, "ib":  4.0, **base},
    ]
    return pd.DataFrame(rows)


def make_donor():
    """FY1989 rows for firm A (loss year) and firm C (not in the panel)."""
    base = {
        "at": 200.0, "lt": 80.0, "act": 60.0, "lct": 30.0,
        "che": 20.0, "oancf": 12.0,
    }
    return pd.DataFrame([
        {"gvkey": "A", "fyear": 1989, "ni": -10.0, "ib": -10.0, **base},
        {"gvkey": "C", "fyear": 1989, "ni":   1.0, "ib":   1.0, **base},
    ])


def test_without_donor_first_year_lags_missing():
    out = build_accounting_features(make_panel(), make_gdp())
    fy1990 = out[out["fyear"] == 1990]
    for col in LAGGED:
        assert fy1990[col].isna().all(), f"{col} should be missing in FY1990"


def test_donor_supplies_genuine_fy1990_lags():
    out = build_accounting_features(make_panel(), make_gdp(),
                                    lag_donor=make_donor())
    a90 = out[(out["gvkey"] == "A") & (out["fyear"] == 1990)].iloc[0]
    # NITA_LAG = donor ib/at = -10/200
    assert a90["NITA_LAG"] == pytest.approx(-10.0 / 200.0)
    # CHIN = (ni_t - ni_{t-1}) / (|ni_t| + |ni_{t-1}|) = (-5 - -10)/15
    assert a90["CHIN"] == pytest.approx((-5.0 + 10.0) / 15.0)
    # INTWO: loss in 1989 AND 1990 -> 1
    assert a90["INTWO"] == 1
    # Firm B has no donor -> FY1990 lags stay missing
    b90 = out[(out["gvkey"] == "B") & (out["fyear"] == 1990)].iloc[0]
    assert pd.isna(b90["NITA_LAG"])


def test_donor_rows_never_enter_the_panel():
    panel = make_panel()
    out = build_accounting_features(panel, make_gdp(), lag_donor=make_donor())
    assert len(out) == len(panel)
    assert int(out["fyear"].min()) == 1990
    assert "C" not in set(out["gvkey"])
    assert "_lag_donor" not in out.columns


def test_donor_with_in_sample_years_rejected():
    donor = make_donor()
    donor.loc[0, "fyear"] = 1990          # collides with the panel start
    with pytest.raises(ValueError, match="strictly before"):
        build_accounting_features(make_panel(), make_gdp(), lag_donor=donor)


def test_non_lag_features_unchanged_by_donor():
    plain = build_accounting_features(make_panel(), make_gdp())
    donated = build_accounting_features(make_panel(), make_gdp(),
                                        lag_donor=make_donor())
    unlagged = ["NITA", "TLTA", "WCTA", "CLCA", "CASHTA", "OCF_TA", "LNTA"]
    left = plain.sort_values(["gvkey", "fyear"])[unlagged].reset_index(drop=True)
    right = donated.sort_values(["gvkey", "fyear"])[unlagged].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
