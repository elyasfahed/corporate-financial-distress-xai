"""
Regression tests for the distress-label window anchor.
======================================================
build_distress_label sets D_{i,t}=1 iff a for-cause delisting falls in the
forward window [A_{i,t}, A_{i,t}+reach], where A_{i,t} is the anchor:

  * anchor="fdate"    -> frozen / primary design. A = actual 10-K filing date,
    reach = horizon_days. A firm that goes dark (delisted for cause > 365d after
    its last 10-K) has NO window reaching the event, so it is NOT labelled
    distressed (documented selection issue).
  * anchor="datadate" -> alternative fiscal-year-end design (Implementation
    Status §10 / proposed future RC). A = fiscal year-end, reach = reach_days.
    A long enough reach recovers the went-dark failure that the 10-K anchor
    misses, at the cost of an *assumed* (not realised) availability date.

These tests pin (a) that the default arguments reproduce the frozen
10-K-anchored label exactly, and (b) that the datadate anchor recovers the
abrupt failure. See src/data/merge_crsp_compustat.py :: build_distress_label.
"""

from __future__ import annotations

import pandas as pd

from src.data.merge_crsp_compustat import build_distress_label

CODES = [400]  # one for-cause code is enough for the logic


def _panel() -> pd.DataFrame:
    """Two firms, each with two fiscal years.

    Firm 1 (gvkey 1) is a went-dark failure: last 10-K filed 2017-03-31, then
    delisted for cause 2018-09-01 (~519 days later -> beyond the 10-K window).
    Firm 2 (gvkey 2) is an ordinary capture: delisted 2017-06-01, inside the
    window of its 2016 10-K filed 2017-03-31.
    """
    rows = [
        # gvkey, permno, fyear, datadate,     fdate
        (1, 10001, 2015, "2015-12-31", "2016-03-31"),
        (1, 10001, 2016, "2016-12-31", "2017-03-31"),
        (2, 20002, 2015, "2015-12-31", "2016-03-31"),
        (2, 20002, 2016, "2016-12-31", "2017-03-31"),
    ]
    df = pd.DataFrame(rows, columns=["gvkey", "permno", "fyear", "datadate", "fdate"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["fdate"] = pd.to_datetime(df["fdate"])
    return df


def _delist() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "permno": [10001, 20002],
            "dlstdt": pd.to_datetime(["2018-09-01", "2017-06-01"]),
            "dlstcd": [400, 400],
        }
    )
    return df


def _distress(df: pd.DataFrame, gvkey: int) -> int:
    """Max distress flag for a firm across its fiscal years."""
    return int(df.loc[df["gvkey"] == gvkey, "distress"].max())


def test_fdate_anchor_misses_went_dark_firm():
    """Frozen 10-K anchor: ordinary firm captured, went-dark firm missed."""
    out = build_distress_label(_panel(), _delist(), codes=CODES)  # default anchor="fdate"
    assert _distress(out, 2) == 1, "ordinary timely failure should be captured"
    assert _distress(out, 1) == 0, "went-dark failure should be missed by 10-K anchor"


def test_default_equals_explicit_fdate_anchor():
    """Default arguments reproduce the frozen 10-K-anchored label byte-for-byte."""
    a = build_distress_label(_panel(), _delist(), codes=CODES)
    b = build_distress_label(_panel(), _delist(), codes=CODES, anchor="fdate")
    pd.testing.assert_series_equal(a["distress"], b["distress"])


def test_datadate_anchor_recovers_went_dark_firm():
    """Fiscal-year-end anchor with a 24-month reach recovers the abrupt failure."""
    out = build_distress_label(
        _panel(), _delist(), codes=CODES, anchor="datadate", reach_days=730
    )
    assert _distress(out, 1) == 1, "datadate anchor (730d) should recover went-dark firm"
    assert _distress(out, 2) == 1, "ordinary firm stays captured"


def test_datadate_short_reach_still_misses():
    """A 12-month fiscal-year-end reach is too short for this went-dark firm."""
    out = build_distress_label(
        _panel(), _delist(), codes=CODES, anchor="datadate", reach_days=365
    )
    assert _distress(out, 1) == 0, "365d reach from FY-end does not reach a 2018 delisting"


def test_invalid_anchor_raises():
    try:
        build_distress_label(_panel(), _delist(), codes=CODES, anchor="filing")
    except ValueError:
        return
    raise AssertionError("invalid anchor should raise ValueError")
