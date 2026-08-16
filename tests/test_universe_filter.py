"""
Regression tests — CIZ letter-field universe filter (§18e)
===========================================================
The original pipeline never applied the SHRCD 10/11 restriction because
the CIZ letter fields were nulled during numeric coercion. These tests
cover the date-aware filter in src/data/universe.py:

  1. policy="frozen" is a no-op (reproduces the pipeline as run).
  2. policy="v2" keeps exactly the US-incorporated ordinary common
     shares and excludes ADRs/units/SBIs, non-US issuers, non-common
     securities, and off-NYSE/AMEX/NASDAQ listings (2026-07-12 fix).
  3. Missing letter fields are conservatively INELIGIBLE under v2.
  4. Missing columns raise with a pointer to the fixed extraction.
  5. apply_universe_filter keeps a PERMNO if any of its security-info
     rows is eligible, and passes frames through untouched under frozen.
  6. Date-ranged mode (date_col="datadate", 2026-07-12 fix) keeps a
     panel row only when an eligible security-info segment is valid on
     that row's date.
"""

import pandas as pd
import pytest

from src.data.universe import apply_universe_filter, universe_eligibility


def make_secinfo():
    return pd.DataFrame({
        "permno": [1, 2, 3, 4, 5, 6, 8, 9],
        "securitytype":    ["EQTY", "EQTY", "EQTY", "EQTY", "FUND", "EQTY", "EQTY", "EQTY"],
        "securitysubtype": ["COM",  "COM",  "COM",  "COM",  "CEF",  "COM",  "COM",  "COM"],
        "sharetype":       ["NS",   "AD",   "NS",   "UG",   "NS",   "NS",   "NS",   "NS"],
        "usincflg":        ["Y",    "Y",    "N",    "Y",    "Y",    pd.NA,  "Y",    "Y"],
        "issuertype":      ["CORP", "CORP", "ACOR", "CORP", "CORP", "CORP", "REIT", "CORP"],
        "exchcd_src":      ["N",    "Q",    "Q",    "A",    "N",    "Q",    "N",    "X"],
    })
    # 1 = eligible US common (NYSE); 2 = ADR; 3 = non-US; 4 = unit;
    # 5 = CEF; 6 = missing incorporation flag; 8 = REIT (excluded by
    # IssuerType); 9 = eligible letters but off-exchange (X)


def test_frozen_policy_is_noop():
    sec = make_secinfo()
    mask = universe_eligibility(sec, policy="frozen")
    assert mask.all()
    df = pd.DataFrame({"permno": [1, 2, 3, 4, 5, 6, 8], "x": range(7)})
    out = apply_universe_filter(df, sec, policy="frozen")
    pd.testing.assert_frame_equal(out, df)


def test_exchange_restriction_excludes_off_exchange():
    sec = make_secinfo()
    mask = universe_eligibility(sec, policy="v2")
    row9 = sec.index[sec["permno"] == 9][0]
    assert not mask.iloc[row9]


def test_v2_policy_keeps_only_us_common():
    sec = make_secinfo()
    mask = universe_eligibility(sec, policy="v2")
    assert list(sec.loc[mask, "permno"]) == [1]


def test_v2_missing_letter_fields_ineligible():
    sec = make_secinfo()
    # permno 6 differs from permno 1 only by the NA incorporation flag
    mask = universe_eligibility(sec, policy="v2")
    assert not mask.iloc[5]


def test_v2_case_and_whitespace_insensitive():
    sec = make_secinfo()
    sec.loc[0, "sharetype"] = " ns "
    sec.loc[0, "usincflg"] = "y"
    mask = universe_eligibility(sec, policy="v2")
    assert mask.iloc[0]


def test_reit_issuer_excluded():
    sec = make_secinfo()
    mask = universe_eligibility(sec, policy="v2")
    reit_row = sec.index[sec["permno"] == 8][0]
    assert not mask.iloc[reit_row]


def test_missing_columns_raise():
    sec = make_secinfo().drop(columns=["usincflg"])
    with pytest.raises(KeyError, match="letter fields"):
        universe_eligibility(sec, policy="v2")


def test_bad_policy_raises():
    with pytest.raises(ValueError):
        universe_eligibility(make_secinfo(), policy="nope")


def test_apply_filter_any_rule_per_permno():
    # permno 7 has one stale ineligible row and one eligible row -> kept
    sec = pd.concat([make_secinfo(), pd.DataFrame({
        "permno": [7, 7],
        "securitytype":    ["EQTY", "EQTY"],
        "securitysubtype": ["COM",  "COM"],
        "sharetype":       ["AD",   "NS"],
        "usincflg":        ["Y",    "Y"],
        "issuertype":      ["CORP", "CORP"],
        "exchcd_src":      ["Q",    "Q"],
    })], ignore_index=True)
    df = pd.DataFrame({"permno": [1, 2, 3, 4, 5, 6, 7, 8], "x": range(8)})
    out = apply_universe_filter(df, sec, policy="v2")
    assert sorted(out["permno"]) == [1, 7]


# ---------------------------------------------------------------------------
# Date-ranged (as-of) mode
# ---------------------------------------------------------------------------

def make_dated_secinfo():
    """permno 10: eligible common share 1995–2000-06, then converts to an
    ADR (ineligible) until end-2010; nothing after."""
    return pd.DataFrame({
        "permno": [10, 10],
        "securitytype":    ["EQTY", "EQTY"],
        "securitysubtype": ["COM",  "COM"],
        "sharetype":       ["NS",   "AD"],
        "usincflg":        ["Y",    "Y"],
        "issuertype":      ["CORP", "CORP"],
        "exchcd_src":      ["N",    "N"],
        "namedt":          ["1995-01-01", "2000-07-01"],
        "nameendt":        ["2000-06-30", "2010-12-31"],
    })


def test_date_ranged_keeps_only_rows_inside_eligible_segments():
    sec = make_dated_secinfo()
    df = pd.DataFrame({
        "permno": [10, 10, 10],
        "datadate": pd.to_datetime(["1998-12-31", "2005-12-31", "2012-12-31"]),
        "x": [1, 2, 3],
    })
    out = apply_universe_filter(df, sec, policy="v2", date_col="datadate")
    # 1998: inside the eligible NS segment -> kept
    # 2005: inside the INELIGIBLE AD segment -> dropped
    # 2012: outside every segment -> dropped (conservative)
    assert list(out["x"]) == [1]


def test_date_ranged_stricter_than_any_time():
    sec = make_dated_secinfo()
    df = pd.DataFrame({
        "permno": [10],
        "datadate": pd.to_datetime(["2005-12-31"]),
        "x": [2],
    })
    any_time = apply_universe_filter(df, sec, policy="v2")
    dated = apply_universe_filter(df, sec, policy="v2", date_col="datadate")
    # Any-time keeps the firm (it was once eligible); as-of does not.
    assert len(any_time) == 1
    assert len(dated) == 0


def test_date_ranged_requires_validity_columns():
    sec = make_secinfo()   # no namedt/nameendt
    df = pd.DataFrame({
        "permno": [1],
        "datadate": pd.to_datetime(["1998-12-31"]),
    })
    with pytest.raises(KeyError, match="namedt"):
        apply_universe_filter(df, sec, policy="v2", date_col="datadate")
