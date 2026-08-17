"""
Regression tests for the OENEG/INTWO missing-input policy
=========================================================
Two silent-zero defects in the binary accounting indicators:

  OENEG = (lt > at).astype(int): a NaN comparison evaluates False, so a
  firm-year with missing lt or at gets OENEG=0 (283 v1 rows) — bypassing
  the ≥8/11 coverage filter and imputation.

  INTWO: a missing LAGGED ni already yields NA, but a missing CURRENT ni
  evaluates `NaN < 0 = False` and silently yields INTWO=0.

missing_policy="frozen" (default) reproduces the original behaviour
byte-for-byte; "strict" (v2 profile) returns NA so the row is handled by
the standard missing-data machinery.
"""

import pandas as pd
import pytest

from src.features.accounting_features import compute_intwo, compute_oeneg


def make_frame():
    return pd.DataFrame({
        "gvkey": ["A", "A", "A", "B", "B"],
        "fyear": [2000, 2001, 2002, 2000, 2001],
        #        lt>at   lt NA   lt>at   at NA   both NA
        "lt":    [50.0,  None,   150.0,  10.0,   None],
        "at":    [100.0, 100.0,  100.0,  None,   None],
        "ni":    [-5.0,  None,   -1.0,   2.0,    -3.0],
    })


# ---------------------------------------------------------------------------
# OENEG
# ---------------------------------------------------------------------------

def test_oeneg_frozen_reproduces_silent_zero():
    df = make_frame()
    oeneg = compute_oeneg(df)                       # default = frozen
    # Missing lt/at silently evaluate to 0 (the documented v1 defect)
    assert list(oeneg) == [0, 0, 1, 0, 0]
    assert str(oeneg.dtype) in ("int32", "int64")   # plain int, no NA


def test_oeneg_strict_returns_na_on_missing_inputs():
    df = make_frame()
    oeneg = compute_oeneg(df, missing_policy="strict")
    assert oeneg.iloc[0] == 0          # lt=50  < at=100
    assert pd.isna(oeneg.iloc[1])      # lt missing
    assert oeneg.iloc[2] == 1          # lt=150 > at=100
    assert pd.isna(oeneg.iloc[3])      # at missing
    assert pd.isna(oeneg.iloc[4])      # both missing


def test_oeneg_bad_policy_raises():
    with pytest.raises(ValueError, match="missing_policy"):
        compute_oeneg(make_frame(), missing_policy="nope")


# ---------------------------------------------------------------------------
# INTWO
# ---------------------------------------------------------------------------

def test_intwo_frozen_missing_current_ni_is_zero():
    df = make_frame()
    intwo = compute_intwo(df)                       # default = frozen
    # A/2001: current ni missing but lagged ni present (-5)
    # -> frozen silently yields 0 (the documented defect)
    assert intwo.iloc[1] == 0


def test_intwo_strict_missing_current_ni_is_na():
    df = make_frame()
    intwo = compute_intwo(df, missing_policy="strict")
    assert pd.isna(intwo.iloc[1])      # current ni missing -> NA
    # First observation of each firm has no lag -> NA under both policies
    assert pd.isna(intwo.iloc[0])
    assert pd.isna(intwo.iloc[3])
    # B/2001: ni=-3 < 0 but lagged ni=2 >= 0 -> genuine 0 stays 0
    assert intwo.iloc[4] == 0


def test_intwo_missing_lagged_ni_na_under_both_policies():
    df = make_frame()
    # A/2002: current ni present, but the fyear-2001 ni VALUE is missing
    # -> lag is NaN -> NA under frozen already (unchanged by strict)
    assert pd.isna(compute_intwo(df).iloc[2])
    assert pd.isna(compute_intwo(df, missing_policy="strict").iloc[2])


def test_intwo_bad_policy_raises():
    with pytest.raises(ValueError, match="missing_policy"):
        compute_intwo(make_frame(), missing_policy="nope")
