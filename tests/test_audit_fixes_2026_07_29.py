"""
Regression tests for the 2026-07-29 independent-audit fixes.

Covers:
  * RC3 SMOTENC binary handling (audit item 5)
  * date-ranged market-index restriction (audit item 4)
  * CUSIP mismatch disposition rules (audit item 1)
  * protocol §19 abort guards (audit items 2, 11)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.cusip_disposition import (
    DISPOSITIONS,
    assert_all_dispositioned,
    classify_cusip_mismatches,
    summarise_dispositions,
)
from src.features.market_features import _restrict_to_eligible
from src.robustness.rc3_smote import (
    BINARY_FEATURES,
    binary_feature_indices,
    smote_resample,
)
from src.utils.guards import (
    assert_binary_features,
    assert_finite_model_input,
    assert_label_maturity,
    assert_no_split_overlap,
    assert_outer_purge,
    assert_permno_fyear_documented,
    assert_tuning_sample_is_not_test,
    assert_unique_gvkey_fyear,
)


# ---------------------------------------------------------------- RC3 / SMOTENC

def test_mb_missing_is_declared_binary():
    """The audit defect: MB_MISSING was absent from the categorical list."""
    assert "MB_MISSING" in BINARY_FEATURES
    assert {"OENEG", "INTWO"} <= set(BINARY_FEATURES)


def test_binary_indices_cover_all_three_in_v2_feature_set():
    from src.config import ALL_FEATURES_V2

    feats = list(ALL_FEATURES_V2)
    idx = binary_feature_indices(feats)
    assert {feats[i] for i in idx} == {"OENEG", "INTWO", "MB_MISSING"}


def test_binary_indices_reject_non_binary_column():
    feats = ["NITA", "OENEG"]
    X = np.array([[0.1, 0.0], [0.2, 0.5], [0.3, 1.0]])   # 0.5 is impossible
    with pytest.raises(ValueError, match="non-binary"):
        binary_feature_indices(feats, X)


def test_smotenc_keeps_all_binaries_integral():
    """Synthetic minority rows must not carry fractional binary values."""
    rng = np.random.default_rng(0)
    n = 400
    feats = ["NITA", "TLTA", "OENEG", "INTWO", "MB_MISSING"]
    X = np.column_stack([
        rng.normal(size=n),
        rng.normal(size=n),
        rng.integers(0, 2, size=n).astype(float),
        rng.integers(0, 2, size=n).astype(float),
        rng.integers(0, 2, size=n).astype(float),
    ])
    y = np.zeros(n, dtype=int)
    y[:40] = 1
    X_res, _ = smote_resample(X, y, feats, corrected=True)
    for f in ("OENEG", "INTWO", "MB_MISSING"):
        col = X_res[:, feats.index(f)]
        assert set(np.unique(col)) <= {0.0, 1.0}, f"{f} was interpolated"


def test_plain_smote_does_interpolate_binaries():
    """Guards the contrast: frozen mode is why the corrected mode exists."""
    rng = np.random.default_rng(1)
    n = 400
    feats = ["NITA", "MB_MISSING"]
    X = np.column_stack([rng.normal(size=n), rng.integers(0, 2, size=n).astype(float)])
    y = np.zeros(n, dtype=int)
    y[:40] = 1
    X_res, _ = smote_resample(X, y, feats, corrected=False)
    col = X_res[:, 1]
    assert not set(np.unique(col)) <= {0.0, 1.0}


# -------------------------------------------------- date-ranged market index

def _msf():
    return pd.DataFrame({
        "permno": [1, 1, 1, 2, 2],
        "date": pd.to_datetime(
            ["2000-01-31", "2005-01-31", "2010-01-31", "2000-01-31", "2010-01-31"]
        ),
        "me": [10.0, 20.0, 30.0, 5.0, 7.0],
    })


def _segments():
    # PERMNO 1 eligible only 1999-2006; PERMNO 2 eligible throughout.
    return pd.DataFrame({
        "permno": [1, 2],
        "namedt": pd.to_datetime(["1999-01-01", "1990-01-01"]),
        "nameendt": pd.to_datetime(["2006-12-31", "2020-12-31"]),
    })


def test_anytime_keeps_rows_outside_validity():
    out = _restrict_to_eligible(_msf(), eligible_permnos={1, 2})
    assert len(out) == 5      # PERMNO 1's 2010 row survives despite ineligibility


def test_date_ranged_drops_rows_outside_validity():
    out = _restrict_to_eligible(_msf(), eligible_permnos={1, 2},
                                eligible_segments=_segments())
    assert len(out) == 4
    kept = set(zip(out["permno"], out["date"].dt.year))
    assert (1, 2010) not in kept
    assert (1, 2005) in kept and (2, 2010) in kept


def test_no_filter_when_both_none_is_identity():
    msf = _msf()
    out = _restrict_to_eligible(msf, None)
    assert out is msf


def test_date_ranged_requires_date_column():
    msf = _msf().drop(columns=["date"])
    with pytest.raises(KeyError, match="date"):
        _restrict_to_eligible(msf, {1}, eligible_segments=_segments())


# ------------------------------------------------------- CUSIP dispositions

def _secnames():
    return pd.DataFrame({
        "permno": [10, 10, 20, 30, 40],
        "ncusip": ["87073810", "82836G10", "87889520", "83001A10", "11111110"],
    })


def test_corroborated_history_wins():
    """Compustat CUSIP appears elsewhere in the same PERMNO's history."""
    m = pd.DataFrame({"permno": [10], "cusip8_comp": ["87073810"],
                      "ncusip8": ["82836G10"]})
    out = classify_cusip_mismatches(m, _secnames())
    assert out["disposition"].iloc[0] == "corroborated_history"


def test_same_issuer_different_issue():
    m = pd.DataFrame({"permno": [20], "cusip8_comp": ["87889510"],
                      "ncusip8": ["87889520"]})
    out = classify_cusip_mismatches(m, _secnames())
    assert out["disposition"].iloc[0] == "same_issuer_different_issue"


def test_reorganisation_when_crsp_shows_multiple_issuers():
    m = pd.DataFrame({"permno": [10], "cusip8_comp": ["99999910"],
                      "ncusip8": ["82836G10"]})
    out = classify_cusip_mismatches(m, _secnames())
    assert out["disposition"].iloc[0] == "reorganisation"


def test_unresolved_is_a_documented_outcome_not_an_abort():
    m = pd.DataFrame({"permno": [40], "cusip8_comp": ["99999910"],
                      "ncusip8": ["11111110"]})
    out = classify_cusip_mismatches(m, _secnames())
    assert out["disposition"].iloc[0] == "unresolved"
    assert_all_dispositioned(out)          # must NOT raise


def test_missing_disposition_column_aborts():
    m = pd.DataFrame({"permno": [40], "cusip8_comp": ["9"], "ncusip8": ["1"]})
    with pytest.raises(ValueError, match="no 'disposition'"):
        assert_all_dispositioned(m)


def test_unrecognised_disposition_aborts():
    m = pd.DataFrame({"permno": [40], "cusip8_comp": ["9"], "ncusip8": ["1"],
                      "disposition": ["probably_fine"]})
    with pytest.raises(ValueError, match="documented disposition"):
        assert_all_dispositioned(m)


def test_summary_covers_every_label():
    m = pd.DataFrame({
        "permno": [10, 20, 40],
        "cusip8_comp": ["87073810", "87889510", "99999910"],
        "ncusip8": ["82836G10", "87889520", "11111110"],
    })
    s = summarise_dispositions(classify_cusip_mismatches(m, _secnames()))
    assert list(s["disposition"]) == list(DISPOSITIONS)
    assert s["firm_years"].sum() == 3


def test_empty_mismatches_is_safe():
    out = classify_cusip_mismatches(pd.DataFrame(), _secnames())
    assert len(out) == 0
    assert_all_dispositioned(out)


# ------------------------------------------------------------------- guards

def test_unique_gvkey_fyear():
    ok = pd.DataFrame({"gvkey": ["1", "1"], "fyear": [2000, 2001]})
    assert_unique_gvkey_fyear(ok)
    bad = pd.DataFrame({"gvkey": ["1", "1"], "fyear": [2000, 2000]})
    with pytest.raises(ValueError, match="uniqueness"):
        assert_unique_gvkey_fyear(bad)


def test_registered_tilray_duplicate_is_allowed():
    df = pd.DataFrame({
        "permno": [17977, 17977],
        "fyear": [2020, 2020],
        "gvkey": ["22387", "33703"],
    })
    assert_permno_fyear_documented(df)      # registered -> no raise


def test_new_duplicate_permno_fyear_aborts():
    df = pd.DataFrame({
        "permno": [999, 999],
        "fyear": [2001, 2001],
        "gvkey": ["1", "2"],
    })
    with pytest.raises(ValueError, match="undocumented duplicate"):
        assert_permno_fyear_documented(df)


def test_registered_exception_cannot_shelter_a_different_gvkey():
    """Identity matching, not counting: a new gvkey on the same pair aborts."""
    df = pd.DataFrame({
        "permno": [17977, 17977],
        "fyear": [2020, 2020],
        "gvkey": ["22387", "99999"],
    })
    with pytest.raises(ValueError, match="undocumented duplicate"):
        assert_permno_fyear_documented(df)


def test_label_maturity_guard():
    df = pd.DataFrame({"fdate": pd.to_datetime(["2023-01-01"])})
    assert_label_maturity(df, pd.Timestamp("2025-12-30"))
    with pytest.raises(ValueError, match="label windows ending after"):
        assert_label_maturity(df, pd.Timestamp("2023-06-30"))


def test_binary_feature_guard():
    assert_binary_features(pd.DataFrame({"OENEG": [0, 1, 1]}), ["OENEG"])
    with pytest.raises(ValueError, match="non-binary"):
        assert_binary_features(pd.DataFrame({"OENEG": [0, 0.5]}), ["OENEG"])
    with pytest.raises(ValueError, match="NaN"):
        assert_binary_features(pd.DataFrame({"OENEG": [0, np.nan]}), ["OENEG"])


def test_finite_model_input_guard():
    assert_finite_model_input(np.array([[1.0, 2.0]]), ["a", "b"])
    with pytest.raises(ValueError, match="NaN"):
        assert_finite_model_input(np.array([[1.0, np.nan]]), ["a", "b"])
    with pytest.raises(ValueError, match="inf"):
        assert_finite_model_input(np.array([[1.0, np.inf]]), ["a", "b"])


def test_split_overlap_guard():
    a = pd.DataFrame({"gvkey": ["1"], "fyear": [2000]})
    b = pd.DataFrame({"gvkey": ["1"], "fyear": [2010]})
    c = pd.DataFrame({"gvkey": ["1"], "fyear": [2020]})
    assert_no_split_overlap(a, b, c)
    with pytest.raises(ValueError, match="both train and test"):
        assert_no_split_overlap(a, b, a)


def test_outer_purge_guard():
    early = pd.DataFrame({"fdate": pd.to_datetime(["2009-07-29"])})
    late_ok = pd.DataFrame({"fdate": pd.to_datetime(["2010-07-30"])})
    assert_outer_purge(early, late_ok)
    late_bad = pd.DataFrame({"fdate": pd.to_datetime(["2010-07-28"])})
    with pytest.raises(ValueError, match="Outer purge violated"):
        assert_outer_purge(early, late_bad)


def test_tuning_never_sees_test():
    tune = pd.DataFrame({"gvkey": ["1"], "fyear": [2000]})
    test = pd.DataFrame({"gvkey": ["1"], "fyear": [2020]})
    assert_tuning_sample_is_not_test(tune, test)
    with pytest.raises(ValueError, match="shares"):
        assert_tuning_sample_is_not_test(test, test)
