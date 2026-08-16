"""
Regression tests — sic_col + per-feature peer counts in imputation
===================================================================
The original pipeline derived the SIC-2 imputation group from `sich`, which
is zero for every row of the original panel, so the
"industry median" silently collapsed to a pooled annual median. The fix
adds sic_col (default "sich" = frozen bytes; "_sic" = corrected) and a
per-feature peer-count rule (default "rows" = frozen).

Tests:
  1. Frozen default reproduces the documented collapse: with sich == 0
     everywhere, the "SIC-2" median equals the pooled annual median.
  2. Frozen default == explicit (sic_col="sich", peer_rule="rows") —
     byte-identical outputs.
  3. sic_col="_sic" conditions on real industries: firms in different
     industries receive different imputed values.
  4. peer_rule="per_feature": a cell with enough rows but too few
     non-missing values of a feature is disqualified for that feature
     only, and falls through to the annual median.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.impute import (
    MIN_INDUSTRY_PEERS,
    apply_imputation,
    compute_imputation_medians,
)

FEATURES = ["NITA", "TLTA"]


def make_panel():
    """
    Two industries × one fiscal year, 6 firms each (≥ MIN_INDUSTRY_PEERS).
    Industry 3600 has NITA centred on 0.10; industry 2800 on -0.20.
    `sich` is zero everywhere (frozen-extract reality); `_sic` holds the
    valid historical codes.
    """
    rows = []
    for sic, nita in ((3600, 0.10), (2800, -0.20)):
        for i in range(6):
            rows.append({
                "gvkey": f"{sic}-{i}", "fyear": 2000, "sich": 0, "_sic": sic,
                "NITA": nita + 0.001 * i, "TLTA": 0.5,
            })
    return pd.DataFrame(rows)


def test_frozen_default_collapses_to_pooled_median():
    train = make_panel()
    sic2_med, sic2_only, annual, glob = compute_imputation_medians(
        train, features=FEATURES
    )
    # One pooled pseudo-industry (sic2 == 0), exactly as observed in the
    # frozen artifacts (§18b)
    assert sic2_med["sic2"].nunique() == 1
    assert sic2_med["sic2"].iloc[0] == 0
    # The "industry" median is just the pooled median of all 12 firms
    assert sic2_med["NITA"].iloc[0] == pytest.approx(train["NITA"].median())


def test_frozen_default_equals_explicit_frozen_args():
    train = make_panel()
    default = compute_imputation_medians(train, features=FEATURES)
    explicit = compute_imputation_medians(
        train, features=FEATURES, sic_col="sich", peer_rule="rows"
    )
    for a, b in zip(default[:3], explicit[:3]):
        pd.testing.assert_frame_equal(a, b)
    assert default[3] == explicit[3]


def test_sic_col_conditions_on_real_industries():
    train = make_panel()
    sic2_med, sic2_only, annual, glob = compute_imputation_medians(
        train, features=FEATURES, sic_col="_sic"
    )
    assert set(sic2_med["sic2"].astype(int)) == {36, 28}

    # A firm with missing NITA in each industry gets its own industry's
    # median, not the pooled one
    target = pd.DataFrame([
        {"gvkey": "x1", "fyear": 2000, "sich": 0, "_sic": 3600,
         "NITA": np.nan, "TLTA": 0.5},
        {"gvkey": "x2", "fyear": 2000, "sich": 0, "_sic": 2800,
         "NITA": np.nan, "TLTA": 0.5},
    ])
    out = apply_imputation(
        target, sic2_med, sic2_only, annual, glob,
        features=FEATURES, sic_col="_sic",
    )
    med_36 = train.loc[train["_sic"] == 3600, "NITA"].median()
    med_28 = train.loc[train["_sic"] == 2800, "NITA"].median()
    assert out.loc[0, "NITA"] == pytest.approx(med_36)
    assert out.loc[1, "NITA"] == pytest.approx(med_28)
    assert out.loc[0, "NITA"] != pytest.approx(out.loc[1, "NITA"])


def test_per_feature_peer_rule_disqualifies_sparse_feature():
    train = make_panel()
    # Industry 3600: enough rows (6), but only 2 non-missing NITA values.
    # TLTA stays fully observed, so the cell must still qualify for TLTA.
    train.loc[train["_sic"] == 3600, "NITA"] = np.nan
    train.loc[train[train["_sic"] == 3600].index[:2], "NITA"] = [0.10, 0.12]
    assert train.loc[train["_sic"] == 3600, "NITA"].notna().sum() < MIN_INDUSTRY_PEERS

    # Frozen "rows" rule: the cell qualifies for NITA anyway (row count = 6)
    med_rows, *_ = compute_imputation_medians(
        train, features=FEATURES, sic_col="_sic", peer_rule="rows"
    )
    row_36 = med_rows[med_rows["sic2"] == 36]
    assert not row_36.empty and row_36["NITA"].notna().all()

    # Corrected rule: NITA is masked for industry 36, TLTA still present
    med_pf, sic2_only, annual, glob = compute_imputation_medians(
        train, features=FEATURES, sic_col="_sic", peer_rule="per_feature"
    )
    row_36 = med_pf[med_pf["sic2"] == 36]
    assert not row_36.empty
    assert row_36["NITA"].isna().all()
    assert row_36["TLTA"].notna().all()

    # And a missing NITA in industry 36 falls through past Level 1/2 to a
    # pooled fallback instead of using the under-supported industry median
    target = pd.DataFrame([{
        "gvkey": "x1", "fyear": 2000, "sich": 0, "_sic": 3600,
        "NITA": np.nan, "TLTA": 0.5,
    }])
    out = apply_imputation(
        target, med_pf, sic2_only, annual, glob,
        features=FEATURES, sic_col="_sic",
    )
    under_supported = train.loc[train["_sic"] == 3600, "NITA"].median()
    assert out.loc[0, "NITA"] == pytest.approx(annual.loc[0, "NITA"])
    assert out.loc[0, "NITA"] != pytest.approx(under_supported)


def test_bad_args_raise():
    train = make_panel()
    with pytest.raises(KeyError):
        compute_imputation_medians(train, features=FEATURES, sic_col="nope")
    with pytest.raises(ValueError):
        compute_imputation_medians(train, features=FEATURES, peer_rule="nope")


def test_nullable_int64_feature_column_imputes():
    """
    Regression (v2 stage-c crash): a feature column with pandas nullable
    Int64 dtype (e.g. a binary indicator round-tripped through parquet)
    must be imputable — Series.update used to raise a putmask coercion
    TypeError when the median series was object-dtyped.
    """
    train = make_panel()
    train["INTWO"] = pd.array([0, 1] * 6, dtype="Int64")
    feats = FEATURES + ["INTWO"]
    med = compute_imputation_medians(train, features=feats, sic_col="_sic",
                                     peer_rule="per_feature")
    target = pd.DataFrame([{
        "gvkey": "x1", "fyear": 2000, "sich": 0, "_sic": 3600,
        "NITA": np.nan, "TLTA": 0.5,
        "INTWO": pd.array([pd.NA], dtype="Int64")[0],
    }])
    target["INTWO"] = target["INTWO"].astype("Int64")
    out = apply_imputation(target, *med, features=feats, sic_col="_sic")
    assert out["INTWO"].notna().all()
    assert out["NITA"].notna().all()
    assert set(out["INTWO"].unique()).issubset({0, 1})


def test_binary_indicator_median_is_never_fractional():
    train = make_panel()
    train["INTWO"] = pd.array([0, 1] * 6, dtype="Int64")
    feats = FEATURES + ["INTWO"]
    med = compute_imputation_medians(
        train, features=feats, sic_col="_sic", peer_rule="per_feature"
    )
    target = pd.DataFrame([{
        "gvkey": "x", "fyear": 2000, "_sic": 3600,
        "NITA": 0.0, "TLTA": 0.5, "INTWO": pd.NA,
    }])
    target["INTWO"] = target["INTWO"].astype("Int64")
    out = apply_imputation(target, *med, features=feats, sic_col="_sic")
    assert out.loc[0, "INTWO"] in (0, 1)


def test_sic_zero_is_unknown_not_a_pseudo_industry():
    train = make_panel()
    train["_sic"] = 0
    med, *_ = compute_imputation_medians(
        train, features=FEATURES, sic_col="_sic", peer_rule="per_feature"
    )
    assert med.empty
