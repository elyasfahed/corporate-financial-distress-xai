"""
Regression tests — fold-safe CV (in-fold preprocessing + purging)
==================================================================
Audit findings on the tuning CV: (a) winsorisation/imputation were fitted
once on all 1990–2009 data before the rolling folds were formed, leaking
future distributions into earlier folds; (b) fyear-based folds are not
point-in-time — training rows filed shortly before the fold origin carry
labels resolved after it. Both fixes live in src/models/fold_safe_cv.py,
OFF by default.

  1. purge_fold_train drops exactly the fold-train rows whose label
     window [fdate, fdate+horizon] reaches the fold origin.
  2. NaT filing dates are purged (conservative).
  3. fold_safe_preprocess fits winsor thresholds on the fold's training
     rows only — an extreme value in the fold-validation year no longer
     shapes the clip bounds.
  4. fold_safe_preprocess imputes from fold-train medians only.
  5. End-to-end: tune_model(fold_safe=True, purge_horizon_days=365) runs
     on a tiny raw frame and returns valid params (wiring proof).
  6. The in-fold >=8-of-11 coverage filter (2026-07-12 second-audit fix)
     drops under-covered rows exactly as the outer pipeline does — the
     raw splits are saved BEFORE the outer filter, so without this the
     tuning folds include firm-years the final training set drops.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.fold_safe_cv import fold_safe_preprocess, purge_fold_train
from src.models.tune import rolling_origin_cv_splits, tune_model

FEATURES = ["NITA", "TLTA"]


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------

def make_purge_frame():
    """
    Fold-train rows in fyear 2004, fold-val rows in fyear 2005.
    Validation origin (earliest val fdate) = 2005-03-01.
    """
    rows = [
        # label window ends 2004-06-30+365 = 2005-06-30 >= origin -> purge
        {"fyear": 2004, "fdate": "2004-06-30", "NITA": 0.1, "distress": 0},
        # ends 2005-01-31 < origin -> keep
        {"fyear": 2004, "fdate": "2004-01-31", "NITA": 0.1, "distress": 0},
        # late filer: fdate after origin -> purge
        {"fyear": 2004, "fdate": "2005-06-15", "NITA": 0.1, "distress": 0},
        # missing filing date -> purge (conservative)
        {"fyear": 2004, "fdate": None, "NITA": 0.1, "distress": 0},
        # validation fold
        {"fyear": 2005, "fdate": "2005-03-01", "NITA": 0.2, "distress": 1},
        {"fyear": 2005, "fdate": "2005-09-01", "NITA": 0.2, "distress": 0},
    ]
    df = pd.DataFrame(rows)
    df["fdate"] = pd.to_datetime(df["fdate"])
    return df


def test_purge_drops_immature_and_late_rows():
    df = make_purge_frame()
    tr_idx = df.index[df["fyear"] == 2004]
    val_idx = df.index[df["fyear"] == 2005]
    kept = purge_fold_train(df, tr_idx, val_idx, horizon_days=365)
    assert list(kept) == [1]          # only the early-filed mature row


def test_purge_keeps_everything_when_horizon_zero_and_all_before_origin():
    df = make_purge_frame().dropna(subset=["fdate"]).reset_index(drop=True)
    tr_idx = df.index[df["fyear"] == 2004]
    val_idx = df.index[df["fyear"] == 2005]
    kept = purge_fold_train(df, tr_idx, val_idx, horizon_days=0)
    # With no horizon, only the late filer (fdate >= origin) is purged
    assert list(kept) == [0, 1]


def test_purge_requires_date_column():
    df = make_purge_frame().drop(columns=["fdate"])
    with pytest.raises(KeyError, match="fdate"):
        purge_fold_train(df, df.index[:2], df.index[2:], horizon_days=365)


# ---------------------------------------------------------------------------
# In-fold preprocessing
# ---------------------------------------------------------------------------

def make_raw_folds():
    """
    Fold-train: NITA uniform-ish around 0.10 (60 rows).
    Fold-val:   one extreme outlier (NITA = 50) plus a missing value.
    Under the frozen (outer) scheme the outlier would enter the winsor
    quantiles; in-fold it must not.
    """
    rng = np.random.default_rng(42)
    tr = pd.DataFrame({
        "fyear": 2000, "sich": 0, "_sic": 3600,
        "NITA": rng.normal(0.10, 0.01, size=60),
        "TLTA": rng.normal(0.50, 0.05, size=60),
    })
    val = pd.DataFrame({
        "fyear": [2001, 2001], "sich": [0, 0], "_sic": [3600, 3600],
        "NITA": [50.0, np.nan], "TLTA": [0.5, 0.5],
    })
    return tr, val


def test_in_fold_winsor_bounds_come_from_fold_train_only():
    tr, val = make_raw_folds()
    tr_p, val_p = fold_safe_preprocess(tr, val, features=FEATURES)
    upper = tr["NITA"].quantile(0.99)
    # The val outlier is clipped to the FOLD-TRAIN upper bound
    assert val_p["NITA"].iloc[0] == pytest.approx(upper)
    assert val_p["NITA"].iloc[0] < 1.0     # nowhere near the raw 50.0


def test_in_fold_imputation_uses_fold_train_median():
    # coverage_filter=False isolates the imputation behaviour: with only
    # 2 of the 11 accounting features present, the NaN row would
    # otherwise be dropped by the coverage filter before imputation.
    tr, val = make_raw_folds()
    tr_p, val_p = fold_safe_preprocess(tr, val, features=FEATURES,
                                       coverage_filter=False)
    assert val_p["NITA"].isna().sum() == 0
    # Imputed from fold-train (≈0.10), not from anything val-side
    assert abs(val_p["NITA"].iloc[1] - 0.10) < 0.05


def test_in_fold_coverage_filter_mirrors_outer_pipeline():
    tr, val = make_raw_folds()
    # Only NITA and TLTA of the 11 accounting features are present, so
    # the requirement is min(8, 2) = 2 non-missing: the val row with
    # missing NITA falls below it and must be dropped in-fold, exactly
    # as build_features stage 6b would drop it from the final splits.
    _, val_on = fold_safe_preprocess(tr, val, features=FEATURES)
    assert len(val_on) == 1
    _, val_off = fold_safe_preprocess(tr, val, features=FEATURES,
                                      coverage_filter=False)
    assert len(val_off) == 2


# ---------------------------------------------------------------------------
# End-to-end wiring through tune_model
# ---------------------------------------------------------------------------

def make_raw_training_panel():
    rng = np.random.default_rng(42)
    frames = []
    for fyear in range(2000, 2008):        # 8 years -> 5 rolling folds
        n = 60
        nita = rng.normal(0.05, 0.10, size=n)
        df = pd.DataFrame({
            "fyear": fyear,
            "fdate": pd.Timestamp(f"{fyear + 1}-03-31"),
            "sich": 0,
            "_sic": rng.choice([2800, 3600], size=n),
            "NITA": np.where(rng.random(n) < 0.05, np.nan, nita),
            "TLTA": rng.normal(0.5, 0.2, size=n),
        })
        df["distress"] = (
            (nita < -0.05) | (rng.random(n) < 0.05)
        ).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_tune_model_fold_safe_end_to_end():
    raw = make_raw_training_panel()
    # sanity: every fold has positives so PR-AUC is defined
    splits = rolling_origin_cv_splits(raw)
    assert len(splits) == 5

    best_params, study = tune_model(
        "logistic_regression", raw, features=FEATURES, n_trials=2,
        fold_safe=True, purge_horizon_days=365,
    )
    assert "C" in best_params
    assert np.isfinite(study.best_value)
    assert study.best_value > 0
