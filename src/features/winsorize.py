"""
Winsorisation of continuous predictors.
=========================================
Implements the leakage-free winsorisation protocol (design §6.3):

  1. Compute upper and lower thresholds at the 1st and 99th percentiles
     using ONLY the training sample observations.
  2. Apply those SAME thresholds (unchanged) to validation and test samples.
  3. Save the thresholds to outputs/models/configs/winsor_thresholds.yaml
     so results are fully reproducible.

This prevents any information from the validation or test periods from
influencing the feature distribution used in model training.

Design reference: §6.3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.config import (
    ALL_FEATURES,
    OUT_MODELS_CONFIGS,
    TRAIN_END_YEAR,
)

# Winsorisation percentiles (1st / 99th)
LOWER_PERCENTILE = 0.01
UPPER_PERCENTILE = 0.99

# Features excluded from winsorisation.
#
# OENEG and INTWO are binary indicators, so winsorisation is meaningless.
#
# PRICE is a different case and the earlier comment here ("already bounded")
# was wrong. Its log(15) cap is an UPPER bound only: the left tail is unbounded
# and long, reaching log($0.0156) = -4.16 in the training sample, which is
# -5.9 training standard deviations. PRICE is therefore the only continuous
# predictor whose tail is not trimmed, and that asymmetry should be stated
# rather than implied.
#
# The exclusion is nonetheless retained, on measured grounds rather than on the
# assumption of boundedness. The left tail carries a large share of the outcome:
# 36 of the 404 test events lie in the 0.89% of test rows below the training 1st
# percentile, a 15.9% distress rate against 1.58% overall (26.0% in training).
# Clipping it destroys that signal, and re-fitting the four models with PRICE
# winsorised moves test PR-AUC by -0.0008 (LR), -0.0018 (RF), -0.0023 (XGB) and
# -0.0216 (NN) -- every model weakly worse, ranking unchanged. See
# src/analysis/supp_winsorisation_convention.py, which regenerates those
# figures, and the thesis methodology chapter, which discloses the asymmetry.
NO_WINSORISE = {"OENEG", "INTWO", "PRICE"}


def compute_thresholds(
    train_df: pd.DataFrame,
    features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute winsorisation thresholds from the training sample.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training-set observations ONLY (fyear <= TRAIN_END_YEAR).
    features : list[str] or None
        Feature columns to winsorise. Defaults to ALL_FEATURES minus
        binary indicators.

    Returns
    -------
    dict
        {feature_name: {'lower': float, 'upper': float}}
    """
    if features is None:
        features = [f for f in ALL_FEATURES if f not in NO_WINSORISE]

    thresholds: dict[str, dict[str, float]] = {}
    for feat in features:
        if feat not in train_df.columns:
            continue
        lower = train_df[feat].quantile(LOWER_PERCENTILE)
        upper = train_df[feat].quantile(UPPER_PERCENTILE)
        thresholds[feat] = {"lower": float(lower), "upper": float(upper)}

    print(f"  Winsorisation thresholds computed for {len(thresholds)} features.")
    return thresholds


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Apply pre-computed winsorisation thresholds to a DataFrame.

    Clips each feature at its stored lower/upper bounds. Thresholds are
    ALWAYS from the training sample — never recomputed on val/test.

    Parameters
    ----------
    df : pd.DataFrame
    thresholds : dict
        Output of compute_thresholds().

    Returns
    -------
    pd.DataFrame
        Copy of df with winsorised feature columns.
    """
    df = df.copy()
    for feat, bounds in thresholds.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lower=bounds["lower"], upper=bounds["upper"])
    return df


def save_thresholds(
    thresholds: dict[str, dict[str, float]],
    path: Path | None = None,
) -> None:
    """
    Save thresholds to YAML for reproducibility.

    Parameters
    ----------
    thresholds : dict
    path : Path or None
        Default: outputs/models/configs/winsor_thresholds.yaml
    """
    if path is None:
        path = OUT_MODELS_CONFIGS / "winsor_thresholds.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(thresholds, f, default_flow_style=False)
    print(f"  Winsorisation thresholds saved → {path}")


def load_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    """
    Load previously saved winsorisation thresholds from YAML.

    Parameters
    ----------
    path : Path or None

    Returns
    -------
    dict
    """
    if path is None:
        path = OUT_MODELS_CONFIGS / "winsor_thresholds.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
