"""
RC2 — Alternative prediction horizon: 6 months instead of 12 months.
======================================================================
Changes: DISTRESS_HORIZON = 182 days (6 months) instead of 365 days.
Fixed  : All other design dimensions unchanged (codes 400–499, same splits).

Tests sensitivity to the prediction horizon assumption.
A shorter horizon may improve precision (fewer false positives) but reduce
coverage (misses distress events that materialise 7–12 months out).

Design reference: §11, RC2
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW_CRSP, ALL_FEATURES, DISTRESS_HORIZON_RC2, DISTRESS_CODES_PRIMARY
from src.data.merge_crsp_compustat import build_distress_label
from src.models.train import train_all_models, evaluate_on_test


def run_rc2(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> pd.DataFrame:
    """
    Execute RC2: 6-month distress prediction horizon.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    features : list[str]

    Returns
    -------
    pd.DataFrame
        Performance metrics under the 6-month horizon.
    """
    print(f"\nRC2: Alternative horizon — {DISTRESS_HORIZON_RC2} days (6 months)")
    delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")

    def relabel(split: pd.DataFrame) -> pd.DataFrame:
        split = split.drop(columns=["distress"], errors="ignore")
        return build_distress_label(
            split, delist,
            codes=DISTRESS_CODES_PRIMARY,
            horizon_days=DISTRESS_HORIZON_RC2,
        )

    train_rc2 = relabel(train)
    val_rc2   = relabel(val)
    test_rc2  = relabel(test)

    from src.models.train import train_all_models, evaluate_on_test
    trained = train_all_models(train_rc2, val_rc2, features=features, spec="rc2")
    # output_stem=None: results persisted via run_all_robustness checkpoint.
    return evaluate_on_test(trained, test_rc2, features=features,
                            output_stem=None, run_label="RC2")
