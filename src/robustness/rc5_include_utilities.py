"""
RC5 — Include regulated utilities (SIC 4900–4999).
====================================================
Changes: Utilities (SIC 4900–4999) are included in the sample.
Fixed  : Financial firms (SIC 6000–6999) remain excluded; all other
         design dimensions unchanged.

Tests whether the exclusion of utilities drives the primary results.
Utilities have regulated capital structures and stable cash flows
that may produce systematically different predictor distributions
from industrial firms, potentially distorting model training.

Blueprint v4 reference: §11, RC5
"""

from __future__ import annotations

import pandas as pd

from src.config import ALL_FEATURES


def _load_rc5_splits():
    """
    Try to load the RC5-specific splits (utilities-included sample).

    Returns
    -------
    tuple (train, val, test) of pd.DataFrame  OR  None if not available.
    """
    from src.config import DATA_ROOT
    rc5_samples_dir = DATA_ROOT / "processed" / "samples_rc5"
    train_p = rc5_samples_dir / "train.parquet"
    val_p   = rc5_samples_dir / "val.parquet"
    test_p  = rc5_samples_dir / "test.parquet"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        return None
    from src.config import DATA_SAMPLES
    primary_test = DATA_SAMPLES / "test.parquet"
    if primary_test.exists() and test_p.stat().st_mtime < primary_test.stat().st_mtime:
        print("  RC5 splits are OLDER than the current primary sample — "
              "rebuild them first:  python -m scripts.build_rc5_dataset")
        return None
    print(f"  Loading RC5-specific splits from {rc5_samples_dir}")
    train = pd.read_parquet(train_p)
    val   = pd.read_parquet(val_p)
    test  = pd.read_parquet(test_p)
    return train, val, test


def run_rc5(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> pd.DataFrame:
    """
    Execute RC5: include utilities in the sample.

    Two execution paths:
      1. If `data/processed/samples_rc5/` exists (built by
         `scripts/build_rc5_dataset.py`), load those alternative splits
         which include utilities. Train + evaluate normally.
      2. Otherwise, fall back to checking the primary splits passed in.
         If they don't contain utilities, record RC5 as 'not_implemented'.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Primary splits. Replaced by RC5-specific splits if available.
    features : list[str]

    Returns
    -------
    pd.DataFrame
        Performance metrics with utilities included, OR a not_implemented marker.
    """
    from src.models.train import train_all_models, evaluate_on_test

    print("\nRC5: Including utilities (SIC 4900–4999) in the sample")

    # --- Path 1: use RC5-specific splits if available ------------------------
    rc5_splits = _load_rc5_splits()
    if rc5_splits is not None:
        train, val, test = rc5_splits
        # Sanity check that utilities really are in the RC5 sample
        if "sich" in train.columns or "_sic" in train.columns:
            sic_col = "_sic" if "_sic" in train.columns else "sich"
            n_util_train = train[sic_col].between(4900, 4999).sum()
            n_util_test  = test[sic_col].between(4900, 4999).sum()
            print(f"  Utility firm-years: train={int(n_util_train):,}, test={int(n_util_test):,}")
        trained = train_all_models(train, val, features=features, spec="rc5")
        return evaluate_on_test(trained, test, features=features,
                                output_stem=None, run_label="RC5")

    # --- Path 2: legacy fallback — check primary splits for utilities --------
    if "sich" in train.columns:
        n_util = (
            (train["sich"] // 100 == 49).sum()
            + (val["sich"] // 100 == 49).sum()
            + (test["sich"] // 100 == 49).sum()
        )
        if n_util == 0:
            print(
                "\n  *** RC5 NOT IMPLEMENTED ***\n"
                "  No utility firm-years (SIC 4900–4999) found in the splits.\n"
                "  Build the RC5 dataset first:\n"
                "      python -m scripts.build_rc5_dataset\n"
                "  RC5 is recorded as 'not_implemented' in the results table.\n"
            )
            import numpy as np
            not_impl = pd.DataFrame({
                "model": ["logistic_regression", "random_forest", "xgboost"],
                "implementation_status": ["not_implemented"] * 3,
                "note": ["Run scripts/build_rc5_dataset.py then re-run Stage 9"] * 3,
                "pr_auc": [np.nan] * 3,
                "roc_auc": [np.nan] * 3,
                "f1": [np.nan] * 3,
            })
            return not_impl
    else:
        print("  Warning: 'sich' column not present; cannot verify utility inclusion.")

    # If we somehow have utilities in the primary splits, train anyway.
    trained = train_all_models(train, val, features=features, spec="rc5")
    return evaluate_on_test(trained, test, features=features,
                            output_stem=None, run_label="RC5")
