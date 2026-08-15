"""
RC1 — Alternative distress definition: formal bankruptcy codes only.
=====================================================================
Changes: DLSTCD codes 572, 574, 584 (Chapter 7 and 11 filings) instead
         of the primary definition (codes 400–499).
Fixed  : All other design dimensions unchanged.

This check tests sensitivity to the distress definition scope.
The bankruptcy-only definition substantially reduces the distressed cohort
and excludes firms that experienced severe financial failure without
formally filing for bankruptcy (Blueprint v4 §5.4, §11).

Blueprint v4 reference: §11, RC1
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_RAW_CRSP, DATA_SAMPLES, DISTRESS_CODES_RC1, ALL_FEATURES
from src.data.merge_crsp_compustat import build_distress_label
from src.models.train import train_all_models, evaluate_on_test


def run_rc1(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    codes: list[int] | None = None,
    delist_path=None,
    spec: str = "rc1",
    run_label: str = "RC1",
    skip_if_saved: bool = False,
    tune_storage_dir=None,
    event_label_col: str | None = None,
    unique_event_assignment: bool = False,
    samples_dir=None,
    manifest_input_files=None,
    manifest_extra: dict | None = None,
    train_raw: pd.DataFrame | None = None,
    tune_fold_safe: bool = False,
    tune_purge_horizon_days: int | None = None,
    tune_sic_col: str = "sich",
    tune_peer_rule: str = "rows",
    tune_impute_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Execute RC1: re-label distress using bankruptcy codes only.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Primary specification splits (re-labeled for RC1).
    features : list[str]
    codes : list[int] or None
        Bankruptcy code set. None (frozen) = DISTRESS_CODES_RC1
        (572/574/584 — under the frozen CIZ mapping these are DISJOINT
        from the 400–499 primary label). The v2
        robustness run passes DISTRESS_CODES_RC1_CORRECTED ([470]),
        which is NESTED in the corrected primary label as the design
        intends.
    delist_path : Path or None
        Delisting parquet. None (frozen) = the frozen extract; v2 passes
        the corrected-mapping raw_v2 extract.

    Returns
    -------
    pd.DataFrame
        Performance metrics under the RC1 distress definition.
    """
    codes = codes if codes is not None else DISTRESS_CODES_RC1
    print("\nRC1: Alternative distress definition — "
          "CRSP bankruptcy-reason proxy")
    delist = pd.read_parquet(delist_path or DATA_RAW_CRSP / "crsp_delisting_raw.parquet")

    def relabel(split: pd.DataFrame) -> pd.DataFrame:
        split = split.drop(columns=["distress"], errors="ignore")
        labeled = build_distress_label(
            split, delist, codes=codes, event_label_col=event_label_col,
            unique_event_assignment=unique_event_assignment,
        )
        return labeled

    train_rc1 = relabel(train)
    val_rc1   = relabel(val)
    test_rc1  = relabel(test)
    train_raw_rc1 = relabel(train_raw) if train_raw is not None else None

    # skip_if_saved / tune_storage_dir (frozen defaults off): the v2 run
    # passes both so a multi-hour re-tune survives interruption — finished
    # models are loaded, in-progress Optuna studies persist and resume.
    trained = train_all_models(train_rc1, val_rc1, features=features, spec=spec,
                               skip_if_saved=skip_if_saved,
                               tune_storage_dir=tune_storage_dir,
                               samples_dir=samples_dir,
                               manifest_input_files=manifest_input_files,
                               manifest_extra=manifest_extra,
                               tune_raw_train=train_raw_rc1,
                               tune_fold_safe=tune_fold_safe,
                               tune_purge_horizon_days=tune_purge_horizon_days,
                               tune_sic_col=tune_sic_col,
                               tune_peer_rule=tune_peer_rule,
                               tune_impute_features=tune_impute_features)
    # output_stem=None: results persisted via run_all_robustness checkpoint.
    return evaluate_on_test(trained, test_rc1, features=features,
                            output_stem=None, run_label=run_label)
