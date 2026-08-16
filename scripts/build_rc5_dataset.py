"""
Build the alternative dataset used by RC5 (include utilities, SIC 4900-4999).
============================================================================

The primary dataset excludes both financials (SIC 6000-6999) and utilities
(SIC 4900-4999). RC5 tests sensitivity to the utility exclusion by including
utilities in the sample.

To avoid disturbing the primary dataset, this script produces a parallel set
of files:

    data/processed/merged/panel_raw_rc5.parquet
    data/processed/features/features_all_rc5.parquet
    data/processed/samples_rc5/{train,val,test}.parquet

The RC5 robustness check (src/robustness/rc5_include_utilities.py) loads
these parallel files when they exist.

Run
---
    python -m scripts.build_rc5_dataset

After running, re-run Stage 9 to populate rc5_results.csv with real numbers:
    Remove-Item outputs/tables/robustness/rc5_results.csv
    python run_pipeline.py --stages 9

Design reference: §11, RC5
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_MERGED, DATA_FEATURES, DATA_ROOT
from src.data.merge_crsp_compustat import main_with_overrides
from src.features.build_features import run_feature_pipeline


# Only financials are excluded — utilities are KEPT for RC5
RC5_EXCLUDED_SIC_RANGES: list[tuple[int, int]] = [(6000, 6999)]


def main() -> None:
    print("\n" + "=" * 60)
    print("  BUILD RC5 DATASET — include utilities (SIC 4900-4999)")
    print("=" * 60)
    print(f"  Excluded SIC ranges: {RC5_EXCLUDED_SIC_RANGES}")
    print()

    # Paths for the RC5 parallel dataset
    rc5_panel_path     = DATA_MERGED / "panel_raw_rc5.parquet"
    rc5_features_path  = DATA_FEATURES / "features_all_rc5.parquet"
    rc5_samples_dir    = DATA_ROOT / "processed" / "samples_rc5"
    rc5_attrition_path = DATA_MERGED / "attrition_table_rc5.csv"
    rc5_mismatches_p   = DATA_MERGED / "cusip_mismatches_rc5.csv"

    # ── Step 1: merge with utilities kept in the sample ──────────────────────
    print("\n--- Step 1: Re-running merge with utilities INCLUDED ---")
    main_with_overrides(
        excluded_sic_ranges=RC5_EXCLUDED_SIC_RANGES,
        output_panel_path=rc5_panel_path,
        output_attrition_path=rc5_attrition_path,
        output_mismatches_path=rc5_mismatches_p,
    )

    # ── Step 2: build features + splits on the RC5 panel ─────────────────────
    print("\n--- Step 2: Building features + splits on the RC5 panel ---")
    # save_winsor_imputation_configs=False so we don't overwrite the primary's
    # winsorisation thresholds / imputation medians in outputs/models/configs/
    run_feature_pipeline(
        panel_path=rc5_panel_path,
        features_path=rc5_features_path,
        samples_dir=rc5_samples_dir,
        save_winsor_imputation_configs=False,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RC5 DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"  Panel       -> {rc5_panel_path}")
    print(f"  Features    -> {rc5_features_path}")
    print(f"  Splits dir  -> {rc5_samples_dir}")
    print()
    print("Next step:")
    print("  Remove-Item outputs/tables/robustness/rc5_results.csv")
    print("  python run_pipeline.py --stages 9")


if __name__ == "__main__":
    main()
