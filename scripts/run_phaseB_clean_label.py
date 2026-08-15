"""
Phase B — cleaned primary label: drop clearly-voluntary delistings.
====================================================================
Removes the unambiguously voluntary / administrative CRSP CIZ delisting reasons
from the PRIMARY distress label:

    CORQ (570)  company request   — voluntary going-private / M&A-adjacent
    MVOT (520)  moved to OTC       — venue change (not distress)

NOTE (2026-07-24): MTMK (550, "Market Makers") was initially read as a venue
change and dropped here, but pinning the mnemonics against CRSP's official CIZ
flag dictionary (MetaFlagInfo) showed it is a listing-standard failure (a
failure to maintain the required number of market makers), of the same kind as
SHLD ("Shareholders"), which the primary label retains. MTMK is therefore
RETAINED in the primary label. The dropped set is now exactly
``VOLUNTARY_ADMIN_LEGACY_CODES`` = {520, 570} (below); this docstring and the
code stay in sync via that constant.

The broad performance label (which retains these) is kept as a documented
sensitivity. This is a CONTAINED refit (author-approved): it reuses the frozen
Optuna hyperparameters from the existing config YAMLs (``reuse_configs=True``)
— NO re-tuning.

Because the split features are label-independent, the split parquets are
relabelled IN PLACE (features byte-identical; only ``distress`` flips 1->0 for
the voluntary firm-years, identified by the window-triggering delisting's
reconstructed legacy code in VOLUNTARY_ADMIN_LEGACY_CODES). The wide-label
artifacts are backed up first (once) so the broad-performance run is retained
and the operation is reversible.

This script covers LR / RF / XGB + Platt + the 3-model test table. Run the
balanced NN refit (``scripts.run_v2_nn --force``) and the A-generators
(secondary / parity / xai + classical_benchmarks) afterwards to regenerate the
4-model table, significance, SHAP/LIME, subperiod and descriptives on the
cleaned label.

Run:
    PYTHONUTF8=1 ./.venv311/Scripts/python.exe -m scripts.run_phaseB_clean_label
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DATA_SAMPLES_V2, OUT_TABLES_MODEL, V2_PROFILE
from src.data.distress_definition import VOLUNTARY_ADMIN_LEGACY_CODES
from src.models.train import (
    apply_lr_platt_calibration,
    evaluate_on_test,
    resolve_artifact_dirs,
    train_all_models,
)

SPEC = V2_PROFILE["spec"]                    # "final_primary"
FEATURES = list(V2_PROFILE["feature_set"])   # 18 predictors incl. MB_MISSING
VOL_CODES = sorted(VOLUNTARY_ADMIN_LEGACY_CODES)   # [520, 570] (MTMK 550 retained)
SPLITS = ["train", "val", "test", "train_raw", "val_raw", "test_raw"]
BACKUP_ROOT = ROOT / "outputs" / "_wide_label_sensitivity"


def _backup_wide_label() -> None:
    """Back up the wide-label splits + models + configs ONCE (skip if present).

    Uses a fixed directory and skips if it already exists, so re-running the
    script never overwrites the true wide-label backup with cleaned data.
    """
    if BACKUP_ROOT.exists():
        print(f"  [skip] wide-label backup already exists -> {BACKUP_ROOT}")
        return
    (BACKUP_ROOT / "samples").mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        src = DATA_SAMPLES_V2 / f"{s}.parquet"
        if src.exists():
            shutil.copy2(src, BACKUP_ROOT / "samples" / f"{s}.parquet")
    saved_dir, configs_dir = resolve_artifact_dirs(SPEC)
    shutil.copytree(saved_dir, BACKUP_ROOT / "models_saved", dirs_exist_ok=True)
    shutil.copytree(configs_dir, BACKUP_ROOT / "models_configs", dirs_exist_ok=True)
    for rel in ("tables/model_results", "tables/robustness", "tables/shap",
                "tables/descriptive"):
        src = ROOT / "outputs" / rel / SPEC
        if src.exists():
            shutil.copytree(src, BACKUP_ROOT / rel / SPEC, dirs_exist_ok=True)
    (BACKUP_ROOT / "README.txt").write_text(
        "Wide (broad-performance) label artifacts backed up before the Phase B "
        "cleaned-label refit (dropped voluntary reasons CORQ/MVOT). These "
        "are the documented broad-performance sensitivity run.\n",
        encoding="utf-8",
    )
    print(f"  Backed up wide-label artifacts -> {BACKUP_ROOT}")


def _relabel_splits() -> None:
    """Flip distress 1->0 where the window-triggering reason is voluntary."""
    total = 0
    for s in SPLITS:
        path = DATA_SAMPLES_V2 / f"{s}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        code = pd.to_numeric(df["delist_code"], errors="coerce")
        vol = (df["distress"] == 1) & code.isin(VOL_CODES)
        n_before = int((df["distress"] == 1).sum())
        df.loc[vol, "distress"] = 0
        n_after = int((df["distress"] == 1).sum())
        if int(vol.sum()):
            df.to_parquet(path, index=False)
        total += int(vol.sum())
        print(f"  {s:10s}: events {n_before} -> {n_after}  "
              f"(dropped {int(vol.sum())} voluntary)")
    print(f"  Total voluntary firm-years relabelled to non-distress: {total}")


def main() -> None:
    if ".venv" not in sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv ({sys.prefix})")
    print("=" * 68)
    print("  PHASE B — cleaned primary label (drop voluntary exits)")
    print("=" * 68)
    print(f"  Voluntary legacy codes dropped: {VOL_CODES} (MVOT/CORQ)")

    _backup_wide_label()
    print("\n[1/3] Relabel split parquets in place")
    _relabel_splits()

    print("\n[2/3] Contained refit (reuse frozen hyperparameters, no re-tune)")
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    trained = train_all_models(
        train, val, features=FEATURES, spec=SPEC,
        reuse_configs=True, skip_if_saved=False,
        samples_dir=DATA_SAMPLES_V2,
    )
    trained = apply_lr_platt_calibration(trained, val, features=FEATURES,
                                         spec=SPEC)

    print("\n[3/3] Test-set evaluation (cleaned label)")
    evaluate_on_test(
        trained, test, features=FEATURES,
        output_stem=(OUT_TABLES_MODEL / SPEC / "model_performance_test_3models"),
        run_label=f"{SPEC}_clean_label",
    )
    print("\nPhase B LR/RF/XGB refit complete.")
    print("Next: python -m scripts.run_v2_nn --force ; then the A-generators.")


if __name__ == "__main__":
    main()
