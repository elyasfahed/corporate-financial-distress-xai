"""
Run all 5 pre-specified robustness checks (RC1–RC5).
======================================================
Each check varies EXACTLY ONE design dimension; all others are held fixed.
Results are presented in a single consolidated table (design §11).

A finding is considered ROBUST if:
  (i)  The best-performing model retains its ranking.
  (ii) The sign and approximate magnitude of the top SHAP features
       remain consistent.

Material changes in any check are discussed SUBSTANTIVELY rather than
dismissed as noise.

Checks (design §11):
  RC1 : Distress definition — bankruptcy codes 572+574+584 vs. 400–499 primary
  RC2 : Prediction horizon — 6 months instead of 12 months
  RC3 : Class imbalance treatment — SMOTE instead of class weights
  RC4 : Accounting-only model — remove all 6 market predictors
  RC5 : Industry coverage — include utilities (SIC 4900–4999)

Resume behaviour
----------------
Each RC saves its result to outputs/tables/robustness/rcN_results.csv
immediately on completion. If the script is interrupted (e.g. shutdown)
and re-run, any RC whose checkpoint file already exists is SKIPPED and
its result is loaded from disk. Delete a checkpoint file to force a re-run
of that specific check.

Design reference: §11
"""

from __future__ import annotations

import shutil

import pandas as pd

from src.config import (
    OUT_TABLES_ROBUSTNESS,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    DATA_SAMPLES,
)
from src.robustness.rc1_bankruptcy_codes import run_rc1
from src.robustness.rc2_horizon_6m import run_rc2
from src.robustness.rc3_smote import run_rc3
from src.robustness.rc4_accounting_only import run_rc4
from src.robustness.rc5_include_utilities import run_rc5
from src.utils.tables import save_table


# Checkpoint filenames — one CSV per RC stored in robustness table directory
_CHECKPOINTS = {
    "RC1": "rc1_results.csv",
    "RC2": "rc2_results.csv",
    "RC3": "rc3_results.csv",
    "RC4": "rc4_results.csv",
    "RC5": "rc5_results.csv",
}


def _save_checkpoint(check_name: str, result_df: pd.DataFrame) -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    path = OUT_TABLES_ROBUSTNESS / _CHECKPOINTS[check_name]
    result_df.to_csv(path, index=False)
    print(f"  Checkpoint saved -> {path.name}")


def _load_checkpoint(check_name: str) -> pd.DataFrame | None:
    path = OUT_TABLES_ROBUSTNESS / _CHECKPOINTS[check_name]
    if not path.exists():
        return None
    # Sample-awareness guard: never silently reuse a checkpoint that predates
    # the current analysis sample. Without this, a re-run after a data-layer
    # change (e.g. the filing-date fix) would glue stale RC results onto a
    # fresh primary, producing an inconsistent consolidated table.
    sample_file = DATA_SAMPLES / "test.parquet"
    if sample_file.exists() and path.stat().st_mtime < sample_file.stat().st_mtime:
        print(f"  Checkpoint {path.name} predates the current sample — "
              f"ignoring and re-running {check_name}.")
        return None
    df = pd.read_csv(path)
    print(f"  Checkpoint found — skipping {check_name}, loading from {path.name}")
    return df


def _backup_primary_configs() -> None:
    """
    Copy primary model config YAMLs to a backup folder before robustness
    checks overwrite them. Restored at the end of run_all_robustness.
    """
    backup_dir = OUT_MODELS_CONFIGS / "primary_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logistic_regression", "random_forest", "xgboost"):
        src = OUT_MODELS_CONFIGS / f"{name}_config.yaml"
        dst = backup_dir / f"{name}_config.yaml"
        if src.exists():
            shutil.copy2(src, dst)
    print(f"  Primary model configs backed up -> {backup_dir}")


def _restore_primary_configs() -> None:
    """
    Restore primary model config YAMLs from backup after robustness checks.

    This is the safety net against the configuration-overwrite defect recorded
    in the 2026-07-11 provenance audit. A missing backup directory means the
    net is not in place, so it raises rather than warning-and-continuing: silently
    skipping the restore is precisely how the primary configs were clobbered
    the first time. ``_backup_primary_configs`` always creates the directory,
    so reaching this branch means it never ran or the directory was moved
    (the 2026-07-24 quarantine relocated it to ``outputs/_superseded/``).
    """
    backup_dir = OUT_MODELS_CONFIGS / "primary_backup"
    if not backup_dir.exists():
        raise FileNotFoundError(
            f"No primary config backup at {backup_dir} — refusing to finish "
            "silently. The robustness battery may have overwritten the primary "
            "config YAMLs with no way to restore them. Restore the backup (or "
            "re-run with per-spec artifact dirs, which avoid the shared YAML "
            "entirely) before trusting any primary artifact."
        )
    for name in ("logistic_regression", "random_forest", "xgboost"):
        src = backup_dir / f"{name}_config.yaml"
        dst = OUT_MODELS_CONFIGS / f"{name}_config.yaml"
        if src.exists():
            shutil.copy2(src, dst)
    print(f"  Primary model configs restored from backup.")


def run_all_robustness(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    primary_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all 5 robustness checks and consolidate results.

    Completed checks are checkpointed to disk immediately. If this function
    is interrupted and re-run, already-completed checks are skipped.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Primary specification splits (used as baseline for each check).
    primary_results : pd.DataFrame
        Performance table from the primary specification (for comparison).

    Returns
    -------
    pd.DataFrame
        Consolidated robustness table with one section per check.
    """
    print("\n" + "="*60)
    print("  ROBUSTNESS CHECKS")
    print("="*60)
    print("  Checkpoints saved after each RC — safe to interrupt and resume.")

    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)

    # ── Backup primary model configs before any RC overwrites them ───────────
    _backup_primary_configs()

    check_results = {}
    check_results["primary"] = primary_results

    # ── RC1 ──────────────────────────────────────────────────────────────────
    cached = _load_checkpoint("RC1")
    if cached is not None:
        check_results["RC1"] = cached
    else:
        print("\n--- RC1: Bankruptcy codes only (572+574+584) ---")
        check_results["RC1"] = run_rc1(train, val, test)
        _save_checkpoint("RC1", check_results["RC1"])

    # ── RC2 ──────────────────────────────────────────────────────────────────
    cached = _load_checkpoint("RC2")
    if cached is not None:
        check_results["RC2"] = cached
    else:
        print("\n--- RC2: 6-month prediction horizon ---")
        check_results["RC2"] = run_rc2(train, val, test)
        _save_checkpoint("RC2", check_results["RC2"])

    # ── RC3 ──────────────────────────────────────────────────────────────────
    cached = _load_checkpoint("RC3")
    if cached is not None:
        check_results["RC3"] = cached
    else:
        print("\n--- RC3: SMOTE class imbalance treatment ---")
        check_results["RC3"] = run_rc3(train, val, test)
        _save_checkpoint("RC3", check_results["RC3"])

    # ── RC4 ──────────────────────────────────────────────────────────────────
    cached = _load_checkpoint("RC4")
    if cached is not None:
        check_results["RC4"] = cached
    else:
        print("\n--- RC4: Accounting-only features (no market variables) ---")
        check_results["RC4"] = run_rc4(train, val, test)
        _save_checkpoint("RC4", check_results["RC4"])

    # ── RC5 ──────────────────────────────────────────────────────────────────
    cached = _load_checkpoint("RC5")
    if cached is not None:
        check_results["RC5"] = cached
    else:
        print("\n--- RC5: Include utilities (SIC 4900–4999) ---")
        check_results["RC5"] = run_rc5(train, val, test)
        _save_checkpoint("RC5", check_results["RC5"])

    # ── Restore primary model configs ────────────────────────────────────────
    _restore_primary_configs()

    # ── Consolidate ──────────────────────────────────────────────────────────
    rows = []
    for check_name, result_df in check_results.items():
        result_df = result_df.copy()
        result_df.insert(0, "check", check_name)
        rows.append(result_df)

    consolidated = pd.concat(rows, ignore_index=True)

    save_table(
        consolidated,
        OUT_TABLES_ROBUSTNESS / "robustness_consolidated",
        caption=(
            "Consolidated robustness results (design §11). "
            "Each row reports out-of-sample test-set metrics for one model "
            "under one robustness condition: \\textbf{primary} = frozen "
            "design; \\textbf{RC1} = bankruptcy-only DLSTCD codes "
            "(572/574/584); \\textbf{RC2} = 6-month prediction horizon; "
            "\\textbf{RC3} = SMOTE imbalance treatment; \\textbf{RC4} = "
            "accounting predictors only (market features removed); "
            "\\textbf{RC5} = utilities (SIC 4900--4999) included in the "
            "sample. PR-AUC confidence intervals are 95\\% bootstrap with "
            "block resampling by firm (1{,}000 resamples). A finding is "
            "robust if the best model retains its ranking and the sign and "
            "approximate magnitude of effects are preserved across checks."
        ),
        label="tab:robustness_consolidated",
    )
    print(f"\nConsolidated robustness table saved -> {OUT_TABLES_ROBUSTNESS}")

    return consolidated


if __name__ == "__main__":
    from src.config import DATA_SAMPLES
    train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    primary_results = pd.read_csv(
        __import__("src.config", fromlist=["OUT_TABLES_MODEL"]).OUT_TABLES_MODEL
        / "model_performance_test.csv"
    )
    run_all_robustness(train, val, test, primary_results)
