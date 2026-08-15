"""
Driver: per-check hyperparameter search for the balanced network (RC1-RC5).
===========================================================================
Removes the last protocol asymmetry in Chapter 7 by giving the network its own
100-trial search in each pre-specified check, as the other three models receive.

⚠ MULTI-DAY. One balanced-network fit takes minutes on this data, so a
100-trial search over five folds is hours to tens of hours per check. The run
is built to be interrupted: each finished check writes its own results CSV and
is skipped afterwards, and each check's Optuna study persists to its own SQLite
file and resumes at its completed-trial count. Re-run the same command to
continue.

Usage
-----
    PYTHONPATH=. python -m scripts.run_rc_nn_retune               # all checks
    PYTHONPATH=. python -m scripts.run_rc_nn_retune --checks rc4  # one check
    PYTHONPATH=. python -m scripts.run_rc_nn_retune --n-trials 1  # timing probe
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import OUT_TABLES_ROBUSTNESS, V2_PROFILE
from src.robustness.rc_nn_final_primary import CHECK_LABEL, get_check_data
from src.robustness.rc_nn_retune import (build_folds, evaluate_check,
                                         prepare_rc3_folds, tune_check,
                                         write_config)
from src.utils.tables import save_table

ROBUSTNESS_DIR = Path(OUT_TABLES_ROBUSTNESS) / V2_PROFILE["spec"]
CHECKPOINT_DIR = ROBUSTNESS_DIR / "_rc_nn_retune_checkpoints"
RETUNE_CHECKS = ["rc1", "rc2", "rc3", "rc4", "rc5"]
FULL_TRIAL_BUDGET = 100


def _checkpoint(check: str) -> Path:
    return CHECKPOINT_DIR / f"{check}.csv"


def run_one(check: str, n_trials: int) -> pd.DataFrame | None:
    ck = _checkpoint(check)
    if ck.exists():
        print(f"\n[{check}] checkpoint exists — skipping.")
        return pd.read_csv(ck)

    print(f"\n{'=' * 70}\n  {check.upper()}: {CHECK_LABEL[check]}\n{'=' * 70}")
    _, _, _, features = get_check_data(check)

    t0 = time.time()
    print("  building fold-safe folds ...")
    folds = build_folds(check, features)
    if check == "rc3":
        print("  pre-scaling and pre-resampling folds (SMOTENC) ...")
        folds = prepare_rc3_folds(folds, features)
    print(f"  {len(folds)} folds ready in {time.time() - t0:.0f}s")

    t1 = time.time()
    params, cv, done = tune_check(check, folds, n_trials=n_trials)
    elapsed = time.time() - t1
    per_trial = elapsed / max(1, done)
    print(f"  search: {elapsed / 60:.1f} min for {done} completed trials "
          f"(~{per_trial:.0f}s/trial)")

    if done < FULL_TRIAL_BUDGET:
        print(f"  [partial: {done}/{FULL_TRIAL_BUDGET} trials] "
              "no config or checkpoint written; re-run to continue.")
        return None

    write_config(check, params, cv, n_trials)
    row = evaluate_check(check, params)
    row["best_cv_pr_auc"] = round(cv, 5)
    df = pd.DataFrame([row])
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ck, index=False)
    print(f"  checkpointed -> {ck}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="+", default=RETUNE_CHECKS,
                    choices=RETUNE_CHECKS)
    ap.add_argument("--n-trials", type=int, default=FULL_TRIAL_BUDGET)
    args = ap.parse_args()

    print("=" * 70)
    print("  PER-CHECK NEURAL-NETWORK RE-TUNE  (multi-day; safe to interrupt)")
    print("=" * 70)

    frames = [f for f in (run_one(c, args.n_trials) for c in args.checks)
              if f is not None]

    if len(frames) < len(RETUNE_CHECKS):
        print(f"\n[{len(frames)}/{len(RETUNE_CHECKS)} checks complete] "
              "consolidated table not written yet; re-run to continue.")
        return

    res = pd.concat(frames, ignore_index=True)
    print()
    print(res.to_string(index=False))
    save_table(
        res, ROBUSTNESS_DIR / "rc_nn_retuned_results",
        caption=(
            "The balanced neural network re-tuned within each pre-specified "
            "robustness check: 100 Optuna trials per check on fold-safe "
            "rolling-origin folds with label-maturity purging, matching the "
            "convention applied to the benchmark and the two tree ensembles. "
            "This removes the protocol asymmetry disclosed in "
            "Section~\\ref{subsec:rc_nn}, where the network reused the primary "
            "specification's hyperparameters. For RC\\textsubscript{1} and "
            "RC\\textsubscript{2} the raw training frame is re-labelled before "
            "the folds are built, so each search optimises against its own "
            "check's outcome; for RC\\textsubscript{3} the in-fold treatment is "
            "SMOTENC, as it is for the other models."),
        label="tab:rc_nn_retuned")
    print("\n  wrote rc_nn_retuned_results.{csv,tex}")


if __name__ == "__main__":
    main()
