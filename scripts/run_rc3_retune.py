"""
Driver: RC3 (SMOTE) with an independent hyperparameter search.
==============================================================
Closes the one protocol asymmetry inside the pre-specified robustness set.
RC1, RC2, RC4 and RC5 each re-tune from scratch; RC3 reused the primary
configurations, which confounds the imbalance method with hyperparameter
compatibility (see ``src/robustness/rc3_smote_retune`` for why that matters
here in particular).

The search is resumable: the Optuna studies are persisted under
``outputs/models/configs/final_primary_rc3/``, so an interrupted run continues
at its completed-trial count rather than starting over.

The frozen ``rc3_results`` table is NOT overwritten. This writes
``rc3_retuned_results.{csv,tex}`` alongside it, so the held-fixed and re-tuned
readings can be compared directly and the published check remains reproducible.

Usage
-----
    PYTHONPATH=. python -m scripts.run_rc3_retune              # 100 trials
    PYTHONPATH=. python -m scripts.run_rc3_retune --n-trials 2 # smoke test
    PYTHONPATH=. python -m scripts.run_rc3_retune --force      # overwrite table
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (ACCOUNTING_FEATURES, ALL_FEATURES_V2,
                        DATA_SAMPLES_V2, DISTRESS_HORIZON_DAYS,
                        MARKET_IMPUTE_FEATURES, OUT_MODELS_CONFIGS,
                        OUT_TABLES_ROBUSTNESS, V2_PROFILE)
from src.robustness.rc3_smote import run_rc3
from src.robustness.rc3_smote_retune import (MODEL_NAMES, RC3_SPEC,
                                             prepare_resampled_folds,
                                             tune_rc3_model, write_config)
from src.utils.tables import save_table

ROBUSTNESS_DIR = Path(OUT_TABLES_ROBUSTNESS) / V2_PROFILE["spec"]
CONFIGS_DIR = Path(OUT_MODELS_CONFIGS) / RC3_SPEC

#: Blueprint v4 §9.3 budget. A run below this is a smoke test and writes nothing.
FULL_TRIAL_BUDGET = 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--force", action="store_true",
                    help="Regenerate the table even if it already exists.")
    args = ap.parse_args()

    # A reduced-trial run is a smoke test, not a robustness check. It must not
    # be able to leave a published table behind that looks like one --- the
    # convention the rest of this repo follows for its --quick paths.
    full_budget = args.n_trials >= FULL_TRIAL_BUDGET
    if not full_budget:
        print(f"[smoke test] {args.n_trials} trials < the {FULL_TRIAL_BUDGET}-trial "
              "budget; no table will be written.")

    out = ROBUSTNESS_DIR / "rc3_retuned_results"
    if full_budget and Path(f"{out}.csv").exists() and not args.force:
        print(f"[skip] {out}.csv exists; pass --force to regenerate.")
        return

    features = list(ALL_FEATURES_V2)
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    train_raw = pd.read_parquet(DATA_SAMPLES_V2 / "train_raw.parquet")

    print("=" * 66)
    print("  RC3 RE-TUNE — SMOTENC with an independent search")
    print("=" * 66)
    print(f"  splits: train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"  trials per model: {args.n_trials}")
    print("\n  Building resampled folds (once; hyperparameter-independent) ...")

    folds = prepare_resampled_folds(
        train_raw, features,
        purge_horizon_days=DISTRESS_HORIZON_DAYS,
        sic_col=V2_PROFILE["sic_col"],
        peer_rule=V2_PROFILE["impute_peer_rule"],
        impute_features=[f for f in ACCOUNTING_FEATURES + MARKET_IMPUTE_FEATURES
                         if f in features],
        corrected=True,
    )

    print("\n  Searching ...")
    for name in MODEL_NAMES:
        params, cv = tune_rc3_model(name, folds, n_trials=args.n_trials,
                                    storage_dir=CONFIGS_DIR)
        if not full_budget:
            print(f"    [smoke test] {name}: best CV {cv:.5f}; config not written")
            continue
        path = write_config(name, params, cv, args.n_trials, CONFIGS_DIR)
        print(f"    wrote {path}")

    if not full_budget:
        print("\n[smoke test] stopping before the test evaluation. The search "
              "trials are persisted, so a full run resumes from them.")
        return

    print("\n  Final fit and single test evaluation (delegated to run_rc3) ...")
    res = run_rc3(train, val, test, features=features, corrected=True,
                  params_dir=CONFIGS_DIR)

    save_table(
        res, out,
        caption=(
            "RC\\textsubscript{3} with an independent hyperparameter search. "
            "SMOTE replaces class weighting, as in the held-fixed version of "
            "this check, but the three models are re-tuned from scratch over "
            f"{args.n_trials} Optuna trials rather than inheriting the primary "
            "configurations, matching the per-check convention of "
            "RC\\textsubscript{1}, RC\\textsubscript{2}, RC\\textsubscript{4} "
            "and RC\\textsubscript{5}. SMOTENC treats the three binary "
            "indicators as categorical; continuous features are standardised "
            "on training statistics before resampling; class weighting is "
            "disabled so the imbalance correction is not applied twice. "
            "Within the search, standardisation and resampling are fitted on "
            "each fold's training rows only and the validation rows are never "
            "resampled, so no synthetic minority point is built from an "
            "observation used to score the fold."),
        label="tab:rc3_retuned")
    print()
    print(res.to_string(index=False))
    print(f"\n  wrote {out}.{{csv,tex}}")


if __name__ == "__main__":
    main()
