"""
Driver: the balanced neural network across RC1-RC5 on the reported run.
=======================================================================
Closes the last gap in the robustness evidence: the network is one of the four
co-primary models but the pre-specified battery under ``final_primary`` covered
only LR, RF and XGBoost, the NN-across-checks artifacts in the repository
belonging to superseded sample and label generations.

Hyperparameters are frozen at the primary specification's and the network is
refit per check --- the supplementary convention used throughout Chapter 7, not
the per-check re-tune the other three models receive. See
``src/robustness/rc_nn_final_primary`` for why, and for how that asymmetry is
bounded.

Read-only with respect to every frozen artifact: no saved estimator, study,
configuration or split is written, and the models fitted here are discarded.

Usage
-----
    PYTHONPATH=. python -m scripts.run_rc_nn_final_primary
    PYTHONPATH=. python -m scripts.run_rc_nn_final_primary --checks primary rc4
    PYTHONPATH=. python -m scripts.run_rc_nn_final_primary --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import OUT_TABLES_ROBUSTNESS, V2_PROFILE
from src.robustness.rc_nn_final_primary import CHECKS, run_battery
from src.utils.tables import save_table

ROBUSTNESS_DIR = Path(OUT_TABLES_ROBUSTNESS) / V2_PROFILE["spec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", nargs="+", default=None, choices=CHECKS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    partial = args.checks is not None and set(args.checks) != set(CHECKS)
    out = ROBUSTNESS_DIR / "rc_nn_final_primary_results"

    if partial:
        print(f"[partial run] {args.checks} — no table will be written.")
    elif Path(f"{out}.csv").exists() and not args.force:
        print(f"[skip] {out}.csv exists; pass --force to regenerate.")
        return

    print("=" * 70)
    print("  BALANCED NEURAL NETWORK ACROSS THE ROBUSTNESS CHECKS")
    print("=" * 70)

    res = run_battery(args.checks)
    print()
    print(res.to_string(index=False))

    if partial:
        print("\n[partial run] table not written.")
        return

    save_table(
        res, out,
        caption=(
            "The balanced neural network across the pre-specified robustness "
            "checks, on the reported run. The network is one of the four "
            "co-primary models, but the battery in "
            "Section~\\ref{sec:robustness_results} re-tunes only the benchmark "
            "and the two tree ensembles. Here the network reuses the "
            "hyperparameters selected for the primary specification and is "
            "refit on each check's data --- the supplementary convention of "
            "this chapter, not the per-check re-tune the other three models "
            "receive, a per-check search for this model being a one-to-three "
            "day computation. Each check's label, horizon, predictor set and "
            "sample are constructed with the same calls the other models use, "
            "so the rows are comparable across models within a check; the "
            "prevalence baseline is reported per check because it varies. "
            "RC\\textsubscript{3} replaces oversampling-to-parity with SMOTENC, "
            "constructed exactly as for the other models."),
        label="tab:rc_nn_final_primary")
    print(f"\n  wrote {out}.{{csv,tex}}")


if __name__ == "__main__":
    main()
