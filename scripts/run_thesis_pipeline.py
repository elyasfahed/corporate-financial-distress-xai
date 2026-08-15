"""Clear, checkpointed entry point for the thesis pipeline.

The public outputs use academic names (primary specification, bankruptcy
proxy, and sensitivity checks).  Historical/internal directory namespaces
are deliberately left unchanged so prior artifacts are not overwritten by
accident.

Run one step at a time from the PyCharm terminal.  There is intentionally no
``--step all`` because the label audit must be reviewed before rebuilding and
the held-out test must not be evaluated during iterative development.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_ROOT,
    DATA_SAMPLES_V2,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    V2_PROFILE,
)
from src.data.distress_definition import (
    NARROW_FINANCIAL_REASONS,
    PERFORMANCE_LEGACY_CODES,
    VOLUNTARY_ADMIN_LEGACY_CODES,
    VOLUNTARY_ADMIN_REASONS,
)


APPROVAL_PATH = (OUT_MODELS_CONFIGS / V2_PROFILE["spec"] /
                 "outcome_definition_approval.yaml")
DELIST_PATH = DATA_ROOT / "raw_final_primary" / "crsp_delisting_raw.parquet"


def _build_approval_payload() -> dict:
    """
    Assemble the outcome-definition approval record.

    Every field that scripts/verify_final_outputs.py checks is derived from
    the same constants that module imports, so the written record and the
    check cannot drift apart. The four fields that were previously absent or
    misnamed -- primary_event_column, performance_family_legacy_codes,
    voluntary_admin_excluded_legacy_codes, and the EXCLUDING/MVOT/CORQ wording
    in primary_rule_short -- are what made a regenerated file fail four of the
    six checks.
    """
    return {
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_by": "thesis_author",
        "primary_definition": "CRSP performance delistings",
        "primary_event_column": V2_PROFILE["distress_event_col"],
        "primary_cleaning": (
            "Cleaned primary label = the performance-delisting family below "
            "MINUS the clearly voluntary / administrative GDR exits (reason "
            "CORQ, reconstructed legacy code 570; reason MVOT, code 520). The "
            "broad performance family that retains these is kept only as a "
            "documented sensitivity (is_distress_performance). MTMK (Market "
            "Makers, code 550) is a listing-standard failure and is retained "
            "in the primary label, confirmed against CRSP's CIZ flag "
            "dictionary (MetaFlagInfo)."
        ),
        "voluntary_admin_excluded_reasons": sorted(VOLUNTARY_ADMIN_REASONS),
        "voluntary_admin_excluded_legacy_codes": sorted(
            VOLUNTARY_ADMIN_LEGACY_CODES),
        "performance_family_legacy_codes": sorted(PERFORMANCE_LEGACY_CODES),
        "primary_rule_short": (
            "performance delistings (reconstructed legacy 400-499, 500, "
            "520-584) EXCLUDING the voluntary/administrative reasons "
            "MVOT (520) and CORQ (570)"
        ),
        "narrow_sensitivity_gdr_reasons": sorted(NARROW_FINANCIAL_REASONS),
        "bankruptcy_proxy_reason": "BKPY",
        "broad_sensitivity_actions": ["GLI", "GDR"],
        "note": (
            "Author-approved final-primary rebuild. The primary outcome is "
            "the cleaned performance-delisting label (performance family "
            "minus the voluntary/administrative reasons CORQ and MVOT; see "
            "primary_cleaning and primary_event_column above), following the "
            "literature-aligned CRSP performance-delisting convention. The "
            "broad performance family and the July 2026 five-reason label "
            "are retained only as sensitivities."
        ),
    }


def _record_label_approval(reapprove: bool = False) -> None:
    """
    Write the outcome-definition approval record, refusing to clobber one.

    The approval file is a SHA-256-hashed ``input_files`` entry in the
    final_primary run manifest as well as the subject of six content checks
    in scripts/verify_final_outputs.py. Because the payload stamps
    ``approved_at_utc``, ANY rewrite changes its digest and breaks the
    manifest even when the content is perfectly correct. An existing record
    is therefore never overwritten silently -- which matters because this
    function is the first thing ``--step data`` does, so the old unconditional
    write destroyed a verified artifact before the rebuild had begun.

    Pass reapprove=True (CLI: --reapprove-label-definition) only when the
    outcome definition itself has genuinely changed and the manifest is
    expected to be refreshed afterwards.
    """
    import yaml
    if APPROVAL_PATH.exists() and not reapprove:
        existing = yaml.safe_load(
            APPROVAL_PATH.read_text(encoding="utf-8")) or {}
        stamp = existing.get("approved_at_utc", "timestamp unknown")
        print(
            f"Outcome definition already approved ({stamp}); leaving "
            f"{APPROVAL_PATH} untouched.\n"
            "  Rewriting it would change its SHA-256 and invalidate the run "
            "manifest. Pass --reapprove-label-definition only if the outcome "
            "definition itself has changed."
        )
        return
    APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with APPROVAL_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_build_approval_payload(), f, sort_keys=False)
    print(f"Recorded author approval -> {APPROVAL_PATH}")


def _require_label_approval() -> None:
    if not APPROVAL_PATH.exists():
        raise SystemExit(
            "STOP: the outcome definition has not been frozen. First run "
            "`python -m scripts.audit_distress_definition`, review the CSV, "
            "then run the data step with `--approve-label-definition`."
        )


def step_label_audit() -> None:
    from scripts.audit_distress_definition import main
    main()


def step_data(approve: bool, reapprove: bool = False) -> None:
    if not approve:
        raise SystemExit(
            "STOP: review the label-audit CSV first. If the proposed reason "
            "list is accepted, repeat this command with "
            "`--approve-label-definition`."
        )
    _record_label_approval(reapprove=reapprove)

    from scripts.run_v2_rebuild import stage_a, stage_a2, stage_b, stage_c
    stage_a(force=True)   # refreshes the delisting extract with raw CIZ flags
    stage_a2()            # explicit universe-quality gate
    stage_b(force=True)   # rebuild label and merged panel
    stage_c(force=True)   # rebuild features and chronological splits


def _assert_current_data() -> None:
    _require_label_approval()
    if not DELIST_PATH.exists():
        raise SystemExit("STOP: delisting extract missing; run --step data.")
    required = {
        "DelActionType", "DelReasonType", "DelStatusType",
        "DelPaymentType", V2_PROFILE["distress_event_col"],
        V2_PROFILE["bankruptcy_event_col"],
        V2_PROFILE["narrow_distress_event_col"],
    }
    cols = set(pd.read_parquet(DELIST_PATH).columns)
    missing = required - cols
    if missing:
        raise SystemExit(
            "STOP: delisting extract is stale and lacks "
            f"{sorted(missing)}. Run --step data."
        )
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        if not (DATA_SAMPLES_V2 / name).exists():
            raise SystemExit(f"STOP: {name} missing; run --step data.")


def step_models(n_trials: int) -> None:
    _assert_current_data()
    print("This step tunes on training data, selects thresholds on validation, "
          "and then evaluates the held-out test once.")
    from scripts.run_v2_rebuild import stage_d
    stage_d(force=True, n_trials=n_trials)


def step_neural_network(n_trials: int) -> None:
    _assert_current_data()
    from scripts.run_v2_nn import main
    main(n_trials=n_trials, force=True)


def step_robustness() -> None:
    _assert_current_data()
    from scripts.run_v2_robustness import (
        stage_consolidated,
        stage_h2,
        stage_narrow_label,
        stage_rc1,
        stage_rc2,
        stage_rc3,
        stage_rc4,
        stage_rc5,
    )
    stage_rc1(force=True)
    stage_rc2(force=True)
    stage_rc3(force=True)
    stage_rc4(force=True)
    stage_rc5(force=True)
    stage_narrow_label(force=True)
    stage_h2(force=True)
    stage_consolidated(force=True)


def step_figures() -> None:
    _assert_current_data()
    if not (OUT_TABLES_MODEL / V2_PROFILE["spec"] /
            "model_performance_test_3models.csv").exists():
        raise SystemExit("STOP: model results missing; run --step models first.")
    from scripts.run_v2_secondary import main as secondary
    from scripts.run_v2_xai import main as xai
    from scripts.run_v2_parity import main as parity
    secondary()
    xai()
    parity()


def step_verify() -> None:
    _assert_current_data()
    from scripts.verify_final_outputs import main
    main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "label-audit", "data", "models", "neural-network",
            "robustness", "figures", "verify",
        ],
    )
    parser.add_argument("--approve-label-definition", action="store_true")
    parser.add_argument(
        "--reapprove-label-definition",
        action="store_true",
        help=("Overwrite an existing outcome-definition approval record. "
              "This changes the file's SHA-256 and therefore invalidates the "
              "run manifest, so use it only when the outcome definition "
              "itself has changed and the manifest will be refreshed."),
    )
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    actions = {
        "label-audit": lambda: step_label_audit(),
        "data": lambda: step_data(args.approve_label_definition,
                                  args.reapprove_label_definition),
        "models": lambda: step_models(args.n_trials),
        "neural-network": lambda: step_neural_network(args.n_trials),
        "robustness": step_robustness,
        "figures": step_figures,
        "verify": step_verify,
    }
    actions[args.step]()


if __name__ == "__main__":
    main()
