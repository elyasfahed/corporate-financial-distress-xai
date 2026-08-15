"""
Read-only completeness and consistency checker for final thesis outputs.

This script never fits a model or writes an output. It checks that the saved
artifacts are present, newer than the frozen sample, and mutually consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_MERGED,
    DATA_SAMPLES,
    OUT_MODELS_SAVED,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    OUT_TABLES_SHAP,
)

OK, BAD, WARN = "  [ OK ]", "  [FAIL]", "  [warn]"
problems: list[str] = []


def _sample_mtime() -> float:
    path = DATA_SAMPLES / "test.parquet"
    return path.stat().st_mtime if path.exists() else 0.0


def _models_mtime() -> float:
    """Newest saved-model timestamp: any table that re-scores the frozen models
    (SHAP, subperiod, out-of-time, significance) must postdate ALL of them."""
    names = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
        "neural_network_balanced.joblib",
    ]
    mtimes = [
        (OUT_MODELS_SAVED / n).stat().st_mtime
        for n in names
        if (OUT_MODELS_SAVED / n).exists()
    ]
    return max(mtimes) if mtimes else 0.0


def check_file(
    path: Path,
    sample_mtime: float,
    optional: bool = False,
    stale_reason: str = "older than frozen test sample",
) -> None:
    if not path.exists():
        if optional:
            print(f"{WARN} {path.name} - not present (optional)")
        else:
            print(f"{BAD} {path.name} - MISSING")
            problems.append(f"missing: {path.name}")
        return
    if path.stat().st_mtime < sample_mtime:
        if optional:
            print(f"{WARN} {path.name} - stale optional output")
        else:
            print(f"{BAD} {path.name} - STALE ({stale_reason})")
            problems.append(f"stale: {path.name}")
    else:
        print(f"{OK} {path.name}")


# Canonical short names so PR-AUC values can be compared across tables that
# use display names ("Logistic Regression (ridge)") vs snake_case keys.
_MODEL_ALIASES = {
    "logistic regression (ridge)": "lr",
    "logistic regression": "lr",
    "logistic_regression": "lr",
    "random forest": "rf",
    "random_forest": "rf",
    "xgboost": "xgb",
    "neural network (mlp)": "nn",
    "neural network": "nn",
    "neural_network": "nn",
    "neural_network_balanced": "nn",
}


def _pr_auc_map(df: pd.DataFrame, model_col: str = "model") -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        key = _MODEL_ALIASES.get(str(row[model_col]).strip().lower())
        if key is not None and "pr_auc" in row.index and pd.notna(row["pr_auc"]):
            out[key] = round(float(row["pr_auc"]), 4)
    return out


def _compare_pr_auc(label: str, ref: dict[str, float], other: dict[str, float],
                    models: tuple[str, ...] = ("lr", "rf", "xgb")) -> None:
    tol = 1.0e-4
    for m in models:
        if m not in other:
            print(f"{BAD} {label}: no {m.upper()} row found")
            problems.append(f"{label}: missing {m} row")
            continue
        if abs(ref[m] - other[m]) > tol:
            print(f"{BAD} {label}: {m.upper()} PR-AUC {other[m]} != headline {ref[m]}")
            problems.append(f"{label}: {m} PR-AUC mismatch")
        else:
            print(f"{OK} {label}: {m.upper()} PR-AUC = {other[m]}")


def main() -> None:
    problems.clear()
    sample_mtime = _sample_mtime()

    print("=" * 68)
    print("FINAL OUTPUT VERIFICATION - READ ONLY")
    print("=" * 68)

    print("\n1) SAMPLE INTEGRITY")
    panel = pd.read_parquet(
        DATA_MERGED / "panel_raw.parquet",
        columns=["fdate_source", "distress"],
    )
    source_share = panel["fdate_source"].value_counts(normalize=True).round(4).to_dict()
    if source_share == {"actual_10k": 1.0}:
        print(f"{OK} panel filing-date source = 100% actual_10k")
    else:
        print(f"{BAD} panel filing-date source = {source_share}")
        problems.append("panel is not 100% actual_10k")

    print("\n2) CORE FROZEN OUTPUTS")
    core = [
        OUT_TABLES_MODEL / "model_performance_test.csv",
        OUT_MODELS_SAVED / "logistic_regression.joblib",
        OUT_MODELS_SAVED / "random_forest.joblib",
        OUT_MODELS_SAVED / "xgboost.joblib",
        OUT_TABLES_SHAP / "shap_ranking_xgboost.csv",
        OUT_TABLES_SHAP / "kendall_tau_xgboost.csv",
        OUT_TABLES_SHAP / "theory_consistency_table.csv",
        OUT_TABLES_ROBUSTNESS / "rc1_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc2_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc3_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc4_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc5_results.csv",
        OUT_TABLES_ROBUSTNESS / "robustness_consolidated.csv",
    ]
    for path in core:
        check_file(path, sample_mtime)

    print("\n3) EXTENSIONS AND POST-FREEZE CO-PRIMARY OUTPUTS")
    extensions = [
        OUT_TABLES_ROBUSTNESS / "rc6_no_rsize_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc7_neural_network_results.csv",
        OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results.csv",
        OUT_TABLES_SHAP / "lime_shap_agreement.csv",
        OUT_TABLES_MODEL / "model_performance_test_4models.csv",
        OUT_TABLES_MODEL / "significance_tests.csv",
        OUT_TABLES_MODEL / "significance_tests_4models.csv",
        OUT_TABLES_MODEL / "subperiod_performance.csv",
        OUT_TABLES_MODEL / "out_of_time_stability.csv",
        OUT_TABLES_MODEL / "calibration_ece_brier_4models_balanced.csv",
    ]
    for path in extensions:
        check_file(path, sample_mtime)

    print("\n3b) RE-INFERENCE OUTPUTS MUST POSTDATE THE SAVED MODELS")
    models_mtime = _models_mtime()
    reinference = [
        OUT_TABLES_SHAP / "shap_ranking_xgboost.csv",
        OUT_TABLES_SHAP / "shap_ranking_logistic_regression.csv",
        OUT_TABLES_SHAP / "shap_ranking_random_forest.csv",
        OUT_TABLES_SHAP / "kendall_tau_xgboost.csv",
        OUT_TABLES_SHAP / "kendall_tau_logistic_regression.csv",
        OUT_TABLES_SHAP / "theory_consistency_table.csv",
        OUT_TABLES_SHAP / "shap_stability_xgboost.csv",
        OUT_TABLES_SHAP / "lime_shap_agreement.csv",
        OUT_TABLES_MODEL / "subperiod_performance.csv",
        OUT_TABLES_MODEL / "out_of_time_stability.csv",
        OUT_TABLES_MODEL / "significance_tests.csv",
        OUT_TABLES_MODEL / "significance_tests_4models.csv",
    ]
    for path in reinference:
        check_file(path, models_mtime,
                   stale_reason="older than the saved models it must describe")

    print("\n4) TEST-SAMPLE FINGERPRINT")
    try:
        primary = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test.csv")
        reference = round(float(primary["prevalence_baseline_pr_auc"].iloc[0]), 4)
        print(f"     primary reference baseline = {reference}")

        checks: list[tuple[str, float]] = []
        consolidated = OUT_TABLES_ROBUSTNESS / "robustness_consolidated.csv"
        if consolidated.exists():
            rc = pd.read_csv(consolidated)
            for check in ("RC3", "RC4"):
                rows = rc.loc[
                    rc["check"] == check,
                    "prevalence_baseline_pr_auc",
                ]
                if len(rows):
                    checks.append((check, round(float(rows.iloc[0]), 4)))

        for label, filename in [
            ("RC6", "rc6_no_rsize_results.csv"),
            ("NN raw sensitivity", "rc7_neural_network_results.csv"),
            ("NN balanced headline", "rc7b_neural_network_balanced_results.csv"),
        ]:
            path = OUT_TABLES_ROBUSTNESS / filename
            if path.exists():
                df = pd.read_csv(path)
                checks.append((
                    label,
                    round(float(df["prevalence_baseline_pr_auc"].iloc[0]), 4),
                ))

        for label, value in checks:
            if value == reference:
                print(f"{OK} {label} baseline = {value}")
            else:
                print(f"{BAD} {label} baseline = {value} (expected {reference})")
                problems.append(f"{label} baseline mismatch")

        headline_path = OUT_TABLES_MODEL / "model_performance_test_4models.csv"
        if headline_path.exists():
            headline = pd.read_csv(headline_path)
            nn_rows = headline[
                headline["model"].astype(str).str.contains(
                    "Neural Network",
                    case=False,
                )
            ]
            if len(headline) == 4 and len(nn_rows) == 1:
                nn_pr = round(float(nn_rows["pr_auc"].iloc[0]), 4)
                if nn_pr == 0.0994:
                    print(f"{OK} headline uses balanced NN PR-AUC = {nn_pr}")
                else:
                    print(f"{BAD} headline NN PR-AUC = {nn_pr} (expected 0.0994)")
                    problems.append("headline does not use balanced NN")
            else:
                print(f"{BAD} headline must contain four rows and one NN row")
                problems.append("invalid headline structure")

        significance_path = OUT_TABLES_MODEL / "significance_tests.csv"
        if significance_path.exists():
            significance = pd.read_csv(significance_path)
            required = {
                "delta_pr_auc_ci_lower",
                "delta_pr_auc_ci_upper",
                "p_value_pr_auc",
            }
            if required.issubset(significance.columns):
                print(f"{OK} corrected PR-AUC difference inference present")
            else:
                print(f"{BAD} significance output uses the obsolete schema")
                problems.append("uncorrected significance output")
    except Exception as exc:
        print(f"{BAD} consistency check failed: {exc}")
        problems.append("consistency check error")

    print("\n5) CROSS-FILE PR-AUC CONSISTENCY (one primary tuple everywhere)")
    try:
        primary = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test.csv")
        ref = _pr_auc_map(primary)
        print(f"     headline tuple: LR={ref.get('lr')} RF={ref.get('rf')} XGB={ref.get('xgb')}")

        headline4_path = OUT_TABLES_MODEL / "model_performance_test_4models.csv"
        if headline4_path.exists():
            _compare_pr_auc("4-model headline",
                            ref, _pr_auc_map(pd.read_csv(headline4_path)))

        sub_path = OUT_TABLES_MODEL / "subperiod_performance.csv"
        if sub_path.exists():
            sub = pd.read_csv(sub_path)
            period_col = "period" if "period" in sub.columns else "window"
            full = sub[sub[period_col].astype(str).str.contains("full", case=False)]
            _compare_pr_auc("subperiod full-test row", ref, _pr_auc_map(full))

        oot_path = OUT_TABLES_MODEL / "out_of_time_stability.csv"
        if oot_path.exists():
            oot = pd.read_csv(oot_path)
            full = oot[oot["window"].astype(str).str.contains("full", case=False)]
            _compare_pr_auc("out-of-time full row", ref, _pr_auc_map(full))

        cons_path = OUT_TABLES_ROBUSTNESS / "robustness_consolidated.csv"
        if cons_path.exists():
            cons = pd.read_csv(cons_path)
            prim_rows = cons[cons["check"].astype(str).str.lower() == "primary"]
            _compare_pr_auc("consolidated primary row", ref, _pr_auc_map(prim_rows))

        # 4-model NN row must equal the balanced RC7b table.
        rc7b_path = OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results.csv"
        if headline4_path.exists() and rc7b_path.exists():
            nn_head = _pr_auc_map(pd.read_csv(headline4_path)).get("nn")
            nn_rc7b = round(float(pd.read_csv(rc7b_path)["pr_auc"].iloc[0]), 4)
            if nn_head is None or abs(nn_head - nn_rc7b) > 1.0e-4:
                print(f"{BAD} headline NN PR-AUC {nn_head} != RC7b {nn_rc7b}")
                problems.append("headline NN vs RC7b mismatch")
            else:
                print(f"{OK} headline NN row matches RC7b = {nn_rc7b}")
    except Exception as exc:
        print(f"{BAD} cross-file PR-AUC check failed: {exc}")
        problems.append("cross-file PR-AUC check error")

    print("\n" + "=" * 68)
    if problems:
        print(f"RESULT: {len(problems)} PROBLEM(S)")
        for problem in problems:
            print(f"  - {problem}")
        print("=" * 68)
        raise SystemExit(1)

    print("RESULT: ALL CHECKS PASSED")
    print("=" * 68)


if __name__ == "__main__":
    from src.utils.superseded import abort_if_superseded
    abort_if_superseded(__file__, "python scripts/verify_final_outputs.py")
    main()
