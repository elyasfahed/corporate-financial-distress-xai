"""Independent, read-only verification of the final-primary thesis run.

The verifier never fits a model.  It recomputes headline metrics from the
saved estimators, checks label timing and provenance hashes, confirms exactly
100 completed Optuna trials for every primary model, and inventories the
publication outputs.  Its only write is the JSON verification report.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    DATA_MERGED_V2,
    DATA_RAW_V2,
    DATA_SAMPLES_V2,
    OUT_MODELS_CONFIGS,
    OUT_MODELS_SAVED,
    ROOT,
    V2_PROFILE,
)
from src.data.distress_definition import (
    PERFORMANCE_LEGACY_CODES,
    VOLUNTARY_ADMIN_LEGACY_CODES,
)
from src.utils.run_manifest import _code_fingerprint


SPEC = V2_PROFILE["spec"]
FEATURES = list(V2_PROFILE["feature_set"])
MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "neural_network_balanced",
)
MODEL_LABELS = {
    "logistic_regression": {"logistic_regression", "logistic regression (ridge)",
                            "logistic regression (platt)"},
    "random_forest": {"random_forest", "random forest"},
    "xgboost": {"xgboost"},
    # "neural network" is the current printed label; the parenthesised form is
    # retained so tables written before the qualifier was dropped still resolve.
    "neural_network_balanced": {"neural_network_balanced",
                                "neural network (balanced)",
                                "neural network"},
}

CONFIG_DIR = OUT_MODELS_CONFIGS / SPEC
MODEL_DIR = OUT_MODELS_SAVED / SPEC
TABLE_ROOT = ROOT / "outputs" / "tables"
FIGURE_ROOT = ROOT / "outputs" / "figures"
MODEL_TABLE_DIR = TABLE_ROOT / "model_results" / SPEC
ROBUSTNESS_DIR = TABLE_ROOT / "robustness" / SPEC
SHAP_DIR = TABLE_ROOT / "shap" / SPEC
DESCRIPTIVE_DIR = TABLE_ROOT / "descriptive" / SPEC
REPORT = ROOT / "outputs" / "verification" / f"{SPEC}_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_code_fingerprint() -> str:
    """Use the manifest writer's canonical producing-code scope.

    Keeping a second implementation here previously omitted requirements and
    an internal provenance file even though the manifest writer included it.
    That made a freshly written, valid manifest impossible to verify.
    """
    return str(_code_fingerprint()["sha256"])


def canonical_model(value: object) -> str | None:
    text = str(value).strip().lower()
    return next((key for key, aliases in MODEL_LABELS.items() if text in aliases), None)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.artifacts: dict[str, dict] = {}

    def check(self, condition: bool, name: str, detail: str = "") -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        print(f"  [{'OK' if passed else 'FAIL':4}] {name}"
              + (f" — {detail}" if detail else ""))

    @staticmethod
    def _label(path: Path) -> str:
        """
        Repo-relative label for an artifact, tolerant of an external data root.

        src/config.py deliberately supports a DATA_ROOT outside the repository
        (THESIS_DATA_ROOT or .env DATA_ROOT — the documented Seafile/Drive
        layout), in which case Path.relative_to(ROOT) raises. Fall back to the
        absolute path so the artifact is still identified and hashed rather than
        crashing the run.
        """
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def file(self, path: Path, name: str, min_bytes: int = 1) -> bool:
        good = path.exists() and path.is_file() and path.stat().st_size >= min_bytes
        label = self._label(path)
        self.check(good, name, label)
        if good:
            self.artifacts[label] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
        return good

    @property
    def failures(self) -> list[dict]:
        return [row for row in self.checks if not row["passed"]]


def verify_data(audit: Audit) -> pd.DataFrame | None:
    expected_years = {
        "train": (1990, 2008),
        "val": (2010, 2013),
        "test": (2015, 2023),
    }
    frames: dict[str, pd.DataFrame] = {}
    for split, bounds in expected_years.items():
        path = DATA_SAMPLES_V2 / f"{split}.parquet"
        if not audit.file(path, f"{split} model sample exists"):
            continue
        frame = pd.read_parquet(path)
        frames[split] = frame
        audit.check(len(frame) > 0, f"{split} is non-empty", f"n={len(frame):,}")
        audit.check(int(frame["distress"].sum()) > 0, f"{split} has events",
                    f"events={int(frame['distress'].sum()):,}")
        audit.check(set(frame["distress"].dropna().unique()) <= {0, 1},
                    f"{split} outcome is binary")
        years = (int(frame["fyear"].min()), int(frame["fyear"].max()))
        audit.check(years == bounds, f"{split} year bounds", str(years))
        audit.check(not frame.duplicated(["gvkey", "fyear"]).any(),
                    f"{split} firm-years are unique")
        missing = [feature for feature in FEATURES if feature not in frame]
        audit.check(not missing, f"{split} has all {len(FEATURES)} predictors",
                    ", ".join(missing))
        if not missing:
            audit.check(np.isfinite(frame[FEATURES].to_numpy(float)).all(),
                        f"{split} predictors are finite")

        raw_path = DATA_SAMPLES_V2 / f"{split}_raw.parquet"
        if audit.file(raw_path, f"{split} raw checkpoint exists"):
            raw = pd.read_parquet(raw_path)
            positive = raw[raw["distress"].eq(1)].copy()
            start = pd.to_datetime(positive["fdate"], errors="coerce")
            event = pd.to_datetime(positive["delist_date"], errors="coerce")
            timing = event.notna() & event.ge(start) & event.le(start + pd.Timedelta(days=365))
            audit.check(bool(timing.all()), f"{split} positive labels obey [F,F+365]",
                        f"checked={len(positive):,}")

    if set(frames) == set(expected_years):
        ids = {
            name: set(zip(frame["gvkey"].astype(str), frame["fyear"].astype(int)))
            for name, frame in frames.items()
        }
        overlap = ((ids["train"] & ids["val"]) | (ids["train"] & ids["test"])
                   | (ids["val"] & ids["test"]))
        audit.check(not overlap, "chronological samples have no firm-year overlap")
    return frames.get("test")


def verify_outcome_and_cusip(audit: Audit) -> None:
    approval = CONFIG_DIR / "outcome_definition_approval.yaml"
    if audit.file(approval, "author outcome-definition approval exists"):
        payload = yaml.safe_load(approval.read_text(encoding="utf-8"))
        audit.check(payload.get("primary_definition") == "CRSP performance delistings",
                    "primary outcome name is exact")
        # The primary label is the CLEANED performance-core outcome: the
        # performance family MINUS the voluntary/administrative exit codes
        # {520 MVOT, 570 CORQ}. The approval manifest must record that family,
        # the excluded codes, and the event column that carries the label.
        audit.check(payload.get("primary_event_column") == "is_distress_performance_core",
                    "primary event column is the cleaned performance-core label")
        audit.check(set(payload.get("performance_family_legacy_codes", [])) ==
                    set(PERFORMANCE_LEGACY_CODES),
                    "performance-delisting legacy-code family is exact")
        audit.check(set(payload.get("voluntary_admin_excluded_legacy_codes", [])) ==
                    set(VOLUNTARY_ADMIN_LEGACY_CODES),
                    "voluntary/administrative excluded codes are exact ({520, 570})")
        audit.check(
            "EXCLUDING" in str(payload.get("primary_rule_short", ""))
            and "MVOT" in str(payload.get("primary_rule_short", ""))
            and "CORQ" in str(payload.get("primary_rule_short", "")),
            "short primary rule states the voluntary-exit exclusion")
        audit.check(payload.get("bankruptcy_proxy_reason") == "BKPY",
                    "RC1 is identified as a bankruptcy-reason proxy")

    # Independent reconstruction: the delisting extract's cleaned-core flag must
    # equal "reconstructed legacy code in (performance family minus {520, 570})",
    # and no positive event may carry an excluded voluntary/administrative code.
    delist_path = DATA_RAW_V2 / "crsp_delisting_raw.parquet"
    if audit.file(delist_path, "delisting extract exists"):
        dl = pd.read_parquet(delist_path)
        needed = {"is_distress_performance_core", "dlstcd_reconstructed"}
        audit.check(needed <= set(dl.columns),
                    "delisting extract carries the core label + reconstructed code")
        if needed <= set(dl.columns):
            core = pd.to_numeric(dl["is_distress_performance_core"],
                                 errors="coerce").fillna(0).astype(int)
            code = pd.to_numeric(dl["dlstcd_reconstructed"], errors="coerce")
            core_set = set(PERFORMANCE_LEGACY_CODES) - set(VOLUNTARY_ADMIN_LEGACY_CODES)
            by_codeset = code.isin(core_set).astype(int)
            audit.check(bool((core.values == by_codeset.values).all()),
                        "core label == performance family minus voluntary codes")
            audit.check(int(code[core == 1].isin(VOLUNTARY_ADMIN_LEGACY_CODES).sum()) == 0,
                        "no core-positive event carries an excluded voluntary code")

    panel_path = DATA_MERGED_V2 / "panel_raw.parquet"
    if audit.file(panel_path, "merged audit panel exists"):
        panel = pd.read_parquet(panel_path)
        needed = {"cusip_check_available", "cusip_match", "cusip_mismatch"}
        audit.check(needed <= set(panel.columns), "CUSIP audit flags are present")
        if needed <= set(panel.columns):
            available = int(panel["cusip_check_available"].sum())
            matched = int(panel["cusip_match"].sum())
            mismatched = int(panel["cusip_mismatch"].sum())
            audit.check(available > 0, "historical CUSIP validation has coverage",
                        f"available={available:,}")
            audit.check(matched + mismatched == available,
                        "CUSIP match/mismatch accounting is complete",
                        f"matched={matched:,}, mismatched={mismatched:,}")
            match_rate = matched / available
            audit.check(match_rate >= 0.90,
                        "point-in-time CUSIP validation rate is at least 90%",
                        f"match_rate={match_rate:.2%}")


def completed_trials(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        # Optuna stores COMPLETE as enum value 1 in current storage; joining
        # the state name is not portable across schema versions, so use the
        # public trial-state encoding recorded in this locked environment.
        rows = connection.execute(
            "SELECT state, COUNT(*) FROM trials GROUP BY state"
        ).fetchall()
    counts = {str(state).upper(): int(n) for state, n in rows}
    return counts.get("COMPLETE", counts.get("1", 0))


def verify_primary_models(audit: Audit, test: pd.DataFrame | None) -> None:
    manifest_path = CONFIG_DIR / "run_manifest.json"
    manifest = None
    if audit.file(manifest_path, "primary run manifest exists"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit.check(manifest.get("spec") == SPEC, "manifest specification matches")
        audit.check(manifest.get("git", {}).get("producing_code_dirty") is False,
                    "producing src/scripts tree was clean at provenance capture")
        recorded_fp = manifest.get("code_fingerprint", {}).get("sha256")
        audit.check(recorded_fp == current_code_fingerprint(),
                    "recorded code fingerprint matches current code")

        extra = manifest.get("extra", {})
        if extra.get("provenance_refreshed_posthoc") is True:
            backup_name = str(extra.get("preserved_original_manifest", ""))
            backup_path = CONFIG_DIR / backup_name
            safe_name = bool(backup_name) and Path(backup_name).name == backup_name
            audit.check(safe_name and backup_path.is_file(),
                        "pre-refresh manifest is preserved", backup_name)
            expected_original = extra.get("original_manifest_sha256")
            audit.check(
                bool(safe_name and backup_path.is_file()
                     and expected_original
                     and sha256(backup_path) == expected_original),
                "pre-refresh manifest hash matches refresh disclosure",
            )
            audit.check(bool(extra.get("provenance_refresh_reason")),
                        "post-hoc provenance refresh reason is disclosed")

    for model_name in MODELS:
        model_path = MODEL_DIR / f"{model_name}.joblib"
        config_path = CONFIG_DIR / f"{model_name}_config.yaml"
        study_path = CONFIG_DIR / f"optuna_{model_name}.db"
        audit.file(model_path, f"saved model: {model_name}", 100)
        if audit.file(config_path, f"model config: {model_name}"):
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            audit.check(int(config.get("optuna_trials_target", -1)) == 100,
                        f"{model_name} Optuna target is 100")
            audit.check(int(config.get("optuna_trials_complete", -1)) == 100,
                        f"{model_name} config records 100 completed trials")
            audit.check("threshold" in config, f"{model_name} threshold is recorded")
        if audit.file(study_path, f"persistent Optuna study: {model_name}"):
            count = completed_trials(study_path)
            audit.check(count == 100, f"{model_name} database has 100 COMPLETE trials",
                        f"complete={count}")

    if manifest:
        data_dir = Path(manifest["data_dir"])
        for name, record in manifest.get("data", {}).items():
            path = data_dir / f"{name}.parquet"
            good = record is not None and path.exists()
            if good:
                good = (sha256(path) == record["sha256"] and
                        path.stat().st_size == int(record["bytes"]))
            audit.check(good, f"manifest data hash: {name}")
        for name, record in manifest.get("models", {}).items():
            path = MODEL_DIR / name
            good = path.exists() and sha256(path) == record["sha256"]
            audit.check(good, f"manifest model hash: {name}")

    if test is not None:
        verify_recomputed_metrics(audit, test)


def verify_recomputed_metrics(audit: Audit, test: pd.DataFrame) -> None:
    table_path = MODEL_TABLE_DIR / "model_performance_test_4models.csv"
    if not audit.file(table_path, "four-model headline table exists"):
        return
    table = pd.read_csv(table_path)
    table["_model"] = table["model"].map(canonical_model)
    audit.check(set(table["_model"].dropna()) == set(MODELS) and len(table) == 4,
                "headline contains exactly the four primary models")
    X = test[FEATURES].to_numpy(float)
    y = test["distress"].to_numpy(int)
    for model_name in MODELS:
        model = joblib.load(MODEL_DIR / f"{model_name}.joblib")
        config = yaml.safe_load(
            (CONFIG_DIR / f"{model_name}_config.yaml").read_text(encoding="utf-8")
        )
        probability = model.predict_proba(X)[:, 1]
        prediction = (probability >= float(config["threshold"])).astype(int)
        recomputed = {
            "pr_auc": average_precision_score(y, probability),
            "roc_auc": roc_auc_score(y, probability),
            "precision": precision_score(y, prediction, zero_division=0),
            "recall": recall_score(y, prediction, zero_division=0),
            "f1": f1_score(y, prediction, zero_division=0),
        }
        row = table.loc[table["_model"].eq(model_name)].iloc[0]
        for metric, value in recomputed.items():
            audit.check(math.isclose(float(row[metric]), float(value), abs_tol=5e-4),
                        f"independent {model_name} {metric}",
                        f"table={float(row[metric]):.4f}, recomputed={value:.4f}")
    baseline = y.mean()
    audit.check(np.allclose(table["prevalence_baseline_pr_auc"], baseline, atol=5e-5),
                "headline prevalence baseline matches the test sample")


def verify_reporting_outputs(audit: Audit) -> None:
    required_csv = {
        MODEL_TABLE_DIR: {
            "model_performance_test_3models.csv", "model_performance_test_4models.csv",
            "significance_tests.csv", "calibration_comparison_4models.csv",
            "subperiod_performance.csv",
        },
        ROBUSTNESS_DIR: {
            "neural_network_balanced_results.csv", "rc1_results.csv",
            "rc2_results.csv", "rc3_results.csv", "rc4_results.csv",
            "rc5_results.csv", "narrow_label_results.csv",
            "robustness_consolidated.csv", "h2_results.csv",
            "h2_design_optimism.csv", "temporal_bootstrap_ci.csv",
            "temporal_bootstrap_significance.csv",
        },
        SHAP_DIR: {
            "shap_importance.csv", "shap_lr_concordance.csv",
            "theory_consistency.csv", "shap_stability.csv",
            "lime_shap_agreement.csv",
        },
        DESCRIPTIVE_DIR: {
            "sample_composition_by_year.csv", "predictor_summary_stats.csv",
            "correlation_matrix.csv", "correlation_high_pairs.csv",
        },
    }
    for directory, names in required_csv.items():
        for name in sorted(names):
            path = directory / name
            good = audit.file(path, f"reporting table: {name}")
            if good:
                try:
                    audit.check(len(pd.read_csv(path)) > 0, f"{name} is non-empty")
                except Exception as error:
                    audit.check(False, f"{name} is readable", str(error))

    fixed_figures = {
        FIGURE_ROOT / "descriptive" / SPEC: {"distress_rate_timeseries"},
        FIGURE_ROOT / "model" / SPEC: {
            "calibration_4models", "pr_curves_4models", "roc_curves_4models",
        },
        FIGURE_ROOT / "shap" / SPEC: {
            "lev_roa_heatmap", "lime_tp", "lime_fn", "lime_tn",
            "shap_waterfall_tp", "shap_waterfall_fn", "shap_waterfall_tn",
            "shap_bar_logistic_regression", "shap_bar_random_forest",
            "shap_bar_xgboost", "shap_beeswarm_logistic_regression",
            "shap_beeswarm_random_forest", "shap_beeswarm_xgboost",
        },
    }
    for directory, stems in fixed_figures.items():
        for stem in sorted(stems):
            for suffix in (".png", ".pdf"):
                audit.file(directory / f"{stem}{suffix}",
                           f"publication figure: {stem}{suffix}", 1_000)
    dependence = FIGURE_ROOT / "shap" / SPEC
    pngs = list(dependence.glob("shap_dependence_*.png")) if dependence.exists() else []
    pdfs = list(dependence.glob("shap_dependence_*.pdf")) if dependence.exists() else []
    audit.check(len(pngs) >= 5 and len(pdfs) >= 5,
                "five nonlinear SHAP dependence figures exist")


def main() -> None:
    print("=" * 78)
    print("FINAL-PRIMARY THESIS OUTPUT VERIFICATION")
    print("=" * 78)
    audit = Audit()
    audit.check(SPEC == "final_primary", "active artifact namespace is final_primary")
    audit.check(len(FEATURES) == 18 and "MB_MISSING" in FEATURES,
                "final primary feature set is the disclosed 18-variable set")
    test = verify_data(audit)
    verify_outcome_and_cusip(audit)
    verify_primary_models(audit, test)
    verify_reporting_outputs(audit)

    payload = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": SPEC,
        "result": "PASS" if not audit.failures else "FAIL",
        "python": sys.version.split()[0],
        "checks": audit.checks,
        "artifacts": audit.artifacts,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if audit.failures:
        print(f"\nFAILED: {len(audit.failures)} check(s). Report: {REPORT}")
        raise SystemExit(1)
    print(f"\nALL {len(audit.checks)} CHECKS PASSED. Report: {REPORT}")


if __name__ == "__main__":
    main()
