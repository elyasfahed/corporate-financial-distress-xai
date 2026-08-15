"""Repair corrected robustness-run provenance without retraining models.

RC1/RC2/RC4/RC5 passed corrected DataFrames to ``train_all_models`` but did
not pass ``samples_dir``.  The models and result tables are valid; only their
manifests therefore defaulted to hashes of the historical sample directory.

This one-time repair is intentionally transparent.  It preserves each
original manifest byte-for-byte as ``run_manifest_pre_repair.json`` and puts
the original timestamp and SHA-256 in the replacement manifest.  No model,
configuration, Optuna study, table, or figure is modified.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import (
    ACCOUNTING_FEATURES,
    ALL_FEATURES_V2,
    DATA_FEATURES_V2,
    DATA_MERGED_V2,
    DATA_ROOT,
    DATA_SAMPLES_V2,
    DISTRESS_HORIZON_DAYS,
    DISTRESS_HORIZON_RC2,
    OUT_MODELS_CONFIGS,
    OUT_MODELS_SAVED,
    V2_PROFILE,
)
from src.utils.run_manifest import write_run_manifest


SPEC = V2_PROFILE["spec"]
DELIST = DATA_ROOT / "raw_primary" / "crsp_delisting_raw.parquet"
COMP = DATA_ROOT / "raw_primary" / "compustat_annual_raw.parquet"
SECNAMES = DATA_ROOT / "raw_primary" / "crsp_security_names.parquet"
MONTHLY = DATA_ROOT / "raw" / "crsp" / "crsp_monthly_raw.parquet"
APPROVAL = OUT_MODELS_CONFIGS / "label_definition_approval.yaml"
RC5_SAMPLES = DATA_ROOT / "processed_primary" / "samples_rc5"
RC5_PANEL = DATA_MERGED_V2 / "panel_raw_rc5.parquet"
RC5_FEATURES = DATA_FEATURES_V2 / "features_all_rc5.parquet"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _model_mtimes(saved_dir: Path) -> dict[str, str]:
    return {
        path.name: datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(timespec="seconds")
        for path in sorted(saved_dir.glob("*.joblib"))
    }


def repair_one(
    spec: str,
    *,
    samples_dir: Path,
    features: list[str],
    input_files: list[Path],
    design: dict,
) -> Path:
    configs_dir = OUT_MODELS_CONFIGS / spec
    saved_dir = OUT_MODELS_SAVED / spec
    manifest_path = configs_dir / "run_manifest.json"
    backup_path = configs_dir / "run_manifest_pre_repair.json"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not saved_dir.exists() or not list(saved_dir.glob("*.joblib")):
        raise FileNotFoundError(f"No saved models in {saved_dir}")

    original_bytes = manifest_path.read_bytes()
    original = json.loads(original_bytes.decode("utf-8"))
    if not backup_path.exists():
        backup_path.write_bytes(original_bytes)
    elif backup_path.read_bytes() != original_bytes:
        # A repeated invocation after repair must retain the first original,
        # not replace it with the already-repaired manifest.
        original_bytes = backup_path.read_bytes()
        original = json.loads(original_bytes.decode("utf-8"))

    extra = {
        "provenance_repaired_posthoc": True,
        "repair_reason": (
            "Training received corrected DataFrames but the original caller "
            "omitted samples_dir, so the manifest defaulted to historical "
            "sample hashes. Models and results were not regenerated."
        ),
        "original_manifest_timestamp_utc": original.get("timestamp_utc"),
        "original_manifest_sha256": _sha256_bytes(original_bytes),
        "saved_model_mtime_utc": _model_mtimes(saved_dir),
        "n_trials_effective": 100,
        **design,
    }
    repaired = write_run_manifest(
        configs_dir=configs_dir,
        saved_dir=saved_dir,
        spec=spec,
        features=features,
        samples_dir=samples_dir,
        input_files=input_files,
        extra=extra,
    )
    print(f"  Repaired provenance only: {repaired}")
    return repaired


def main() -> None:
    base_inputs = [DELIST, APPROVAL]
    repair_one(
        f"{SPEC}_rc1",
        samples_dir=DATA_SAMPLES_V2,
        features=list(ALL_FEATURES_V2),
        input_files=base_inputs,
        design={
            "outcome_variant": "GDR DelReasonType=BKPY bankruptcy-reason proxy",
            "horizon_days": DISTRESS_HORIZON_DAYS,
        },
    )
    repair_one(
        f"{SPEC}_rc2",
        samples_dir=DATA_SAMPLES_V2,
        features=list(ALL_FEATURES_V2),
        input_files=base_inputs,
        design={
            "outcome_variant": "primary financial-distress label",
            "horizon_days": DISTRESS_HORIZON_RC2,
        },
    )
    repair_one(
        f"{SPEC}_rc4",
        samples_dir=DATA_SAMPLES_V2,
        features=list(ACCOUNTING_FEATURES),
        input_files=[APPROVAL],
        design={
            "predictor_variant": "11 accounting predictors only",
            "horizon_days": DISTRESS_HORIZON_DAYS,
        },
    )
    repair_one(
        f"{SPEC}_rc5",
        samples_dir=RC5_SAMPLES,
        features=list(ALL_FEATURES_V2),
        input_files=[
            COMP,
            DELIST,
            SECNAMES,
            MONTHLY,
            APPROVAL,
            RC5_PANEL,
            RC5_FEATURES,
        ],
        design={
            "universe_variant": "utilities included; financials excluded",
            "horizon_days": DISTRESS_HORIZON_DAYS,
        },
    )
    print("Corrected RC manifest repair complete; no models were retrained.")


if __name__ == "__main__":
    main()
