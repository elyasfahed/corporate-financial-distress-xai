"""
Run manifest — provenance record for every training run.
=========================================================
Addresses a provenance gap: the 2026-06-29 primary re-run reused a
hyperparameter config whose origin could not be reconstructed afterwards,
because nothing recorded which code, data, and configuration produced a
given set of model artifacts.

write_run_manifest() captures, at the moment models are saved:

  - git commit hash + dirty flag (the code state)
  - Python and key package versions (the environment)
  - SHA-256 of the train/val/test parquets (the data state)
  - SHA-256 of every model artifact written (the outputs)
  - the feature list and its hash, spec tag, and caller-supplied extras

The manifest is written as JSON next to the config YAMLs of the same
specification, so a saved model can always be traced back to exactly one
(code, data, config) triple. Best-effort by design: callers wrap it in
try/except so a provenance failure never kills a multi-hour run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_SAMPLES, ROOT as PROJECT_ROOT

MANIFEST_NAME = "run_manifest.json"

_PACKAGES = ["numpy", "pandas", "sklearn", "xgboost", "optuna", "shap", "imblearn"]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        if not commit:
            return {"commit": None, "dirty": None}
        status = run("status", "--porcelain")
        producing_status = run("status", "--porcelain", "--", "src", "scripts")
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, check=False,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
            "producing_code_dirty": bool(producing_status),
            "producing_code_status": producing_status.splitlines(),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {"commit": None, "dirty": None}


def _package_versions() -> dict:
    import importlib

    versions = {"python": sys.version.split()[0]}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            versions[pkg] = None
    return versions


def _data_hashes(samples_dir: Path) -> dict:
    out = {}
    for name in ("train", "val", "test", "train_raw", "val_raw", "test_raw"):
        p = samples_dir / f"{name}.parquet"
        if p.exists():
            out[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
        else:
            out[name] = None
    return out


def _input_hashes(paths) -> dict:
    out = {}
    for raw in paths or []:
        p = Path(raw)
        out[str(p)] = (
            {"sha256": _sha256(p), "bytes": p.stat().st_size}
            if p.exists() and p.is_file() else None
        )
    return out


def _code_fingerprint() -> dict:
    """Hash every producing source file, including untracked new modules."""
    roots = [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.py"))
    files.extend(
        p for p in (
            PROJECT_ROOT / "requirements.txt",
            PROJECT_ROOT / "requirements-lock.txt",
        ) if p.exists()
    )
    file_hashes = {
        str(p.relative_to(PROJECT_ROOT)): _sha256(p)
        for p in sorted(files, key=lambda x: str(x).lower())
    }
    combined = hashlib.sha256()
    for name, digest in file_hashes.items():
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {"sha256": combined.hexdigest(), "files": file_hashes}


def write_run_manifest(
    configs_dir: Path,
    saved_dir: Path,
    spec: str,
    features: list[str],
    extra: dict | None = None,
    samples_dir: Path | None = None,
    input_files: list[Path] | None = None,
) -> Path:
    """
    Write the provenance manifest for one training run.

    Returns the path of the manifest written.
    """
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    saved_dir = Path(saved_dir)

    models = {}
    if saved_dir.exists():
        for p in sorted(saved_dir.glob("*.joblib")):
            models[p.name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": spec,
        "git": _git_state(),
        "code_fingerprint": _code_fingerprint(),
        "versions": _package_versions(),
        # Record WHICH samples directory was hashed, so a v2 manifest can
        # never be silently mistaken for one hashing the v1 splits
        # (2026-07-12 second-audit fix).
        "data_dir": str(samples_dir or DATA_SAMPLES),
        "data": _data_hashes(samples_dir or DATA_SAMPLES),
        "input_files": _input_hashes(input_files),
        "features": list(features),
        "features_hash": hashlib.md5(
            ",".join(sorted(features)).encode()
        ).hexdigest()[:8],
        "models": models,
        "extra": extra or {},
    }

    out_path = configs_dir / MANIFEST_NAME
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Run manifest written -> {out_path}")
    return out_path
