"""
Regression tests — spec-aware artifact directories + primary-overwrite guard
============================================================================
A previous implementation let every train_all_models() caller write models
and config YAMLs to the same shared primary paths, so a robustness
re-tune silently replace the primary hyperparameter configs. These tests
pin the fix:

  1. resolve_artifact_dirs: primary -> frozen top-level dirs (unchanged
     layout); any other spec -> its own subdirectory.
  2. The guard blocks spec="primary" when primary artifacts exist, unless
     allow_primary_overwrite=True; non-primary specs are never blocked.
  3. An end-to-end (tiny, reuse_configs) run under spec="rcX" writes
     models + configs + run manifest into the rcX subdirs and leaves the
     primary files byte-identical.
  4. The run manifest records git state, package versions, features and
     model checksums.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import src.models.train as train_mod
from src.models.train import (
    MODEL_NAMES,
    resolve_artifact_dirs,
    _guard_primary_artifacts,
    train_all_models,
)
from src.config import OUT_MODELS_SAVED, OUT_MODELS_CONFIGS


# ---------------------------------------------------------------------------
# 1. Path resolution
# ---------------------------------------------------------------------------

def test_primary_spec_resolves_to_frozen_paths():
    saved, configs = resolve_artifact_dirs("primary")
    assert saved == OUT_MODELS_SAVED
    assert configs == OUT_MODELS_CONFIGS


def test_non_primary_spec_resolves_to_subdirs():
    saved, configs = resolve_artifact_dirs("rc6")
    assert saved == OUT_MODELS_SAVED / "rc6"
    assert configs == OUT_MODELS_CONFIGS / "rc6"


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", ".."])
def test_invalid_spec_rejected(bad):
    with pytest.raises(ValueError):
        resolve_artifact_dirs(bad)


# ---------------------------------------------------------------------------
# 2. Guard behaviour
# ---------------------------------------------------------------------------

def test_guard_blocks_primary_overwrite(tmp_path):
    saved = tmp_path / "saved"
    configs = tmp_path / "configs"
    saved.mkdir()
    configs.mkdir()
    (saved / f"{MODEL_NAMES[0]}.joblib").write_bytes(b"frozen")

    with pytest.raises(RuntimeError, match="allow_primary_overwrite"):
        _guard_primary_artifacts(saved, configs, "primary",
                                 allow_primary_overwrite=False)

    # Explicit opt-in passes
    _guard_primary_artifacts(saved, configs, "primary",
                             allow_primary_overwrite=True)


def test_guard_never_blocks_non_primary(tmp_path):
    saved = tmp_path / "saved"
    configs = tmp_path / "configs"
    saved.mkdir()
    (saved / f"{MODEL_NAMES[0]}.joblib").write_bytes(b"anything")
    _guard_primary_artifacts(saved, configs, "rc1",
                             allow_primary_overwrite=False)


def test_guard_allows_fresh_primary(tmp_path):
    _guard_primary_artifacts(tmp_path / "s", tmp_path / "c", "primary",
                             allow_primary_overwrite=False)


# ---------------------------------------------------------------------------
# 3 + 4. End-to-end spec isolation (tiny synthetic run, reuse_configs)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_splits():
    rng = np.random.default_rng(42)
    n = 240
    features = ["f1", "f2", "f3"]

    def make(n):
        X = rng.normal(size=(n, 3))
        y = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.2).astype(int)
        y[:3] = 1  # ensure both classes present
        df = pd.DataFrame(X, columns=features)
        df["distress"] = y
        return df

    return make(n), make(80), features


def _write_tiny_configs(configs_dir):
    configs_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "logistic_regression": {"C": 1.0},
        "random_forest": {"n_estimators": 10, "max_depth": 3},
        "xgboost": {"n_estimators": 10, "max_depth": 2,
                    "learning_rate": 0.3},
    }
    for name, p in params.items():
        with open(configs_dir / f"{name}_config.yaml", "w") as f:
            yaml.dump({"model": name, "best_params": p, "threshold": 0.5}, f)


def test_spec_run_writes_own_subdir_and_manifest(tmp_path, monkeypatch, tiny_splits):
    train, val, features = tiny_splits

    saved_root = tmp_path / "saved"
    configs_root = tmp_path / "configs"
    monkeypatch.setattr(train_mod, "OUT_MODELS_SAVED", saved_root)
    monkeypatch.setattr(train_mod, "OUT_MODELS_CONFIGS", configs_root)

    # Pre-existing frozen primary artifacts that must remain untouched
    saved_root.mkdir(parents=True)
    configs_root.mkdir(parents=True)
    primary_model = saved_root / "logistic_regression.joblib"
    primary_cfg = configs_root / "logistic_regression_config.yaml"
    primary_model.write_bytes(b"FROZEN-MODEL-BYTES")
    primary_cfg.write_text("frozen: true\n")

    # Tiny tuned configs for the rcX spec so reuse_configs skips Optuna
    _write_tiny_configs(configs_root / "rcX")

    out = train_all_models(train, val, features=features,
                           reuse_configs=True, spec="rcX")

    # Models + configs land in the spec subdirs
    for name in MODEL_NAMES:
        assert (saved_root / "rcX" / f"{name}.joblib").exists()
        assert (configs_root / "rcX" / f"{name}_config.yaml").exists()
        assert name in out

    # Primary artifacts byte-identical
    assert primary_model.read_bytes() == b"FROZEN-MODEL-BYTES"
    assert primary_cfg.read_text() == "frozen: true\n"

    # Run manifest written into the spec's config dir with provenance keys
    manifest_path = configs_root / "rcX" / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["spec"] == "rcX"
    assert set(manifest["features"]) == set(features)
    assert "git" in manifest and "commit" in manifest["git"]
    assert manifest["versions"].get("sklearn")
    assert any(k.endswith(".joblib") for k in manifest["models"])
    assert all(v["sha256"] for v in manifest["models"].values())


def test_primary_guard_end_to_end(tmp_path, monkeypatch, tiny_splits):
    train, val, features = tiny_splits

    saved_root = tmp_path / "saved"
    configs_root = tmp_path / "configs"
    monkeypatch.setattr(train_mod, "OUT_MODELS_SAVED", saved_root)
    monkeypatch.setattr(train_mod, "OUT_MODELS_CONFIGS", configs_root)

    saved_root.mkdir(parents=True)
    configs_root.mkdir(parents=True)
    (saved_root / "xgboost.joblib").write_bytes(b"FROZEN")

    with pytest.raises(RuntimeError, match="allow_primary_overwrite"):
        train_all_models(train, val, features=features,
                         reuse_configs=True, spec="primary")
