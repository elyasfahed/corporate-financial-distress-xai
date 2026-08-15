"""
Tests for spec-aware artifact loading in src/utils/io.py.
=========================================================

``src.utils.io._resolve_artifact_dirs`` deliberately mirrors
``src.models.train.resolve_artifact_dirs`` instead of importing it (that module
pulls in scikit-learn and XGBoost, far too heavy for resolving a path). These
tests pin the two implementations together so the duplication cannot drift.

Background: before 2026-07-28, ``load_model``/``load_config`` were hardcoded to
the top-level ``outputs/models/{saved,configs}/`` directories. The 2026-07-24
provenance quarantine moved the v1 artifacts to ``outputs/_superseded/``, so
those helpers pointed at paths that no longer exist, while every current entry
point had already moved to per-spec namespaces.
"""

import pytest

from src.config import OUT_MODELS_SAVED, OUT_MODELS_CONFIGS
from src.models.train import resolve_artifact_dirs
from src.utils.io import _resolve_artifact_dirs, load_model, load_config


SPECS = ["primary", "final_primary", "final_primary_rc1", "v2", "v2_rc5"]


@pytest.mark.parametrize("spec", SPECS)
def test_io_resolution_matches_train_resolution(spec):
    """The mirrored helper must agree with the canonical one for every spec."""
    assert _resolve_artifact_dirs(spec) == resolve_artifact_dirs(spec)


def test_primary_spec_keeps_frozen_toplevel_layout():
    """spec='primary' must resolve to the un-namespaced v1 directories."""
    saved, configs = _resolve_artifact_dirs("primary")
    assert saved == OUT_MODELS_SAVED
    assert configs == OUT_MODELS_CONFIGS


def test_non_primary_spec_is_namespaced():
    saved, configs = _resolve_artifact_dirs("final_primary")
    assert saved == OUT_MODELS_SAVED / "final_primary"
    assert configs == OUT_MODELS_CONFIGS / "final_primary"


@pytest.mark.parametrize("bad", ["", "..", "a/b", "a\\b"])
def test_invalid_spec_rejected(bad):
    """Path-traversal and empty spec names must raise, as in train.py."""
    with pytest.raises(ValueError):
        _resolve_artifact_dirs(bad)


def test_missing_model_raises_with_actionable_message():
    """
    A missing artifact must fail loudly and name the per-spec layout, rather
    than surfacing a bare joblib/OS error.
    """
    with pytest.raises(FileNotFoundError) as exc:
        load_model("definitely_not_a_model", spec="final_primary")
    msg = str(exc.value)
    assert "spec=" in msg
    assert "_superseded" in msg


def test_missing_config_raises_with_actionable_message():
    with pytest.raises(FileNotFoundError) as exc:
        load_config("definitely_not_a_config", spec="final_primary")
    msg = str(exc.value)
    assert "spec=" in msg
    assert "_superseded" in msg
