"""
Tests for the repeated-test-evaluation runtime guard.

Covers the four behaviours the guard exists to provide:
  1. a repeat evaluation is refused;
  2. an approved override proceeds;
  3. the override reason is recorded in the manifest;
  4. legitimately different specifications are not blocked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation_guard import (GUARD_COLUMNS, RepeatedEvaluationError,
                                         check_evaluation_permitted,
                                         evaluation_identity,
                                         existing_identities, feature_hash)
# aliased: a module-level name starting with "test_" would be collected as a test
from src.models.evaluation_guard import test_data_hash as data_hash

FEATURES = ["NITA", "TLTA", "SIGMA"]


def _frame(n=50, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gvkey": np.repeat(np.arange(n // 5), 5),
        "fyear": np.tile(np.arange(2015, 2020), n // 5),
        "distress": rng.integers(0, 2, n),
        "NITA": rng.normal(size=n) + shift,
        "TLTA": rng.normal(size=n),
        "SIGMA": rng.normal(size=n),
    })


def _identity(test, spec="final_primary", label="is_distress_performance_core",
              horizon=365, features=FEATURES):
    return evaluation_identity(spec, label, horizon, features, test)


def _manifest(tmp_path, identities):
    p = tmp_path / "evaluation_manifest.csv"
    rows = []
    for ident in identities:
        row = {"timestamp_utc": "2026-08-01T00:00:00+00:00", "run_label": "primary",
               "output_path": "in-memory", "n_test": 50, "n_distress": 25,
               "n_features": 3, "features_hash": "abcd1234"}
        row.update({k: ident.get(k, "") for k in GUARD_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# --- 1. refusal ------------------------------------------------------------
def test_repeat_evaluation_is_refused(tmp_path):
    test = _frame()
    ident = _identity(test)
    path = _manifest(tmp_path, [ident])
    with pytest.raises(RepeatedEvaluationError) as exc:
        check_evaluation_permitted(ident, path)
    msg = str(exc.value)
    assert "Refusing to evaluate the test set again" in msg
    assert ident["evaluation_identity"] in msg
    # the message must tell the author how to proceed legitimately
    assert "override_reason" in msg


def test_first_evaluation_is_permitted(tmp_path):
    test = _frame()
    path = _manifest(tmp_path, [])
    assert check_evaluation_permitted(_identity(test), path) is None


def test_missing_manifest_is_permitted(tmp_path):
    assert check_evaluation_permitted(
        _identity(_frame()), tmp_path / "does_not_exist.csv") is None


def test_legacy_manifest_without_identity_column_does_not_block(tmp_path):
    """Rows written before the guard existed carry no identity and must not
    retroactively block a first guarded evaluation."""
    p = tmp_path / "evaluation_manifest.csv"
    pd.DataFrame([{"timestamp_utc": "2026-05-26T00:00:00+00:00",
                   "run_label": "primary", "output_path": "in-memory",
                   "n_test": 50, "n_distress": 25, "n_features": 3,
                   "features_hash": "abcd1234"}]).to_csv(p, index=False)
    assert existing_identities(p) == set()
    assert check_evaluation_permitted(_identity(_frame()), p) is None


# --- 2 & 3. override -------------------------------------------------------
def test_approved_override_proceeds_and_returns_reason(tmp_path):
    test = _frame()
    ident = _identity(test)
    path = _manifest(tmp_path, [ident])
    reason = "Re-scoring after the delisting extract was refreshed by CRSP."
    assert check_evaluation_permitted(ident, path, override_reason=reason) == reason


def test_override_reason_must_be_substantive(tmp_path):
    ident = _identity(_frame())
    path = _manifest(tmp_path, [ident])
    for bad in ("", "   ", "oops"):
        with pytest.raises(ValueError, match="substantive"):
            check_evaluation_permitted(ident, path, override_reason=bad)


def test_override_reason_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    from src.models import train as train_mod
    monkeypatch.setattr(train_mod, "OUT_TABLES_MODEL", tmp_path)
    ident = _identity(_frame())
    reason = "Deliberate re-evaluation documented in the review appendix."
    train_mod._append_evaluation_manifest(
        run_label="primary", n_test=50, n_distress=25, features=FEATURES,
        output_stem=None, identity=ident, override_reason=reason)
    df = pd.read_csv(tmp_path / "evaluation_manifest.csv")
    assert df.loc[0, "override_reason"] == reason
    assert df.loc[0, "evaluation_identity"] == ident["evaluation_identity"]
    assert df.loc[0, "spec"] == "final_primary"
    assert int(df.loc[0, "horizon_days"]) == 365


def test_append_preserves_legacy_rows_when_schema_grows(tmp_path, monkeypatch):
    from src.models import train as train_mod
    monkeypatch.setattr(train_mod, "OUT_TABLES_MODEL", tmp_path)
    p = tmp_path / "evaluation_manifest.csv"
    pd.DataFrame([{"timestamp_utc": "2026-05-26T00:00:00+00:00",
                   "run_label": "legacy", "output_path": "in-memory",
                   "n_test": 1, "n_distress": 0, "n_features": 3,
                   "features_hash": "old00000"}]).to_csv(p, index=False)
    train_mod._append_evaluation_manifest(
        run_label="primary", n_test=50, n_distress=25, features=FEATURES,
        output_stem=None, identity=_identity(_frame()), override_reason=None)
    df = pd.read_csv(p)
    assert len(df) == 2
    assert df.loc[0, "run_label"] == "legacy"          # legacy row intact
    assert pd.isna(df.loc[0, "evaluation_identity"])   # and un-backfilled
    assert isinstance(df.loc[1, "evaluation_identity"], str)


# --- 4. different legitimate specifications --------------------------------
@pytest.mark.parametrize("kwargs", [
    {"spec": "final_primary_rc1"},
    {"label": "is_bankruptcy_proxy"},
    {"horizon": 180},
    {"features": FEATURES + ["MB_MISSING"]},
])
def test_different_specification_is_not_blocked(tmp_path, kwargs):
    test = _frame()
    path = _manifest(tmp_path, [_identity(test)])
    assert check_evaluation_permitted(_identity(test, **kwargs), path) is None


def test_corrected_data_is_a_new_evaluation(tmp_path):
    """Re-scoring a corrected sample under the same spec is legitimate."""
    original = _frame()
    corrected = _frame(shift=0.25)
    path = _manifest(tmp_path, [_identity(original)])
    assert data_hash(original, FEATURES) != data_hash(corrected, FEATURES)
    assert check_evaluation_permitted(_identity(corrected), path) is None


def test_identity_is_stable_and_order_insensitive():
    test = _frame()
    assert _identity(test)["evaluation_identity"] == _identity(test)["evaluation_identity"]
    assert feature_hash(["b", "a"]) == feature_hash(["a", "b"])
    # row order is part of the split's identity, so a reordered frame differs
    assert data_hash(test, FEATURES) != data_hash(
        test.iloc[::-1].reset_index(drop=True), FEATURES)
