"""
Tests for the outcome-definition approval record.
=================================================

``scripts.run_thesis_pipeline._record_label_approval`` is the FIRST thing
``--step data`` does. Before 2026-08-14 it opened
``outputs/models/configs/final_primary/outcome_definition_approval.yaml`` in
write mode unconditionally and emitted a payload that failed four of the six
checks in ``scripts/verify_final_outputs.py:196-217``:

  * ``primary_event_column``                     -- absent
  * ``performance_family_legacy_codes``          -- written as ``primary_legacy_codes``
  * ``voluntary_admin_excluded_legacy_codes``    -- absent
  * ``primary_rule_short``                       -- lacked EXCLUDING / MVOT / CORQ

The file is also a SHA-256-hashed ``input_files`` entry in the final_primary run
manifest, and because the payload stamps ``approved_at_utc`` every rewrite
changes the digest even when the content is correct. So the first command of a
rebuild destroyed a verified artifact and guaranteed ``--step verify`` failed.

These tests pin both halves of the fix: the payload now satisfies every
verifier assertion, and an existing record is not overwritten without an
explicit opt-in. The assertions below deliberately compare against the same
constants ``verify_final_outputs`` imports rather than against string literals,
so the record and the check cannot drift apart.
"""

import yaml

import pytest

from src.config import V2_PROFILE
from src.data.distress_definition import (
    NARROW_FINANCIAL_REASONS,
    PERFORMANCE_LEGACY_CODES,
    VOLUNTARY_ADMIN_LEGACY_CODES,
    VOLUNTARY_ADMIN_REASONS,
)
from scripts.run_thesis_pipeline import (
    APPROVAL_PATH,
    _build_approval_payload,
    _record_label_approval,
)


# --------------------------------------------------------------------------
# 1. The generated payload satisfies every verifier check
# --------------------------------------------------------------------------

def test_payload_primary_definition_is_exact():
    """verify_final_outputs: 'primary outcome name is exact'."""
    assert _build_approval_payload()["primary_definition"] == \
        "CRSP performance delistings"


def test_payload_names_the_cleaned_event_column():
    """verify_final_outputs: 'primary event column is the cleaned ... label'."""
    payload = _build_approval_payload()
    assert payload["primary_event_column"] == V2_PROFILE["distress_event_col"]
    assert payload["primary_event_column"] == "is_distress_performance_core"


def test_payload_performance_family_is_exact():
    """verify_final_outputs: 'performance-delisting legacy-code family is exact'."""
    payload = _build_approval_payload()
    assert set(payload["performance_family_legacy_codes"]) == \
        set(PERFORMANCE_LEGACY_CODES)


def test_payload_voluntary_admin_codes_are_exact():
    """verify_final_outputs: 'voluntary/administrative excluded codes are exact'."""
    payload = _build_approval_payload()
    assert set(payload["voluntary_admin_excluded_legacy_codes"]) == \
        set(VOLUNTARY_ADMIN_LEGACY_CODES)
    assert set(payload["voluntary_admin_excluded_reasons"]) == \
        set(VOLUNTARY_ADMIN_REASONS)


@pytest.mark.parametrize("token", ["EXCLUDING", "MVOT", "CORQ"])
def test_payload_rule_short_states_the_exclusion(token):
    """verify_final_outputs: 'short primary rule states the voluntary-exit exclusion'.

    This is the check the old literal "400-490, 500, 520-584" failed on all
    three tokens.
    """
    assert token in _build_approval_payload()["primary_rule_short"]


def test_payload_identifies_the_bankruptcy_proxy():
    """verify_final_outputs: 'RC1 is identified as a bankruptcy-reason proxy'."""
    assert _build_approval_payload()["bankruptcy_proxy_reason"] == "BKPY"


def test_payload_retains_the_narrow_sensitivity_reasons():
    """Not verifier-checked, but the checked-in record carries it."""
    assert set(_build_approval_payload()["narrow_sensitivity_gdr_reasons"]) == \
        set(NARROW_FINANCIAL_REASONS)


# --------------------------------------------------------------------------
# 2. The generated schema covers what the checked-in record carries
# --------------------------------------------------------------------------

def test_generated_schema_covers_the_checked_in_record():
    """
    A regenerated record must not silently lose fields the committed one has.

    Skips rather than fails when the artifact is absent, so the suite still
    runs in a clean checkout without outputs/.
    """
    if not APPROVAL_PATH.exists():
        pytest.skip(f"{APPROVAL_PATH} not present in this checkout")
    committed = yaml.safe_load(APPROVAL_PATH.read_text(encoding="utf-8"))
    missing = set(committed) - set(_build_approval_payload())
    assert not missing, f"regenerated payload would drop keys: {sorted(missing)}"


# --------------------------------------------------------------------------
# 3. An existing record is not clobbered
# --------------------------------------------------------------------------

def test_existing_record_is_left_untouched(tmp_path, monkeypatch, capsys):
    """The regression test for the hazard itself: bytes must not change."""
    target = tmp_path / "outcome_definition_approval.yaml"
    original = "approved_at_utc: '2026-07-20T23:52:03.943092+00:00'\nfoo: bar\n"
    target.write_text(original, encoding="utf-8")
    monkeypatch.setattr("scripts.run_thesis_pipeline.APPROVAL_PATH", target)

    _record_label_approval()

    assert target.read_text(encoding="utf-8") == original
    assert "leaving" in capsys.readouterr().out


def test_reapprove_flag_rewrites_the_record(tmp_path, monkeypatch):
    """The opt-in must actually write, and write a verifier-valid record."""
    target = tmp_path / "outcome_definition_approval.yaml"
    target.write_text("approved_at_utc: 'old'\nfoo: bar\n", encoding="utf-8")
    monkeypatch.setattr("scripts.run_thesis_pipeline.APPROVAL_PATH", target)

    _record_label_approval(reapprove=True)

    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert "foo" not in written
    assert written["primary_event_column"] == V2_PROFILE["distress_event_col"]
    assert set(written["performance_family_legacy_codes"]) == \
        set(PERFORMANCE_LEGACY_CODES)
    assert "EXCLUDING" in written["primary_rule_short"]


def test_first_approval_writes_when_absent(tmp_path, monkeypatch):
    """A genuine first approval must still work, including creating the dir."""
    target = tmp_path / "nested" / "outcome_definition_approval.yaml"
    monkeypatch.setattr("scripts.run_thesis_pipeline.APPROVAL_PATH", target)

    _record_label_approval()

    assert target.exists()
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written["primary_definition"] == "CRSP performance delistings"
