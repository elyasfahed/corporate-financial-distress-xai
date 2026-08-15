"""Regression tests for final-primary provenance verification."""

from scripts.verify_final_outputs import current_code_fingerprint
from src.utils.run_manifest import _code_fingerprint


def test_verifier_uses_manifest_writers_canonical_code_scope() -> None:
    """A newly written manifest must be verifiable without scope drift."""
    assert current_code_fingerprint() == _code_fingerprint()["sha256"]
