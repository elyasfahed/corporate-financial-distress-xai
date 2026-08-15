"""
Runtime guard on repeated test-set evaluation.
==============================================
The frozen design promises the test sample is *evaluated exactly once* per
specification. Until now that promise was documented and audited after the fact
but never enforced: ``evaluate_on_test`` appended a manifest row and returned,
so a second call on an identical specification was recorded but not prevented.

This module adds the missing enforcement. It is a **forward-looking safeguard
only**. The rows already in the manifest predate it and carry no evaluation
identity, so nothing historical is retroactively blocked, and the guard does
**not** repair or excuse the evaluate-once departure already disclosed in the
methodology chapter. It prevents the next one.

Evaluation identity
-------------------
Two evaluations are "the same" when all five of these agree:

* ``spec``            — artifact namespace (``final_primary``, ``v2_rc1``, ...)
* ``outcome_label``   — the event column defining the outcome
* ``horizon_days``    — the outcome window length
* ``feature_hash``    — SHA-256 over the sorted feature list
* ``test_data_hash``  — SHA-256 over the test split's identity, outcome, and
  feature values

Including the data hash is what makes the guard usable: re-scoring a *corrected*
sample is a legitimately new evaluation and is allowed, whereas re-scoring the
same sample under the same specification is refused.

Overriding
----------
An override requires an explicit, non-empty reason, which is recorded in the
manifest alongside the timestamp and destination. There is deliberately no
environment-variable or boolean escape hatch: a repeat evaluation should cost
the author a sentence explaining why.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns added to the evaluation manifest by the guard.
GUARD_COLUMNS = [
    "spec", "outcome_label", "horizon_days", "test_data_hash",
    "evaluation_identity", "override_reason",
]


class RepeatedEvaluationError(RuntimeError):
    """Raised when a (spec, label, horizon, features, data) tuple recurs."""


def _sha(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def feature_hash(features: list[str]) -> str:
    return _sha(",".join(sorted(features)))[:16]


def test_data_hash(test: pd.DataFrame, features: list[str],
                   label_col: str = "distress") -> str:
    """
    Content hash of the test split: shape, identity keys, outcome, and the
    feature matrix rounded to 10 decimals (so float formatting noise across
    platforms does not spuriously create a "new" evaluation).
    """
    cols = [c for c in ("gvkey", "fyear") if c in test.columns]
    ident = pd.util.hash_pandas_object(test[cols], index=False).to_numpy() \
        if cols else np.array([0])
    y = test[label_col].to_numpy(dtype=int) if label_col in test.columns else np.array([0])
    X = np.round(test[[f for f in features if f in test.columns]]
                 .to_numpy(dtype=float), 10)
    h = hashlib.sha256()
    h.update(np.asarray(test.shape, dtype=np.int64).tobytes())
    h.update(ident.astype(np.uint64).tobytes())
    h.update(y.astype(np.int64).tobytes())
    h.update(np.ascontiguousarray(X).tobytes())
    return h.hexdigest()[:16]


def evaluation_identity(spec: str, outcome_label: str, horizon_days: int,
                        features: list[str], test: pd.DataFrame,
                        label_col: str = "distress") -> dict:
    """Build the identity dict and its hash."""
    fh = feature_hash(features)
    dh = test_data_hash(test, features, label_col)
    return {
        "spec": str(spec),
        "outcome_label": str(outcome_label),
        "horizon_days": int(horizon_days),
        "feature_hash": fh,
        "test_data_hash": dh,
        "evaluation_identity": _sha(spec, outcome_label, horizon_days, fh, dh)[:16],
    }


def existing_identities(manifest_path: Path) -> set[str]:
    """Identities already recorded. Legacy rows have none and are ignored."""
    path = Path(manifest_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:      # header-less / zero-byte manifest
        return set()
    if "evaluation_identity" not in df.columns:
        return set()
    return set(df["evaluation_identity"].dropna().astype(str)) - {""}


def check_evaluation_permitted(identity: dict, manifest_path: Path,
                               override_reason: str | None = None) -> str | None:
    """
    Refuse a repeat evaluation unless an explicit reason is supplied.

    Returns
    -------
    str or None
        The recorded override reason, or None when this is a first evaluation.

    Raises
    ------
    RepeatedEvaluationError
        If the identity is already present and no reason was given.
    ValueError
        If a reason is supplied but is blank or uninformatively short.
    """
    ident = identity["evaluation_identity"]
    seen = ident in existing_identities(manifest_path)

    if override_reason is not None:
        reason = str(override_reason).strip()
        if len(reason) < 10:
            raise ValueError(
                "override_reason must be a substantive explanation "
                f"(at least 10 characters); got {override_reason!r}."
            )
        return reason

    if seen:
        raise RepeatedEvaluationError(
            "Refusing to evaluate the test set again under an identical "
            f"specification.\n"
            f"  spec           : {identity['spec']}\n"
            f"  outcome label  : {identity['outcome_label']}\n"
            f"  horizon (days) : {identity['horizon_days']}\n"
            f"  feature hash   : {identity['feature_hash']}\n"
            f"  test data hash : {identity['test_data_hash']}\n"
            f"  identity       : {ident}\n"
            "This tuple is already recorded in the evaluation manifest. The "
            "frozen design evaluates the test sample once per specification. "
            "If the repeat is genuinely warranted, pass an explicit "
            "override_reason=... to evaluate_on_test(); it will be recorded in "
            "the manifest. Changing the data, the label, the horizon or the "
            "feature set produces a new identity and needs no override."
        )
    return None
