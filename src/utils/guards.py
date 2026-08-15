"""
Pipeline abort guards for the corrected-run protocol.
======================================================
Every guard here raises. None of them warn-and-continue: the failure modes this
repository has actually suffered (stale-output retraction and configuration
overwrite) were cases where a check existed but returned rather than raised,
so "not verified" silently read as "verified".

Each function maps to a defined abort condition in the frozen protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: §19.2 — the sole registered duplicate PERMNO-fiscal-year exception.
#: Tilray/Aphria reverse merger: PERMNO 17977 carries both the surviving
#: entity (gvkey 22387, FYE 2021-05-31 -> fyear 2020 under the June rule) and
#: the predecessor (gvkey 33703, FYE 2020-12-31). Matched by identity, never by
#: count, so a NEW duplicate can never be absorbed by this allowance.
REGISTERED_PERMNO_FYEAR_EXCEPTIONS: set[tuple[int, int, str]] = {
    (17977, 2020, "22387"),
    (17977, 2020, "33703"),
}


def assert_unique_gvkey_fyear(panel: pd.DataFrame) -> None:
    """§19.1 — gvkey-fyear must be unique."""
    dup = panel.duplicated(["gvkey", "fyear"], keep=False)
    if dup.any():
        ex = panel.loc[dup, ["gvkey", "fyear"]].head(5).to_dict("records")
        raise ValueError(
            f"{int(dup.sum()):,} rows violate gvkey-fyear uniqueness "
            f"(protocol §19.1). Examples: {ex}"
        )


def assert_permno_fyear_documented(panel: pd.DataFrame) -> None:
    """
    §19.2 — any duplicate permno-fiscal-year mapping must be registered.

    Registered pairs are matched on (permno, fyear, gvkey) identity, so an
    unrelated new duplicate cannot hide behind the allowance.
    """
    if "permno" not in panel.columns:
        return
    dup = panel.duplicated(["permno", "fyear"], keep=False)
    if not dup.any():
        return
    offending = panel.loc[dup, ["permno", "fyear", "gvkey"]].copy()
    offending["permno"] = pd.to_numeric(offending["permno"], errors="coerce").astype("Int64")
    keys = {
        (int(p), int(f), str(g))
        for p, f, g in zip(offending["permno"], offending["fyear"], offending["gvkey"])
        if pd.notna(p)
    }
    unregistered = keys - REGISTERED_PERMNO_FYEAR_EXCEPTIONS
    if unregistered:
        raise ValueError(
            f"{len(unregistered)} undocumented duplicate permno-fyear mappings "
            f"(protocol §19.2): {sorted(unregistered)[:5]}. Register them with a "
            "written justification or resolve the linkage."
        )


def assert_label_maturity(
    panel: pd.DataFrame,
    coverage_end: pd.Timestamp,
    horizon_days: int = 365,
    date_col: str = "fdate",
) -> None:
    """
    §19.4 — no label window may extend beyond the outcome data's coverage.

    A firm-year whose window [fdate, fdate + horizon] ends after the delisting
    extract stops is coded 0 on incomplete follow-up, which is right-censoring
    masquerading as a negative.
    """
    if date_col not in panel.columns:
        raise KeyError(f"assert_label_maturity needs {date_col!r}")
    end = pd.to_datetime(panel[date_col]) + pd.Timedelta(days=horizon_days)
    immature = end > pd.Timestamp(coverage_end)
    if immature.any():
        raise ValueError(
            f"{int(immature.sum()):,} firm-years have label windows ending after "
            f"outcome coverage {pd.Timestamp(coverage_end).date()} "
            f"(protocol §19.4); latest window end {end.max().date()}."
        )


def assert_binary_features(
    df: pd.DataFrame,
    binary_features: list[str],
    allow_nan: bool = False,
) -> None:
    """§19.6 — declared binary features must take values in {0, 1}."""
    for f in binary_features:
        if f not in df.columns:
            continue
        col = pd.to_numeric(df[f], errors="coerce")
        if not allow_nan and col.isna().any():
            raise ValueError(
                f"Binary feature {f!r} contains {int(col.isna().sum()):,} NaN "
                "(protocol §19.6)."
            )
        vals = np.unique(col.dropna().values)
        offending = np.setdiff1d(vals, np.array([0.0, 1.0]))
        if offending.size:
            raise ValueError(
                f"Binary feature {f!r} takes non-binary values "
                f"{offending[:5]} (protocol §19.6)."
            )


def assert_finite_model_input(X, features: list[str] | None = None) -> None:
    """§19.7 — no NaN or +/-inf may reach an estimator."""
    arr = np.asarray(X, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    if n_nan or n_inf:
        detail = ""
        if features is not None and arr.ndim == 2 and arr.shape[1] == len(features):
            bad = [
                features[j]
                for j in range(arr.shape[1])
                if np.isnan(arr[:, j]).any() or np.isinf(arr[:, j]).any()
            ]
            detail = f" Offending features: {bad}"
        raise ValueError(
            f"Model input carries {n_nan:,} NaN and {n_inf:,} inf values "
            f"(protocol §19.7).{detail}"
        )


def assert_no_split_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    keys: tuple[str, str] = ("gvkey", "fyear"),
) -> None:
    """Supporting check for §19.8 — splits must be disjoint on their keys."""
    def _k(d):
        return set(zip(*(d[k] for k in keys)))
    tr, va, te = _k(train), _k(val), _k(test)
    for a, b, na, nb in ((tr, va, "train", "val"),
                         (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        inter = a & b
        if inter:
            raise ValueError(
                f"{len(inter):,} {keys} keys appear in both {na} and {nb} "
                "(protocol §19.8)."
            )


def assert_outer_purge(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    horizon_days: int = 365,
    date_col: str = "fdate",
    earlier_name: str = "train",
    later_name: str = "val",
) -> None:
    """
    §19.8 — the earlier split's label windows must close strictly before the
    later split's first prediction date, or the earlier labels encode the
    later period.
    """
    e = pd.to_datetime(earlier[date_col]).max() + pd.Timedelta(days=horizon_days)
    l = pd.to_datetime(later[date_col]).min()
    if not (e < l):
        raise ValueError(
            f"Outer purge violated (protocol §19.8): {earlier_name} max "
            f"{date_col}+{horizon_days}d = {e.date()} is not strictly before "
            f"{later_name} min {date_col} = {l.date()}."
        )


def assert_tuning_sample_is_not_test(
    tuning_frame: pd.DataFrame,
    test: pd.DataFrame,
    keys: tuple[str, str] = ("gvkey", "fyear"),
) -> None:
    """§19.10 — tuning code must never see the final evaluation sample."""
    shared = set(zip(*(tuning_frame[k] for k in keys))) & set(
        zip(*(test[k] for k in keys))
    )
    if shared:
        raise ValueError(
            f"Tuning frame shares {len(shared):,} {keys} keys with the test "
            "sample (protocol §19.10)."
        )


def assert_model_spec(manifest: dict, expected_spec: str) -> None:
    """§19.11 — refuse artifacts belonging to another specification."""
    got = manifest.get("spec")
    if got != expected_spec:
        raise ValueError(
            f"Model artifacts belong to spec {got!r}, expected "
            f"{expected_spec!r} (protocol §19.11)."
        )
