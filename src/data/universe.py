"""
Universe eligibility — CIZ letter-field equivalent of SHRCD 10/11.
===================================================================
The pre-specified design restricts the universe to CRSP share codes 10/11
(US-incorporated ordinary common shares). The local CIZ-format data
drop has no numeric SHRCD; share type and incorporation are encoded in
letter fields (ShareType, USIncFlg, SecurityType, SecuritySubType),
which the original extraction nulled via numeric coercion. The filter therefore
did not bind (measured contamination ≈3.2% non-US
firm-years plus ADR/unit/SBI, REIT and CEF components).

This module provides two explicit eligibility modes:

    universe_eligibility(secinfo, policy="frozen")  -> all True
    universe_eligibility(secinfo, policy="v2")      -> CIZ equivalent of
                                                       SHRCD 10/11

The v2 mapping follows CRSP's published SHRCD <-> CIZ crosswalk:
shrcd 10/11 corresponds to an ordinary common share (SecurityType EQTY,
SecuritySubType COM), no special share type (ShareType NS), issued by a
US-incorporated company (USIncFlg Y), and — per the design's universe
statement — listed on NYSE/AMEX/NASDAQ (exchcd_src N/A/Q). Rows with
missing letter fields are
treated as INELIGIBLE under v2 (conservative: eligibility must be
demonstrated, not assumed).

Eligibility can be evaluated in two ways:

  PERMNO-level ("any-time"): a PERMNO is eligible if any of its
    security-info rows passes. Used for the market-index / RSIZE
    denominator, where the contaminants (ETFs, CEFs, derivatives) are
    ineligible in every segment, so time-variation is immaterial.
  Date-ranged (as-of): a panel row is eligible only if a security-info
    row that passes the letter-field mask is valid on the row's date
    (namedt <= date <= nameendt). Used for the panel itself, so a firm
    that moves off an eligible exchange or share class contributes only
    the firm-years during which it actually qualified. Pass
    date_col="datadate" to apply_universe_filter.

The official CIZ value dictionary (MetaFlagInfo) is not in the local data
drop. The eligible values below are configuration constants
(src/config.py: CIZ_UNIVERSE_ELIGIBLE) and should be checked against CRSP's
flag dictionary before running the v2 rebuild.
"""

from __future__ import annotations

import pandas as pd

from src.config import CIZ_UNIVERSE_ELIGIBLE

#: secinfo column name -> config key
_FIELD_MAP = {
    "securitytype": "SecurityType",
    "securitysubtype": "SecuritySubType",
    "sharetype": "ShareType",
    "usincflg": "USIncFlg",
    "issuertype": "IssuerType",
    "exchcd_src": "Exchange",   # NYSE/AMEX/NASDAQ letter code (N/A/Q)
}


def universe_eligibility(
    secinfo: pd.DataFrame,
    policy: str = "frozen",
    eligible: dict | None = None,
) -> pd.Series:
    """
    Boolean eligibility mask over a security-info frame.

    Parameters
    ----------
    secinfo : pd.DataFrame
        Security-information rows carrying the CIZ letter fields
        (lower-case names as written by build_crsp_security_names:
        'securitytype', 'securitysubtype', 'sharetype', 'usincflg').
    policy : str
        "frozen"  — no filter; every row eligible (reproduces the
                    pipeline as run).
        "v2"      — CIZ letter-field equivalent of SHRCD 10/11; rows
                    with missing letter fields are ineligible.
    eligible : dict or None
        Override for the eligible-value sets; defaults to
        config.CIZ_UNIVERSE_ELIGIBLE ({config key: [allowed values]}).

    Returns
    -------
    pd.Series of bool, aligned to secinfo.index.
    """
    if policy == "frozen":
        return pd.Series(True, index=secinfo.index)
    if policy != "v2":
        raise ValueError(f"policy must be 'frozen' or 'v2', got {policy!r}")

    eligible = eligible if eligible is not None else CIZ_UNIVERSE_ELIGIBLE

    missing_cols = [c for c in _FIELD_MAP if c not in secinfo.columns]
    if missing_cols:
        raise KeyError(
            f"secinfo lacks the CIZ letter fields {missing_cols}. "
            "Re-run the extraction with the F5-fixed "
            "build_crsp_security_names() so the letter fields are "
            "preserved (src/data/load_local_rds.py)."
        )

    mask = pd.Series(True, index=secinfo.index)
    for col, key in _FIELD_MAP.items():
        allowed = eligible.get(key)
        if not allowed:
            continue
        values = secinfo[col].astype("string").str.strip().str.upper()
        # Missing letter field -> ineligible (conservative)
        mask &= values.isin([v.upper() for v in allowed]).fillna(False)
    return mask.astype(bool)


def apply_universe_filter(
    df: pd.DataFrame,
    secinfo: pd.DataFrame,
    policy: str = "frozen",
    on: str = "permno",
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    Filter a PERMNO-keyed frame to the eligible universe.

    Two modes are available (see the module docstring):

    date_col=None (PERMNO-level, "any-time"): a PERMNO is eligible if
      ANY of its security-info rows is eligible. Appropriate for the
      market-index / RSIZE denominator (ETFs/CEFs/derivatives are
      ineligible in every segment).
    date_col="datadate" (date-ranged, as-of): a row of `df` is kept only
      if an eligible security-info row is valid on that row's date
      (namedt <= df[date_col] <= nameendt). Rows whose date falls
      outside every eligible validity segment are excluded —
      conservative, matching the missing-letter-field rule.

    Under policy="frozen" the frame is returned unchanged.
    """
    if policy == "frozen":
        return df

    mask = universe_eligibility(secinfo, policy=policy)
    before = len(df)

    if date_col is None:
        eligible_ids = set(
            pd.to_numeric(secinfo.loc[mask, on], errors="coerce")
            .dropna().astype("int64")
        )
        ids = pd.to_numeric(df[on], errors="coerce").astype("Int64")
        out = df[ids.isin(list(eligible_ids))].copy()   # NA permno -> excluded
        print(
            f"  Universe filter (policy={policy}, any-time): kept "
            f"{len(out):,} of {before:,} rows ({before - len(out):,} "
            f"excluded; {len(eligible_ids):,} eligible {on}s)"
        )
        return out

    # Date-ranged mode: match each df row to the eligible validity
    # segments of its PERMNO and require the row date to fall inside one.
    for col in ("namedt", "nameendt"):
        if col not in secinfo.columns:
            raise KeyError(
                f"Date-ranged universe filter needs {col!r} in secinfo; "
                "re-run the extraction with build_crsp_security_names()."
            )
    if date_col not in df.columns:
        raise KeyError(f"date_col={date_col!r} not found in the panel frame")

    seg = secinfo.loc[mask, [on, "namedt", "nameendt"]].copy()
    seg[on] = pd.to_numeric(seg[on], errors="coerce")
    seg = seg.dropna(subset=[on])
    seg[on] = seg[on].astype("int64")
    seg["namedt"] = pd.to_datetime(seg["namedt"], errors="coerce")
    seg["nameendt"] = pd.to_datetime(seg["nameendt"], errors="coerce")
    seg = seg.dropna(subset=["namedt", "nameendt"])

    probe = pd.DataFrame({
        "_row": df.index,
        on: pd.to_numeric(df[on], errors="coerce"),
        "_date": pd.to_datetime(df[date_col], errors="coerce"),
    }).dropna(subset=[on, "_date"])
    probe[on] = probe[on].astype("int64")

    joined = probe.merge(seg, on=on, how="inner")
    in_range = joined[
        (joined["_date"] >= joined["namedt"])
        & (joined["_date"] <= joined["nameendt"])
    ]
    keep_rows = pd.Index(in_range["_row"].unique())
    out = df.loc[df.index.isin(keep_rows)].copy()
    print(
        f"  Universe filter (policy={policy}, as-of {date_col}): kept "
        f"{len(out):,} of {before:,} rows ({before - len(out):,} excluded; "
        f"{seg[on].nunique():,} {on}s with eligible segments)"
    )
    return out
