"""
CUSIP mismatch disposition — closing audit item 1 (2026-07-29).
================================================================
``merge_crsp_compustat.validate_cusip`` flags firm-years where the CCM link
and the CUSIP cross-check disagree, writes them to ``cusip_mismatches.csv``
"for manual review", and **nothing ever reads the flag again**. On the reported
sample 2,699 flagged firm-years entered the modelling panel with no recorded
disposition.

This module supplies the documented rules the protocol (§5) requires, so every
flagged pair is classified and an unclassified one becomes an abort condition
rather than a silent pass.

Why a mismatch is NOT a link rejection
--------------------------------------
CCM is the authoritative WRDS-maintained link and handles reorganisations and
share-class selection by design; CUSIP equality does not. Measured on the
reported sample, the flagged set is dominated by identifier churn:

  * Swift Energy -> SilverBow Resources (post-Chapter-11 rename, new CUSIP)
    keeps PERMNO 66739 throughout — CRSP tracks the change point-in-time while
    the Compustat value stays stale.
  * Tecumseh Products 87889510 vs 87889520 — same issuer, different issue
    (share class).
  * Six Flags 83001P10 vs 83001A10 — issuer code reassigned at reorganisation.

Coverage caveat that must be reported with any use of this flag: the check is
**inoperative before ~2010**. Only 49,241 of 117,183 firm-years (42.0%) are
checkable at all; the remainder fail solely because the Compustat CUSIP is
empty — 100% empty in the 1990s, 73.1% in the 2000s, 0% from 2010. The layer
therefore validates the test window and essentially none of the training window.
"""

from __future__ import annotations

import pandas as pd

#: Disposition labels, ordered from strongest to weakest evidence.
DISPOSITIONS = (
    "corroborated_history",
    "same_issuer_different_issue",
    "reorganisation",
    "unresolved",
)


def _issuer(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().str[:6]


def build_permno_cusip_history(secnames: pd.DataFrame) -> dict[int, set[str]]:
    """
    Map each PERMNO to every 8-character CUSIP CRSP ever associates with it.

    Parameters
    ----------
    secnames : pd.DataFrame
        CRSP security names with 'permno' and 'ncusip'.

    Returns
    -------
    dict[int, set[str]]
    """
    sn = secnames[["permno", "ncusip"]].copy()
    sn["permno"] = pd.to_numeric(sn["permno"], errors="coerce")
    sn = sn.dropna(subset=["permno"])
    sn["permno"] = sn["permno"].astype("int64")
    sn["c8"] = sn["ncusip"].fillna("").astype(str).str.strip().str.upper().str[:8]
    sn = sn[sn["c8"].ne("")]
    return sn.groupby("permno")["c8"].apply(set).to_dict()


def classify_cusip_mismatches(
    mismatches: pd.DataFrame,
    secnames: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign every flagged mismatch one of the protocol §5 dispositions.

    Rules, applied in order:

    ``corroborated_history``
        The Compustat CUSIP appears elsewhere in the SAME PERMNO's CRSP name
        history. CCM and CUSIP agree on identity and disagree only on vintage.
    ``same_issuer_different_issue``
        First 6 characters (the issuer code) are equal — share class or issue
        number differs, not the firm.
    ``reorganisation``
        Issuer codes differ AND the PERMNO's CRSP history contains at least two
        distinct issuers, i.e. CRSP itself records an issuer reassignment.
    ``unresolved``
        None of the above. Retained in the sample, reported, and counted.

    Parameters
    ----------
    mismatches : pd.DataFrame
        Rows flagged by ``validate_cusip`` (needs permno, cusip8_comp, ncusip8).
    secnames : pd.DataFrame
        CRSP security names.

    Returns
    -------
    pd.DataFrame
        ``mismatches`` plus a 'disposition' column.
    """
    if len(mismatches) == 0:
        out = mismatches.copy()
        out["disposition"] = pd.Series(dtype="object")
        return out

    required = {"permno", "cusip8_comp", "ncusip8"}
    missing = required - set(mismatches.columns)
    if missing:
        raise KeyError(f"mismatches frame lacks {sorted(missing)}")

    hist = build_permno_cusip_history(secnames)
    issuers = {p: {c[:6] for c in cs} for p, cs in hist.items()}

    out = mismatches.copy()
    permno = pd.to_numeric(out["permno"], errors="coerce")

    comp = out["cusip8_comp"].fillna("").astype(str).str.strip().str.upper()
    crsp = out["ncusip8"].fillna("").astype(str).str.strip().str.upper()

    corroborated = [
        (c in hist.get(p, set())) if pd.notna(p) else False
        for c, p in zip(comp, permno)
    ]
    same_issuer = _issuer(comp).values == _issuer(crsp).values
    multi_issuer = [
        (len(issuers.get(p, set())) >= 2) if pd.notna(p) else False
        for p in permno
    ]

    disposition = []
    for corr, same, multi in zip(corroborated, same_issuer, multi_issuer):
        if corr:
            disposition.append("corroborated_history")
        elif same:
            disposition.append("same_issuer_different_issue")
        elif multi:
            disposition.append("reorganisation")
        else:
            disposition.append("unresolved")
    out["disposition"] = disposition
    return out


def summarise_dispositions(classified: pd.DataFrame) -> pd.DataFrame:
    """Counts and shares per disposition, in the canonical order."""
    counts = (
        classified["disposition"]
        .value_counts()
        .reindex(list(DISPOSITIONS))
        .fillna(0)
        .astype(int)
    )
    total = int(counts.sum())
    return pd.DataFrame({
        "disposition": counts.index,
        "firm_years": counts.values,
        "share": (counts.values / total).round(4) if total else 0.0,
    })


def assert_all_dispositioned(classified: pd.DataFrame) -> None:
    """
    Protocol §19.3 abort condition: every flagged mismatch must carry a
    disposition drawn from :data:`DISPOSITIONS`.

    ``unresolved`` is a *documented* disposition and does not abort — it is the
    honest residual that must be reported. A missing or unrecognised label does.
    """
    if len(classified) == 0:
        return
    if "disposition" not in classified.columns:
        raise ValueError(
            "CUSIP mismatches carry no 'disposition' column — run "
            "classify_cusip_mismatches() before the panel is used (protocol §19.3)."
        )
    bad = classified.loc[
        ~classified["disposition"].isin(list(DISPOSITIONS))
        | classified["disposition"].isna()
    ]
    if len(bad):
        raise ValueError(
            f"{len(bad):,} CUSIP mismatches lack a documented disposition "
            f"(protocol §19.3). Offending labels: "
            f"{sorted(set(bad['disposition'].dropna().unique()))[:5]}"
        )
