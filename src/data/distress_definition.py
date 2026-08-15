"""Academically explicit CRSP CIZ delisting definitions.

CRSP CIZ does not provide the old numeric ``DLSTCD`` as a single field.
Instead it unpacks the information into action, reason, status, and payment
fields.  A ``GDR`` action means *Dropped*; it is not itself a bankruptcy or
financial-distress code.  The reason field determines why the security was
dropped.

The final-primary label follows the performance-delisting convention used
in the CRSP literature: all liquidation codes (400--490), code 500, and
codes 520--584.  Because CIZ stores the components rather than ``DLSTCD``,
the legacy code is first reconstructed from the official CRSP crosswalk and
then tested against that pre-specified set.

The narrower five-reason definition used by the July 2026 candidate rebuild
is retained as an explicitly supplementary sensitivity label.  It is not
silently discarded and is never called the primary outcome.

The bankruptcy robustness label uses ``BKPY`` only.  It is described as a
CRSP bankruptcy-reason proxy, not as a verified Chapter 7/11 court filing.
The broad label includes every GDR event and is retained only as a noisy
sensitivity definition.

Source: CRSP's official legacy-delisting-code to CIZ crosswalk
https://www.crsp.org/wp-content/uploads/DelistCode.html
"""

from __future__ import annotations

import pandas as pd


NARROW_FINANCIAL_REASONS = frozenset({
    "BKPY",  # Bankruptcy
    "FING",  # Financial Guidelines
    "INSC",  # Insufficient Capital
    "EQRQ",  # Equity Requirements
    "LP",    # Low Price
})

PERFORMANCE_LEGACY_CODES = frozenset(
    list(range(400, 500)) + [500] + list(range(520, 585))
)

# Clearly voluntary / administrative GDR reasons that are NOT financial
# distress: company-request delistings and pure venue changes. The cleaned
# PRIMARY label excludes them; the broad performance label (which retains
# them) is kept as a documented sensitivity. Reason mnemonics follow CRSP's
# published delisting-code crosswalk and were pinned against the CIZ
# MetaFlagInfo dictionary for the final-primary run.
VOLUNTARY_ADMIN_REASONS = frozenset({
    "CORQ",  # "Company Request" — voluntary (often going-private / M&A-adjacent)
    "MVOT",  # "Moved to OTC" — a venue change, not a failure
})
# Their exact reconstructed legacy codes (MVOT=520, CORQ=570). No GLI
# liquidation status maps into this set, so a firm-year is a voluntary /
# administrative exit iff its reconstructed code is one of these — which makes
# the code-level and reason-level exclusions exactly equivalent.
#
# NOTE — MTMK (legacy 550) is deliberately NOT in this set. It was initially
# read as "moved to market" (a venue change) and excluded; pinning the mnemonics
# against CRSP's official MetaFlagInfo dictionary showed its definition is
# "Market Makers", i.e. a failure to maintain the required number of market
# makers. That is an exchange listing-standard failure of the same kind as SHLD
# ("Shareholders"), which the primary label retains, so MTMK belongs in the
# primary label too. Corrected 2026-07-24 after the dictionary pin.
VOLUNTARY_ADMIN_LEGACY_CODES = frozenset({520, 570})

PRIMARY_EVENT_COL = "is_distress_performance"          # broad performance (sensitivity)
CORE_PRIMARY_EVENT_COL = "is_distress_performance_core"  # cleaned PRIMARY (excl. voluntary)
NARROW_EVENT_COL = "is_distress_narrow_financial"
BANKRUPTCY_EVENT_COL = "is_bankruptcy_proxy"
BROAD_EVENT_COL = "is_broad_for_cause"

# Backward-compatible import name for older audit scripts.  New code should
# use NARROW_FINANCIAL_REASONS so the estimand is explicit.
PRIMARY_FINANCIAL_REASONS = NARROW_FINANCIAL_REASONS


# Exact reason-to-code rows for DelActionType == GDR in CRSP's crosswalk.
GDR_REASON_TO_LEGACY_CODE = {
    "UNAV": 500,
    "MVNM": 502,
    "MVMF": 505,
    "MVB": 510,
    "MVCHI": 513,
    "MVMO": 514,
    "MVPAC": 516,
    "MVPH": 517,
    "MVTO": 519,
    "MVOT": 520,
    "MTMK": 550,
    "SHLD": 551,
    "LP": 552,
    "INSC": 560,
    "INSF": 561,
    "CORQ": 570,
    "DERE": 573,
    "BKPY": 574,
    "OFFRE": 575,
    "DELQ": 580,
    "FARG": 581,
    "EQRQ": 582,
    "DEEX": 583,
    "FING": 584,
    "PUBI": 585,
    "VIO": 587,
    "FDCV": 588,
    "UNL": 589,
    "SERQ": 591,
}

# For liquidation rows the completion-status field distinguishes the old
# 400-series code.  This is used for audit/provenance, not to define the
# direct CIZ label.
GLI_STATUS_TO_LEGACY_CODE = {
    "UNAV": 400,
    "FPAY": 450,
    "NFC": 460,
    "NFP": 470,
    "NDP": 480,
    "NDC": 490,
}


def _normalise_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[column].astype("string").str.strip().str.upper()


def reconstruct_legacy_dlstcd(df: pd.DataFrame) -> pd.Series:
    """Reconstruct the legacy code for the GLI/GDR rows used in the study.

    Unknown or non-GLI/GDR combinations remain missing.  They must never be
    silently assigned a generic code, because that would mix unrelated exit
    mechanisms into the outcome definition.
    """
    action = _normalise_flag(df, "DelActionType")
    reason = _normalise_flag(df, "DelReasonType")
    status = _normalise_flag(df, "DelStatusType")

    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    gdr = action.eq("GDR")
    mapped_gdr = reason.map(GDR_REASON_TO_LEGACY_CODE).astype("Int64")
    out.loc[gdr] = mapped_gdr.loc[gdr]

    gli = action.eq("GLI")
    mapped_gli = status.map(GLI_STATUS_TO_LEGACY_CODE).astype("Int64")
    out.loc[gli] = mapped_gli.loc[gli]
    return out


def classify_delisting_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add direct CIZ outcome flags and an auditable reconstructed code."""
    out = df.copy()
    action = _normalise_flag(out, "DelActionType")
    reason = _normalise_flag(out, "DelReasonType")
    legacy_code = reconstruct_legacy_dlstcd(out)

    out[PRIMARY_EVENT_COL] = legacy_code.isin(PERFORMANCE_LEGACY_CODES).astype("int8")
    # Cleaned primary: performance delisting AND not a voluntary/administrative
    # GDR exit (company request or venue move).
    out[CORE_PRIMARY_EVENT_COL] = (
        legacy_code.isin(PERFORMANCE_LEGACY_CODES)
        & ~(action.eq("GDR") & reason.isin(VOLUNTARY_ADMIN_REASONS))
    ).astype("int8")
    out[NARROW_EVENT_COL] = (
        action.eq("GLI")
        | (action.eq("GDR") & reason.isin(NARROW_FINANCIAL_REASONS))
    ).astype("int8")
    out[BANKRUPTCY_EVENT_COL] = (
        action.eq("GDR") & reason.eq("BKPY")
    ).astype("int8")
    out[BROAD_EVENT_COL] = action.isin(["GLI", "GDR"]).astype("int8")
    out["dlstcd_reconstructed"] = legacy_code
    return out


def definition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact action/reason audit with all three label counts."""
    labeled = classify_delisting_events(df)
    for col in ("DelActionType", "DelReasonType", "DelStatusType",
                "DelPaymentType"):
        if col not in labeled:
            labeled[col] = pd.NA
    return (
        labeled.groupby(
            ["DelActionType", "DelReasonType", "DelStatusType",
             "DelPaymentType", "dlstcd_reconstructed"],
            dropna=False,
        )
        .agg(
            n_events=("DelActionType", "size"),
            n_primary=(PRIMARY_EVENT_COL, "sum"),
            n_narrow=(NARROW_EVENT_COL, "sum"),
            n_bankruptcy_proxy=(BANKRUPTCY_EVENT_COL, "sum"),
            n_broad=(BROAD_EVENT_COL, "sum"),
        )
        .reset_index()
        .sort_values(["n_primary", "n_events"], ascending=False)
        .reset_index(drop=True)
    )
