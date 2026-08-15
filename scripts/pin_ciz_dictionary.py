"""
Pin the CRSP CIZ delisting mnemonics to the official flag-value dictionary.
============================================================================
The distress label is built from the CIZ ``DelActionType`` / ``DelReasonType``
mnemonics (GDR, GLI, CORQ, MTMK, MVOT, BKPY, ...). The local data drop contains
two of the three CIZ metadata tables -- ``MetaItemInfo`` (item/field
definitions) and ``MetaColumnInfo`` (column layout) -- but NOT the flag-value
dictionary that maps a coded value to its official description. Until that
dictionary is supplied, the mnemonic interpretations rest on CRSP's published
delisting-code crosswalk rather than on CRSP's own value table.

``MetaItemInfo`` pins the FIELDS and gives the flag-type keys needed to request
the values:

    DelActionType  "Delisting Corporate Action Type"   FlagType = CD
    DelReasonType  "Delisting Reason Type"             FlagType = DR
    DelStatusType  "Delisting Completion Status Type"  FlagType = CS
    DelPaymentType "Delisting Payment Summary Type"    FlagType = PT

WHAT TO OBTAIN
--------------
The CIZ flag-value dictionary, usually delivered as ``MetaFlagInfo`` alongside
the two Meta tables already present. Any of these works:

  * ``data/raw/crsp/MetaFlagInfo.rds``  (same export that produced the other
    two Meta tables -- easiest route, since that source clearly had access)
  * ``data/raw/crsp/MetaFlagInfo.csv``  (e.g. exported from WRDS, or typed up
    from the CRSP Data Descriptions Guide appendix)

It only needs three usable columns: a flag-type column, a flag-value column and
a description column. Column names are auto-detected.

WHAT THIS SCRIPT DOES
---------------------
1. Loads the dictionary and restricts it to the delisting flag types.
2. Cross-checks EVERY action/reason mnemonic that actually occurs in the
   delisting extract against the dictionary, and FAILS LOUDLY on any mnemonic
   the dictionary does not define (that is exactly the exposure being closed).
3. Writes an appendix table mapping each mnemonic to its official CRSP
   description, its reconstructed legacy code, its event count, and the
   classification this thesis assigns it.

Run:
    PYTHONUTF8=1 ./.venv311/Scripts/python.exe -m scripts.pin_ciz_dictionary
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DATA_RAW_V2, OUT_TABLES_DESCRIPTIVE, V2_PROFILE
from src.data.distress_definition import (
    NARROW_FINANCIAL_REASONS,
    PERFORMANCE_LEGACY_CODES,
    VOLUNTARY_ADMIN_REASONS,
    reconstruct_legacy_dlstcd,
)
from src.utils.tables import save_table

RAW_CRSP = ROOT / "data" / "raw" / "crsp"
DELIST = DATA_RAW_V2 / "crsp_delisting_raw.parquet"
DELIST_FLAG_TYPES = {"CD": "DelActionType", "DR": "DelReasonType",
                     "CS": "DelStatusType", "PT": "DelPaymentType"}


def _load_dictionary() -> pd.DataFrame | None:
    """Load MetaFlagInfo from .rds or .csv; return None if absent."""
    rds, csv = RAW_CRSP / "MetaFlagInfo.rds", RAW_CRSP / "MetaFlagInfo.csv"
    if rds.exists():
        import pyreadr
        df = list(pyreadr.read_r(str(rds)).values())[0]
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return None

    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            for lc, orig in cols.items():
                if c in lc:
                    return orig
        return None
    c_type = pick("flagtype", "type")
    c_val  = pick("flagvalue", "value", "code")
    c_desc = pick("flagdesc", "desc", "definition", "label")
    if not all([c_type, c_val, c_desc]):
        raise SystemExit(
            f"Could not identify flag-type/value/description columns in "
            f"{rds if rds.exists() else csv}. Columns found: {list(df.columns)}"
        )
    out = df[[c_type, c_val, c_desc]].copy()
    out.columns = ["flag_type", "flag_value", "official_description"]
    for c in out.columns:
        out[c] = out[c].astype("string").str.strip()
    return out


def _classify(action: str, reason: str, code) -> str:
    """The classification this thesis assigns to an (action, reason) pair."""
    in_perf = pd.notna(code) and int(code) in PERFORMANCE_LEGACY_CODES
    if not in_perf:
        return "--"
    if action == "GDR" and reason in VOLUNTARY_ADMIN_REASONS:
        return "excl. (vol.)"
    labels = ["P"]
    if action == "GLI" or (action == "GDR" and reason in NARROW_FINANCIAL_REASONS):
        labels.append("N")
    if action == "GDR" and reason == "BKPY":
        labels.append("B")
    return "+".join(labels)


def main() -> None:
    dic = _load_dictionary()
    pinned = dic is not None
    if pinned:
        dic = dic[dic["flag_type"].isin(DELIST_FLAG_TYPES)]
        print(f"Dictionary loaded: {len(dic)} delisting flag values "
              f"({sorted(dic['flag_type'].unique())})")
    else:
        print("NOTE: MetaFlagInfo not found in "
              f"{RAW_CRSP} — writing the PRELIMINARY table (mnemonics, "
              "reconstructed codes, counts and this thesis's classification) "
              "without CRSP's official descriptions.\n"
              "      Supply MetaFlagInfo.{rds,csv} and re-run to pin the "
              "mnemonics and add the official wording.")

    dl = pd.read_parquet(DELIST)
    for c in ("DelActionType", "DelReasonType"):
        dl[c] = dl[c].astype("string").str.strip().str.upper()
    dl["legacy"] = reconstruct_legacy_dlstcd(dl)

    grp = (dl.groupby(["DelActionType", "DelReasonType", "legacy"], dropna=False)
             .size().reset_index(name="n_events"))

    grp["thesis_classification"] = [
        _classify(a, r, c) for a, r, c in
        zip(grp["DelActionType"], grp["DelReasonType"], grp["legacy"])
    ]

    cols = ["action", "reason", "legacy_code", "n_events", "thesis_classification"]
    if pinned:
        act = dic[dic["flag_type"] == "CD"].set_index("flag_value")["official_description"]
        rsn = dic[dic["flag_type"] == "DR"].set_index("flag_value")["official_description"]
        grp["action_official"] = grp["DelActionType"].map(act)
        grp["reason_official"] = grp["DelReasonType"].map(rsn)

        # --- the actual pin: every mnemonic in use must be defined ---------
        missing_a = sorted(set(grp.loc[grp["action_official"].isna(),
                                       "DelActionType"].dropna()))
        missing_r = sorted(set(grp.loc[grp["reason_official"].isna(),
                                       "DelReasonType"].dropna()))
        if missing_a or missing_r:
            print("\n*** PIN FAILED — mnemonics present in the data but absent "
                  "from the dictionary ***")
            if missing_a: print("  DelActionType:", missing_a)
            if missing_r: print("  DelReasonType:", missing_r)
            raise SystemExit(
                "Resolve these before citing the dictionary as authoritative.")
        print("\nPIN OK — every action/reason mnemonic in the delisting extract "
              "is defined by the official CRSP dictionary.")
        cols = ["action", "action_official", "reason", "reason_official",
                "legacy_code", "n_events", "thesis_classification"]

    # integer legacy codes ("--" where the action carries none), then
    # human-readable column headers for the appendix table
    code = pd.to_numeric(grp["legacy"], errors="coerce")
    grp["legacy"] = [("--" if pd.isna(v) else str(int(v))) for v in code]
    out = (grp.sort_values("n_events", ascending=False)
              .rename(columns={"DelActionType": "action",
                               "DelReasonType": "reason",
                               "legacy": "legacy_code"})[cols])
    out = out.rename(columns={
        "action": "Action", "reason": "Reason",
        "action_official": "CRSP action description",
        "reason_official": "CRSP reason description",
        "legacy_code": "Legacy code", "n_events": "Events",
        "thesis_classification": "Label"})

    caption = (
        "CRSP CIZ delisting corporate-action and reason codes occurring in the "
        "sample, with the reconstructed legacy delisting code, the number of "
        "delisting events, and the label each combination feeds. Label key: "
        "\\textbf{P} $=$ primary (cleaned performance) label; \\textbf{N} $=$ "
        "also in the narrow five-reason financial sensitivity; \\textbf{B} $=$ "
        "also the RC$_1$ bankruptcy-reason proxy; \\textit{excl.\\ (vol.)} $=$ "
        "inside the performance family but excluded from the primary label as a "
        "voluntary or administrative exit; \\textit{--} $=$ never enters any "
        "label (mergers, exchange transfers, venue moves outside the "
        "performance range).")
    caption += (
        " Action and reason wording is CRSP's own, taken from the CIZ metadata "
        "dictionary (flag types CD and DR)." if pinned else
        " The mnemonic interpretations follow CRSP's published delisting-code "
        "crosswalk; confirmation against CRSP's own CIZ flag-value dictionary "
        "is outstanding and is recorded as a limitation in "
        "Chapter~\\ref{ch:conclusion}.")

    dest = OUT_TABLES_DESCRIPTIVE / V2_PROFILE["spec"] / "ciz_code_dictionary"
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_table(out, dest, caption=caption, label="tab:ciz_code_dictionary")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
