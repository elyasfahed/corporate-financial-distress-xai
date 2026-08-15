"""Audit the CRSP CIZ delisting definition before any model is fitted.

Run from the PyCharm terminal:

    python -m scripts.audit_distress_definition

This command reads the original StkDelists RDS, preserves all four CIZ
classification fields, and writes neutral publication/audit filenames.  It
does not build features, tune models, or inspect test performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ROOT, V2_PROFILE
from src.data.distress_definition import (
    BANKRUPTCY_EVENT_COL,
    BROAD_EVENT_COL,
    NARROW_EVENT_COL,
    PRIMARY_EVENT_COL,
    NARROW_FINANCIAL_REASONS,
    PERFORMANCE_LEGACY_CODES,
    definition_summary,
)
from src.data.load_local_rds import read_stkdelists


def main() -> None:
    print("=" * 72)
    print("  CRSP DELISTING-DEFINITION AUDIT")
    print("=" * 72)
    delist = read_stkdelists(mapping="academic")

    out_dir = ROOT / "outputs" / "tables" / "data_validation" / V2_PROFILE["spec"]
    out_dir.mkdir(parents=True, exist_ok=True)

    event_path = out_dir / "delisting_events_classified.csv"
    summary_path = out_dir / "delisting_definition_audit.csv"
    delist.to_csv(event_path, index=False)
    summary = definition_summary(delist)
    summary.to_csv(summary_path, index=False)

    print("\nFinal-primary performance-delisting legacy-code rule:")
    print("  400--490, 500, and 520--584")
    print(f"  Explicit code count: {len(PERFORMANCE_LEGACY_CODES)}")
    print("\nNarrow supplementary GDR reasons:")
    print("  " + ", ".join(sorted(NARROW_FINANCIAL_REASONS)))
    print("\nEvent counts in the complete delisting file:")
    print(f"  Final primary         : {int(delist[PRIMARY_EVENT_COL].sum()):,}")
    print(f"  Narrow sensitivity    : {int(delist[NARROW_EVENT_COL].sum()):,}")
    print(f"  Bankruptcy proxy      : {int(delist[BANKRUPTCY_EVENT_COL].sum()):,}")
    print(f"  Broad sensitivity     : {int(delist[BROAD_EVENT_COL].sum()):,}")
    print(f"\nDetailed events -> {event_path}")
    print(f"Definition audit -> {summary_path}")
    print("\nSTOP HERE. Review the audit and freeze the primary reason list before "
          "rebuilding data or fitting models.")


if __name__ == "__main__":
    main()
