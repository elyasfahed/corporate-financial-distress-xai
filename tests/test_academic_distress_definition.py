from __future__ import annotations

import pandas as pd

from src.data.distress_definition import (
    BANKRUPTCY_EVENT_COL,
    BROAD_EVENT_COL,
    NARROW_EVENT_COL,
    PRIMARY_EVENT_COL,
    classify_delisting_events,
    reconstruct_legacy_dlstcd,
)
from src.data.merge_crsp_compustat import build_distress_label


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        "DelActionType": ["GDR", "GDR", "GDR", "GDR", "GLI", "MER"],
        "DelReasonType": ["BKPY", "FING", "LP", "MVOT", "UNAV", "UNAV"],
        "DelStatusType": ["VCL", "VCL", "VCL", "VCL", "UNAV", "FPAY"],
        "DelPaymentType": ["PRCF", "PRCF", "PRCF", "PRCF", "UNAV", "CASH"],
    })


def test_gdr_is_classified_by_reason_not_as_one_code():
    codes = reconstruct_legacy_dlstcd(_events())
    assert codes.iloc[0] == 574  # BKPY
    assert codes.iloc[1] == 584  # FING
    assert codes.iloc[2] == 552  # Low price
    assert codes.iloc[3] == 520  # Move to OTC, not 450


def test_primary_follows_performance_delisting_convention():
    out = classify_delisting_events(_events())
    # BKPY/FING/LP, move-to-OTC (520), and liquidation are all included.
    assert out[PRIMARY_EVENT_COL].tolist() == [1, 1, 1, 1, 1, 0]
    assert out[NARROW_EVENT_COL].tolist() == [1, 1, 1, 0, 1, 0]
    assert out[BANKRUPTCY_EVENT_COL].tolist() == [1, 0, 0, 0, 0, 0]
    assert (out[BANKRUPTCY_EVENT_COL] <= out[PRIMARY_EVENT_COL]).all()


def test_broad_definition_is_explicitly_wider():
    out = classify_delisting_events(_events())
    assert out[BROAD_EVENT_COL].tolist() == [1, 1, 1, 1, 1, 0]


def test_panel_label_uses_direct_ciz_event_flag():
    panel = pd.DataFrame({
        "gvkey": ["1", "2"],
        "fyear": [2020, 2020],
        "permno": [10001, 10002],
        "fdate": pd.to_datetime(["2021-03-01", "2021-03-01"]),
    })
    delist = pd.DataFrame({
        "permno": [10001, 10002],
        "dlstdt": pd.to_datetime(["2021-06-01", "2021-06-01"]),
        # Both bankruptcy and move-to-OTC are performance delistings.
        PRIMARY_EVENT_COL: [1, 0],
        "dlstcd_reconstructed": pd.Series([574, 520], dtype="Int64"),
    })
    labeled = build_distress_label(
        panel, delist, event_label_col=PRIMARY_EVENT_COL
    )
    assert labeled["distress"].tolist() == [1, 0]


def test_narrow_sensitivity_excludes_move_to_otc():
    panel = pd.DataFrame({
        "gvkey": ["1", "2"],
        "fyear": [2020, 2020],
        "permno": [10001, 10002],
        "fdate": pd.to_datetime(["2021-03-01", "2021-03-01"]),
    })
    delist = classify_delisting_events(_events().iloc[[0, 3]].copy())
    delist["permno"] = [10001, 10002]
    delist["dlstdt"] = pd.to_datetime(["2021-06-01", "2021-06-01"])
    labeled = build_distress_label(panel, delist, event_label_col=NARROW_EVENT_COL)
    assert labeled["distress"].tolist() == [1, 0]


def test_one_event_is_assigned_to_most_recent_eligible_filing():
    panel = pd.DataFrame({
        "gvkey": ["1", "1"],
        "fyear": [2019, 2020],
        "permno": [10001, 10001],
        "fdate": pd.to_datetime(["2020-04-01", "2021-03-15"]),
    })
    delist = pd.DataFrame({
        "permno": [10001],
        "dlstdt": pd.to_datetime(["2021-03-20"]),
        PRIMARY_EVENT_COL: [1],
        "dlstcd_reconstructed": pd.Series([574], dtype="Int64"),
    })
    overlapping = build_distress_label(
        panel, delist, event_label_col=PRIMARY_EVENT_COL,
        unique_event_assignment=False,
    )
    unique = build_distress_label(
        panel, delist, event_label_col=PRIMARY_EVENT_COL,
        unique_event_assignment=True,
    )
    assert overlapping["distress"].tolist() == [1, 1]
    assert unique["distress"].tolist() == [0, 1]
