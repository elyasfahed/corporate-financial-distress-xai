from __future__ import annotations

import pandas as pd

from src.data.load_local_rds import _SN_NCUSIP, _pick
from src.data.merge_crsp_compustat import (
    AttritionTracker,
    merge_ccm_primary,
    validate_cusip,
)


def test_ccm_tie_break_prefers_primary_link_over_consolidated_link():
    comp = pd.DataFrame({
        "gvkey": ["1"],
        "fyear": [2020],
        "datadate": pd.to_datetime(["2020-12-31"]),
    })
    ccm = pd.DataFrame({
        "gvkey": ["1", "1"],
        "permno": [11111, 22222],
        "permco": [111, 222],
        "linktype": ["LC", "LC"],
        "linkprim": ["C", "P"],
        "linkdt": pd.to_datetime(["2010-01-01", "2010-01-01"]),
        "linkenddt": pd.to_datetime(["2030-12-31", "2030-12-31"]),
    })
    out = merge_ccm_primary(comp, ccm, AttritionTracker())
    assert out.loc[0, "permno"] == 22222
    assert out.loc[0, "linkprim"] == "P"


def test_cusip_validation_distinguishes_match_mismatch_and_unavailable():
    panel = pd.DataFrame({
        "gvkey": ["1", "2", "3"],
        "fyear": [2020, 2020, 2020],
        "permno": [11111, 22222, 33333],
        "datadate": pd.to_datetime(["2020-12-31"] * 3),
        "cusip": ["12345678", "87654321", pd.NA],
    })
    names = pd.DataFrame({
        "permno": [11111, 22222, 33333],
        "ncusip": ["12345678", "00000000", "99999999"],
        "namedt": pd.to_datetime(["2010-01-01"] * 3),
        "nameendt": pd.to_datetime(["2030-12-31"] * 3),
    })
    out, mismatches = validate_cusip(panel, names)
    assert out["cusip_check_available"].tolist() == [True, True, False]
    assert out["cusip_match"].tolist() == [True, False, False]
    assert out["cusip_mismatch"].tolist() == [False, True, False]
    assert mismatches[["gvkey", "fyear"]].values.tolist() == [["2", 2020]]


def test_crsp_cusip_resolution_prefers_historical_value_over_header_value():
    frame = pd.DataFrame({
        "HdrCUSIP": ["CURRENT1"],
        "CUSIP": ["HISTORY1"],
    })
    assert _pick(frame, _SN_NCUSIP) == "CUSIP"
