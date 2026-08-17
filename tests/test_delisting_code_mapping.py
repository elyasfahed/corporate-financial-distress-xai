"""
Regression tests for the CIZ delisting-code synthesis.
=======================================================
The CRSP CIZ v2 delisting file has no numeric DLSTCD; the pipeline
synthesizes one from DelActionType/DelReasonType strings
(src/data/load_local_rds.synthesize_dlstcd). The frozen mapping placed
bankruptcies at 572 (outside the primary 400--499 label range) and let
liquidations ('GLI') fall through to the 500 default -- so the primary
distress label excluded exactly the two least ambiguous distress classes,
and RC1 (bankruptcy-only) was DISJOINT from the primary label instead of
the nested subset in the pre-specified design. See Implementation Status
paragraph 17.

These tests pin the frozen mapping byte-for-byte and verify that the
corrected mapping (GLI -> 400, GDR -> 450, BKPY -> 470) repairs both
defects without changing any other assignment.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    DISTRESS_CODES_PRIMARY,
    DISTRESS_CODES_RC1,
    DISTRESS_CODES_RC1_CORRECTED,
)
from src.data.load_local_rds import synthesize_dlstcd


def _frame():
    """One row per relevant (action, reason) class observed in StkDelists."""
    return pd.DataFrame({
        "DelActionType": ["GDR",  "GDR", "GDR",  "GLI",  "GEX",  "MER",  "LOS"],
        "DelReasonType": ["BKPY", "LP",  "CORQ", "UNAV", "UNAV", "UNAV", "UNAV"],
    })


# ---------------------------------------------------------------------------
# Test 1 -- frozen mapping reproduces the original construction exactly
# ---------------------------------------------------------------------------

def test_frozen_mapping_reproduces_original():
    codes = synthesize_dlstcd(_frame(), mapping="frozen").tolist()
    #        BKPY GDR-LP GDR-CORQ GLI  GEX  MER  LOS
    assert codes == [572, 450, 450, 500, 500, 500, 500]


def test_frozen_mapping_excludes_bankruptcy_and_liquidation_from_primary():
    """Documents the defect: under the frozen mapping the primary 400-499
    label contains neither bankruptcies (572) nor liquidations (500)."""
    codes = synthesize_dlstcd(_frame(), mapping="frozen")
    bkpy = codes.iloc[0]   # GDR + BKPY row
    gli = codes.iloc[3]    # GLI row
    assert bkpy not in DISTRESS_CODES_PRIMARY
    assert gli not in DISTRESS_CODES_PRIMARY


# ---------------------------------------------------------------------------
# Test 2 -- corrected mapping: liquidation and bankruptcy inside 400-499
# ---------------------------------------------------------------------------

def test_corrected_mapping_codes():
    codes = synthesize_dlstcd(_frame(), mapping="corrected").tolist()
    #        BKPY GDR-LP GDR-CORQ GLI  GEX  MER  LOS
    assert codes == [470, 450, 450, 400, 500, 500, 500]


def test_corrected_mapping_primary_label_captures_all_distress_classes():
    codes = synthesize_dlstcd(_frame(), mapping="corrected")
    assert codes.iloc[0] in DISTRESS_CODES_PRIMARY   # bankruptcy
    assert codes.iloc[1] in DISTRESS_CODES_PRIMARY   # exchange drop
    assert codes.iloc[3] in DISTRESS_CODES_PRIMARY   # liquidation
    # non-distress exits stay outside the primary range
    assert codes.iloc[4] not in DISTRESS_CODES_PRIMARY  # went to other exchange
    assert codes.iloc[5] not in DISTRESS_CODES_PRIMARY  # merger


# ---------------------------------------------------------------------------
# Test 3 -- RC1 nesting: bankruptcy subset of the primary label
# ---------------------------------------------------------------------------

def test_rc1_nested_under_corrected_mapping():
    assert all(c in DISTRESS_CODES_PRIMARY for c in DISTRESS_CODES_RC1_CORRECTED), \
        "corrected bankruptcy code(s) must lie inside the primary 400-499 range"


def test_rc1_disjoint_under_frozen_mapping():
    """Documents the defect the correction fixes: the frozen RC1 codes
    (572/574/584) cannot be a subset of the 400-499 primary range."""
    assert not any(c in DISTRESS_CODES_PRIMARY for c in DISTRESS_CODES_RC1)


# ---------------------------------------------------------------------------
# Test 4 -- bankruptcy overwrite precedence holds under both mappings
# ---------------------------------------------------------------------------

def test_bkpy_overwrites_gdr():
    df = pd.DataFrame({"DelActionType": ["GDR"], "DelReasonType": ["BKPY"]})
    assert synthesize_dlstcd(df, mapping="frozen").iloc[0] == 572
    assert synthesize_dlstcd(df, mapping="corrected").iloc[0] == 470


# ---------------------------------------------------------------------------
# Test 5 -- invalid mapping raises
# ---------------------------------------------------------------------------

def test_invalid_mapping_raises():
    with pytest.raises(ValueError):
        synthesize_dlstcd(_frame(), mapping="bogus")
