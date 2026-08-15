"""
Regression tests for the corrected H2 training-side design.

The defect these guard against is specific and was silent: the predecessor
design (``h2_leakage_sensitivity.chronological_test_contaminated_train``) was
built to contaminate the training sample with future observations, but excluded
every firm owning a post-cutoff row, so its training pool contained none. The
experiment reported "no effect" because it applied no treatment. The first test
below is the assertion whose absence allowed that to go unnoticed for months.

These tests use a synthetic panel: the arm-construction logic is pure and does
not need the real data, and keeping it synthetic means the suite stays fast and
runs without the processed parquets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.supp_h2_training_channel import (ARMS, EVAL_RANGE,
                                                   FUTURE_MIN, SEEDED_ARMS,
                                                   TRAIN_MAX, VAL_RANGE,
                                                   _subsample, build_arm)


def _panel(n_firms: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    A synthetic panel spanning every block.

    Three firm types are represented deliberately, because the design turns on
    the distinction between them: long-lived firms that straddle the evaluation
    and future blocks (these populate ``R_same``), firms that exit before the
    evaluation block, and **late entrants that first appear in FY2020 or later**
    (these populate ``R_diff``). Omitting the third type leaves ``R_diff`` empty
    and silently disables the firm-disjoint and size-matched arms.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_firms):
        if g % 4 == 3:                      # late entrant -> R_diff
            start, end = 2020 + (g % 3), 2023
        else:                               # straddler or early exit
            start = 1995 + (g % 25)
            end = min(2023, start + 5 + (g % 20))
        for fy in range(start, end + 1):
            rows.append({"gvkey": g, "fyear": fy,
                         "distress": int(rng.random() < 0.03)})
    return pd.DataFrame(rows)


def test_synthetic_panel_exercises_all_three_firm_types():
    """Guard the fixture itself: an empty R_diff would vacuously pass the arms."""
    b = _blocks(_panel())
    assert len(b["R_same"]) > 0
    assert len(b["R_diff"]) > 0
    assert len(b["T"]) > 0 and len(b["V"]) > 0 and len(b["E"]) > 0


def _blocks(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    b = {
        "T": panel[panel.fyear <= TRAIN_MAX],
        "V": panel[panel.fyear.between(*VAL_RANGE)],
        "E": panel[panel.fyear.between(*EVAL_RANGE)],
        "R": panel[panel.fyear >= FUTURE_MIN],
    }
    ef = set(b["E"].gvkey.unique())
    b["R_same"] = b["R"][b["R"].gvkey.isin(ef)]
    b["R_diff"] = b["R"][~b["R"].gvkey.isin(ef)]
    return {k: v.copy() for k, v in b.items()}


# ---------------------------------------------------------------------------
# The assertion the predecessor design would have failed
# ---------------------------------------------------------------------------

def test_contaminated_arms_actually_contain_future_rows():
    """Every arm that claims to add future data must in fact add some.

    This is the test the old design would have failed: its training pool held
    zero post-cutoff rows, so the treatment was never applied.
    """
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    for arm in ["future_all", "future_firm_disjoint", "future_matched"]:
        train = build_arm(b, arm, n_add, seed=0)
        n_future = (train.fyear >= FUTURE_MIN).sum()
        assert n_future > 0, f"arm {arm!r} claims contamination but adds none"


def test_baseline_and_recent_past_arms_contain_no_future_rows():
    """The two control arms must be clean of the future block."""
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    for arm in ["chronological", "recent_past_matched"]:
        train = build_arm(b, arm, n_add, seed=0)
        assert (train.fyear >= FUTURE_MIN).sum() == 0


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------

def test_no_arm_contains_an_evaluation_row():
    """Training on an evaluation row would be trivial leakage, not the channel
    under study. Every arm must be disjoint from the evaluation block."""
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    for arm in ARMS:
        train = build_arm(b, arm, n_add, seed=0)
        assert not train.fyear.between(*EVAL_RANGE).any(), arm


def test_firm_disjoint_arm_excludes_every_evaluated_firm():
    """The firm-disjoint arm may add future rows only from unevaluated firms."""
    b = _blocks(_panel())
    train = build_arm(b, "future_firm_disjoint", len(b["R_diff"]), seed=0)
    eval_firms = set(b["E"].gvkey.unique())
    added = train[train.fyear >= FUTURE_MIN]
    assert not added.gvkey.isin(eval_firms).any()
    # ... and it must still be a non-trivial addition.
    assert len(added) > 0


def test_matched_arms_add_the_same_number_of_rows():
    """The size control only works if the two matched arms are equal in size."""
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    base = len(build_arm(b, "chronological", n_add, seed=0))
    fut = len(build_arm(b, "future_matched", n_add, seed=0))
    past = len(build_arm(b, "recent_past_matched", n_add, seed=0))
    assert fut == past == base + n_add


def test_all_arms_share_the_base_training_block():
    """Arms differ only by their addition; the base block is common."""
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    base_rows = len(b["T"])
    for arm in ARMS:
        train = build_arm(b, arm, n_add, seed=0)
        assert (train.fyear <= TRAIN_MAX).sum() == base_rows, arm


# ---------------------------------------------------------------------------
# Subsampling behaviour
# ---------------------------------------------------------------------------

def test_seeded_arms_vary_with_seed_and_unseeded_arms_do_not():
    b = _blocks(_panel())
    n_add = len(b["R_diff"])
    for arm in SEEDED_ARMS:
        a = build_arm(b, arm, n_add, seed=0)
        c = build_arm(b, arm, n_add, seed=1)
        assert len(a) == len(c)
        assert not a.reset_index(drop=True).equals(c.reset_index(drop=True))
    for arm in set(ARMS) - SEEDED_ARMS:
        a = build_arm(b, arm, n_add, seed=0)
        c = build_arm(b, arm, n_add, seed=1)
        pd.testing.assert_frame_equal(a.reset_index(drop=True),
                                      c.reset_index(drop=True))


def test_subsample_is_reproducible_and_without_replacement():
    df = pd.DataFrame({"x": range(500)})
    a = _subsample(df, 100, seed=7)
    assert len(a) == 100
    assert a.x.is_unique
    pd.testing.assert_frame_equal(a, _subsample(df, 100, seed=7))


def test_subsample_returns_everything_when_pool_is_too_small():
    df = pd.DataFrame({"x": range(10)})
    assert len(_subsample(df, 50, seed=0)) == 10


def test_unknown_arm_raises():
    b = _blocks(_panel())
    with pytest.raises(ValueError):
        build_arm(b, "not_an_arm", 10, seed=0)
