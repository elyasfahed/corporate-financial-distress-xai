"""
Regression test: each check's tuning folds must carry that check's own label.

The failure this guards against is silent and is the same class as the H2
defect --- a treatment named but not applied. Fold-safe cross-validation
consumes the RAW training frame, so if ``get_check_raw_train`` returned the
primary-labelled frame for RC1 or RC2, every hyperparameter search would
optimise against the *primary* outcome and then be evaluated on a re-labelled
test set. Nothing would error, the tables would look plausible, and the
reported per-check configurations would be answers to the wrong question.

These tests read the real split files, because the property under test is a
property of the label construction rather than of a pure function. They skip
rather than fail when the processed data is absent, so a clean checkout without
the parquets still runs the suite.
"""

from __future__ import annotations

import pytest

from src.config import DATA_SAMPLES_V2

pytestmark = pytest.mark.skipif(
    not (DATA_SAMPLES_V2 / "train_raw.parquet").exists(),
    reason="processed final_primary splits not present",
)


@pytest.fixture(scope="module")
def raw_events():
    """Distress-event count in each check's raw training frame."""
    from src.robustness.rc_nn_final_primary import get_check_raw_train
    return {c: int(get_check_raw_train(c)["distress"].sum())
            for c in ("primary", "rc1", "rc2", "rc3", "rc4")}


def test_rc1_tuning_frame_is_relabelled(raw_events):
    """RC1 restricts to the bankruptcy-reason proxy, a strict subset."""
    assert raw_events["rc1"] < raw_events["primary"], (
        "RC1's tuning folds carry the primary label — the search would "
        "optimise against the wrong outcome"
    )
    assert raw_events["rc1"] > 0


def test_rc2_tuning_frame_is_relabelled(raw_events):
    """RC2 halves the outcome window, so it must lose events."""
    assert raw_events["rc2"] < raw_events["primary"], (
        "RC2's tuning folds carry the 12-month label"
    )
    assert raw_events["rc2"] > 0


def test_checks_that_share_the_primary_label_are_untouched(raw_events):
    """RC3 and RC4 vary the imbalance treatment and the predictor set only."""
    assert raw_events["rc3"] == raw_events["primary"]
    assert raw_events["rc4"] == raw_events["primary"]


def test_rc1_is_nested_inside_rc2_and_primary(raw_events):
    """The bankruptcy proxy is the narrowest outcome of the three."""
    assert raw_events["rc1"] < raw_events["rc2"] < raw_events["primary"]


def test_rc4_changes_the_predictor_set_not_the_label():
    """RC4's distinguishing feature is the 11-predictor set."""
    from src.config import ACCOUNTING_FEATURES
    from src.robustness.rc_nn_final_primary import get_check_data

    _, _, _, feats = get_check_data("rc4")
    assert len(feats) == len(ACCOUNTING_FEATURES) == 11
    assert not any(f.startswith("MB_MISS") for f in feats)
