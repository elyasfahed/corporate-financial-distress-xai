from __future__ import annotations

import pandas as pd

from src.features.build_features import purge_outer_split_labels


def _frame(dates, distress):
    return pd.DataFrame({
        "fdate": pd.to_datetime(dates),
        "distress": distress,
        "fyear": range(2000, 2000 + len(dates)),
    })


def test_outer_boundaries_drop_only_labels_not_mature_at_next_origin():
    train = _frame(["2008-01-01", "2009-07-01"], [0, 1])
    val = _frame(["2010-03-01", "2014-07-01"], [0, 1])
    test = _frame(["2015-03-01"], [0])

    tr, va, te = purge_outer_split_labels(train, val, test, horizon_days=365)

    assert tr["fdate"].dt.strftime("%Y-%m-%d").tolist() == ["2008-01-01"]
    assert va["fdate"].dt.strftime("%Y-%m-%d").tolist() == ["2010-03-01"]
    pd.testing.assert_frame_equal(te, test)
