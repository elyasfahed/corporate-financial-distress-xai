"""
Regression tests — eligible-universe market aggregates (§17(ii))
=================================================================
The frozen pipeline computed the value-weighted index (vwretd, used by
EXRET) and the total-market-cap denominator (used by RSIZE) over the
UNFILTERED monthly file, so ETFs/CEFs/derivatives inflate both. The fix
adds eligible_permnos to compute_vwretd / compute_total_market_cap /
build_market_features, default None = frozen bytes.

  1. eligible_permnos=None reproduces the frozen aggregation exactly.
  2. Passing an eligible set removes the excluded securities from both
     the index weightings and the total-cap denominator.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.market_features import compute_total_market_cap, compute_vwretd


def make_msf():
    """
    Two months, three securities: PERMNO 1 & 2 are common stocks,
    PERMNO 9 is a huge 'ETF' that should be excluded under v2.
    """
    dates = [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    rows = []
    for permno, me0, ret_feb in ((1, 100.0, 0.10), (2, 300.0, -0.10),
                                 (9, 4000.0, 0.02)):
        rows.append({"permno": permno, "date": dates[0], "ret": 0.0, "me": me0})
        rows.append({"permno": permno, "date": dates[1], "ret": ret_feb,
                     "me": me0 * (1 + ret_feb)})
    return pd.DataFrame(rows)


def test_default_none_reproduces_frozen_aggregation():
    msf = make_msf()
    vw_default = compute_vwretd(msf)
    vw_explicit_all = compute_vwretd(msf, eligible_permnos=None)
    pd.testing.assert_frame_equal(vw_default, vw_explicit_all)

    tot_default = compute_total_market_cap(msf)
    tot_explicit = compute_total_market_cap(msf, eligible_permnos=None)
    pd.testing.assert_frame_equal(tot_default, tot_explicit)

    # Frozen February vwretd includes the ETF weight
    feb = vw_default.loc[vw_default["date"] == pd.Timestamp("2020-02-29"),
                         "vwretd"].iloc[0]
    expected = np.average([0.10, -0.10, 0.02], weights=[100.0, 300.0, 4000.0])
    assert feb == pytest.approx(expected)


def test_eligible_set_excludes_funds_from_index():
    msf = make_msf()
    vw = compute_vwretd(msf, eligible_permnos={1, 2})
    feb = vw.loc[vw["date"] == pd.Timestamp("2020-02-29"), "vwretd"].iloc[0]
    expected = np.average([0.10, -0.10], weights=[100.0, 300.0])
    assert feb == pytest.approx(expected)
    # And it genuinely differs from the polluted frozen value
    frozen = compute_vwretd(msf)
    frozen_feb = frozen.loc[frozen["date"] == pd.Timestamp("2020-02-29"),
                            "vwretd"].iloc[0]
    assert feb != pytest.approx(frozen_feb)


def test_eligible_set_excludes_funds_from_total_cap():
    msf = make_msf()
    tot = compute_total_market_cap(msf, eligible_permnos={1, 2})
    jan = tot.loc[(tot["_year"] == 2020) & (tot["_month"] == 1),
                  "total_me"].iloc[0]
    assert jan == pytest.approx(400.0)     # 100 + 300, no 4000 ETF

    frozen = compute_total_market_cap(msf)
    jan_frozen = frozen.loc[(frozen["_year"] == 2020) & (frozen["_month"] == 1),
                            "total_me"].iloc[0]
    assert jan_frozen == pytest.approx(4400.0)
