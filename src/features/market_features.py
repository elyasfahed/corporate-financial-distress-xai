"""
Construct the 6 market-based predictors.
=========================================
All variables follow Campbell, Hilscher & Szilagyi (2008).
Market data comes from the CRSP Monthly Stock File.

Variable definitions (design §7):
  EXRET  = 12-month cumulative excess return over CRSP value-weighted index
  SIGMA  = Std dev of raw monthly returns over trailing 12 months
  LNMK   = log(market cap in constant 2012 USD)
  RSIZE  = log(firm market cap / total CRSP market cap)
  MB     = Market cap / Book equity (market-to-book ratio)
  PRICE  = log(Price per share), capped at log(15)

Timing convention:
  All 12-month windows end at the fiscal year-end calendar month.
  The match is: MSF observation where (year, month) == (datadate.year, datadate.month).
  These predictors reflect information available to investors at fiscal year-end,
  before the 10-K is filed on F_{i,t}.

Design reference: §7 (Market-Based Predictors)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Internal helpers ─────────────────────────────────────────────────────────

def _add_year_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add integer year and month columns from a date column."""
    df = df.copy()
    df["_year"]  = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month
    return df


# ── Value-weighted market return (needed before EXRET) ───────────────────────

def _restrict_to_eligible(
    msf: pd.DataFrame,
    eligible_permnos,
    eligible_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    §17(ii) fix — optionally restrict the monthly file to the eligible
    equity universe before market-wide aggregation. The frozen pipeline
    aggregated over the UNFILTERED MSF, so the reconstructed index and
    the RSIZE denominator include ~6,900 ETFs, ~1,300 CEFs and other
    non-common securities (ETFs ≈10.6% of the Dec-2020 "total cap",
    trending upward through the test decade). eligible_permnos=None
    reproduces the frozen behaviour byte-for-byte.

    Two eligibility modes:

    PERMNO-level "any-time" (``eligible_segments=None``) — a PERMNO enters the
      aggregate in every month it has an MSF row, provided ANY of its
      security-info segments is eligible. This is what `final_primary` ran.
    Date-ranged (``eligible_segments`` supplied) — a (PERMNO, month) pair enters
      only if an eligible segment is valid in that month (namedt <= date <=
      nameendt), so a security that moves off an eligible exchange or share
      class leaves the aggregate at the moment it does.

    Measured difference on the reported sample (`final_primary`, 1989–2024):
    85,852 of 2,239,834 index rows (3.83%) are any-time-eligible but NOT
    eligible as of their own month; their share of aggregate market cap rises
    secularly from 0.59% (1990s) to 2.97% (2010s). The resulting VWRETD series
    correlate at 0.99997 (mean |diff| 1.8e-4/month) and the total-market-cap
    log-ratio averages 0.00735. Because RSIZE = log(me_firm) - log(total_me),
    the denominator error is a MONTH-COMMON ADDITIVE SHIFT: within-month
    cross-sectional ordering of RSIZE is exactly unchanged, and PR-AUC/ROC-AUC
    are rank-based. The date-ranged path is therefore available as an
    alternative and disclosed rather than adopted; adopting it
    would require a full feature rebuild for an effect an order of magnitude
    inside the ±0.025 bootstrap CI.

    Parameters
    ----------
    msf : pd.DataFrame
        Monthly file with 'permno' and (for date-ranged mode) 'date'.
    eligible_permnos : set/sequence of int or None
    eligible_segments : pd.DataFrame or None
        Eligible security-info segments with columns permno, namedt, nameendt.
        Supplying this switches on date-ranged restriction.
    """
    if eligible_permnos is None and eligible_segments is None:
        return msf

    before = msf["permno"].nunique()

    if eligible_segments is None:
        eligible = set(pd.Series(list(eligible_permnos)).astype("int64"))
        out = msf[msf["permno"].astype("int64").isin(eligible)]
        print(f"  Market aggregate restricted to eligible universe "
              f"(any-time): {out['permno'].nunique():,} of {before:,} "
              "PERMNOs kept")
        return out

    if "date" not in msf.columns:
        raise KeyError(
            "Date-ranged market-aggregate restriction needs a 'date' column "
            "in the monthly file."
        )
    for col in ("permno", "namedt", "nameendt"):
        if col not in eligible_segments.columns:
            raise KeyError(
                f"eligible_segments lacks {col!r}; supply permno/namedt/nameendt."
            )

    seg = eligible_segments[["permno", "namedt", "nameendt"]].copy()
    seg["permno"] = pd.to_numeric(seg["permno"], errors="coerce")
    seg = seg.dropna(subset=["permno"])
    seg["permno"] = seg["permno"].astype("int64")
    seg["namedt"] = pd.to_datetime(seg["namedt"], errors="coerce")
    seg["nameendt"] = pd.to_datetime(seg["nameendt"], errors="coerce")
    seg = seg.dropna(subset=["namedt", "nameendt"])

    probe = pd.DataFrame({
        "_row": msf.index,
        "permno": pd.to_numeric(msf["permno"], errors="coerce"),
        "_date": pd.to_datetime(msf["date"], errors="coerce"),
    }).dropna(subset=["permno", "_date"])
    probe["permno"] = probe["permno"].astype("int64")

    joined = probe.merge(seg, on="permno", how="inner")
    keep = joined.loc[
        (joined["_date"] >= joined["namedt"])
        & (joined["_date"] <= joined["nameendt"]),
        "_row",
    ].unique()
    out = msf.loc[msf.index.isin(keep)]
    print(f"  Market aggregate restricted to eligible universe "
          f"(date-ranged): {len(out):,} of {len(msf):,} rows, "
          f"{out['permno'].nunique():,} of {before:,} PERMNOs kept")
    return out


def compute_vwretd(msf: pd.DataFrame, eligible_permnos=None,
                   eligible_segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute the CRSP value-weighted market return by calendar month.

    The weight for each security in month t is its lagged (t-1) market
    capitalisation, following the standard CRSP methodology.

    Parameters
    ----------
    msf : pd.DataFrame
        Full CRSP Monthly Stock File. Must contain: permno, date, ret, me.
    eligible_permnos : set/sequence of int or None, default None
        None (frozen) aggregates over the unfiltered file. The v2 rebuild
        passes the eligible-universe PERMNOs (src/data/universe.py) so the
        index excludes ETFs/CEFs/derivatives (§17(ii)).

    Returns
    -------
    pd.DataFrame
        Columns: date, vwretd.
        One row per calendar month.
    """
    msf = _restrict_to_eligible(msf, eligible_permnos, eligible_segments)
    msf = msf.copy().sort_values(["permno", "date"])

    # Lagged market cap as weight
    msf["me_lag"] = msf.groupby("permno")["me"].shift(1)

    # Drop rows with missing ret or weight
    valid = msf["ret"].notna() & msf["me_lag"].notna() & (msf["me_lag"] > 0)
    msf_valid = msf[valid].copy()

    # Value-weighted return = sum(weight × ret) / sum(weight)
    vw = (
        msf_valid.groupby("date")
        .apply(
            lambda g: np.average(g["ret"], weights=g["me_lag"]),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "vwretd"})
    )
    return vw


def compute_total_market_cap(msf: pd.DataFrame, eligible_permnos=None,
                             eligible_segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute total CRSP market cap by calendar year (fiscal year-end month).

    Sums market cap across all firms in each calendar month.
    Used to compute RSIZE = log(firm me / total me).

    Parameters
    ----------
    msf : pd.DataFrame
        Full CRSP Monthly Stock File with 'me' and 'date' columns.
    eligible_permnos : set/sequence of int or None, default None
        None (frozen) sums over the unfiltered file. The v2 rebuild passes
        the eligible-universe PERMNOs so the RSIZE denominator excludes
        ETFs/CEFs/derivatives, removing the secular fund-growth component
        from RSIZE (§17(ii)).

    Returns
    -------
    pd.DataFrame
        Columns: _year, _month, total_me.
    """
    msf = _restrict_to_eligible(msf, eligible_permnos, eligible_segments)
    msf = _add_year_month(msf)
    total = (
        msf[msf["me"].notna()]
        .groupby(["_year", "_month"])["me"]
        .sum()
        .reset_index()
        .rename(columns={"me": "total_me"})
    )
    return total


# ── Rolling-window market features ───────────────────────────────────────────

def compute_exret(
    msf: pd.DataFrame,
    vwretd: pd.DataFrame,
    min_obs: int = 6,
) -> pd.DataFrame:
    """
    Cumulative 12-month excess return over CRSP value-weighted index.

    For each permno × calendar month-end:
      EXRET = prod(1 + ret_firm, 12m) - prod(1 + ret_market, 12m)

    Parameters
    ----------
    msf : pd.DataFrame
        CRSP Monthly Stock File. Must contain: permno, date, ret.
    vwretd : pd.DataFrame
        Value-weighted market return. Columns: date, vwretd.
    min_obs : int, default 6
        Minimum number of valid monthly returns required in the trailing
        12-month window; firm-months with fewer are returned as NaN.

        ``min_obs=6`` reproduces the frozen pipeline. Note that with
        ``min_obs < 12`` a firm-month with 6--11 valid returns yields a
        *partial-window* EXRET: a cumulative excess return over fewer than
        twelve months, mislabelled as a 12-month figure and not comparable
        across firms (the missing months are additionally treated as zero
        return below). ``min_obs=12`` is the principled full-window setting
        — it makes EXRET a genuine 12-month excess return for every non-NaN
        firm-month and eliminates the within-window zero-fill, at the cost of
        marking the partial-window firm-years (recent listings) as missing,
        where they are handled by the standard market-feature missing-value
        treatment. The construction-noise footprint and a four-model
        sensitivity are documented in
        ``src/robustness/exret_sigma_construction_sensitivity.py``.

    Returns
    -------
    pd.DataFrame
        Columns: permno, _year, _month, EXRET.
    """
    # Merge market return onto msf
    msf = msf.merge(vwretd, on="date", how="left")
    msf = _add_year_month(msf).sort_values(["permno", "date"])

    # Log returns for cumulative product via sum; treat missing ret as 0
    msf["_log_ret"] = np.log1p(msf["ret"].fillna(0))
    msf["_log_mkt"] = np.log1p(msf["vwretd"].fillna(0))

    # Rolling 12-month cumulative returns within each permno group
    MIN_OBS = min_obs

    msf["_cum_firm"] = (
        msf.groupby("permno")["_log_ret"]
        .transform(lambda x: x.rolling(12, min_periods=MIN_OBS).sum())
    )
    # Market return is the same for all firms — compute once
    mkt_cum = (
        vwretd.sort_values("date")["vwretd"]
        .apply(lambda v: np.log1p(v if pd.notna(v) else 0))
        .rolling(12, min_periods=MIN_OBS)
        .sum()
    )
    mkt_cum_df = vwretd[["date"]].copy()
    mkt_cum_df["_cum_mkt"] = mkt_cum.values
    msf = msf.merge(mkt_cum_df, on="date", how="left")

    # Convert log-cumsum back to cumulative return
    msf["EXRET"] = np.expm1(msf["_cum_firm"]) - np.expm1(msf["_cum_mkt"])

    # NaN where fewer than MIN_OBS months available
    n_valid = (
        msf.groupby("permno")["ret"]
        .transform(lambda x: x.notna().rolling(12, min_periods=1).sum())
    )
    msf.loc[n_valid < MIN_OBS, "EXRET"] = np.nan

    result = msf[["permno", "_year", "_month", "EXRET"]].drop_duplicates(
        subset=["permno", "_year", "_month"]
    )
    return result


def compute_sigma(msf: pd.DataFrame, min_obs: int = 6) -> pd.DataFrame:
    """
    Standard deviation of raw monthly returns over trailing 12 months.

    Higher SIGMA → higher equity volatility → closer to default boundary
    in the Merton (1974) distance-to-default framework.

    Parameters
    ----------
    msf : pd.DataFrame
        Must contain: permno, date, ret.
    min_obs : int, default 6
        Minimum number of valid monthly returns required in the trailing
        12-month window; firm-months with fewer are returned as NaN.
        ``min_obs=6`` reproduces the frozen pipeline; ``min_obs=12`` requires
        a full window. Note that even at ``min_obs=12`` this is a *monthly*
        volatility estimate based on at most twelve observations, which is
        inherently coarse: a precise SIGMA would be estimated from CRSP daily
        returns. The monthly construction is retained as a documented
        limitation (see the thesis robustness chapter); the daily-data
        reconstruction is left as future work.

    Returns
    -------
    pd.DataFrame
        Columns: permno, _year, _month, SIGMA.
    """
    msf = _add_year_month(msf).sort_values(["permno", "date"]).copy()

    MIN_OBS = min_obs
    msf["SIGMA"] = (
        msf.groupby("permno")["ret"]
        .transform(lambda x: x.rolling(12, min_periods=MIN_OBS).std())
    )

    result = msf[["permno", "_year", "_month", "SIGMA"]].drop_duplicates(
        subset=["permno", "_year", "_month"]
    )
    return result


def compute_price(msf: pd.DataFrame) -> pd.DataFrame:
    """
    Log share price at each calendar month-end, capped at log(15).

    Penny-stock distress signal: very low share prices indicate severe
    market distress. The cap at log(15) ≈ 2.71 prevents high-price stocks
    from distorting the variable (Shumway 2001).

    Parameters
    ----------
    msf : pd.DataFrame
        Must contain: permno, date, prc (price, absolute value).

    Returns
    -------
    pd.DataFrame
        Columns: permno, _year, _month, PRICE.
    """
    CAP = np.log(15)

    msf = _add_year_month(msf).copy()
    prc = msf["prc"].abs()
    prc[prc <= 0] = np.nan
    msf["PRICE"] = np.log(prc).clip(upper=CAP)

    result = msf[["permno", "_year", "_month", "PRICE"]].drop_duplicates(
        subset=["permno", "_year", "_month"]
    )
    return result


# ── Panel-level features (require panel context) ─────────────────────────────

def compute_lnmk(
    panel: pd.DataFrame,
    gdp_deflator: pd.DataFrame,
    base_year: int = 2012,
) -> pd.Series:
    """
    Log market capitalisation in constant base_year USD.

    Larger LNMK → lower distress probability (size effect; CHS 2008).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'me_fyend' (fiscal year-end market cap, $M) and 'fyear'.
    gdp_deflator : pd.DataFrame
        Annual GDP deflator. Columns: ['year', 'deflator'].

    Returns
    -------
    pd.Series
    """
    deflator  = gdp_deflator.set_index("year")["deflator"]
    base_def  = deflator.get(base_year, 1.0)

    panel = panel.copy()
    deflator_year = (
        pd.to_datetime(panel["datadate"], errors="coerce").dt.year
        if "datadate" in panel.columns else panel["fyear"]
    )
    panel["_def"] = deflator_year.map(deflator)
    me_real = panel["me_fyend"] * (base_def / panel["_def"])
    me_real[me_real <= 0] = np.nan
    return np.log(me_real)


def compute_rsize(
    panel: pd.DataFrame,
    total_mkt_cap: pd.DataFrame,
) -> pd.Series:
    """
    Relative size: log(firm market cap / total CRSP market cap).

    Controls for secular market growth. Computed using fiscal year-end
    month market cap for both numerator and denominator.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'me_fyend', '_year', '_month'.
    total_mkt_cap : pd.DataFrame
        Output of compute_total_market_cap(). Columns: _year, _month, total_me.

    Returns
    -------
    pd.Series
    """
    panel = panel.merge(
        total_mkt_cap, on=["_year", "_month"], how="left"
    )
    ratio = panel["me_fyend"].div(panel["total_me"].replace(0, np.nan))
    ratio[ratio <= 0] = np.nan
    return np.log(ratio)


def compute_mb(panel: pd.DataFrame) -> pd.Series:
    """
    Market-to-book ratio: market cap / book equity.

    Low MB signals financial stress (distress discount). Firms trading
    below book value often face market expectations of earnings erosion
    or capital impairment (Fama & French 1992).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'me_fyend' (market cap) and 'ceq' (common equity).

    Returns
    -------
    pd.Series
        NaN where ceq <= 0.
    """
    ceq = panel["ceq"].copy()
    ceq[ceq <= 0] = np.nan
    return panel["me_fyend"].div(ceq)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def build_market_features(
    panel: pd.DataFrame,
    msf: pd.DataFrame,
    gdp_deflator: pd.DataFrame,
    market_min_obs: int = 6,
    eligible_permnos=None,
    eligible_segments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all 6 market-based predictors and attach to panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Merged panel (output of merge_crsp_compustat), sorted by
        ['gvkey', 'fyear']. Must contain 'me_fyend', 'ceq', 'permno',
        'datadate'.
    msf : pd.DataFrame
        CRSP Monthly Stock File.
    gdp_deflator : pd.DataFrame
        Annual GDP deflator with columns ['year', 'deflator'].
    market_min_obs : int, default 6
        Minimum valid monthly returns required for the EXRET/SIGMA trailing
        12-month window (see ``compute_exret``). ``6`` reproduces the frozen
        pipeline; ``12`` is the principled full-window construction.
    eligible_permnos : set/sequence of int or None, default None
        Restricts the MARKET-WIDE aggregates (value-weighted index and the
        RSIZE total-cap denominator) to the eligible equity universe
        (§17(ii)). None reproduces the frozen unfiltered aggregation.
        Per-firm features are always computed for the panel's own PERMNOs.
    eligible_segments : pd.DataFrame or None, default None
        Eligible security-info segments (permno, namedt, nameendt). When
        supplied, the market-wide aggregates use DATE-RANGED eligibility
        instead of the any-time PERMNO rule — see ``_restrict_to_eligible``
        for the measured difference and why `final_primary` keeps any-time.

    Returns
    -------
    pd.DataFrame
        Input panel with 6 new columns:
        EXRET, SIGMA, LNMK, RSIZE, MB, PRICE.
    """
    print("Computing market features ...")

    # ── Pre-compute market-level aggregates ───────────────────────────────
    vwretd    = compute_vwretd(msf, eligible_permnos=eligible_permnos,
                               eligible_segments=eligible_segments)
    total_mkt = compute_total_market_cap(msf, eligible_permnos=eligible_permnos,
                                         eligible_segments=eligible_segments)

    # ── Compute rolling per-permno statistics ─────────────────────────────
    exret_df = compute_exret(msf, vwretd, min_obs=market_min_obs)  # permno, _year, _month, EXRET
    sigma_df = compute_sigma(msf, min_obs=market_min_obs)          # permno, _year, _month, SIGMA
    price_df = compute_price(msf)            # permno, _year, _month, PRICE

    # ── Add fiscal year-end year and month to panel for the merge ─────────
    panel = panel.copy()
    panel["_year"]  = panel["datadate"].dt.year
    panel["_month"] = panel["datadate"].dt.month

    # ── Merge rolling stats onto panel on permno × fiscal-year-end month ──
    panel = panel.merge(
        exret_df.rename(columns={"_year": "_year", "_month": "_month"}),
        on=["permno", "_year", "_month"],
        how="left",
    )
    panel = panel.merge(
        sigma_df,
        on=["permno", "_year", "_month"],
        how="left",
    )
    panel = panel.merge(
        price_df,
        on=["permno", "_year", "_month"],
        how="left",
    )

    # ── Panel-level market features ───────────────────────────────────────
    panel["LNMK"]  = compute_lnmk(panel, gdp_deflator)
    panel["RSIZE"] = compute_rsize(panel, total_mkt)
    panel["MB"]    = compute_mb(panel)

    # ── Drop merge helpers ────────────────────────────────────────────────
    panel = panel.drop(columns=["_year", "_month", "total_me"], errors="ignore")

    computed = [c for c in ["EXRET", "SIGMA", "LNMK", "RSIZE", "MB", "PRICE"]
                if c in panel.columns]
    print(f"  {len(computed)}/6 market features computed.")
    return panel
