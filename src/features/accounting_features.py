"""
Construct the 11 accounting-based predictors.
==============================================
All variables follow the definitions in Campbell, Hilscher & Szilagyi (2008)
and Shumway (2001). Variable names, formulae, and theoretical rationale are
documented below and in the thesis appendix variable construction table.

Variable definitions (Blueprint v4 §7):
  NITA     = ib / at           (Net income / Total assets — ROA)
  TLTA     = lt / at           (Total liabilities / Total assets — Leverage)
  WCTA     = (act - lct) / at  (Working capital / Total assets)
  CLCA     = lct / act         (Current liabilities / Current assets — inv. liquidity)
  OENEG    = 1 if lt > at      (Negative book equity indicator)
  NITA_LAG = NITA at t-1       (Lagged profitability)
  CHIN     = (ni_t - ni_{t-1}) / (|ni_t| + |ni_{t-1}|)  (Change in net income)
  INTWO    = 1 if ni < 0 in both t and t-1               (Sustained losses)
  CASHTA   = che / at          (Cash / Total assets)
  OCF_TA   = oancf / at        (Operating cash flow / Total assets)
  LNTA     = log(at_2012usd)   (Log total assets in constant 2012 USD)

Notes
-----
- NITA_LAG, CHIN, and INTWO require at least 2 consecutive observations per firm.
  Missing prior-year observations yield NaN (handled in imputation step).
- LNTA is deflated to constant 2012 USD using the GDP deflator.
- All variables are winsorised AFTER construction using training-set thresholds
  (see src/features/winsorize.py).

Blueprint v4 reference: §7 (Accounting-Based Predictors)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lag_within_firm_by_fyear(
    df: pd.DataFrame,
    col: str,
    periods: int = 1,
) -> pd.Series:
    """
    Lag ``col`` by exactly ``periods`` *fiscal year(s)* within each firm.

    This is the fiscal-year-aware replacement for ``groupby('gvkey')[col]
    .shift(periods)``. A positional ``shift`` moves by *row position*, so when
    a firm's fyear sequence has a gap (e.g. 1993 → 1996 because FY1994–95 are
    absent), it silently treats the 1993 value as the t-1 of 1996 — fabricating
    a lag that crosses a multi-year gap. Here the lag is aligned strictly on
    ``fyear``: the value is taken only from the genuine prior fiscal year
    (``fyear - periods``). When that year is not observed for the firm, the lag
    is ``NaN`` — the honest "no prior-year observation" encoding, which then
    flows into the standard missing-value imputation step.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'gvkey', 'fyear', and ``col``. ``(gvkey, fyear)`` is
        assumed unique (firm-year panel); duplicates are de-duplicated on the
        source side (last kept) so the lookup can never fan out the panel.
    col : str
        Column to lag.
    periods : int
        Number of fiscal years to lag (default 1).

    Returns
    -------
    pd.Series
        ``col`` lagged by ``periods`` fiscal years, indexed like ``df``;
        ``NaN`` where the prior fiscal year is not observed for the firm.
    """
    # A row at fyear t supplies the lagged value for the row at fyear t+periods.
    src = pd.DataFrame({
        "gvkey": df["gvkey"].values,
        "fyear": df["fyear"].values + periods,
        "_lag_val": df[col].values,
    }).drop_duplicates(subset=["gvkey", "fyear"], keep="last")

    merged = df[["gvkey", "fyear"]].merge(src, on=["gvkey", "fyear"], how="left")
    return pd.Series(merged["_lag_val"].values, index=df.index)


def compute_nita(df: pd.DataFrame) -> pd.Series:
    """
    Net income / Total assets (ROA).

    Profitability signal: lower NITA → less cash flow to service debt
    → higher distress probability (Altman 1968).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'ib' (income before extraordinary items)
        and 'at' (total assets).

    Returns
    -------
    pd.Series
        NITA values; NaN where at == 0 or missing.
    """
    return df["ib"].div(df["at"].replace(0, np.nan))


def compute_tlta(df: pd.DataFrame) -> pd.Series:
    """
    Total liabilities / Total assets (Book leverage).

    Solvency signal: higher leverage → greater refinancing risk
    → higher distress probability (Ohlson 1980).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lt' (total liabilities) and 'at' (total assets).

    Returns
    -------
    pd.Series
    """
    return df["lt"].div(df["at"].replace(0, np.nan))


def compute_wcta(df: pd.DataFrame) -> pd.Series:
    """
    Working capital / Total assets.

    Liquidity buffer: lower WCTA → less liquid cushion against
    short-term obligations → higher distress probability (Altman 1968).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'act' (current assets), 'lct' (current liabilities),
        and 'at' (total assets).

    Returns
    -------
    pd.Series
    """
    wc = df["act"].sub(df["lct"])
    return wc.div(df["at"].replace(0, np.nan))


def compute_clca(df: pd.DataFrame) -> pd.Series:
    """
    Current liabilities / Current assets (inverse current ratio).

    Higher CLCA → less covered short-term obligations
    → higher distress probability (Ohlson 1980).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lct' (current liabilities) and 'act' (current assets).

    Returns
    -------
    pd.Series
        NaN where act == 0.
    """
    return df["lct"].div(df["act"].replace(0, np.nan))


def compute_oeneg(df: pd.DataFrame, missing_policy: str = "frozen") -> pd.Series:
    """
    Negative equity indicator (1 if total liabilities > total assets).

    A firm with lt > at has negative book equity — a severe distress
    signal associated with insolvency (Ohlson 1980).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'lt' and 'at'.
    missing_policy : str
        "frozen" (default) reproduces the original behaviour: a NaN
        comparison evaluates False, so missing lt/at silently yields
        OENEG=0 (283 v1 rows in the 2026-07-12 audit).
        "strict" returns NA when lt or at is missing, so the row flows
        into the ≥8/11 coverage filter and imputation like any other
        missing accounting feature (v2 profile).

    Returns
    -------
    pd.Series
        Integer 0/1 indicator ("frozen") or nullable Int64 ("strict").
    """
    if missing_policy == "frozen":
        return (df["lt"] > df["at"]).astype(int)
    if missing_policy != "strict":
        raise ValueError(
            f"missing_policy must be 'frozen' or 'strict', got {missing_policy!r}"
        )
    oeneg = (df["lt"] > df["at"]).astype("Int64")
    oeneg[df["lt"].isna() | df["at"].isna()] = pd.NA
    return oeneg


def compute_nita_lag(df: pd.DataFrame) -> pd.Series:
    """
    Lagged profitability: NITA at t-1.

    Persistent earnings deterioration is a leading indicator of distress
    (Campbell, Hilscher & Szilagyi 2008).
    Lagged within firm on the genuine prior *fiscal year* (not by row
    position), so a gap in a firm's fyear sequence yields NaN rather than a
    stale earlier-year value (see ``lag_within_firm_by_fyear``).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'gvkey', 'fyear', and a pre-computed 'NITA' column.

    Returns
    -------
    pd.Series
        NaN for the first observation of each firm and whenever the prior
        fiscal year (fyear - 1) is not observed.
    """
    return lag_within_firm_by_fyear(df, "NITA")


def compute_chin(df: pd.DataFrame) -> pd.Series:
    """
    Change in net income: (ni_t - ni_{t-1}) / (|ni_t| + |ni_{t-1}|).

    Normalised change measure bounded in [-1, 1]. Negative CHIN indicates
    deteriorating earnings momentum → higher distress probability (Ohlson 1980).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'gvkey', 'fyear', and 'ni' (net income).

    Returns
    -------
    pd.Series
        NaN for the first observation of each firm, when the prior fiscal
        year (fyear - 1) is not observed, or when the denominator = 0.
    """
    ni      = df["ni"]
    ni_lag  = lag_within_firm_by_fyear(df, "ni")
    denom   = ni.abs().add(ni_lag.abs())
    chin    = ni.sub(ni_lag).div(denom.replace(0, np.nan))
    return chin


def compute_intwo(df: pd.DataFrame, missing_policy: str = "frozen") -> pd.Series:
    """
    Sustained losses indicator: 1 if net income < 0 in both t and t-1.

    Persistent negative earnings → high distress probability (Ohlson 1980).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'gvkey', 'fyear', and 'ni' (net income).
    missing_policy : str
        "frozen" (default) reproduces the original behaviour: a missing
        LAGGED ni already yields NA, but a missing CURRENT ni evaluates
        `NaN < 0 = False` and silently yields INTWO=0 (2026-07-12 audit).
        "strict" also returns NA when the current-year ni is missing.

    Returns
    -------
    pd.Series
        Integer 0/1 indicator; NaN for the first observation of each firm and
        whenever the prior fiscal year (fyear - 1) is not observed.
    """
    if missing_policy not in ("frozen", "strict"):
        raise ValueError(
            f"missing_policy must be 'frozen' or 'strict', got {missing_policy!r}"
        )
    ni_lag     = lag_within_firm_by_fyear(df, "ni")
    ni_neg     = df["ni"] < 0
    ni_lag_neg = ni_lag < 0
    intwo      = (ni_neg & ni_lag_neg).astype("Int64")
    # Set to NA where the prior fiscal year is not available
    intwo[ni_lag.isna()] = pd.NA
    if missing_policy == "strict":
        intwo[df["ni"].isna()] = pd.NA
    return intwo


def compute_cashta(df: pd.DataFrame) -> pd.Series:
    """
    Cash and short-term investments / Total assets.

    Cash reserve relative to asset base: higher CASHTA → more immediate
    liquidity → lower distress probability (Zmijewski 1984).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'che' (cash & short-term investments) and 'at'.

    Returns
    -------
    pd.Series
    """
    return df["che"].div(df["at"].replace(0, np.nan))


def compute_ocf_ta(df: pd.DataFrame) -> pd.Series:
    """
    Operating cash flow / Total assets.

    Cash generation quality: operating CF is harder to manipulate than
    accrual-based net income. Low OCF_TA signals inability to generate
    cash from operations (Shumway 2001).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'oancf' (operating cash flow) and 'at'.
        Falls back to 'fopt' if 'oancf' is missing.

    Returns
    -------
    pd.Series
    """
    ocf = df["oancf"] if "oancf" in df.columns else df.get("fopt")
    if ocf is None:
        return pd.Series(np.nan, index=df.index)
    return ocf.div(df["at"].replace(0, np.nan))


def compute_lnta(
    df: pd.DataFrame,
    gdp_deflator: pd.DataFrame,
    base_year: int = 2012,
) -> pd.Series:
    """
    Log total assets in constant base_year USD.

    Size control: smaller firms are more vulnerable to distress and have
    fewer refinancing options (Shumway 2001). GDP deflation ensures
    comparability across years.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'at' (total assets, $M) and 'fyear'.
    gdp_deflator : pd.DataFrame
        Annual GDP deflator index with columns ['year', 'deflator'].
        Base year deflator value = 1.0 (all other years relative to base_year).
    base_year : int
        Reference year for constant-dollar deflation.

    Returns
    -------
    pd.Series
        Log of total assets in constant base_year USD.
    """
    deflator = gdp_deflator.set_index("year")["deflator"]
    base_def = deflator.get(base_year, 1.0)

    df = df.copy()
    deflator_year = (
        pd.to_datetime(df["datadate"], errors="coerce").dt.year
        if "datadate" in df.columns else df["fyear"]
    )
    df["_deflator"] = deflator_year.map(deflator)
    # Deflate: at_real = at * (base_deflator / year_deflator)
    at_real = df["at"] * (base_def / df["_deflator"])
    at_real = at_real.replace(0, np.nan)
    at_real[at_real <= 0] = np.nan
    return np.log(at_real)


def build_accounting_features(
    panel: pd.DataFrame,
    gdp_deflator: pd.DataFrame,
    missing_policy: str = "frozen",
    lag_donor: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all 11 accounting-based predictors and attach to panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must be sorted by ['gvkey', 'fyear']. Contains raw Compustat items.
    gdp_deflator : pd.DataFrame
        GDP deflator (see compute_lnta docstring).
    missing_policy : str
        "frozen" (default) or "strict" — OENEG/INTWO handling of missing
        raw inputs (see compute_oeneg / compute_intwo; v2 passes "strict").
    lag_donor : pd.DataFrame or None
        Optional raw Compustat rows from the fiscal year immediately
        BEFORE the sample start (FY1989 for the 1990–… panel), used only
        as lag donors so the first sample year's lagged predictors
        (NITA_LAG, CHIN, INTWO) are genuine rather than needlessly
        missing (2026-07-12 second-audit fix: the loader keeps
        SAMPLE_START_YEAR−1 for exactly this purpose, but the merge
        dropped it before features were built). Donor rows are appended
        for the lag computation and REMOVED before returning — they can
        never enter the modelling panel. None (frozen) = no donors.

    Returns
    -------
    pd.DataFrame
        Input panel with 11 new columns:
        NITA, TLTA, WCTA, CLCA, OENEG, NITA_LAG, CHIN, INTWO, CASHTA,
        OCF_TA, LNTA. Row set identical to the input panel.
    """
    print("Computing accounting features ...")
    panel = panel.sort_values(["gvkey", "fyear"]).copy()

    if lag_donor is not None and len(lag_donor):
        donor = lag_donor.copy()
        min_fyear = int(pd.to_numeric(panel["fyear"], errors="coerce").min())
        donor_years = pd.to_numeric(donor["fyear"], errors="coerce")
        if (donor_years >= min_fyear).any():
            raise ValueError(
                "lag_donor may only contain fiscal years strictly before "
                f"the panel start ({min_fyear}); got years up to "
                f"{int(donor_years.max())}."
            )
        # Only donors that can actually serve a panel firm matter
        donor = donor[donor["gvkey"].isin(panel["gvkey"].unique())]
        keep_cols = [c for c in donor.columns if c in panel.columns]
        donor = donor[keep_cols]
        panel["_lag_donor"] = False
        donor["_lag_donor"] = True
        panel = (
            pd.concat([panel, donor], ignore_index=True)
            .sort_values(["gvkey", "fyear"])
            .reset_index(drop=True)
        )
        print(f"  Appended {int(panel['_lag_donor'].sum()):,} lag-donor rows "
              f"(fyear < {min_fyear}) for lagged-feature construction.")

    panel["NITA"]     = compute_nita(panel)
    panel["TLTA"]     = compute_tlta(panel)
    panel["WCTA"]     = compute_wcta(panel)
    panel["CLCA"]     = compute_clca(panel)
    panel["OENEG"]    = compute_oeneg(panel, missing_policy=missing_policy)
    panel["NITA_LAG"] = compute_nita_lag(panel)   # requires NITA column
    panel["CHIN"]     = compute_chin(panel)
    panel["INTWO"]    = compute_intwo(panel, missing_policy=missing_policy)
    panel["CASHTA"]   = compute_cashta(panel)
    panel["OCF_TA"]   = compute_ocf_ta(panel)
    panel["LNTA"]     = compute_lnta(panel, gdp_deflator)

    if "_lag_donor" in panel.columns:
        n_donor = int(panel["_lag_donor"].sum())
        panel = panel[~panel["_lag_donor"]].drop(columns=["_lag_donor"])
        panel = panel.reset_index(drop=True)
        print(f"  Removed {n_donor:,} lag-donor rows after lag construction.")

    computed = [c for c in ["NITA","TLTA","WCTA","CLCA","OENEG",
                             "NITA_LAG","CHIN","INTWO","CASHTA","OCF_TA","LNTA"]
                if c in panel.columns]
    print(f"  {len(computed)}/11 accounting features computed.")
    return panel
