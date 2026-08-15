"""
Market-index / RSIZE denominator eligibility sensitivity.
=========================================================
**Classification: post-hoc supplementary (corrective validation of a disclosure).**

`final_primary` restricts the market-wide aggregate to the eligible equity
universe at **PERMNO level** ("any-time eligible"): a security enters the index
in every month it has a monthly-file row, provided *any* of its security-info
segments is eligible. The stricter convention is **date-ranged**: a
(PERMNO, month) pair enters only if an eligible segment is valid in that month.

The distinction matters because RSIZE is
``log(me_firm) - log(total_market_cap)``, so any error in the denominator is a
**month-common additive shift** in RSIZE. This module measures the shift and
bounds its effect on the reported models, holding everything else fixed.

Two things are reported separately, because they are different claims:

* the within-month cross-sectional **ordering** of RSIZE, which the additive
  structure leaves exactly unchanged;
* the **level** of RSIZE, which does move, and which the fitted models are not
  invariant to — trees split on absolute values and the linear model reads the
  level directly. A rank-based metric does not by itself make the change
  harmless, so the effect is measured by re-scoring the frozen models on shifted
  RSIZE rather than asserted.

Run::

    PYTHONPATH=. python -m src.analysis.supp_rsize_denominator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.analysis.supplementary_common import (DISPLAY, FEATS, MODEL_ORDER,
                                               assert_headline_reproduces,
                                               load_frozen_scores, write_table)
from src.features.market_features import (compute_total_market_cap,
                                          compute_vwretd)


def _eligible(sn: pd.DataFrame) -> tuple[set[int], pd.DataFrame]:
    sn = sn.copy()
    sn["permno"] = pd.to_numeric(sn["permno"], errors="coerce")
    for c in ["sharetype", "usincflg", "securitytype", "securitysubtype",
              "issuertype", "exchcd_src"]:
        sn[c] = sn[c].astype(str).str.strip()
    E = C.CIZ_UNIVERSE_ELIGIBLE
    ok = (sn.securitytype.isin(E["SecurityType"])
          & sn.securitysubtype.isin(E["SecuritySubType"])
          & sn.sharetype.isin(E["ShareType"])
          & sn.usincflg.isin(E["USIncFlg"])
          & sn.issuertype.isin(E["IssuerType"])
          & sn.exchcd_src.isin(E["Exchange"]))
    seg = sn.loc[ok, ["permno", "namedt", "nameendt"]].dropna()
    return set(seg.permno.astype("int64")), seg


def main() -> None:
    msf = pd.read_parquet(C.DATA_RAW_CRSP / "crsp_monthly_raw.parquet")
    msf["date"] = pd.to_datetime(msf["date"])
    sn = pd.read_parquet(C.DATA_ROOT / "raw_final_primary" / "crsp_security_names.parquet")
    perm, seg = _eligible(sn)
    print(f"eligible PERMNOs (any-time): {len(perm):,}; eligible segments: {len(seg):,}\n")

    # --- footprint: rows any-time eligible but not eligible as of their month
    sub = msf[msf.permno.astype("int64").isin(perm)][["permno", "date", "me"]].copy()
    s = seg.copy()
    s["permno"] = s.permno.astype("int64")
    s["namedt"] = pd.to_datetime(s.namedt)
    s["nameendt"] = pd.to_datetime(s.nameendt)
    probe = sub.reset_index().rename(columns={"index": "_row"})
    probe["permno"] = probe.permno.astype("int64")
    j = probe.merge(s, on="permno", how="inner")
    keep = set(j.loc[(j.date >= j.namedt) & (j.date <= j.nameendt), "_row"])
    affected = len(sub) - len(keep)
    print(f"index rows any-time eligible: {len(sub):,}")
    print(f"  of which NOT eligible as of their own month: {affected:,} "
          f"({100*affected/len(sub):.2f}%)\n")

    # --- the two aggregates
    vw_any = compute_vwretd(msf, eligible_permnos=perm)
    vw_dat = compute_vwretd(msf, eligible_permnos=perm, eligible_segments=seg)
    tc_any = compute_total_market_cap(msf, eligible_permnos=perm)
    tc_dat = compute_total_market_cap(msf, eligible_permnos=perm, eligible_segments=seg)

    v = vw_any.merge(vw_dat, on="date", suffixes=("_any", "_dat")).dropna()
    corr = float(np.corrcoef(v.vwretd_any, v.vwretd_dat)[0, 1])
    mad = float(np.mean(np.abs(v.vwretd_any - v.vwretd_dat)))

    # compute_total_market_cap is keyed on (_year, _month), not a date column
    t = tc_any.merge(tc_dat, on=["_year", "_month"],
                     suffixes=("_any", "_dat")).dropna()
    logratio = np.log(t["total_me_any"]) - np.log(t["total_me_dat"])

    print(f"VWRETD correlation           : {corr:.6f}   mean |diff| {mad:.2e}")
    print(f"total-market-cap log-ratio   : mean {logratio.mean():.5f} "
          f"(sd {logratio.std():.5f}, min {logratio.min():.5f}, "
          f"max {logratio.max():.5f})\n")

    # --- effect on RSIZE and on the frozen models -------------------------
    df, y, scores = load_frozen_scores("test")
    assert_headline_reproduces(y, scores)
    shift = pd.DataFrame({"_y": t["_year"].to_numpy(), "_m": t["_month"].to_numpy(),
                          "delta": logratio.to_numpy()})
    dd = pd.to_datetime(df["datadate"])
    key = pd.DataFrame({"_y": dd.dt.year.to_numpy(), "_m": dd.dt.month.to_numpy()})
    delta = key.merge(shift, on=["_y", "_m"], how="left")["delta"].to_numpy()
    delta = np.nan_to_num(delta, nan=float(logratio.mean()))

    rsize_sd = float(df["RSIZE"].std())
    print(f"per-firm-year RSIZE shift: mean {delta.mean():+.5f}, "
          f"max |shift| {np.abs(delta).max():.5f}; "
          f"test RSIZE sd {rsize_sd:.4f} "
          f"(shift is {100*np.abs(delta).mean()/rsize_sd:.2f}% of one sd)")

    # within-month ordering must be exactly preserved (additive month constant)
    flips = 0
    for _, g in df.assign(_ym=dd.dt.to_period("M"), _d=delta).groupby("_ym"):
        if g["_d"].nunique() > 1:
            flips += 1
    print(f"months where the shift is not constant within the month: {flips} "
          f"(0 confirms within-month RSIZE ordering is exactly preserved)\n")

    import joblib
    from src.analysis.supplementary_common import MODELS
    X = df[FEATS].to_numpy(float)
    Xs = X.copy()
    Xs[:, FEATS.index("RSIZE")] += delta
    rows = []
    for name in MODEL_ORDER:
        m = joblib.load(MODELS / f"{name}.joblib")
        p0, p1 = m.predict_proba(X)[:, 1], m.predict_proba(Xs)[:, 1]
        rows.append({
            "model": DISPLAY[name],
            "pr_auc_frozen_denominator": round(float(average_precision_score(y, p0)), 4),
            "pr_auc_date_ranged_denominator": round(float(average_precision_score(y, p1)), 4),
            "pr_auc_delta": round(float(average_precision_score(y, p1)
                                        - average_precision_score(y, p0)), 5),
            "roc_auc_frozen": round(float(roc_auc_score(y, p0)), 4),
            "roc_auc_date_ranged": round(float(roc_auc_score(y, p1)), 4),
        })
    res = pd.DataFrame(rows)
    print(res.to_string(index=False), "\n")

    foot = pd.DataFrame([{
        "index_rows_any_time_eligible": len(sub),
        "rows_not_eligible_as_of_month": affected,
        "share_affected_pct": round(100 * affected / len(sub), 2),
        "vwretd_correlation": round(corr, 6),
        "vwretd_mean_abs_diff": float(f"{mad:.3g}"),
        "total_market_cap_log_ratio_mean": round(float(logratio.mean()), 5),
        "total_market_cap_log_ratio_sd": round(float(logratio.std()), 5),
        "rsize_shift_as_pct_of_one_sd": round(100 * float(np.abs(delta).mean()) / rsize_sd, 2),
        "within_month_rsize_ordering_changed": bool(flips),
    }])
    write_table(foot, "supp_rsize_denominator_footprint",
                "Market-index / RSIZE denominator eligibility: footprint "
                "(post-hoc supplementary). The reported run restricts the "
                "aggregate to the eligible universe at PERMNO level; the "
                "stricter alternative requires eligibility as of each month. "
                "Because RSIZE is a log ratio, the difference is a month-common "
                "additive shift, so the within-month cross-sectional ordering of "
                "RSIZE is unchanged by construction.",
                "tab:supp_rsize_footprint", float_format="%.6g")
    write_table(res, "supp_rsize_denominator_sensitivity",
                "Effect of the RSIZE denominator convention on test performance "
                "(post-hoc supplementary). The frozen estimators are re-scored "
                "on RSIZE shifted by the measured month-specific denominator "
                "difference; no model is refitted. This holds the fitted "
                "parameters fixed and therefore bounds the convention's effect "
                "on the reported metrics rather than simulating a full rebuild.",
                "tab:supp_rsize_sensitivity", float_format="%.5g")


if __name__ == "__main__":
    main()
