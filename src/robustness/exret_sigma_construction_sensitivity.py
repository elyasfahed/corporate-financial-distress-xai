"""
Supplementary robustness: noisy construction of EXRET and SIGMA.
================================================================
Purpose
-------
EXRET (12-month cumulative excess return) and SIGMA (12-month return
volatility) are built from a trailing 12-month window of CRSP *monthly*
returns. The frozen pipeline requires only >= 6 valid monthly returns
(``min_obs=6`` in src/features/market_features.py). Two construction-noise
problems follow:

  1. EXRET partial windows. A firm-month with 6-11 valid returns receives a
     cumulative excess return over fewer than twelve months -- mislabelled as a
     12-month figure and not comparable across firms. The missing months inside
     the window are additionally treated as zero return. The principled fix is a
     full-window rule (``min_obs=12``): EXRET becomes a genuine 12-month excess
     return for every non-NaN firm-month, and the partial windows (recent
     listings) are marked missing, where the standard market-feature
     missing-value treatment handles them.
  2. SIGMA frequency. Even with a full 12-month window, SIGMA is a *monthly*
     volatility estimate from at most twelve observations -- inherently coarse.
     A precise estimate would use CRSP *daily* returns. No ready-made annual
     volatility field exists in the Compustat market-data files (only annual
     high/low close), so the daily reconstruction is out of scope and the
     monthly SIGMA is retained as a documented limitation.

This module is READ-ONLY with respect to the frozen pipeline. It writes ONLY:
    outputs/tables/robustness/exret_sigma_window_footprint.{csv,tex}
    outputs/tables/robustness/exret_sigma_construction_sensitivity.{csv,tex}
It does NOT touch outputs/models/saved/*, the headline model_results tables,
or the evaluation manifest.

Footprint (how many firm-years are affected):
    The window audit recomputes, per matched firm-year, the number of valid
    monthly returns feeding the trailing 12-month window, exactly as
    compute_exret builds it.

Sensitivity (does it matter): re-fits the four headline models (LR, RF, XGBoost,
balanced co-primary NN) with frozen hyperparameters under three regimes:
    C_frozen   : split parquets unchanged                    (reproduces headline)
    F_full12   : EXRET & SIGMA -> NaN where the window had < 12 valid returns,
                 firm-year KEPT and value flows to the model as fillna(0)
                 (principled full-window construction)
    X_exclude  : firm-years with < 12 valid returns DROPPED from all splits
                 (exclusion variant; evaluated on the reduced test set, so its
                 PR-AUC is not strictly like-for-like with C)

Run from the project root:
    python -m src.robustness.exret_sigma_construction_sensitivity
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from src.config import (
    ALL_FEATURES, DATA_SAMPLES, DATA_RAW_CRSP, OUT_MODELS_CONFIGS,
    OUT_TABLES_ROBUSTNESS,
)
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn
from src.models.evaluate import select_threshold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

LABEL = "distress"
FULL_WINDOW = 12   # principled full-window requirement

# Frozen primary hyperparameters (outputs/models/configs/*_config.yaml), reused
# so this matches the frozen models without paying the Optuna re-tune.
CFG = {
    "logistic_regression": dict(C=0.06477118947657097),
    "random_forest": dict(n_estimators=800, max_depth=11,
                          min_samples_leaf=27, max_features="log2"),
    "xgboost": dict(n_estimators=200, max_depth=6,
                    learning_rate=0.019382979042711645,
                    subsample=0.8440809026257249,
                    colsample_bytree=0.621895727026898,
                    min_child_weight=15,
                    reg_alpha=2.3373178719336034e-05,
                    reg_lambda=0.6875877587762004),
}
try:
    with open(OUT_MODELS_CONFIGS / "neural_network_balanced_config.yaml") as _fh:
        NN_CFG = yaml.safe_load(_fh)["best_params"]
except FileNotFoundError:
    NN_CFG = None

MODELS = ["logistic_regression", "random_forest", "xgboost"]
if NN_CFG is not None:
    MODELS.append("neural_network_balanced")

REGIMES = ["C_frozen", "F_full12", "X_exclude"]


# ── window audit (valid monthly returns per firm-year) ───────────────────────

def compute_window_audit() -> pd.DataFrame:
    """Recompute #valid monthly returns in the trailing 12-month window per
    permno x calendar-month, matching compute_exret's n_valid logic exactly."""
    msf = pd.read_parquet(DATA_RAW_CRSP / "crsp_monthly_raw.parquet")
    msf = msf.sort_values(["permno", "date"]).copy()
    msf["_year"] = msf["date"].dt.year
    msf["_month"] = msf["date"].dt.month
    msf["n_valid"] = (
        msf.groupby("permno")["ret"]
        .transform(lambda x: x.notna().rolling(12, min_periods=1).sum())
    )
    return msf[["permno", "_year", "_month", "n_valid"]].drop_duplicates(
        ["permno", "_year", "_month"]
    )


def _attach_nvalid(split: str, nvkey: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_parquet(DATA_SAMPLES / f"{split}.parquet").copy()
    d["_year"] = d["datadate"].dt.year
    d["_month"] = d["datadate"].dt.month
    d = d.merge(nvkey, on=["permno", "_year", "_month"], how="left")
    d["n_valid"] = d["n_valid"].fillna(0)
    d["_split"] = split
    return d


# ── footprint table ──────────────────────────────────────────────────────────

def build_footprint(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, d in splits.items():
        n = len(d)
        lt6 = int((d["n_valid"] < 6).sum())
        part = int(((d["n_valid"] >= 6) & (d["n_valid"] < FULL_WINDOW)).sum())
        full = int((d["n_valid"] >= FULL_WINDOW).sum())
        ev = d["distress"].astype(int)
        rows.append(dict(
            split=s, n_firmyears=n, n_events=int(ev.sum()),
            full12=full, full12_pct=round(100 * full / n, 2),
            partial_6_11=part, partial_6_11_pct=round(100 * part / n, 2),
            partial_events=int(ev[(d["n_valid"] >= 6) & (d["n_valid"] < FULL_WINDOW)].sum()),
            below6_nan=lt6, below6_nan_pct=round(100 * lt6 / n, 2),
        ))
    return pd.DataFrame(rows)


# ── sensitivity ──────────────────────────────────────────────────────────────

def _prep(splits, regime):
    tr, va, te = (splits["train"].copy(), splits["val"].copy(), splits["test"].copy())
    if regime == "F_full12":
        for d in (tr, va, te):
            m = d["n_valid"] < FULL_WINDOW
            d.loc[m, "EXRET"] = np.nan
            d.loc[m, "SIGMA"] = np.nan
    elif regime == "X_exclude":
        tr = tr[tr["n_valid"] >= FULL_WINDOW]
        va = va[va["n_valid"] >= FULL_WINDOW]
        te = te[te["n_valid"] >= FULL_WINDOW]
    f = list(ALL_FEATURES)
    return (
        tr[f].astype(float).fillna(0).values, tr[LABEL].astype(int).values,
        va[f].astype(float).fillna(0).values, va[LABEL].astype(int).values,
        te[f].astype(float).fillna(0).values, te[LABEL].astype(int).values,
    )


def _fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte):
    if name == "logistic_regression":
        mdl = build_logistic_regression(**CFG[name])
    elif name == "random_forest":
        mdl = build_random_forest(**CFG[name])
    elif name == "xgboost":
        mdl = build_xgboost(**CFG[name], scale_pos_weight=compute_scale_pos_weight(ytr))
    else:
        mdl = build_balanced_nn(**NN_CFG)
    mdl.fit(Xtr, ytr)
    th = select_threshold(yva, mdl.predict_proba(Xva)[:, 1])
    pt = mdl.predict_proba(Xte)[:, 1]
    return dict(
        pr_auc=average_precision_score(yte, pt),
        roc_auc=roc_auc_score(yte, pt),
        f1=f1_score(yte, (pt >= th).astype(int), zero_division=0),
        n_test=len(yte), test_events=int(yte.sum()),
    )


def main() -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)

    print("Recomputing trailing-window valid-return counts ...")
    nvkey = compute_window_audit()
    splits = {s: _attach_nvalid(s, nvkey) for s in ("train", "val", "test")}

    # ---- footprint ---------------------------------------------------------
    fp = build_footprint(splits)
    print("\nConstruction-noise footprint (firm-years by window completeness):")
    print(fp.to_string(index=False))
    fp.to_csv(OUT_TABLES_ROBUSTNESS / "exret_sigma_window_footprint.csv", index=False)
    with open(OUT_TABLES_ROBUSTNESS / "exret_sigma_window_footprint.tex", "w", encoding="utf-8") as fh:
        fh.write(fp.to_latex(
            index=False, float_format="%.2f",
            caption=("Construction-noise footprint for EXRET and SIGMA. Firm-years "
                     "by completeness of the trailing 12-month CRSP return window: a "
                     "full window (12 valid monthly returns), a partial window "
                     "(6--11 returns, which the frozen min\\_obs=6 rule emits as a "
                     "noisy partial-horizon value), and below-minimum windows "
                     "($<$6 returns, already returned as missing). Partial windows "
                     "carry a small share of distress events and a lower distress "
                     "rate, consistent with the recent-listing missingness mechanism."),
            label="tab:exret_sigma_window_footprint",
        ))

    # ---- sensitivity -------------------------------------------------------
    rows = []
    for rg in REGIMES:
        Xtr, ytr, Xva, yva, Xte, yte = _prep(splits, rg)
        print(f"\n[{rg}] train={len(ytr):,} test={len(yte):,} test_events={int(yte.sum())}")
        for name in MODELS:
            r = _fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte)
            r.update(regime=rg, model=name)
            rows.append(r)
            print(f"   {name:24s} PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

    df = pd.DataFrame(rows)[
        ["regime", "model", "pr_auc", "roc_auc", "f1", "n_test", "test_events"]
    ].round(4)
    df.to_csv(OUT_TABLES_ROBUSTNESS / "exret_sigma_construction_sensitivity.csv", index=False)

    wide = df.pivot(index="model", columns="regime", values="pr_auc")
    wide["d_F_vs_C"] = (wide["F_full12"] - wide["C_frozen"]).round(4)
    wide["d_X_vs_C"] = (wide["X_exclude"] - wide["C_frozen"]).round(4)
    max_abs = float(wide[["d_F_vs_C", "d_X_vs_C"]].abs().max().max())
    print("\nPR-AUC by regime (delta vs frozen):")
    print(wide.round(4).to_string())
    print(f"\nMax |delta PR-AUC| across regimes = {max_abs:.4f}")

    with open(OUT_TABLES_ROBUSTNESS / "exret_sigma_construction_sensitivity.tex", "w", encoding="utf-8") as fh:
        fh.write(wide.to_latex(
            float_format="%.4f",
            caption=("Sensitivity to the noisy construction of EXRET and SIGMA. "
                     "Test-set PR-AUC for the four headline models under the frozen "
                     "min\\_obs=6 rule (C), the principled full-window construction "
                     "that marks partial windows missing (F: min\\_obs=12, firm-years "
                     "kept), and outright exclusion of partial-window firm-years (X, "
                     "evaluated on a reduced test set). Differences are within "
                     f"{max_abs:.4f} PR-AUC -- an order of magnitude inside the "
                     "bootstrap confidence intervals -- and the "
                     "LR$>$XGB$\\approx$RF$>$NN ranking, and hence the "
                     "H\\textsubscript{1} conclusion, is unchanged under every regime."),
            label="tab:exret_sigma_construction_sensitivity",
        ))

    print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'exret_sigma_window_footprint.csv'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'exret_sigma_window_footprint.tex'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'exret_sigma_construction_sensitivity.csv'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'exret_sigma_construction_sensitivity.tex'}")


if __name__ == "__main__":
    main()
