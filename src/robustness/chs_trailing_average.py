"""
Supplementary robustness: CHS-style trailing-average features.
===========================================================================
Methodological motivation:
Campbell, Hilscher & Szilagyi (2008, JF) do not feed the *spot* values of
profitability and excess return into their hazard model. They use geometrically
declining trailing averages -- NIMTAAVG and EXRETAVG -- on the grounds that a
firm's recent history of earnings and market performance, not just its latest
reading, carries the distress signal (the spot value is noisy; the trailing
average smooths transitory shocks and captures persistence).

The frozen 17-variable set uses the spot NITA (profitability) and spot EXRET
(12-month excess return). This module constructs the two CHS-style trailing
averages and tests whether *augmenting* the frozen set with them changes the
out-of-sample PR-AUC ranking or the H1 conclusion. It is deliberately ADDITIVE:
the 17 frozen predictors are left untouched and the two trailing averages are
appended (17 -> 19), so the check cannot disturb the frozen primary.

Construction (look-ahead-safe; all inputs observable at the 10-K filing date):
    For a firm-year at fiscal year t, the trailing average of x is a
    geometrically weighted mean of x over fiscal years {t, t-1, ..., t-(W-1)},
    weight decay^k on the year k steps back (k=0 is the current year). The
    average is fiscal-year-aligned (a gap in the firm's fyear sequence skips the
    missing year, it is not back-filled), and requires at least MIN_OBS observed
    years or the value is NaN -> standard fillna(0) at model-input time, exactly
    like the frozen market features.

        NITA_AVG  : CHS NIMTAAVG analogue (trailing average profitability)
        EXRET_AVG : CHS EXRETAVG analogue (trailing average excess return)

    W = 3 fiscal years, decay = 2**(-1) (weight halves each year back), MIN_OBS = 2.
    CHS use quarterly data with decay 2**(-1/3); here the panel is annual, so the
    window and decay are the natural annual analogue, documented as a design choice.

SAFETY -- READ-ONLY with respect to the frozen pipeline. Trailing averages are
built in memory from the frozen split parquets (which already carry NITA, EXRET,
gvkey, fyear); models are re-fit in memory with the frozen hyperparameters; the
frozen .joblib files in outputs/models/saved/ are NEVER touched. This script
writes ONLY:
    outputs/tables/robustness/chs_trailing_average.{csv,tex}

Caveat (stated in ch07): the frozen hyperparameters were tuned on the 17-feature
set. They are reused here (no re-tune), so this isolates the marginal value of
the two trailing-average features under the frozen configuration, not the value
they would have under a fresh hyperparameter search.

Run from the project root:
    python -m src.robustness.chs_trailing_average
    python -m src.robustness.chs_trailing_average --quick   # LR+XGB only (smoke test)
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from src.config import (
    ALL_FEATURES, DATA_SAMPLES, OUT_MODELS_CONFIGS, OUT_TABLES_ROBUSTNESS,
)
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn
from src.models.evaluate import select_threshold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

LABEL = "distress"
WINDOW = 3                 # trailing window in fiscal years (t, t-1, t-2)
DECAY = 0.5                # geometric weight: halves each year back
MIN_OBS = 2                # minimum observed years required, else NaN
TRAILING_SPEC = {"NITA": "NITA_AVG", "EXRET": "EXRET_AVG"}
EXTRA_FEATURES = list(TRAILING_SPEC.values())

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

ALL_MODELS = ["logistic_regression", "random_forest", "xgboost"]
if NN_CFG is not None:
    ALL_MODELS.append("neural_network_balanced")


def trailing_weighted_avg_by_fyear(
    df: pd.DataFrame, col: str, window: int = WINDOW,
    decay: float = DECAY, min_obs: int = MIN_OBS,
) -> pd.Series:
    """
    Geometrically weighted, fiscal-year-aligned trailing average of ``col``.

    For each row at fiscal year t, averages ``col`` over fiscal years
    {t, t-1, ..., t-(window-1)} with weight ``decay**k`` on the year k steps
    back (k=0 = current year). Years absent from the firm's fyear sequence are
    skipped (not back-filled); rows with fewer than ``min_obs`` observed,
    non-missing years return NaN. Uses only current/past data -> look-ahead-safe.

    Requires 'gvkey', 'fyear', and ``col``; (gvkey, fyear) assumed unique.
    """
    src = df[["gvkey", "fyear", col]].drop_duplicates(["gvkey", "fyear"], keep="last")
    base = df[["gvkey", "fyear"]].copy()

    wsum = np.zeros(len(df))            # sum of weight * value (over present years)
    nwt = np.zeros(len(df))            # sum of weights (over present years)
    cnt = np.zeros(len(df))            # count of present, non-missing years

    for k in range(window):
        w = decay ** k
        lk = base.copy()
        lk["fyear"] = lk["fyear"] - k
        merged = lk.merge(src, on=["gvkey", "fyear"], how="left")
        vals = merged[col].values.astype(float)
        present = ~np.isnan(vals)
        wsum[present] += w * vals[present]
        nwt[present] += w
        cnt[present] += 1

    out = np.where(nwt > 0, wsum / np.where(nwt == 0, np.nan, nwt), np.nan)
    out[cnt < min_obs] = np.nan
    return pd.Series(out, index=df.index)


def add_trailing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the CHS-style trailing averages to a split (in memory)."""
    df = df.copy()
    for src_col, new_col in TRAILING_SPEC.items():
        df[new_col] = trailing_weighted_avg_by_fyear(df, src_col)
    return df


def _build(name: str, y_fit: np.ndarray):
    if name == "logistic_regression":
        return build_logistic_regression(**CFG[name])
    if name == "random_forest":
        return build_random_forest(**CFG[name])
    if name == "xgboost":
        return build_xgboost(**CFG[name], scale_pos_weight=compute_scale_pos_weight(y_fit))
    return build_balanced_nn(**NN_CFG)


def _fit_eval(name, splits, features):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    Xtr = tr[features].astype(float).fillna(0).values
    ytr = tr[LABEL].astype(int).values
    Xva = va[features].astype(float).fillna(0).values
    yva = va[LABEL].astype(int).values
    Xte = te[features].astype(float).fillna(0).values
    yte = te[LABEL].astype(int).values
    mdl = _build(name, ytr)
    mdl.fit(Xtr, ytr)
    th = select_threshold(yva, mdl.predict_proba(Xva)[:, 1])
    pt = mdl.predict_proba(Xte)[:, 1]
    return dict(
        pr_auc=average_precision_score(yte, pt),
        roc_auc=roc_auc_score(yte, pt),
        f1=f1_score(yte, (pt >= th).astype(int), zero_division=0),
        n_test=len(yte), test_events=int(yte.sum()),
    )


def main(quick: bool = False) -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    base = {s: pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in ("train", "val", "test")}
    aug = {s: add_trailing_features(base[s]) for s in base}

    # Coverage diagnostic for the two new features (test split).
    te = aug["test"]
    for col in EXTRA_FEATURES:
        cov = te[col].notna().mean()
        print(f"   {col}: non-missing on test = {100*cov:.1f}%")

    models = ["logistic_regression", "xgboost"] if quick else ALL_MODELS
    feature_sets = {
        "F17_frozen": list(ALL_FEATURES),
        "A19_trailavg": list(ALL_FEATURES) + EXTRA_FEATURES,
    }

    rows = []
    for fs_name, feats in feature_sets.items():
        print(f"\n=== Feature set {fs_name} ({len(feats)} features) ===")
        for name in models:
            r = _fit_eval(name, aug, feats)
            r.update(feature_set=fs_name, model=name)
            rows.append(r)
            print(f"   {name:24s} PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}")

    df = pd.DataFrame(rows)[
        ["feature_set", "model", "pr_auc", "roc_auc", "f1", "n_test", "test_events"]
    ].round(4)
    if not quick:
        df.to_csv(OUT_TABLES_ROBUSTNESS / "chs_trailing_average.csv", index=False)

    wide = df.pivot(index="model", columns="feature_set", values="pr_auc").reindex(models)
    wide["d_A_vs_F"] = (wide["A19_trailavg"] - wide["F17_frozen"]).round(4)
    max_abs = float(wide["d_A_vs_F"].abs().max())
    print("\nTest PR-AUC by feature set (delta = augmented - frozen):")
    print(wide.round(4).to_string())
    print(f"\nMax |delta PR-AUC| (augmented vs frozen) = {max_abs:.4f}")
    print("Ranking under A19_trailavg:",
          " > ".join(wide["A19_trailavg"].sort_values(ascending=False).index))

    if quick:
        print("\n[quick mode] .tex NOT written (smoke test only)")
        return

    with open(OUT_TABLES_ROBUSTNESS / "chs_trailing_average.tex", "w", encoding="utf-8") as fh:
        fh.write(wide.to_latex(
            float_format="%.4f",
            caption=("CHS-style trailing-average features: test-set PR-AUC for the four "
                     "headline models on the frozen 17-feature set (F17) and on the same set "
                     "augmented with geometrically weighted trailing averages of profitability "
                     "(NITA\\_AVG) and excess return (EXRET\\_AVG) (A19). Both re-fit in memory "
                     "with the frozen hyperparameters; the frozen saved models are not modified."),
            label="tab:chs_trailing_average",
        ))
    print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'chs_trailing_average.csv'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'chs_trailing_average.tex'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LR+XGB only (smoke test)")
    main(**vars(ap.parse_args()))
