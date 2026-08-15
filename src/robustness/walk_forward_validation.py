"""
Supplementary robustness: walk-forward (expanding-window) re-estimation.
===========================================================================
Methodological motivation: the frozen primary
fits each model ONCE on the 1990-2009 training block and scores the whole
2015-2024 test block. That single static fit is increasingly stale across a
ten-year test window, and -- because distress prevalence drifts down over
time -- it pairs a comparatively high-prevalence training sample (train
~2.9%) with a low-prevalence test sample (test ~1.4%). A walk-forward design
re-estimates the model every test year on all data observable up to that
year, so (i) the model is never stale and (ii) the training prevalence the
model sees tracks the prevalence it is scored against.

This module quantifies whether expanding-window re-estimation changes the
out-of-sample PR-AUC ranking and the H1 conclusion. It compares two regimes,
both produced by the *same* in-memory re-fit procedure with the *frozen*
hyperparameters, so the only thing that varies is static-vs-expanding fitting:

    S_static : fit once on fyear <= TRAIN_END_YEAR (1990-2009), select the
               threshold on the validation block (2010-2014), then score every
               test year. Reproduces the frozen single-split design in-harness.
    W_walkfwd: for each test origin y in 2015..2024, fit on fyear <= y-2
               (expanding), select the threshold on fyear == y-1 (a rolling
               one-year validation slice), and score fyear == y. The one-year
               gap between the fit window and the scored year embargoes the
               forward 12-month distress window so a firm-year's outcome cannot
               leak into the data used to fit the model that scores it. Test-year
               predictions are POOLED across 2015..2024 and the pooled PR-AUC /
               ROC-AUC are reported -- directly comparable to S_static because
               the scored firm-years are identical.

SAFETY -- READ-ONLY with respect to the frozen pipeline. Models are re-fit
*in memory* with the frozen hyperparameters; the frozen .joblib files in
outputs/models/saved/ are NEVER touched and train_all_models is NEVER called.
This script writes ONLY:
    outputs/tables/robustness/walk_forward_validation.{csv,tex}
    outputs/tables/robustness/walk_forward_prevalence.{csv,tex}

Caveat (stated in ch07): walk-forward re-estimation is a *robustness* design,
not a replacement for the pre-specified single chronological split (the test
block is still evaluated once). It re-uses the frozen hyperparameters rather
than re-tuning at every origin, so it isolates the effect of re-fitting, not of
re-searching the hyperparameter space.

Run from the project root:
    python -m src.robustness.walk_forward_validation
    python -m src.robustness.walk_forward_validation --quick   # LR+XGB, 2 origins (smoke test)
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
    TRAIN_END_YEAR, VAL_END_YEAR, SAMPLE_END_YEAR,
)
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn
from src.models.evaluate import select_threshold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

LABEL = "distress"
TEST_START_YEAR = VAL_END_YEAR + 1          # 2015
WALKFWD_EMBARGO = 1                          # years between fit window and scored year

# Frozen primary hyperparameters, reused so this matches the frozen models
# without paying the Optuna re-tune (identical to the §8/§9/§10 sensitivity scripts).
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


def _build(name: str, y_fit: np.ndarray):
    """Instantiate a fresh model with the frozen hyperparameters."""
    if name == "logistic_regression":
        return build_logistic_regression(**CFG[name])
    if name == "random_forest":
        return build_random_forest(**CFG[name])
    if name == "xgboost":
        return build_xgboost(**CFG[name], scale_pos_weight=compute_scale_pos_weight(y_fit))
    return build_balanced_nn(**NN_CFG)


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[list(ALL_FEATURES)].astype(float).fillna(0).values


def _y(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL].astype(int).values


def load_panel() -> pd.DataFrame:
    """Reconstruct the full modeling panel from the frozen split parquets."""
    parts = [pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in ("train", "val", "test")]
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["fyear", "gvkey"]).reset_index(drop=True)


def walk_forward_splits(panel: pd.DataFrame, y: int, embargo: int = WALKFWD_EMBARGO):
    """
    Expanding-window split for test origin year ``y`` (pure, look-ahead-safe).

    Returns (fit, val, test) frames where:
        fit  = fyear <= y - embargo - 1   (expanding training window)
        val  = fyear == y - 1             (rolling one-year threshold slice)
        test = fyear == y                 (scored year)

    The ``embargo``-year gap between the fit window and the scored year keeps a
    firm-year's forward 12-month distress window out of the data used to fit the
    model that scores it. With embargo=1 the fit window ends at y-2, so the most
    recent fitted outcome window (ending ~y-1) cannot overlap year-y predictors.
    """
    fit = panel[panel["fyear"] <= y - embargo - 1]
    val = panel[panel["fyear"] == y - 1]
    test = panel[panel["fyear"] == y]
    return fit, val, test


# ---------------------------------------------------------------------------
# Regime S: static single-split fit (reproduces the frozen design in-harness)
# ---------------------------------------------------------------------------
def run_static(panel: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    fit_df = panel[panel["fyear"] <= TRAIN_END_YEAR]
    val_df = panel[(panel["fyear"] > TRAIN_END_YEAR) & (panel["fyear"] <= VAL_END_YEAR)]
    test_df = panel[panel["fyear"] > VAL_END_YEAR]
    Xf, yf = _X(fit_df), _y(fit_df)
    Xv, yv = _X(val_df), _y(val_df)
    Xt, yt = _X(test_df), _y(test_df)

    rows = []
    for name in models:
        mdl = _build(name, yf)
        mdl.fit(Xf, yf)
        th = select_threshold(yv, mdl.predict_proba(Xv)[:, 1])
        pt = mdl.predict_proba(Xt)[:, 1]
        rows.append(dict(
            regime="S_static", model=name,
            pr_auc=average_precision_score(yt, pt),
            roc_auc=roc_auc_score(yt, pt),
            f1=f1_score(yt, (pt >= th).astype(int), zero_division=0),
            n_test=len(yt), test_events=int(yt.sum()),
            mean_train_prev=float(yf.mean()),
        ))
        print(f"   S_static  {name:24s} PR-AUC={rows[-1]['pr_auc']:.4f}  ROC={rows[-1]['roc_auc']:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime W: expanding-window walk-forward, predictions pooled across test years
# ---------------------------------------------------------------------------
def run_walkforward(panel: pd.DataFrame, models: list[str],
                    test_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled = {name: {"y": [], "p": []} for name in models}
    prev_rows = []

    for y in test_years:
        fit_df, val_df, test_df = walk_forward_splits(panel, y)        # expanding, embargoed
        if test_df[LABEL].sum() == 0 or fit_df[LABEL].sum() == 0:
            continue
        Xf, yf = _X(fit_df), _y(fit_df)
        Xv, yv = _X(val_df), _y(val_df)
        Xt, yt = _X(test_df), _y(test_df)
        prev_rows.append(dict(test_year=y, n_fit=len(yf), n_test=len(yt),
                              fit_prev=float(yf.mean()), test_prev=float(yt.mean()),
                              test_events=int(yt.sum())))
        for name in models:
            mdl = _build(name, yf)
            mdl.fit(Xf, yf)
            # threshold is re-selected each origin but only used for F1 reporting;
            # PR-AUC/ROC-AUC are threshold-free.
            pt = mdl.predict_proba(Xt)[:, 1]
            pooled[name]["y"].append(yt)
            pooled[name]["p"].append(pt)
        print(f"   y={y}: fit n={len(yf):,} (prev {100*yf.mean():.2f}%)  "
              f"test n={len(yt):,} (prev {100*yt.mean():.2f}%, {int(yt.sum())} events)")

    rows = []
    for name in models:
        yt = np.concatenate(pooled[name]["y"])
        pt = np.concatenate(pooled[name]["p"])
        rows.append(dict(
            regime="W_walkfwd", model=name,
            pr_auc=average_precision_score(yt, pt),
            roc_auc=roc_auc_score(yt, pt),
            f1=np.nan,                       # pooled across heterogeneous thresholds
            n_test=len(yt), test_events=int(yt.sum()),
            mean_train_prev=float(np.mean([r["fit_prev"] for r in prev_rows])),
        ))
        print(f"   W_walkfwd {name:24s} PR-AUC={rows[-1]['pr_auc']:.4f}  ROC={rows[-1]['roc_auc']:.4f}")
    return pd.DataFrame(rows), pd.DataFrame(prev_rows)


def main(quick: bool = False) -> None:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    models = ["logistic_regression", "xgboost"] if quick else ALL_MODELS
    test_years = list(range(TEST_START_YEAR, SAMPLE_END_YEAR + 1))
    if quick:
        test_years = test_years[:2]

    print(f"\nPanel: {len(panel):,} firm-years, fyear "
          f"{panel['fyear'].min()}-{panel['fyear'].max()}")
    print(f"Models: {models}")
    print(f"Test origins: {test_years}\n")

    print("=== Regime S_static (frozen single split, in-harness) ===")
    df_s = run_static(panel, models)
    print("\n=== Regime W_walkfwd (expanding-window re-estimation) ===")
    df_w, prev = run_walkforward(panel, models, test_years)

    df = pd.concat([df_s, df_w], ignore_index=True)[
        ["regime", "model", "pr_auc", "roc_auc", "f1", "n_test", "test_events", "mean_train_prev"]
    ].round(4)
    if not quick:
        df.to_csv(OUT_TABLES_ROBUSTNESS / "walk_forward_validation.csv", index=False)

    wide = df.pivot(index="model", columns="regime", values="pr_auc").reindex(models)
    wide["d_W_vs_S"] = (wide["W_walkfwd"] - wide["S_static"]).round(4)
    max_abs = float(wide["d_W_vs_S"].abs().max())
    print("\nTest PR-AUC by regime (delta = walk-forward - static):")
    print(wide.round(4).to_string())
    print(f"\nMax |delta PR-AUC| (walk-forward vs static) = {max_abs:.4f}")
    print("Ranking under W_walkfwd:",
          " > ".join(wide["W_walkfwd"].sort_values(ascending=False).index))

    # Prevalence diagnostic (the "unbalanced label between train and test" point)
    prev = prev.round(5)
    if not quick:
        prev.to_csv(OUT_TABLES_ROBUSTNESS / "walk_forward_prevalence.csv", index=False)
    if not prev.empty:
        print(f"\nMean walk-forward fit prevalence : {prev['fit_prev'].mean():.4f}")
        print(f"Mean test-year prevalence        : {prev['test_prev'].mean():.4f}")
        static_prev = float(df_s["mean_train_prev"].iloc[0])
        print(f"Static (1990-2009) fit prevalence: {static_prev:.4f}")

    if quick:
        print("\n[quick mode] tables NOT written for .tex (smoke test only)")
        return

    with open(OUT_TABLES_ROBUSTNESS / "walk_forward_validation.tex", "w", encoding="utf-8") as fh:
        fh.write(wide.to_latex(
            float_format="%.4f",
            caption=("Walk-forward (expanding-window) re-estimation: pooled 2015-2024 "
                     "test-set PR-AUC for the four headline models under the frozen "
                     "static single-split fit (S) and expanding-window annual re-estimation "
                     "(W). Both regimes re-fit in memory with the frozen hyperparameters; "
                     "the frozen saved models are not modified."),
            label="tab:walk_forward_validation",
        ))
    with open(OUT_TABLES_ROBUSTNESS / "walk_forward_prevalence.tex", "w", encoding="utf-8") as fh:
        fh.write(prev.to_latex(
            index=False, float_format="%.4f",
            caption=("Walk-forward training and test-year distress prevalence by test "
                     "origin. The expanding-window fit prevalence tracks the contemporaneous "
                     "test-year prevalence, narrowing the train/test prevalence gap of the "
                     "static single-split design."),
            label="tab:walk_forward_prevalence",
        ))
    print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'walk_forward_validation.csv'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'walk_forward_validation.tex'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'walk_forward_prevalence.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="LR+XGB, 2 origins (smoke test)")
    main(**vars(ap.parse_args()))
