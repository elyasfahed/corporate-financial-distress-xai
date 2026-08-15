"""
Supplementary robustness (Level B): does relaxing the 10-K criterion change the
result? -- the alternative fiscal-year-end-anchored distress label.
===========================================================================
This is the executable counterpart to severe_case_selection_footprint.py. The
footprint script only *counts* the for-cause delistings the 10-K-anchored label
misses (documented selection issue / Implementation Status §10). This script re-labels
the sample with the alternative fiscal-year-end anchor that recovers those
went-dark failures, re-fits the four headline models, and reports whether the
PR-AUC ranking and the H1 conclusion move.

SAFETY -- this module is READ-ONLY with respect to the frozen pipeline. It re-fits
models *in memory* using the frozen hyperparameters and NEVER writes to
outputs/models/saved/*. It does NOT call train_all_models (which would overwrite
the frozen .joblib files). It writes ONLY:
    outputs/tables/robustness/severe_case_selection_sensitivity.{csv,tex}

Regimes (all re-fit with the same frozen hyperparameters, for a like-for-like
within-harness comparison):
    F_fdate    : frozen primary label (anchor = actual 10-K filing date, 365d).
                 Re-labels via build_distress_label(anchor="fdate") so the
                 baseline is produced by the exact same re-fit procedure.
    D_datadate : alternative label (anchor = fiscal year-end, reach = 730d).
                 Recovers ~75% of the orphaned went-dark for-cause delistings.

Caveat (stated in ch07): the datadate anchor pairs the outcome with accounting
data that, for went-dark firms, was never actually filed in time -- it trades the
point-in-time guarantee for outcome coverage. This check is therefore supplementary
/ future-work, NOT part of the frozen specification.

Run from the project root:
    python -m src.robustness.severe_case_selection_sensitivity
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from src.config import (
    ALL_FEATURES, DATA_SAMPLES, DATA_RAW_CRSP, OUT_MODELS_CONFIGS,
    OUT_TABLES_ROBUSTNESS, DISTRESS_CODES_PRIMARY,
)
from src.data.merge_crsp_compustat import build_distress_label
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn
from src.models.evaluate import select_threshold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

LABEL = "distress"
DATADATE_REACH = 730  # 24-month fiscal-year-end reach (recovers ~75% of orphans)

# Frozen primary hyperparameters, reused so this matches the frozen models
# without paying the Optuna re-tune (same values as the §8/§9 sensitivity scripts).
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


def _relabel(split: pd.DataFrame, delist: pd.DataFrame, **kw) -> pd.DataFrame:
    """Re-label one split under a given anchor; drop stale label columns first."""
    s = split.drop(columns=["distress", "delist_date", "delist_code"], errors="ignore")
    return build_distress_label(s, delist, codes=DISTRESS_CODES_PRIMARY, **kw)


def _prep(splits):
    f = list(ALL_FEATURES)
    tr, va, te = splits["train"], splits["val"], splits["test"]
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
    delist = pd.read_parquet(DATA_RAW_CRSP / "crsp_delisting_raw.parquet")
    base = {s: pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in ("train", "val", "test")}

    regimes = {
        "F_fdate":    dict(anchor="fdate"),
        "D_datadate": dict(anchor="datadate", reach_days=DATADATE_REACH),
    }

    rows = []
    for rg, kw in regimes.items():
        print(f"\n=== Regime {rg}: {kw} ===")
        splits = {s: _relabel(base[s], delist, **kw) for s in base}
        for s in splits:
            n, d = len(splits[s]), int(splits[s][LABEL].sum())
            print(f"   {s:5s}: {n:,} firm-years, {d} distress ({100*d/n:.2f}%)")
        Xtr, ytr, Xva, yva, Xte, yte = _prep(splits)
        for name in MODELS:
            r = _fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte)
            r.update(regime=rg, model=name)
            rows.append(r)
            print(f"   {name:24s} PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  "
                  f"events={r['test_events']}")

    df = pd.DataFrame(rows)[
        ["regime", "model", "pr_auc", "roc_auc", "f1", "n_test", "test_events"]
    ].round(4)
    df.to_csv(OUT_TABLES_ROBUSTNESS / "severe_case_selection_sensitivity.csv", index=False)

    wide = df.pivot(index="model", columns="regime", values="pr_auc")
    wide = wide.reindex(MODELS)
    wide["d_D_vs_F"] = (wide["D_datadate"] - wide["F_fdate"]).round(4)
    max_abs = float(wide["d_D_vs_F"].abs().max())
    print("\nTest PR-AUC by label regime (delta = datadate - fdate):")
    print(wide.round(4).to_string())
    print(f"\nMax |delta PR-AUC| (datadate vs fdate) = {max_abs:.4f}")
    print("Ranking under D_datadate:",
          " > ".join(wide["D_datadate"].sort_values(ascending=False).index))

    with open(OUT_TABLES_ROBUSTNESS / "severe_case_selection_sensitivity.tex", "w", encoding="utf-8") as fh:
        fh.write(wide.to_latex(
            float_format="%.4f",
            caption=("Sensitivity to relaxing the 10-K criterion: test-set PR-AUC for "
                     "the four headline models under the frozen 10-K-filing-date label "
                     "(F: anchor = filing date, 365d) and the alternative "
                     "fiscal-year-end-anchored label (D: anchor = fiscal year-end, 730d) "
                     "that recovers the went-dark for-cause delistings. Both regimes "
                     "re-fit in memory with the frozen hyperparameters; the frozen saved "
                     "models are not modified."),
            label="tab:severe_selection_sensitivity",
        ))
    print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'severe_case_selection_sensitivity.csv'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'severe_case_selection_sensitivity.tex'}")


if __name__ == "__main__":
    main()
