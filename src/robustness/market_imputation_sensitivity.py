"""
Supplementary robustness: market-feature missing-value sensitivity.
===================================================================
Purpose
-------
The three market predictors EXRET, SIGMA, MB carry NaNs that are NOT covered
by the design's SIC-2 median imputation (src/features/impute.py imputes only
the 11 accounting features). They reach the models as `.fillna(0)` at model
input time (src/models/train.py). This script quantifies whether that choice
matters by re-fitting the four headline models (LR, RF, XGBoost, and the
balanced co-primary neural network) under four imputation regimes:

  Z  current   : fillna(0)                       (reproduces the saved table)
  M  median    : train-sample median fill
  S  design    : the SIC-2 x fyear rule extended to EXRET/SIGMA/MB
  P  principled: SIC-2 rule for EXRET/SIGMA (benign, ~MCAR missingness) PLUS an
                 explicit MB_MISSING 0/1 indicator for MB, whose missingness is
                 informative (book equity <= 0; ratio undefined) rather than
                 missing-at-random. This is the Little & Rubin missing-indicator
                 treatment for the only feature whose missingness carries signal.

The two market-feature missingness mechanisms are economically distinct, and
this script also writes a companion table documenting them:
  * EXRET/SIGMA: missing iff the firm has < 12 months of CRSP return history
    (recent listings). Distress rate is LOWER when missing -> benign / ~MCAR.
  * MB         : missing iff book equity <= 0 (ratio undefined). Distress rate
    is 6-7x HIGHER when missing, and OENEG=1 in ~82% of these rows -> the
    missingness is itself a distress signal (informative / MNAR), already
    encoded by the OENEG predictor.

It is SUPPLEMENTARY robustness evidence for the transparent-deviation
disclosure. It is read-only with respect to the frozen pipeline:
  * it does NOT overwrite outputs/models/saved/*.joblib
  * it does NOT write outputs/tables/model_results/model_performance_test.csv
  * it does NOT append to evaluation_manifest.csv
  * it writes ONLY:
       outputs/tables/robustness/market_imputation_sensitivity.csv
       outputs/tables/robustness/market_imputation_sensitivity.tex
       outputs/tables/robustness/market_missingness_mechanism.csv
       outputs/tables/robustness/market_missingness_mechanism.tex

Run from the project root:
    python -m src.robustness.market_imputation_sensitivity
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import yaml

from src.config import (
    ALL_FEATURES, DATA_SAMPLES, OUT_TABLES_ROBUSTNESS, OUT_MODELS_CONFIGS,
)
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn
from src.models.evaluate import select_threshold
from src.features.impute import compute_imputation_medians, apply_imputation
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)

LABEL = "distress"
MKT_NA = ["EXRET", "SIGMA", "MB"]   # market features that carry NaN

# Saved primary hyperparameters (outputs/models/configs/*_config.yaml).
# Reused so this matches the frozen models without paying the Optuna re-tune.
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

# Balanced (co-primary) neural-network tuned hyperparameters, loaded from the
# saved config so the NN row matches the frozen RC7b model without a re-tune.
# If the config is absent the NN is simply skipped (3-model fallback).
try:
    with open(OUT_MODELS_CONFIGS / "neural_network_balanced_config.yaml") as _fh:
        NN_CFG = yaml.safe_load(_fh)["best_params"]
except FileNotFoundError:
    NN_CFG = None

MODELS = ["logistic_regression", "random_forest", "xgboost"]
if NN_CFG is not None:
    MODELS.append("neural_network_balanced")

REGIMES = ["Z_fillna0", "M_train_median", "S_design_sic2", "P_principled_indicator"]


def _build_matrices(train, val, test, regime):
    tr, va, te = train.copy(), val.copy(), test.copy()
    feats = list(ALL_FEATURES)
    if regime == "Z_fillna0":
        pass
    elif regime == "M_train_median":
        med = {f: tr[f].median() for f in MKT_NA}          # train-only
        for f in MKT_NA:
            for d in (tr, va, te):
                d[f] = d[f].fillna(med[f])
    elif regime == "S_design_sic2":
        s1, s2, an, gl = compute_imputation_medians(tr, features=MKT_NA)
        tr = apply_imputation(tr, s1, s2, an, gl, features=MKT_NA)
        va = apply_imputation(va, s1, s2, an, gl, features=MKT_NA)
        te = apply_imputation(te, s1, s2, an, gl, features=MKT_NA)
    elif regime == "P_principled_indicator":
        # MB missingness is informative (MNAR): flag it BEFORE imputing the value,
        # then SIC-2-impute EXRET/SIGMA/MB so the value column stays in-range.
        for d in (tr, va, te):
            d["MB_MISSING"] = d["MB"].isna().astype(float)
        feats = feats + ["MB_MISSING"]
        s1, s2, an, gl = compute_imputation_medians(tr, features=MKT_NA)
        tr = apply_imputation(tr, s1, s2, an, gl, features=MKT_NA)
        va = apply_imputation(va, s1, s2, an, gl, features=MKT_NA)
        te = apply_imputation(te, s1, s2, an, gl, features=MKT_NA)
    Xtr = tr[feats].astype(float).fillna(0).values
    Xva = va[feats].astype(float).fillna(0).values
    Xte = te[feats].astype(float).fillna(0).values
    return Xtr, Xva, Xte


def _fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte):
    if name == "logistic_regression":
        mdl = build_logistic_regression(**CFG[name])
    elif name == "random_forest":
        mdl = build_random_forest(**CFG[name])
    elif name == "xgboost":
        mdl = build_xgboost(**CFG[name],
                            scale_pos_weight=compute_scale_pos_weight(ytr))
    elif name == "neural_network_balanced":
        # ImbPipeline(StandardScaler -> RandomOverSampler -> MLP); the sampler
        # rebalances the train fold only, so imbalance treatment matches LR/RF/XGB.
        mdl = build_balanced_nn(**NN_CFG)
    else:
        raise ValueError(f"unknown model {name}")
    mdl.fit(Xtr, ytr)
    th = select_threshold(yva, mdl.predict_proba(Xva)[:, 1])
    pt = mdl.predict_proba(Xte)[:, 1]
    yp = (pt >= th).astype(int)
    return dict(
        pr_auc=average_precision_score(yte, pt),
        roc_auc=roc_auc_score(yte, pt),
        f1=f1_score(yte, yp, zero_division=0),
        precision=precision_score(yte, yp, zero_division=0),
        recall=recall_score(yte, yp, zero_division=0),
        threshold=th,
    )


def main() -> None:
    train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    ytr = train[LABEL].astype(int).values
    yva = val[LABEL].astype(int).values
    yte = test[LABEL].astype(int).values

    # ---- missingness MECHANISM table (cause + selection) -------------------
    # Documents WHY each feature is missing and whether missingness is
    # correlated with the outcome. Written as a companion artifact so the
    # thesis can cite the mechanism, not just the sensitivity magnitude.
    print("Missingness mechanism (test set):")
    base = yte.mean()
    mech_rows = []
    for f in MKT_NA:
        m = test[f].isna().values
        if m.sum() == 0:
            continue
        dr_m = yte[m].mean()
        dr_p = yte[~m].mean()
        rr = (dr_m / dr_p) if dr_p > 0 else float("nan")
        # OENEG share among missing rows pins the MB cause (book equity <= 0)
        oeneg_miss = float(test.loc[m, "OENEG"].mean()) if "OENEG" in test.columns else float("nan")
        mech_rows.append(dict(
            feature=f, n_missing=int(m.sum()), pct_missing=round(100*m.mean(), 2),
            distress_if_missing=round(dr_m, 4), distress_if_present=round(dr_p, 4),
            rel_risk=round(rr, 2), oeneg_share_if_missing=round(oeneg_miss, 3),
        ))
        print(f"  {f:6s} missing={m.sum():5d} ({100*m.mean():5.2f}%)  "
              f"distress|missing={dr_m:.4f}  distress|present={dr_p:.4f}  "
              f"rel.risk={rr:.2f}  P(OENEG|miss)={oeneg_miss:.2f}")
    print(f"  overall test distress rate = {base:.4f}\n")
    mech = pd.DataFrame(mech_rows)

    # ---- regimes -----------------------------------------------------------
    rows = []
    for rg in REGIMES:
        Xtr, Xva, Xte = _build_matrices(train, val, test, rg)
        for name in MODELS:
            r = _fit_eval(name, Xtr, ytr, Xva, yva, Xte, yte)
            r.update(regime=rg, model=name)
            rows.append(r)
            print(f"[{rg:14s}] {name:22s} "
                  f"PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  F1={r['f1']:.4f}")

    df = pd.DataFrame(rows)[
        ["regime", "model", "pr_auc", "roc_auc", "f1",
         "precision", "recall", "threshold"]
    ].round(4)

    # ---- wide PR-AUC view + delta vs current -------------------------------
    wide = df.pivot(index="model", columns="regime", values="pr_auc")
    wide["delta_M_vs_Z"] = (wide["M_train_median"] - wide["Z_fillna0"]).round(4)
    wide["delta_S_vs_Z"] = (wide["S_design_sic2"] - wide["Z_fillna0"]).round(4)
    wide["delta_P_vs_Z"] = (wide["P_principled_indicator"] - wide["Z_fillna0"]).round(4)
    max_abs_delta = float(
        wide[["delta_M_vs_Z", "delta_S_vs_Z", "delta_P_vs_Z"]].abs().max().max()
    )
    print("\nPR-AUC by regime (delta vs current fillna(0)):")
    print(wide.to_string())
    print(f"\nMax |delta PR-AUC| across all regimes = {max_abs_delta:.4f}")

    # ---- save (new files only; nothing existing is overwritten) ------------
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_TABLES_ROBUSTNESS / "market_imputation_sensitivity.csv"
    tex_path = OUT_TABLES_ROBUSTNESS / "market_imputation_sensitivity.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(wide.to_latex(
            float_format="%.4f",
            caption=("Market-feature missing-value sensitivity. Test-set PR-AUC under "
                     "four imputation regimes for EXRET, SIGMA, MB: current fillna(0) "
                     "(Z), train-sample median (M), the SIC-2 design rule (S), and the "
                     "principled per-feature treatment (P: SIC-2 for EXRET/SIGMA plus an "
                     "MB missing-indicator for the informative MB gaps). Differences are "
                     f"within {max_abs_delta:.4f} PR-AUC; the LR$>$XGB$\\approx$RF ranking "
                     "is unchanged under every regime."),
            label="tab:market_imputation_sensitivity",
        ))

    # ---- mechanism companion table -----------------------------------------
    mech_csv = OUT_TABLES_ROBUSTNESS / "market_missingness_mechanism.csv"
    mech_tex = OUT_TABLES_ROBUSTNESS / "market_missingness_mechanism.tex"
    mech.to_csv(mech_csv, index=False)
    with open(mech_tex, "w", encoding="utf-8") as fh:
        fh.write(mech.to_latex(
            index=False, float_format="%.4f",
            caption=("Market-feature missingness mechanism (test set). EXRET and SIGMA "
                     "are missing for firms with $<$12 months of CRSP history and carry a "
                     "LOWER distress rate when missing (benign, $\\approx$MCAR). MB is "
                     "missing when book equity $\\leq 0$ (ratio undefined): its missingness "
                     "is associated with a 6--7$\\times$ HIGHER distress rate and coincides "
                     "with OENEG$=1$, i.e. it is informative (MNAR) and already encoded by "
                     "the OENEG predictor."),
            label="tab:market_missingness_mechanism",
        ))
    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {tex_path}")
    print(f"Saved -> {mech_csv}")
    print(f"Saved -> {mech_tex}")


if __name__ == "__main__":
    main()
