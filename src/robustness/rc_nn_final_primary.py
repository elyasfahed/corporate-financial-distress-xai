"""
The balanced neural network across RC1-RC5, on the reported run.
================================================================
**Classification: supplementary robustness evidence for a co-primary model.**

Why this exists
---------------
The neural network was promoted to co-primary status alongside the benchmark
and the two tree ensembles, but the pre-specified robustness battery under
``final_primary`` covers only LR, RF and XGBoost. The NN-across-checks
artifacts in the repository belong to superseded sample and label generations,
so one of the four reported models has had no current robustness evidence at
all. This module supplies it.

Protocol asymmetry --- stated plainly
-------------------------------------
LR, RF and XGBoost are **re-tuned from scratch** in each pre-specified check
(100 Optuna trials per model per check). The network here is **not**: it reuses
the hyperparameters selected for the primary specification and is refit on each
check's data. A per-check re-tune of the network is a one-to-three day
computation on this data (roughly 105 s per fit, times five folds, times 100
trials, times five checks), which is why the supplementary convention used
throughout Chapter 7 --- frozen hyperparameters, refit only --- is applied here
too.

The asymmetry is bounded rather than assumed away. The RC3 re-tune measured
what an independent search is worth to the other three models under a changed
design: +0.0023, +0.0058 and +0.0039 PR-AUC. Against the network's 0.027 deficit
to the random forest on the primary specification, a gain of that order would
not reorder anything. That is the argument for tolerating the asymmetry; it is
not a proof, and the tables say so.

What each check does to the network
-----------------------------------
RC1, RC2, RC4 and RC5 change the label, the horizon, the predictor set or the
sample, and the network is simply refit on the changed data with the balanced
(oversampling-to-parity) treatment it carries in the primary specification.

RC3 is different, because it changes the imbalance treatment itself --- the very
thing that makes this network the "balanced" one. Its analogue for the network
is to replace oversampling-to-parity with SMOTENC, and it is constructed exactly
as ``rc3_smote.run_rc3`` constructs it for the other models: continuous features
standardised on training statistics, binary indicators left in {0, 1} and
declared categorical, resampling applied to training rows only. The MLP itself
is taken from the same builder, so its architecture, penalty and learning rate
are identical to the primary network's.

Run
---
    PYTHONPATH=. python -m scripts.run_rc_nn_final_primary
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import (ACCOUNTING_FEATURES, ALL_FEATURES_V2, DATA_RAW_V2,
                        DATA_ROOT, DATA_SAMPLES_V2, DISTRESS_CODES_RC1_CORRECTED,
                        DISTRESS_HORIZON_DAYS, DISTRESS_HORIZON_RC2,
                        OUT_MODELS_CONFIGS, V2_PROFILE)

DELIST_PATH = DATA_RAW_V2 / "crsp_delisting_raw.parquet"
RC5_SAMPLES_DIR = DATA_ROOT / "processed_final_primary" / "samples_rc5"
NN_CONFIG = (Path(OUT_MODELS_CONFIGS) / V2_PROFILE["spec"]
             / "neural_network_balanced_config.yaml")

CHECKS = ["primary", "rc1", "rc2", "rc3", "rc4", "rc5"]

CHECK_LABEL = {
    "primary": "Primary specification",
    "rc1": "RC1 bankruptcy-reason proxy",
    "rc2": "RC2 six-month horizon",
    "rc3": "RC3 SMOTENC",
    "rc4": "RC4 accounting-only",
    "rc5": "RC5 utilities included",
}


def load_frozen_nn_params() -> dict:
    """The hyperparameters selected for the primary balanced network."""
    with open(NN_CONFIG) as fh:
        return (yaml.safe_load(fh) or {}).get("best_params", {})


def _load_splits(samples_dir=DATA_SAMPLES_V2):
    return (pd.read_parquet(Path(samples_dir) / "train.parquet"),
            pd.read_parquet(Path(samples_dir) / "val.parquet"),
            pd.read_parquet(Path(samples_dir) / "test.parquet"))


def get_check_data(check: str):
    """
    Return ``(train, val, test, features)`` for one check.

    The label, horizon, predictor set and sample are constructed with the same
    calls the LR/RF/XGBoost rows use, so the network's row is comparable to
    theirs rather than to a re-derivation of them.
    """
    from src.data.merge_crsp_compustat import build_distress_label

    features = list(ALL_FEATURES_V2)

    if check in ("primary", "rc3"):
        train, val, test = _load_splits()
        return train, val, test, features

    if check == "rc4":
        train, val, test = _load_splits()
        return train, val, test, list(ACCOUNTING_FEATURES)

    if check == "rc5":
        train, val, test = _load_splits(RC5_SAMPLES_DIR)
        return train, val, test, features

    delist = pd.read_parquet(DELIST_PATH)

    if check == "rc1":
        def relabel(split):
            split = split.drop(columns=["distress"], errors="ignore")
            return build_distress_label(
                split, delist,
                codes=DISTRESS_CODES_RC1_CORRECTED,
                event_label_col=V2_PROFILE["bankruptcy_event_col"],
                unique_event_assignment=V2_PROFILE["unique_event_assignment"],
            )
    elif check == "rc2":
        def relabel(split):
            split = split.drop(columns=["distress"], errors="ignore")
            return build_distress_label(
                split, delist,
                horizon_days=DISTRESS_HORIZON_RC2,
                event_label_col=V2_PROFILE["distress_event_col"],
                unique_event_assignment=V2_PROFILE["unique_event_assignment"],
            )
    else:
        raise ValueError(f"Unknown check {check!r}")

    train, val, test = _load_splits()
    return relabel(train), relabel(val), relabel(test), features


def get_check_raw_train(check: str) -> pd.DataFrame:
    """
    The RAW (pre-winsorisation) training frame for one check.

    Fold-safe cross-validation re-fits winsorisation, the coverage filter and
    the imputation medians inside each fold, so it must consume the raw frame.
    RC1 and RC2 relabel it exactly as they relabel the modelling splits ---
    omitting that would tune on primary-label folds and evaluate on a
    re-labelled test set.
    """
    from src.data.merge_crsp_compustat import build_distress_label

    samples = RC5_SAMPLES_DIR if check == "rc5" else DATA_SAMPLES_V2
    raw = pd.read_parquet(Path(samples) / "train_raw.parquet")

    if check in ("primary", "rc3", "rc4", "rc5"):
        return raw

    delist = pd.read_parquet(DELIST_PATH)
    raw = raw.drop(columns=["distress"], errors="ignore")
    if check == "rc1":
        return build_distress_label(
            raw, delist,
            codes=DISTRESS_CODES_RC1_CORRECTED,
            event_label_col=V2_PROFILE["bankruptcy_event_col"],
            unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        )
    if check == "rc2":
        return build_distress_label(
            raw, delist,
            horizon_days=DISTRESS_HORIZON_RC2,
            event_label_col=V2_PROFILE["distress_event_col"],
            unique_event_assignment=V2_PROFILE["unique_event_assignment"],
        )
    raise ValueError(f"Unknown check {check!r}")


def _fit_score_balanced(train, val, test, features, params):
    """Refit the balanced (oversampling-to-parity) network on this check."""
    from src.robustness.rc7b_neural_network_balanced import build_balanced_nn

    X_tr = train[features].astype(float).fillna(0).values
    X_va = val[features].astype(float).fillna(0).values
    X_te = test[features].astype(float).fillna(0).values

    model = build_balanced_nn(**params)
    model.fit(X_tr, train["distress"].astype(int).values)
    return (model.predict_proba(X_va)[:, 1], model.predict_proba(X_te)[:, 1])


def _fit_score_smotenc(train, val, test, features, params):
    """
    RC3 analogue: SMOTENC replaces oversampling-to-parity.

    Mirrors ``rc3_smote.run_rc3`` exactly --- continuous features standardised
    on training statistics, binaries preserved in {0, 1} and declared
    categorical, resampling on training rows only --- so the network's RC3 row
    is built the same way the other models' are. The MLP is taken from the
    shared builder so its architecture and regularisation are unchanged.
    """
    from src.robustness.rc3_smote import _scale_continuous, smote_resample
    from src.robustness.rc7b_neural_network_balanced import build_balanced_nn

    X_tr = train[features].astype(float).fillna(0).values
    X_va = val[features].astype(float).fillna(0).values
    X_te = test[features].astype(float).fillna(0).values

    X_tr, X_va, X_te, _ = _scale_continuous(X_tr, X_va, X_te, features)
    X_res, y_res = smote_resample(X_tr, train["distress"].astype(int).values,
                                  features, corrected=True)

    # The sampler and scaler steps are already applied above, so the bare
    # MLP is pulled out of the shared pipeline rather than rebuilt by hand.
    mlp = build_balanced_nn(**params).named_steps["clf"]
    mlp.fit(X_res, y_res)
    return (mlp.predict_proba(X_va)[:, 1], mlp.predict_proba(X_te)[:, 1])


def run_check(check: str, params: dict) -> dict:
    """Refit and evaluate the network for one check."""
    from src.models.evaluate import (bootstrap_pr_auc_ci, compute_all_metrics,
                                     select_threshold)

    train, val, test, features = get_check_data(check)
    y_va = val["distress"].astype(int).values
    y_te = test["distress"].astype(int).values

    print(f"\n  [{check}] train {len(train):,} ({int(train['distress'].sum())} events) | "
          f"val {len(val):,} ({int(y_va.sum())}) | "
          f"test {len(test):,} ({int(y_te.sum())}) | {len(features)} features")

    scorer = _fit_score_smotenc if check == "rc3" else _fit_score_balanced
    p_val, p_test = scorer(train, val, test, features, params)

    threshold = select_threshold(y_va, p_val)
    metrics = compute_all_metrics(y_te, p_test, threshold,
                                  model_name="neural_network_balanced")
    lo, hi = bootstrap_pr_auc_ci(y_te, p_test, test["gvkey"].values)

    row = {
        "check": CHECK_LABEL[check],
        "n_test": len(test),
        "test_events": int(y_te.sum()),
        "prevalence_baseline_pr_auc": round(float(y_te.mean()), 4),
        "pr_auc": metrics["pr_auc"],
        "pr_auc_ci_lower": round(lo, 4),
        "pr_auc_ci_upper": round(hi, 4),
        "roc_auc": metrics["roc_auc"],
        "f1": metrics["f1"],
        "n_features": len(features),
    }
    print(f"    PR-AUC={row['pr_auc']:.4f} "
          f"[{row['pr_auc_ci_lower']:.4f}, {row['pr_auc_ci_upper']:.4f}]  "
          f"ROC={row['roc_auc']:.4f}  baseline={row['prevalence_baseline_pr_auc']:.4f}")
    return row


def run_battery(checks: list[str] | None = None) -> pd.DataFrame:
    """Run the network through every requested check."""
    params = load_frozen_nn_params()
    print(f"  frozen NN hyperparameters: {params}")
    rows = [run_check(c, params) for c in (checks or CHECKS)]
    return pd.DataFrame(rows)
