"""
RC3 — Alternative class imbalance treatment: SMOTE.
====================================================
Changes: Synthetic Minority Over-sampling Technique (SMOTE) applied to
         the training set instead of class weights.
Fixed  : All other design dimensions unchanged.

Tests sensitivity of results to the imbalance correction method.
SMOTE is NOT the primary approach because it generates synthetic observations
that may not correspond to real firm characteristics. Class weights are
preferred as they do not alter the data distribution (Blueprint v4 §9.2).

Corrected mode (2026-07-12 audit remediation, Phase 5)
------------------------------------------------------
The frozen implementation has three defects, disclosed and fixed behind
``corrected=True`` (default False reproduces the frozen behaviour):

1. Plain SMOTE interpolates the binary indicators (OENEG, INTWO) into
   impossible fractional values. Corrected: SMOTENC, which treats them
   as categorical (synthetic values by neighbour majority vote).
2. SMOTE's kNN ran in RAW feature space, where distances are dominated
   by the large-scale size variables (LNTA, LNMK). Corrected: continuous
   features are standardised (train-fitted scaler) before resampling and
   the models are trained/evaluated in the same standardised space.
3. Models used library-default hyperparameters, so RC3 was not
   comparable to the tuned primary. Corrected: the tuned per-model
   configurations are loaded from ``params_dir`` (class weighting is
   still disabled — SMOTE replaces it, which is the point of RC3).

Reference: Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling
Technique." JAIR, 16, 321–357.

Blueprint v4 reference: §11, RC3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE, SMOTENC

from src.config import ALL_FEATURES, OUT_MODELS_CONFIGS, RANDOM_SEED

#: Binary predictors that SMOTE must not interpolate.
#:
#: 2026-07-29 audit fix — ``MB_MISSING`` was absent from this list while the
#: final-primary feature set (ALL_FEATURES_V2 = 17 + MB_MISSING) carries it as
#: a 0/1 indicator. SMOTENC therefore treated it as continuous and interpolated
#: it to fractional values, i.e. exactly the defect this module documents itself
#: as fixing, applied to only two of the three binaries. The list is now the
#: union over every supported feature set and is validated against the data by
#: :func:`binary_feature_indices`, so a future feature addition cannot silently
#: reintroduce the bug.
BINARY_FEATURES = ["OENEG", "INTWO", "MB_MISSING"]


def binary_feature_indices(features, X=None):
    """
    Column indices of the declared binary predictors present in ``features``.

    When ``X`` is supplied, each candidate column is validated to contain only
    {0, 1} (ignoring NaN). A declared binary carrying any other value is a
    protocol abort condition (§19.6) rather than something to coerce silently.

    Parameters
    ----------
    features : list[str]
    X : np.ndarray or None
        Feature matrix aligned to ``features`` (columns in the same order).

    Returns
    -------
    list[int]
    """
    idx = [i for i, f in enumerate(features) if f in BINARY_FEATURES]
    if X is not None:
        for i in idx:
            col = np.asarray(X[:, i], dtype=float)
            col = col[~np.isnan(col)]
            offending = np.setdiff1d(np.unique(col), np.array([0.0, 1.0]))
            if offending.size:
                raise ValueError(
                    f"Declared binary feature {features[i]!r} carries "
                    f"non-binary values {offending[:5]} — refusing to treat it "
                    "as categorical on corrupted input (protocol §19.6)."
                )
    return idx


def _scale_continuous(X_train, X_val, X_test, features):
    """
    Standardise the CONTINUOUS features on train statistics; binary
    columns are left untouched (0/1). Returns transformed copies and
    the list of binary column indices (for SMOTENC).
    """
    from sklearn.preprocessing import StandardScaler

    binary_idx = binary_feature_indices(features, X_train)
    cont_idx = [i for i in range(len(features)) if i not in binary_idx]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    if cont_idx:
        scaler = StandardScaler().fit(X_train[:, cont_idx])
        X_train[:, cont_idx] = scaler.transform(X_train[:, cont_idx])
        X_val[:, cont_idx] = scaler.transform(X_val[:, cont_idx])
        X_test[:, cont_idx] = scaler.transform(X_test[:, cont_idx])
    return X_train, X_val, X_test, binary_idx


def smote_resample(X_train, y_train, features, corrected: bool = False):
    """
    Resample the training set with SMOTE (frozen) or SMOTENC (corrected).

    Corrected mode keeps binary features in {0, 1}; frozen mode
    reproduces the original plain-SMOTE behaviour, fractional binaries
    included.
    """
    binary_idx = binary_feature_indices(features, X_train)
    if corrected and binary_idx:
        sampler = SMOTENC(categorical_features=binary_idx,
                          random_state=RANDOM_SEED)
    else:
        sampler = SMOTE(random_state=RANDOM_SEED)
    return sampler.fit_resample(X_train, y_train)


def _load_tuned_params(params_dir: Path) -> dict:
    """Load the tuned best_params for each model from config YAMLs."""
    params = {}
    for name in ("logistic_regression", "random_forest", "xgboost"):
        cfg_path = Path(params_dir) / f"{name}_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"RC3 corrected mode needs tuned configs; missing {cfg_path}"
            )
        with open(cfg_path, "r") as f:
            params[name] = (yaml.safe_load(f) or {}).get("best_params", {})
    return params


def run_rc3(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    corrected: bool = False,
    params_dir=None,
) -> pd.DataFrame:
    """
    Execute RC3: apply SMOTE to training set; retrain and evaluate.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    features : list[str]
    corrected : bool, default False
        False reproduces the frozen behaviour (plain SMOTE in raw space,
        library-default models). True applies the audit fixes: SMOTENC,
        standardised continuous space, tuned hyperparameters.
    params_dir : Path or None
        Where corrected mode loads the tuned configs from. None = the
        primary config dir; the v2 robustness run passes configs/v2.

    Returns
    -------
    pd.DataFrame
        Performance metrics under SMOTE.
    """
    from src.models.logistic_regression import build_logistic_regression
    from src.models.random_forest import build_random_forest
    from src.models.xgboost_model import build_xgboost
    from src.models.evaluate import compute_all_metrics, select_threshold, bootstrap_pr_auc_ci

    mode = "corrected (SMOTENC, scaled, tuned params)" if corrected else "frozen"
    print(f"\nRC3: SMOTE imbalance correction (instead of class weights) [{mode}]")

    X_train = train[features].astype(float).fillna(0).values
    y_train = train["distress"].astype(int).values
    X_val   = val[features].astype(float).fillna(0).values
    y_val   = val["distress"].astype(int).values
    X_test  = test[features].astype(float).fillna(0).values
    y_test  = test["distress"].astype(int).values
    firm_ids = test["gvkey"].values

    if corrected:
        X_train, X_val, X_test, _ = _scale_continuous(
            X_train, X_val, X_test, features)

    # Oversample the minority class in the training set only
    X_res, y_res = smote_resample(X_train, y_train, features,
                                  corrected=corrected)
    print(f"  SMOTE: {len(y_train):,} -> {len(y_res):,} training obs "
          f"({int(y_res.sum()):,} positive)")

    # Build models without class weighting — SMOTE handles imbalance
    if corrected:
        tuned = _load_tuned_params(params_dir or OUT_MODELS_CONFIGS)
        model_builders = {
            "logistic_regression": lambda: build_logistic_regression(
                **tuned["logistic_regression"], class_weight=None),
            "random_forest": lambda: build_random_forest(
                **tuned["random_forest"], class_weight=None),
            "xgboost": lambda: build_xgboost(
                **tuned["xgboost"], scale_pos_weight=1.0),
        }
    else:
        model_builders = {
            "logistic_regression": lambda: build_logistic_regression(class_weight=None),
            "random_forest":       lambda: build_random_forest(class_weight=None),
            "xgboost":             lambda: build_xgboost(scale_pos_weight=1.0),
        }

    rows = []
    for name, builder in model_builders.items():
        model = builder()
        model.fit(X_res, y_res)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        threshold  = select_threshold(y_val, y_prob_val)

        y_prob_test = model.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, y_prob_test, threshold, model_name=name)

        ci_lo, ci_hi = bootstrap_pr_auc_ci(y_test, y_prob_test, firm_ids)
        metrics["pr_auc_ci_lower"] = round(ci_lo, 4)
        metrics["pr_auc_ci_upper"] = round(ci_hi, 4)
        rows.append(metrics)
        print(f"  {name:25s}  PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}")

    return pd.DataFrame(rows)
