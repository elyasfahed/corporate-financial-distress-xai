"""
Refit-based SHAP stability — audit item 9.
===========================================
What the existing analysis measures, and what it does not
---------------------------------------------------------
``src/explainability/shap_stability.py`` bootstraps the **test rows** against a
**single fitted model**. The estimator is never refitted, so the quantity it
reports is the *conditional* sampling variability of the explanation given that
one model. That is a real quantity, but it is not what "SHAP instability"
normally means, and it explains the implausibly clean result on the reported run:
rank standard deviation of exactly 0 for every top-ten feature. With the trees
held fixed, the mean-|SHAP| ordering is a near-deterministic function of the
model, so resampling test rows barely perturbs it.

The economically interesting question is different: **would a differently-drawn
training sample have produced a different explanation?** Blueprint v4 §10.5
requires material ranking changes to be reported rather than suppressed, and a
fixed-model bootstrap structurally cannot surface them.

Method
------
Two refit schemes, both holding the tuned hyperparameters fixed (this is a
stability analysis, not a re-tune — re-tuning would confound estimator variance
with search variance):

``seed``
    Refit on the same training data with different estimator random seeds.
    Isolates pure algorithmic non-determinism (bagging draws, column subsampling).
``firm_bootstrap``
    Resample training FIRMS with replacement, refit, then explain the SAME fixed
    test set. Isolates sensitivity to the training draw — the notion of
    stability that matters for whether the economic reading is an artefact of
    one sample.

In both cases SHAP is computed on the identical test matrix for every refit, so
the test sample contributes no variance and the spread is attributable to the
refit alone. Reported per feature: mean rank, rank SD, rank range, the 95%
interval, and a ``stable`` flag (rank interval spanning at most 3 positions,
matching the existing convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.config import RANDOM_SEED

#: Rank-interval width beyond which a feature is flagged UNSTABLE.
MAX_STABLE_SPAN = 3


def _mean_abs_shap(model, X: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                      # older shap: per-class list
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # (n, features, classes)
        sv = sv[:, :, -1]
    return np.abs(sv).mean(axis=0)


def _refit_xgboost(params: dict, X: np.ndarray, y: np.ndarray, seed: int):
    from src.models.xgboost_model import build_xgboost

    p = dict(params)
    p.pop("random_state", None)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    spw = (n_neg / n_pos) if n_pos else 1.0
    model = build_xgboost(**p, scale_pos_weight=spw)
    try:
        model.set_params(random_state=seed)
    except Exception:
        pass
    model.fit(X, y)
    return model


def run_refit_shap_stability(
    params: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    scheme: str = "firm_bootstrap",
    n_refits: int = 25,
    seed: int = RANDOM_SEED,
    firm_col: str = "gvkey",
    label_col: str = "distress",
) -> pd.DataFrame:
    """
    Rank stability of mean |SHAP| across model refits.

    Parameters
    ----------
    params : dict
        Tuned hyperparameters, held FIXED across refits.
    train, test : pd.DataFrame
    features : list[str]
    scheme : {"firm_bootstrap", "seed"}
    n_refits : int
        Number of refits (25 is ample for rank spread and keeps runtime sane;
        each refit is a full XGBoost fit plus a TreeExplainer pass).
    seed : int
    firm_col, label_col : str

    Returns
    -------
    pd.DataFrame
        One row per feature: mean_rank, std_rank, min/max rank, 95% interval,
        mean |SHAP|, and the stability flag.
    """
    if scheme not in {"firm_bootstrap", "seed"}:
        raise ValueError(f"scheme must be 'firm_bootstrap' or 'seed', got {scheme!r}")

    X_test = test[features].astype(float).values
    y_train_all = train[label_col].astype(int).values
    X_train_all = train[features].astype(float).values

    rng = np.random.default_rng(seed)
    firms = pd.unique(train[firm_col])
    firm_pos = {f: np.where(train[firm_col].values == f)[0] for f in firms}

    rank_rows, mag_rows = [], []
    for k in range(n_refits):
        if scheme == "seed":
            Xb, yb = X_train_all, y_train_all
        else:
            pick = rng.choice(firms, size=len(firms), replace=True)
            idx = np.concatenate([firm_pos[f] for f in pick])
            Xb, yb = X_train_all[idx], y_train_all[idx]
            if yb.min() == yb.max():              # degenerate draw
                continue

        model = _refit_xgboost(params, Xb, yb, seed=seed + k)
        mag = _mean_abs_shap(model, X_test)
        # rank 1 = most important
        order = np.argsort(-mag)
        ranks = np.empty(len(features), dtype=float)
        ranks[order] = np.arange(1, len(features) + 1)
        rank_rows.append(ranks)
        mag_rows.append(mag)

    R = np.vstack(rank_rows)
    M = np.vstack(mag_rows)
    out = pd.DataFrame({
        "feature": features,
        "mean_rank": R.mean(axis=0).round(2),
        "std_rank": R.std(axis=0, ddof=1).round(2),
        "min_rank": R.min(axis=0).astype(int),
        "max_rank": R.max(axis=0).astype(int),
        "rank_ci_lower": np.percentile(R, 2.5, axis=0).round(1),
        "rank_ci_upper": np.percentile(R, 97.5, axis=0).round(1),
        "mean_abs_shap": M.mean(axis=0),
        "n_refits": len(R),
        "scheme": scheme,
    })
    out["rank_span"] = out["rank_ci_upper"] - out["rank_ci_lower"]
    out["stable"] = out["rank_span"] <= MAX_STABLE_SPAN
    return out.sort_values("mean_rank").reset_index(drop=True)


def compare_conditional_vs_refit(
    conditional: pd.DataFrame,
    refit: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Put the fixed-model and refit-based rank spreads side by side.

    The comparison is the point of the exercise: if the conditional analysis
    reports SD 0 while the refit analysis reports a materially larger spread,
    the original figure was measuring the wrong source of variation.
    """
    c = conditional.rename(columns={
        "std_rank": "std_rank_conditional", "mean_rank": "mean_rank_conditional"
    })[["feature", "mean_rank_conditional", "std_rank_conditional"]]
    r = refit.rename(columns={
        "std_rank": "std_rank_refit", "mean_rank": "mean_rank_refit"
    })[["feature", "mean_rank_refit", "std_rank_refit", "rank_span", "stable"]]
    out = c.merge(r, on="feature", how="outer")
    out = out.sort_values("mean_rank_refit").head(top_n).reset_index(drop=True)
    out["std_rank_increase"] = (
        out["std_rank_refit"] - out["std_rank_conditional"]
    ).round(2)
    return out
