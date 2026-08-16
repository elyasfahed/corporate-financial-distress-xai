"""
Paired firm-cluster inference for ROC-AUC differences.
======================================================================
Why this exists
---------------
PR-AUC in this thesis already carries three uncertainty schemes: firm-block,
year-block and two-way (Cameron–Gelbach–Miller) — see
``src/robustness/temporal_bootstrap.py``. **ROC-AUC has only DeLong.**

``delong_test`` is correct and fast (vectorised midrank, Sun & Xu 2014), but the
DeLong variance is derived under **independent observations**. The reported test
sample is 25,512 firm-years drawn from 4,393 firms — roughly 5.8 correlated
observations per firm. Repeated firm-years are not independent draws.

This matters more than a generic caveat suggests: under the reported run the
**only comparisons surviving Holm correction are the ROC ones**
(XGB−LR Holm p = .0037, XGB−RF Holm p = .0029), while no PR-AUC pair survives.
The statistically significant findings therefore rest precisely on the inference
method whose independence assumption is violated. A clustered alternative is not
a refinement here; it is the check that decides whether those findings hold.

Method
------
Paired firm-cluster bootstrap. Firms are resampled with replacement; every
observation of a resampled firm travels with it, so the within-firm dependence
is preserved rather than assumed away. Both models are scored on the *same*
resample, so the difference is paired and the between-model correlation (the
thing DeLong models analytically) is captured empirically.

Two p-values are reported:

``p_percentile``
    Share of resamples in which the difference changes sign relative to the
    observed difference, doubled (equal-tailed). Assumption-light.
``p_normal``
    Wald p-value using the bootstrap standard error of the difference. Directly
    comparable to the DeLong z-test, which makes the inflation factor readable
    as ``se_cluster / se_delong``.

The observed difference is NOT recentred: unlike the PR-AUC significance test,
this is a CI-based test of ``H0: ΔAUC = 0``, and recentring would discard the
paired structure the bootstrap is here to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import BOOTSTRAP_REPS, RANDOM_SEED


def _firm_index_map(firm_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    firms = pd.unique(pd.Series(firm_ids))
    order = {f: i for i, f in enumerate(firms)}
    buckets: list[list[int]] = [[] for _ in firms]
    for pos, f in enumerate(firm_ids):
        buckets[order[f]].append(pos)
    return np.asarray(firms), [np.asarray(b, dtype=np.int64) for b in buckets]


def firm_cluster_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    firm_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict:
    """Firm-block bootstrap CI for a single model's ROC-AUC."""
    firms, buckets = _firm_index_map(np.asarray(firm_ids))
    rng = np.random.default_rng(seed)
    n_firms = len(firms)
    draws = []
    for _ in range(n_reps):
        pick = rng.integers(0, n_firms, size=n_firms)
        idx = np.concatenate([buckets[i] for i in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        draws.append(roc_auc_score(yt, y_prob[idx]))
    draws = np.asarray(draws)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ci_lower": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "se": float(draws.std(ddof=1)),
        "n_valid_resamples": int(len(draws)),
    }


def paired_firm_cluster_roc_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    firm_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Paired firm-cluster bootstrap test for ``AUC(a) - AUC(b)``.

    Returns observed delta, clustered CI and SE, and both p-values.
    """
    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    firms, buckets = _firm_index_map(np.asarray(firm_ids))
    rng = np.random.default_rng(seed)
    n_firms = len(firms)

    observed = float(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b))

    deltas = []
    for _ in range(n_reps):
        pick = rng.integers(0, n_firms, size=n_firms)
        idx = np.concatenate([buckets[i] for i in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        deltas.append(
            roc_auc_score(yt, y_prob_a[idx]) - roc_auc_score(yt, y_prob_b[idx])
        )
    deltas = np.asarray(deltas)
    se = float(deltas.std(ddof=1))

    # Equal-tailed percentile p-value: how often does the resampled difference
    # fall on the opposite side of zero from the observed one?
    if observed >= 0:
        tail = float((deltas <= 0).mean())
    else:
        tail = float((deltas >= 0).mean())
    p_percentile = float(min(1.0, 2 * tail))

    z = observed / se if se > 0 else np.nan
    p_normal = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

    return {
        "delta_auc": observed,
        "ci_lower": float(np.percentile(deltas, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        "se_cluster": se,
        "z_cluster": float(z) if np.isfinite(z) else np.nan,
        "p_percentile": p_percentile,
        "p_normal": p_normal,
        "n_valid_resamples": int(len(deltas)),
        "n_firms": int(n_firms),
        "n_obs": int(len(y_true)),
    }


def holm(pvals: list[float]) -> list[float]:
    """
    Holm-Bonferroni adjusted p-values, computed on FULL-PRECISION input.

    Rounding before adjustment is how a p just under .05 becomes one just over
    it; the protocol (§17) requires unrounded input.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def _paired_delta_draws(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    cluster_ids: np.ndarray,
    n_reps: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap draws of the paired AUC difference, resampling whole clusters."""
    clusters, buckets = _firm_index_map(np.asarray(cluster_ids))
    rng = np.random.default_rng(seed)
    n = len(clusters)
    out = []
    for _ in range(n_reps):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([buckets[i] for i in pick])
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        out.append(
            roc_auc_score(yt, y_prob_a[idx]) - roc_auc_score(yt, y_prob_b[idx])
        )
    return np.asarray(out)


def paired_twoway_roc_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    firm_ids: np.ndarray,
    year_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Firm, year and two-way (Cameron-Gelbach-Miller) inference for an AUC difference.

    Motivation, established empirically on the reported run: the firm-cluster SE
    for a PAIRED AUC difference is essentially identical to the DeLong SE
    (inflation 0.96-1.00). Within-firm dependence moves both models' scores
    together, so it cancels in the difference. The dimension that actually
    carries uncertainty here is TIME -- exactly what
    ``src/robustness/temporal_bootstrap.py`` found for PR-AUC
    (SE_twoway ~ SE_year ~ 2 x SE_firm). ROC-AUC previously had no year-clustered
    counterpart at all, so this closes the real gap rather than the assumed one.

    ``Var_2way = Var_firm + Var_year - Var_iid``, floored at zero (CGM 2011).
    """
    from src.robustness.temporal_bootstrap import twoway_variance

    y_true = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)
    observed = float(roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b))

    obs_ids = np.arange(len(y_true))
    d_firm = _paired_delta_draws(y_true, y_prob_a, y_prob_b, firm_ids, n_reps, seed)
    d_year = _paired_delta_draws(y_true, y_prob_a, y_prob_b, year_ids, n_reps, seed + 1)
    d_iid = _paired_delta_draws(y_true, y_prob_a, y_prob_b, obs_ids, n_reps, seed + 2)

    v_firm = float(np.var(d_firm, ddof=1))
    v_year = float(np.var(d_year, ddof=1))
    v_iid = float(np.var(d_iid, ddof=1))
    se2 = float(np.sqrt(twoway_variance(v_firm, v_year, v_iid)))

    z2 = observed / se2 if se2 > 0 else np.nan
    p2 = float(2 * (1 - stats.norm.cdf(abs(z2)))) if np.isfinite(z2) else np.nan
    crit = stats.norm.ppf(0.975)

    return {
        "delta_auc": observed,
        "se_firm": float(np.sqrt(v_firm)),
        "se_year": float(np.sqrt(v_year)),
        "se_iid": float(np.sqrt(v_iid)),
        "se_twoway": se2,
        "twoway_ci_lower": observed - crit * se2,
        "twoway_ci_upper": observed + crit * se2,
        "p_twoway": p2,
        "n_years": int(pd.Series(year_ids).nunique()),
    }


def compare_models_clustered(
    y_true: np.ndarray,
    probs: dict[str, np.ndarray],
    firm_ids: np.ndarray,
    year_ids: np.ndarray | None = None,
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = RANDOM_SEED,
    delong: bool = True,
) -> pd.DataFrame:
    """
    All pairwise ROC-AUC comparisons under every available inference scheme.

    Parameters
    ----------
    probs : dict
        {display name: predicted probabilities}. Order defines pair order.
    year_ids : array or None
        Fiscal years. When supplied, year-cluster and two-way (CGM) inference
        are added -- the schemes that actually matter (see
        :func:`paired_twoway_roc_test`).
    delong : bool
        Also run the i.i.d. DeLong test so the schemes can be read side by side.

    Returns
    -------
    pd.DataFrame with one row per pair, Holm-adjusted within each scheme.
    """
    from src.models.evaluate import delong_test

    names = list(probs)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = paired_firm_cluster_roc_test(
                y_true, probs[a], probs[b], firm_ids, n_reps=n_reps, seed=seed
            )
            row = {
                "comparison": f"{a} vs {b}",
                "delta_auc": round(res["delta_auc"], 4),
                "cluster_ci_lower": round(res["ci_lower"], 4),
                "cluster_ci_upper": round(res["ci_upper"], 4),
                "se_cluster": res["se_cluster"],
                "p_cluster_normal": res["p_normal"],
                "p_cluster_percentile": res["p_percentile"],
            }
            if delong:
                z, p = delong_test(y_true, probs[a], probs[b])
                row["delong_z"] = round(float(z), 3)
                row["p_delong"] = float(p)
                se_delong = abs(res["delta_auc"] / z) if z else np.nan
                row["se_delong"] = se_delong
                row["se_inflation"] = (
                    round(res["se_cluster"] / se_delong, 2)
                    if np.isfinite(se_delong) and se_delong > 0 else np.nan
                )
            if year_ids is not None:
                tw = paired_twoway_roc_test(
                    y_true, probs[a], probs[b], firm_ids, year_ids,
                    n_reps=n_reps, seed=seed,
                )
                row.update({
                    "se_year": tw["se_year"],
                    "se_twoway": tw["se_twoway"],
                    "twoway_ci_lower": round(tw["twoway_ci_lower"], 4),
                    "twoway_ci_upper": round(tw["twoway_ci_upper"], 4),
                    "p_twoway": tw["p_twoway"],
                    "twoway_se_inflation": (
                        round(tw["se_twoway"] / tw["se_firm"], 2)
                        if tw["se_firm"] > 0 else np.nan
                    ),
                })
            rows.append(row)

    out = pd.DataFrame(rows)
    out["p_cluster_holm"] = holm(out["p_cluster_normal"].tolist())
    if delong:
        out["p_delong_holm"] = holm(out["p_delong"].tolist())
    if year_ids is not None:
        out["p_twoway_holm"] = holm(out["p_twoway"].tolist())
    return out
