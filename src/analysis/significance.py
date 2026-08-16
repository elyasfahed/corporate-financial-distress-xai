"""
Formal statistical significance testing for H1.
=================================================
Produces pairwise DeLong (1988) test results comparing AUC-ROC across
the three frozen-design models and a null-recentred paired firm-block
bootstrap test for PR-AUC differences.

DeLong et al. (1988) test:
  H0: AUC(model_a) = AUC(model_b)
  Accounts for the correlation between AUC estimates from the same test set.

PR-AUC bootstrap test:
  H0: PR-AUC(model_a) = PR-AUC(model_b)
  1,000 bootstrap resamples by firm (block bootstrap, consistent with
  the primary CI estimation approach).

Output: one table with z-statistic, p-value, and significance stars
for every pairwise comparison (RF vs LR, XGB vs LR, RF vs XGB).

Design-specification reference: §9.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import OUT_TABLES_MODEL, BOOTSTRAP_REPS, RANDOM_SEED
from src.models.evaluate import delong_test
from src.utils.tables import save_table


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""


def _bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    firm_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float, float]:
    """
    Block-bootstrap two-sided test for H0: PR-AUC(a) = PR-AUC(b).

    Uses block resampling by firm (consistent with the primary CI approach).
    Returns (p_value, ci_lower, ci_upper) for the PR-AUC difference.

    Why the recentering matters
    ---------------------------
    A raw bootstrap distribution of the difference is centred on the OBSERVED
    delta, not on zero. Counting how many resampled deltas exceed the observed
    delta in absolute value therefore returns ~0.5 by construction regardless of
    the true effect — which is exactly the ~0.49–0.51 artefact in the earlier
    table. To obtain a valid two-sided p-value we recentre the resampled deltas
    to zero (subtract their mean) so the reference distribution approximates the
    sampling distribution of the difference *under the null*, then count tail
    mass beyond |observed_delta|. Equivalently, the difference is "significant"
    when its percentile bootstrap CI excludes zero (also returned).
    """
    rng = np.random.default_rng(seed=RANDOM_SEED)
    unique_firms = np.unique(firm_ids)
    n_firms = len(unique_firms)

    firm_to_idx = {f: np.where(firm_ids == f)[0] for f in unique_firms}

    observed_delta = (average_precision_score(y_true, y_prob_a)
                      - average_precision_score(y_true, y_prob_b))

    boot_deltas = []
    for _ in range(n_reps):
        sampled = rng.choice(unique_firms, size=n_firms, replace=True)
        idx = np.concatenate([firm_to_idx[f] for f in sampled])
        y_b = y_true[idx]
        if y_b.sum() == 0:
            continue
        d = (average_precision_score(y_b, y_prob_a[idx])
             - average_precision_score(y_b, y_prob_b[idx]))
        boot_deltas.append(d)

    boot_deltas = np.array(boot_deltas)

    # Percentile bootstrap CI for the difference (excludes 0 <=> significant)
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    # Recentre under H0 (mean -> 0), then two-sided tail count beyond |observed|.
    # +1 smoothing keeps the p-value strictly positive (Davison & Hinkley 1997).
    centered = boot_deltas - boot_deltas.mean()
    p_value = float((np.sum(np.abs(centered) >= np.abs(observed_delta)) + 1)
                    / (len(boot_deltas) + 1))
    return p_value, ci_lower, ci_upper


def run_significance_tests(
    trained_models: dict,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Run pairwise DeLong AUC tests and bootstrap PR-AUC tests.

    Parameters
    ----------
    trained_models : dict
        Output of train_all_models() — must contain y_prob for each model.
    test : pd.DataFrame
    features : list[str]

    Returns
    -------
    pd.DataFrame
        Pairwise comparison table.
    """
    import itertools

    y_true   = test["distress"].astype(int).values
    firm_ids = test["gvkey"].values

    model_names = list(trained_models.keys())
    probs = {
        name: info["y_prob"]
        for name, info in trained_models.items()
        if "y_prob" in info
    }

    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost",
    }

    rows = []
    pairs = [
        ("random_forest",       "logistic_regression"),
        ("xgboost",             "logistic_regression"),
        ("random_forest",       "xgboost"),
    ]

    for name_a, name_b in pairs:
        if name_a not in probs or name_b not in probs:
            continue

        y_a = probs[name_a]
        y_b = probs[name_b]

        # DeLong test on ROC-AUC
        z, p_roc = delong_test(y_true, y_a, y_b)

        # Bootstrap test on PR-AUC (recentred under H0; returns CI too)
        delta_pr = (average_precision_score(y_true, y_a)
                    - average_precision_score(y_true, y_b))
        p_pr, ci_lo, ci_hi = _bootstrap_pr_auc_diff(y_true, y_a, y_b, firm_ids)

        rows.append({
            "comparison":           f"{labels[name_a]} vs {labels[name_b]}",
            "delta_pr_auc":         round(delta_pr, 4),
            "delta_pr_auc_ci_lower": round(ci_lo, 4),
            "delta_pr_auc_ci_upper": round(ci_hi, 4),
            "p_value_pr_auc":       round(p_pr,  4),
            "sig_pr_auc":           _stars(p_pr),
            "delong_z":             round(z,     4),
            "p_value_roc_auc":      round(p_roc, 4),
            "sig_roc_auc":          _stars(p_roc),
        })

    results = pd.DataFrame(rows)

    save_table(
        results,
        OUT_TABLES_MODEL / "significance_tests",
        caption=(
            "Pairwise model comparison: DeLong (1988) test for AUC-ROC differences "
            "and block-bootstrap test for PR-AUC differences (1,000 resamples by firm; "
            "two-sided $p$-value from the null-recentred bootstrap distribution, with the "
            "95\\% percentile CI of the difference). "
            "$\\Delta$ = row model minus column model; a CI excluding zero indicates a "
            "significant PR-AUC difference. "
            "Significance: $\\dagger$ $p<0.10$, * $p<0.05$, ** $p<0.01$, *** $p<0.001$."
        ),
        label="tab:significance",
    )

    print("\n  Pairwise significance tests:")
    print(results.to_string(index=False))
    return results
