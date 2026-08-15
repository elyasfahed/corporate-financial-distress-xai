"""
Model evaluation — all metrics required by Blueprint v4 §9.4.
==============================================================
Primary metric : PR-AUC (most informative for severely imbalanced data;
                 random-classifier baseline = prevalence rate ~3–5%)
Co-primary     : AUC-ROC (comparability with published benchmarks)
Secondary      : Precision, Recall, F1 (at optimal threshold)
                 KS statistic (standard in credit risk)
                 Brier Score (probability calibration)

Key design rules (Blueprint v4):
  - Accuracy is NOT reported (misleading under class imbalance).
  - All tables must state the prevalence-baseline PR-AUC explicitly.
  - AUC-ROC significance: DeLong et al. (1988) test.
  - PR-AUC confidence intervals: bootstrap with block resampling by firm
    (1,000 resamples).
  - Classification threshold: selected on VALIDATION set to maximise F1;
    locked before test evaluation.

Blueprint v4 reference: §9.4
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import BOOTSTRAP_REPS, RANDOM_SEED, THRESHOLD_METRIC


# ---------------------------------------------------------------------------
# Threshold selection (on VALIDATION set only)
# ---------------------------------------------------------------------------

def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = THRESHOLD_METRIC,
) -> float:
    """
    Select the classification threshold that maximises `metric` on the
    validation set. Lock this threshold before evaluating on the test set.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted distress probabilities.
    metric : str
        'f1' (default, per Blueprint v4 §9.2) or 'precision' or 'recall'.

    Returns
    -------
    float
        Optimal threshold in [0, 1].
    """
    # Search the exact set of decision boundaries induced by validation
    # predictions.  A coarse fixed grid can miss the true F1 optimum when
    # probabilities are concentrated, particularly under severe imbalance.
    thresholds = np.unique(np.concatenate(([0.0], np.asarray(y_prob), [1.0])))
    best_thresh = 0.5
    best_score  = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if score > best_score:
            best_score  = score
            best_thresh = t

    return float(best_thresh)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute PR-AUC (area under the precision-recall curve).

    This is the PRIMARY metric. The random-classifier baseline is the
    prevalence rate (fraction of distressed firm-years in the sample).

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray

    Returns
    -------
    float
    """
    return average_precision_score(y_true, y_prob)


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute AUC-ROC.

    Co-primary metric for comparability with prior literature.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray

    Returns
    -------
    float
    """
    return roc_auc_score(y_true, y_prob)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic.

    Standard metric in credit risk / distress prediction models.
    KS = max(TPR - FPR) across all thresholds.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray

    Returns
    -------
    float
        KS statistic in [0, 1]. Higher is better.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier Score (probability calibration).

    Lower is better. A perfectly calibrated model minimises the Brier score.
    Useful for detecting over- or under-confidence in predicted probabilities.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray

    Returns
    -------
    float
    """
    return brier_score_loss(y_true, y_prob)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str = "",
) -> dict:
    """
    Compute the full metric set required by Blueprint v4 §9.4.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted distress probabilities.
    threshold : float
        Classification threshold (selected on validation set).
    model_name : str
        For labelling in output tables.

    Returns
    -------
    dict
        Keys: model, prevalence_baseline_pr_auc, pr_auc, roc_auc,
              precision, recall, f1, ks_stat, brier_score, threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    prevalence = y_true.mean()

    return {
        "model":                    model_name,
        "prevalence_baseline_pr_auc": round(prevalence, 4),
        "pr_auc":                   round(compute_pr_auc(y_true, y_prob), 4),
        "roc_auc":                  round(compute_roc_auc(y_true, y_prob), 4),
        "precision":                round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":                   round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ks_stat":                  round(compute_ks_statistic(y_true, y_prob), 4),
        "brier_score":              round(compute_brier_score(y_true, y_prob), 4),
        "threshold":                round(threshold, 4),
    }


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple[float, float]:
    """
    DeLong et al. (1988) test for the difference in AUC-ROC between two models.

    H0: AUC(model_a) = AUC(model_b)

    Parameters
    ----------
    y_true : np.ndarray
    y_prob_a : np.ndarray
        Predicted probabilities of model A.
    y_prob_b : np.ndarray
        Predicted probabilities of model B.

    Returns
    -------
    z_stat : float
    p_value : float

    References
    ----------
    DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988).
    "Comparing the Areas under Two or More Correlated Receiver Operating
    Characteristic Curves: A Nonparametric Approach."
    Biometrics, 44(3), 837–845.
    """
    def _auc_components(y_true, y_prob):
        """Compute structural components of the DeLong estimator.

        The placement (structural) values are computed via a vectorised
        midrank construction (sort + ``np.searchsorted``) in
        O((n1 + n0) log(n1 + n0)) time. This is mathematically identical to
        the naive O(n1 * n0) pairwise definition — each placement value is an
        exact count of strict wins plus half the ties divided by the opposing
        class size — but avoids the per-observation Python loop. See Sun & Xu
        (2014), "Fast Implementation of DeLong's Algorithm", IEEE SPL 21(11).
        """
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        n1, n0  = len(pos_idx), len(neg_idx)

        p_pos = y_prob[pos_idx]
        p_neg = y_prob[neg_idx]

        # Placement of each positive among the negatives:
        #   v10[i] = ( #{neg < p_pos[i]} + 0.5 * #{neg == p_pos[i]} ) / n0
        neg_sorted = np.sort(p_neg)
        lt_neg = np.searchsorted(neg_sorted, p_pos, side="left")    # neg strictly below
        le_neg = np.searchsorted(neg_sorted, p_pos, side="right")   # neg below or equal
        v10 = (lt_neg + 0.5 * (le_neg - lt_neg)) / n0

        # Placement of each negative among the positives:
        #   v01[j] = ( #{pos > p_neg[j]} + 0.5 * #{pos == p_neg[j]} ) / n1
        pos_sorted = np.sort(p_pos)
        lt_pos = np.searchsorted(pos_sorted, p_neg, side="left")    # pos strictly below
        le_pos = np.searchsorted(pos_sorted, p_neg, side="right")   # pos below or equal
        v01 = ((n1 - le_pos) + 0.5 * (le_pos - lt_pos)) / n1

        auc = v10.mean()
        var = (np.var(v10, ddof=1) / n1) + (np.var(v01, ddof=1) / n0)
        return auc, var, v10, v01

    auc_a, var_a, v10_a, v01_a = _auc_components(y_true, y_prob_a)
    auc_b, var_b, v10_b, v01_b = _auc_components(y_true, y_prob_b)

    # Covariance between the two AUC estimates (DeLong 1988 Eq. 6)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0  = len(pos_idx), len(neg_idx)
    cov = (
        np.cov(v10_a, v10_b, ddof=1)[0, 1] / n1
        + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n0
    )

    delta_auc = auc_a - auc_b
    var_delta  = var_a + var_b - 2 * cov
    if var_delta <= 0:
        return 0.0, 1.0

    z_stat  = delta_auc / np.sqrt(var_delta)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_value)


def bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    firm_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for PR-AUC with block resampling by firm.

    Block resampling by firm (rather than by observation) accounts for the
    panel structure: all observations from a resampled firm are included
    together. This avoids artificially inflating precision by splitting
    a firm's time-series across bootstrap train/test (Blueprint v4 §9.4).

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray
    firm_ids : np.ndarray
        Array of firm identifiers (e.g., gvkey or permno) — one per obs.
    n_reps : int
        Number of bootstrap resamples (default 1,000).
    alpha : float
        Significance level for CI (default 0.05 → 95% CI).

    Returns
    -------
    ci_lower, ci_upper : float
        95% percentile bootstrap CI bounds.
    """
    rng = np.random.default_rng(seed=42)
    unique_firms = np.unique(firm_ids)
    n_firms = len(unique_firms)
    pr_aucs = []

    # Pre-build firm → observation indices lookup (avoids repeated np.where calls)
    firm_to_idx: dict[object, np.ndarray] = {
        f: np.where(firm_ids == f)[0] for f in unique_firms
    }

    for _ in range(n_reps):
        # Resample firms with replacement; a firm drawn k times contributes k copies
        sampled_firms = rng.choice(unique_firms, size=n_firms, replace=True)
        indices = np.concatenate([firm_to_idx[f] for f in sampled_firms])
        y_b = y_true[indices]
        p_b = y_prob[indices]

        # Need at least one positive case to compute PR-AUC
        if y_b.sum() == 0:
            continue
        pr_aucs.append(average_precision_score(y_b, p_b))

    pr_aucs = np.array(pr_aucs)
    ci_lower = float(np.percentile(pr_aucs, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(pr_aucs, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper
