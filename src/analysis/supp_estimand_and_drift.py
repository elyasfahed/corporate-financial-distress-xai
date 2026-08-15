"""
Firm-balanced estimand and prevalence drift.
============================================
**Classification: post-hoc supplementary.**

Two questions the reported analysis does not separate.

**(E) Whose performance is being measured?** Clustering the standard errors fixes
the *variance* of the reported metrics but not the *estimand*. The reported
PR-AUC weights every firm-year equally, so a firm present for all nine test years
contributes nine times the weight of a firm present once. Long-lived firms are
systematically different from short-lived ones — they are larger, older and less
distress-prone — so the firm-year-weighted metric describes a population tilted
toward survivors. This module recomputes the headline metrics giving every
*firm* equal total weight and reports whether the ranking moves.

**(H) Prevalence drift.** Training prevalence is 3.01%, validation 1.31% and test
1.58%. This module measures the drift and separates the metrics it does and does
not touch, so the thesis can state the consequence precisely rather than
gesturing at it.

Run::

    PYTHONPATH=. python -m src.analysis.supp_estimand_and_drift
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.supplementary_common import (DISPLAY, MODEL_ORDER,
                                               assert_headline_reproduces,
                                               load_frozen_scores, load_split,
                                               write_table)


def _weighted_ap(y, s, w):
    """Weighted average precision, computed from the weighted PR path."""
    order = np.argsort(-s, kind="stable")
    y, s, w = y[order], s[order], w[order]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * (1 - y))
    total_pos = tp[-1]
    if total_pos <= 0:
        return np.nan
    precision = tp / np.maximum(tp + fp, 1e-300)
    recall = tp / total_pos
    # step integral, summing only where recall advances (ties collapse)
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(precision * d_recall))


def _weighted_auc(y, s, w):
    """Weighted ROC-AUC via the weighted Mann-Whitney statistic (ties at 0.5)."""
    order = np.argsort(s, kind="stable")
    y, s, w = y[order], s[order], w[order]
    pos, neg = y == 1, y == 0
    wp, wn = w[pos], w[neg]
    sp, sn = s[pos], s[neg]
    if wp.sum() == 0 or wn.sum() == 0:
        return np.nan
    order_n = np.argsort(sn, kind="stable")
    sn_sorted, wn_sorted = sn[order_n], wn[order_n]
    cum = np.concatenate([[0.0], np.cumsum(wn_sorted)])
    lo = np.searchsorted(sn_sorted, sp, side="left")
    hi = np.searchsorted(sn_sorted, sp, side="right")
    less = cum[lo]
    equal = cum[hi] - cum[lo]
    return float(np.sum(wp * (less + 0.5 * equal)) / (wp.sum() * wn.sum()))


def estimand_table(df, y, scores, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    counts = df.groupby("gvkey")["gvkey"].transform("size").to_numpy(float)
    w_year = np.ones(len(df))
    w_firm = 1.0 / counts          # every firm gets total weight 1
    rows = []
    for name in MODEL_ORDER:
        s = scores[name]
        ap_y = _weighted_ap(y, s, w_year)
        ap_f = _weighted_ap(y, s, w_firm)
        au_y = _weighted_auc(y, s, w_year)
        au_f = _weighted_auc(y, s, w_firm)
        rows.append({
            "model": DISPLAY[name],
            "pr_auc_firm_year_weighted": round(ap_y, 4),
            "pr_auc_firm_balanced": round(ap_f, 4),
            "pr_auc_delta": round(ap_f - ap_y, 4),
            "roc_auc_firm_year_weighted": round(au_y, 4),
            "roc_auc_firm_balanced": round(au_f, 4),
            "roc_auc_delta": round(au_f - au_y, 4),
        })
    out = pd.DataFrame(rows)
    out["rank_firm_year"] = out.pr_auc_firm_year_weighted.rank(ascending=False).astype(int)
    out["rank_firm_balanced"] = out.pr_auc_firm_balanced.rank(ascending=False).astype(int)

    # Firm-block bootstrap on the firm-balanced metric. Without this the table
    # would invite exactly the error this review criticises elsewhere: reading a
    # reordering off point estimates that carry substantial sampling error.
    from src.analysis.supplementary_common import bootstrap_indices, firm_blocks
    blocks = firm_blocks(df)[1]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(MODEL_ORDER)))
    for b in range(n_boot):
        idx = bootstrap_indices(blocks, rng)
        yb, cb = y[idx], counts[idx]
        if yb.sum() == 0:
            draws[b, :] = np.nan
            continue
        for j, name in enumerate(MODEL_ORDER):
            draws[b, j] = _weighted_ap(yb, scores[name][idx], 1.0 / cb)
    out["pr_auc_firm_balanced_ci_lower"] = np.round(
        np.nanpercentile(draws, 2.5, axis=0), 4)
    out["pr_auc_firm_balanced_ci_upper"] = np.round(
        np.nanpercentile(draws, 97.5, axis=0), 4)
    lr = MODEL_ORDER.index("logistic_regression")
    for j, name in enumerate(MODEL_ORDER):
        if j == lr:
            out.loc[j, "p_vs_logistic_firm_balanced"] = np.nan
            continue
        d = draws[:, j] - draws[:, lr]
        obs = out.pr_auc_firm_balanced[j] - out.pr_auc_firm_balanced[lr]
        out.loc[j, "p_vs_logistic_firm_balanced"] = round(
            float(np.nanmean(np.abs(d - np.nanmean(d)) >= abs(obs))), 4)
    return out


def drift_table() -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        d = load_split(split)
        ev = int(d.distress.sum())
        rows.append({
            "split": split,
            "fiscal_years": f"{int(d.fyear.min())}-{int(d.fyear.max())}",
            "observations": len(d),
            "events": ev,
            "prevalence": round(ev / len(d), 5),
            "no_skill_pr_auc": round(ev / len(d), 5),
        })
    out = pd.DataFrame(rows)
    tr = out.loc[out.split == "train", "prevalence"].iloc[0]
    out["ratio_to_training_prevalence"] = (out.prevalence / tr).round(3)
    return out


def drift_by_year() -> pd.DataFrame:
    d = load_split("test")
    g = d.groupby("fyear").agg(observations=("distress", "size"),
                               events=("distress", "sum"))
    g["prevalence"] = (g.events / g.observations).round(5)
    return g.reset_index()


def metric_sensitivity_table() -> pd.DataFrame:
    """Which reported quantities move with the base rate and which do not."""
    return pd.DataFrame([
        {"quantity": "PR-AUC", "sensitive": "yes (baseline shifts)",
         "consequence": "The no-skill baseline equals the prevalence, so PR-AUC "
                        "levels are not comparable across periods or label "
                        "definitions; only the gap to the baseline is."},
        {"quantity": "ROC-AUC", "sensitive": "no",
         "consequence": "Invariant to the base rate; comparable across periods."},
        {"quantity": "F1 and the operating threshold", "sensitive": "yes",
         "consequence": "The threshold is tuned on a 1.31% validation base rate "
                        "and applied to a 1.58% test base rate, having been "
                        "trained under class weights targeting 3.01%."},
        {"quantity": "Brier score and calibration", "sensitive": "yes",
         "consequence": "Absolute probabilities inherit the prior of the sample "
                        "the map was fitted on; validation-fitted Platt scaling "
                        "corrects the score scale only partially."},
        {"quantity": "Class weighting", "sensitive": "targets training",
         "consequence": "Rebalancing targets the 3.01% training base rate, "
                        "roughly twice the rate the model is scored against."},
        {"quantity": "Precision at fixed capacity", "sensitive": "yes",
         "consequence": "Precision scales roughly with the base rate at a fixed "
                        "review budget, so capacity results are period-specific."},
    ])


def main() -> None:
    df, y, scores = load_frozen_scores("test")
    assert_headline_reproduces(y, scores)

    est = estimand_table(df, y, scores)
    drift = drift_table()
    byyr = drift_by_year()
    sens = metric_sensitivity_table()

    print(est.to_string(index=False), "\n")
    print(drift.to_string(index=False), "\n")
    print(byyr.to_string(index=False), "\n")

    n_firms = df.gvkey.nunique()
    spell = df.groupby("gvkey").size()
    print(f"test firms: {n_firms}; firm-years per firm "
          f"min {spell.min()} median {spell.median():.0f} max {spell.max()}\n")

    write_table(est, "supp_firm_balanced_estimand",
                "Firm-year-weighted versus firm-balanced test metrics (post-hoc "
                "supplementary). The reported metrics weight every firm-year "
                "equally, so a firm observed in all nine test years contributes "
                "nine times the weight of a firm observed once; the "
                "firm-balanced column gives every firm total weight one. "
                "Clustering corrects the variance of the reported estimand, not "
                "the estimand itself, so this is a distinct question. Computed "
                "from the frozen score vectors; no model is refitted.",
                "tab:supp_estimand")
    # Both take the default max-width treatment: the drift table is wider than
    # \textwidth and is shrunk to fit, the by-year table is narrower and is left
    # at its natural size. Under the previous \resizebox wrapper the latter was
    # scaled UP to the full text width and typeset at roughly twice body size.
    write_table(drift, "supp_prevalence_drift",
                "Distress prevalence across the three chronological splits "
                "(post-hoc supplementary). The no-skill PR-AUC baseline equals "
                "the prevalence by construction.",
                "tab:supp_prevalence_drift")
    write_table(byyr, "supp_test_prevalence_by_year",
                "Test-sample distress prevalence by fiscal year (post-hoc "
                "supplementary).",
                "tab:supp_test_prevalence_by_year")
    write_table(sens, "supp_prevalence_metric_sensitivity",
                "Which reported quantities are sensitive to the base rate "
                "(post-hoc supplementary). Prevalence drift is disclosed and "
                "measured; it is not used to make a prospective recalibration "
                "claim, because doing so would require the test prevalence to "
                "be known in advance.",
                "tab:supp_prevalence_sensitivity",
                # Widths are chosen so that content + 6 x \tabcolsep stays inside
                # \textwidth, and the table is NOT resized: the p{} widths are
                # already correct, so rescaling would only distort the type size.
                column_format="p{0.22\\textwidth}p{0.15\\textwidth}p{0.53\\textwidth}",
                resize=False)


if __name__ == "__main__":
    main()
