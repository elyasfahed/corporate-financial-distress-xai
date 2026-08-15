"""
Operational and calibration evaluation of the frozen models.
=============================================================
**Classification: post-hoc supplementary (economic / operational evaluation).**

The reported evaluation is entirely rank-based (PR-AUC, ROC-AUC) plus one
validation-tuned F1 threshold. Two questions a finance reader will ask are
therefore unanswered:

* **Under a realistic review budget, how good is the screen?** PR-AUC integrates
  over the whole recall range, but Chapter 6 concedes the four curves are
  indistinguishable beyond recall ~0.3 — i.e. the ranking is decided in the
  low-recall region that PR-AUC dilutes. Precision at 1%, 5% and 10% review
  capacity targets exactly that region.
* **What happens when the two error types cost different amounts?** Expected
  misclassification cost over a labelled range of false-negative-to-false-positive
  ratios.

Discipline
----------
* Every operating threshold is selected **on the validation split only** and then
  applied unchanged to test. Nothing is optimised on test.
* Cost ratios are **scenarios**, not estimated economic facts. No Basel claim is
  made: exposure at default, loss given default and a legally defensible default
  definition are all unavailable here, so the analysis is stated as a screening
  trade-off, not as regulatory capital.
* Calibration uses the **common validation-fitted Platt map** for all four models,
  because the class-weighted native outputs are ranking scores rather than event
  probabilities and their raw Brier scores are not cross-model comparable.

Run::

    PYTHONPATH=. python -m src.analysis.supp_economic_calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.analysis.supplementary_common import (DISPLAY, FEATS, MODEL_ORDER,
                                               assert_headline_reproduces,
                                               bootstrap_indices, firm_blocks,
                                               load_frozen_scores, load_split,
                                               write_table)

CAPACITIES = (0.01, 0.05, 0.10)
COST_RATIOS = (5, 10, 20, 50, 100)
N_BOOT = 1000
SEED = 42


# ---------------------------------------------------------------------------
def _platt_probabilities() -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """
    Validation-fitted Platt map applied to validation and test, for all four
    models on a common footing. The saved logistic model is already Platt
    wrapped, so it is unwrapped first to avoid calibrating twice.
    """
    import joblib
    from src.analysis.lr_calibration import PlattScaledModel
    from src.analysis.supplementary_common import MODELS

    val, test = load_split("val"), load_split("test")
    Xv, yv = val[FEATS].to_numpy(float), val["distress"].to_numpy(int)
    Xt, yt = test[FEATS].to_numpy(float), test["distress"].to_numpy(int)

    pv, pt = {}, {}
    for name in MODEL_ORDER:
        m = joblib.load(MODELS / f"{name}.joblib")
        base = m._base if isinstance(m, PlattScaledModel) else m
        raw_v = base.predict_proba(Xv)[:, 1].reshape(-1, 1)
        raw_t = base.predict_proba(Xt)[:, 1].reshape(-1, 1)
        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt.fit(raw_v, yv)
        pv[name] = platt.predict_proba(raw_v)[:, 1]
        pt[name] = platt.predict_proba(raw_t)[:, 1]
    return yv, pv, yt, pt


# ---------------------------------------------------------------------------
def operational_table(df, y, scores, val, yv, sv) -> pd.DataFrame:
    """Precision / recall / lift / number-needed-to-review at fixed capacity."""
    rows = []
    prevalence = y.mean()
    for name in MODEL_ORDER:
        s_te, s_va = scores[name], sv[name]
        for cap in CAPACITIES:
            # threshold that spends exactly `cap` of the VALIDATION budget
            thr = float(np.quantile(s_va, 1 - cap))
            flag = s_te >= thr
            n_flag = int(flag.sum())
            tp = int(y[flag].sum())
            prec = tp / n_flag if n_flag else np.nan
            rec = tp / y.sum()
            rows.append({
                "model": DISPLAY[name],
                "review_capacity": f"{cap:.0%}",
                "threshold_source": "validation quantile",
                "threshold": round(thr, 6),
                "n_reviewed_test": n_flag,
                "realised_capacity": round(n_flag / len(y), 4),
                "precision": round(prec, 4),
                "recall_capture": round(rec, 4),
                "lift_over_prevalence": round(prec / prevalence, 2),
                "number_needed_to_review": round(1 / prec, 1) if prec else np.nan,
            })
    return pd.DataFrame(rows)


def ideal_capacity_table(y, scores) -> pd.DataFrame:
    """
    Exact top-k% of the test sample. Reported as the idealised benchmark: the
    capacity is fixed a priori, so nothing is optimised, but the cut point uses
    the test score distribution and is therefore not attainable prospectively.
    """
    rows = []
    prevalence = y.mean()
    for name in MODEL_ORDER:
        s = scores[name]
        order = np.argsort(-s, kind="stable")
        for cap in CAPACITIES:
            k = max(1, int(round(cap * len(y))))
            tp = int(y[order[:k]].sum())
            prec = tp / k
            rows.append({
                "model": DISPLAY[name],
                "review_capacity": f"{cap:.0%}",
                "n_reviewed_test": k,
                "precision_at_k": round(prec, 4),
                "recall_at_k": round(tp / y.sum(), 4),
                "lift_over_prevalence": round(prec / prevalence, 2),
                "number_needed_to_review": round(1 / prec, 1) if prec else np.nan,
            })
    return pd.DataFrame(rows)


def cost_table(y, scores, yv, sv) -> pd.DataFrame:
    """
    Expected misclassification cost at validation-selected thresholds.

    Cost is normalised per firm-year and reported against the two trivial
    policies (review nobody / review everybody) so the reader can see whether
    the model earns its place at that cost ratio at all.
    """
    rows = []
    n, npos = len(y), int(y.sum())
    for ratio in COST_RATIOS:
        c_fn, c_fp = float(ratio), 1.0
        base_none = c_fn * npos / n                 # miss every event
        base_all = c_fp * (n - npos) / n            # review everything
        for name in MODEL_ORDER:
            grid = np.unique(np.quantile(sv[name], np.linspace(0.0, 1.0, 2001)))
            costs = [(c_fn * int(((sv[name] < t) & (yv == 1)).sum())
                      + c_fp * int(((sv[name] >= t) & (yv == 0)).sum())) / len(yv)
                     for t in grid]
            thr = float(grid[int(np.argmin(costs))])
            fn = int(((scores[name] < thr) & (y == 1)).sum())
            fp = int(((scores[name] >= thr) & (y == 0)).sum())
            cost = (c_fn * fn + c_fp * fp) / n
            best_trivial = min(base_none, base_all)
            rows.append({
                "cost_ratio_FN_to_FP": f"{ratio}:1",
                "model": DISPLAY[name],
                "threshold_from_validation": round(thr, 6),
                "test_false_negatives": fn,
                "test_false_positives": fp,
                "expected_cost_per_firm_year": round(cost, 5),
                "cost_review_none": round(base_none, 5),
                "cost_review_all": round(base_all, 5),
                "savings_vs_best_trivial_pct":
                    round(100 * (best_trivial - cost) / best_trivial, 1),
            })
    return pd.DataFrame(rows)


def calibration_table(yt, pt, df) -> pd.DataFrame:
    """Brier skill score, calibration intercept and slope, observed vs predicted."""
    rows = []
    prevalence = float(yt.mean())
    brier_ref = float(brier_score_loss(yt, np.full_like(pt[MODEL_ORDER[0]], prevalence)))
    blocks = firm_blocks(df)[1]
    rng = np.random.default_rng(SEED)
    draws = [bootstrap_indices(blocks, rng) for _ in range(N_BOOT)]

    for name in MODEL_ORDER:
        p = np.clip(pt[name], 1e-9, 1 - 1e-9)
        brier = float(brier_score_loss(yt, p))
        bss = 1 - brier / brier_ref
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        # slope: regress outcome on the predicted logit; intercept: offset model
        slope_fit = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        slope_fit.fit(logit, yt)
        slope = float(slope_fit.coef_[0][0])
        # calibration-in-the-large: intercept with the logit as a fixed offset
        off = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000,
                                 fit_intercept=True)
        off.fit(np.zeros((len(yt), 1)), yt)
        intercept = float(np.mean(logit.ravel()) * 0)  # placeholder, replaced below
        # solve the offset intercept by 1-D Newton on the score equation
        a = 0.0
        for _ in range(100):
            q = 1 / (1 + np.exp(-(a + logit.ravel())))
            g = np.sum(yt - q)
            h = -np.sum(q * (1 - q))
            if abs(h) < 1e-12:
                break
            step = g / h
            a -= step
            if abs(step) < 1e-12:
                break
        intercept = float(a)
        boot_bss = [1 - brier_score_loss(yt[i], p[i]) /
                    brier_score_loss(yt[i], np.full(len(i), yt[i].mean()))
                    for i in draws if 0 < yt[i].sum() < len(i)]
        rows.append({
            "model": DISPLAY[name],
            "observed_event_rate": round(prevalence, 5),
            "mean_predicted_probability": round(float(p.mean()), 5),
            "obs_minus_pred": round(prevalence - float(p.mean()), 5),
            "brier": round(brier, 6),
            "brier_prevalence_only": round(brier_ref, 6),
            "brier_skill_score": round(bss, 4),
            "bss_ci_lower_firm_block": round(float(np.percentile(boot_bss, 2.5)), 4),
            "bss_ci_upper_firm_block": round(float(np.percentile(boot_bss, 97.5)), 4),
            "calibration_intercept": round(intercept, 4),
            "calibration_slope": round(slope, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    df, y, scores = load_frozen_scores("test")
    assert_headline_reproduces(y, scores)
    val = load_split("val")
    yv, pv, yt, pt = _platt_probabilities()
    # raw (uncalibrated) validation scores drive the operating thresholds, since
    # thresholding is monotone and does not require calibrated probabilities
    import joblib
    from src.analysis.lr_calibration import PlattScaledModel
    from src.analysis.supplementary_common import MODELS
    Xv = val[FEATS].to_numpy(float)
    sv = {}
    for name in MODEL_ORDER:
        m = joblib.load(MODELS / f"{name}.joblib")
        sv[name] = m.predict_proba(Xv)[:, 1]

    op = operational_table(df, y, scores, val, val["distress"].to_numpy(int), sv)
    ideal = ideal_capacity_table(y, scores)
    cost = cost_table(y, scores, val["distress"].to_numpy(int), sv)
    cal = calibration_table(yt, pt, df)

    print(op.to_string(index=False), "\n")
    print(ideal.to_string(index=False), "\n")
    print(cost.to_string(index=False), "\n")
    print(cal.to_string(index=False), "\n")

    write_table(op, "supp_operational_capacity",
                "Screening performance at fixed review capacity (post-hoc "
                "supplementary). The score cut for each capacity is the "
                "corresponding quantile of the \\emph{validation} score "
                "distribution and is applied unchanged to test, so the realised "
                "test capacity differs from the nominal budget. Lift is relative "
                "to the 1.58\\% test prevalence; the number needed to review is "
                "the reciprocal of precision. \\textbf{The precision column is "
                "not comparable across models within a capacity row}, because "
                "each model realises a different test capacity at its own "
                "validation quantile (2.98\\% to 5.44\\% at the nominal 1\\% "
                "budget), so the entries are read at different points of four "
                "different precision--recall curves. Cross-model comparison at "
                "a genuinely common budget is given in "
                "Table~\\ref{tab:supp_precision_at_k}; this table's purpose is "
                "to show how far a validation-derived threshold misses its "
                "intended budget out of period.",
                "tab:supp_operational")
    write_table(ideal, "supp_precision_at_k",
                "Precision and capture at exactly the top $k$\\% of the test "
                "sample (post-hoc supplementary). The capacity is fixed a "
                "priori, so nothing is optimised on test, but the cut point uses "
                "the test score distribution and is therefore an idealised upper "
                "benchmark rather than a prospectively attainable operating point.",
                "tab:supp_precision_at_k")
    write_table(cost, "supp_expected_cost",
                "Expected misclassification cost per firm-year over a range of "
                "false-negative-to-false-positive cost ratios (post-hoc "
                "supplementary). The cost-minimising threshold is selected on "
                "the validation split at each ratio and applied unchanged to "
                "test. The ratios are \\emph{scenarios} chosen to span plausible "
                "asymmetries, not estimated economic quantities; no exposure at "
                "default or loss given default is available, so no regulatory "
                "capital interpretation is claimed.",
                "tab:supp_expected_cost")
    write_table(cal, "supp_calibration_quality",
                "Calibration of the four models under the common "
                "validation-fitted Platt map (post-hoc supplementary). The Brier "
                "skill score is relative to a constant forecast at the test "
                "prevalence; a perfectly calibrated model has intercept 0 and "
                "slope 1. Confidence intervals are firm-block bootstrap "
                f"({N_BOOT} resamples, seed {SEED}).",
                "tab:supp_calibration")


if __name__ == "__main__":
    main()
