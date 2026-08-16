"""
Classical distress prediction benchmarks: Ohlson (1980) O-score.
=================================================================
Applies the original Ohlson (1980) logit model coefficients out-of-sample
to the held-out test set (2015–2024). This directly tests whether a model
estimated on 1970s data retains predictive power on modern data — a common
robustness check in the financial distress literature.

Ohlson (1980) O-score
---------------------
O = -1.32 - 0.407*SIZE + 6.03*TLTA - 1.43*WCTA + 0.076*CLCA
        - 1.72*OENEG   - 2.37*NITA  - 1.83*FUTL + 0.285*INTWO
        - 0.521*CHIN

P(distress) = 1 / (1 + exp(-O))

Variable mapping from predictor set:
  SIZE  = LNTA   (log total assets in 2012 USD; Ohlson used GNP-deflated TA —
                  2012 USD is the closest feasible approximation)
  TLTA  = TLTA   (total liabilities / total assets)
  WCTA  = WCTA   (working capital / total assets)
  CLCA  = CLCA   (current liabilities / current assets)
  OENEG = OENEG  (1 if total liabilities > total assets)
  NITA  = NITA   (net income / total assets)
  FUTL  = OCF_TA / TLTA  (operating cash flow / total liabilities;
                           approximates Ohlson's "funds from operations / TL")
  INTWO = INTWO  (1 if net income < 0 for both of last 2 years)
  CHIN  = CHIN   (change in net income normalised by absolute income sum)

Limitation: Ohlson (1980) was estimated on a 1970–1976 US sample under
different accounting standards. Out-of-sample performance on a 2015–2024
sample tests temporal generalisability, not the original model's validity.
Results are presented as a comparative baseline, not a re-estimation.

References
----------
Ohlson, J.A. (1980). "Financial Ratios and the Probabilistic Prediction of
Bankruptcy." Journal of Accounting Research, 18(1), 109–131.

Design reference: §6.3 (benchmark comparison)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OUT_TABLES_MODEL
from src.models.evaluate import compute_all_metrics, select_threshold, bootstrap_pr_auc_ci
from src.utils.tables import save_table


# Ohlson (1980) original logit coefficients — Table 4, Model 1
_OHLSON_INTERCEPT = -1.32
_OHLSON_COEFS = {
    "LNTA":     -0.407,   # SIZE proxy
    "TLTA":      6.03,
    "WCTA":     -1.43,
    "CLCA":      0.076,
    "OENEG":    -1.72,
    "NITA":     -2.37,
    "FUTL":     -1.83,    # OCF_TA / TLTA (see module docstring)
    "INTWO":     0.285,
    "CHIN":     -0.521,
}


def compute_ohlson_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Ohlson (1980) O-score for each observation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in _OHLSON_COEFS (except FUTL,
        which is derived from OCF_TA and TLTA).

    Returns
    -------
    np.ndarray
        Predicted distress probabilities P(distress=1).
    """
    data = df.copy()

    # Derive FUTL = OCF_TA / TLTA (funds from operations / total liabilities)
    # Guard against division by zero when TLTA = 0
    data["FUTL"] = np.where(
        data["TLTA"].abs() > 1e-6,
        data["OCF_TA"] / data["TLTA"],
        0.0,
    )

    # Compute linear combination
    score = _OHLSON_INTERCEPT
    for var, coef in _OHLSON_COEFS.items():
        if var not in data.columns:
            print(f"  WARNING: Ohlson variable {var!r} not found — treated as 0.")
            continue
        score = score + coef * data[var].fillna(0).values

    # Logistic transformation → probability
    prob = 1.0 / (1.0 + np.exp(-score))
    return prob.astype(float)


# Altman (1968) Z-score — original five-ratio model, Table 5.
_ALTMAN_COEFS = (1.2, 1.4, 3.3, 0.6, 0.999)   # weights on X1..X5


def _safe_ratio(num, den) -> np.ndarray:
    """Elementwise num/den with zero-denominator and NaN/inf → 0."""
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    out = np.where(den.abs() > 1e-6, num / den, 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_altman_z(df: pd.DataFrame) -> np.ndarray:
    """
    Altman (1968) Z-score from raw Compustat items (faithful, five ratios).

    X1 = working capital / total assets          (wcap / at)
    X2 = retained earnings / total assets          (re / at)
    X3 = EBIT / total assets                        ((oibdp - dp) / at,
                                                     fallback ni + xint + txt)
    X4 = market value of equity / total liabilities (me_fyend / lt)
    X5 = sales / total assets                        (sale / at)

    Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 0.999 X5

    Higher Z = healthier; lower Z = higher distress risk. Returns the raw Z
    SCORE (not a probability). Missing / zero-denominator ratios are set to 0,
    consistent with the Ohlson benchmark's treatment. Unlike the CHS-style
    predictor set, all five Altman inputs are available on the merged panel,
    so the original coefficients are applied without proxy substitution.
    """
    d = df
    at, lt = d["at"], d["lt"]
    sales = d["sale"].fillna(d["revt"]) if "revt" in d.columns else d["sale"]
    ebit = (pd.to_numeric(d["oibdp"], errors="coerce")
            - pd.to_numeric(d["dp"], errors="coerce"))
    if {"ni", "xint", "txt"}.issubset(d.columns):
        ebit = ebit.fillna(
            pd.to_numeric(d["ni"], errors="coerce")
            + pd.to_numeric(d["xint"], errors="coerce").fillna(0)
            + pd.to_numeric(d["txt"], errors="coerce").fillna(0)
        )
    x1 = _safe_ratio(d["wcap"], at)
    x2 = _safe_ratio(d["re"], at)
    x3 = _safe_ratio(ebit, at)
    x4 = _safe_ratio(d["me_fyend"], lt)
    x5 = _safe_ratio(sales, at)
    c = _ALTMAN_COEFS
    z = c[0]*x1 + c[1]*x2 + c[2]*x3 + c[3]*x4 + c[4]*x5
    return np.asarray(z, dtype=float)


def _platt_prob(scores_val, y_val, scores_test) -> np.ndarray:
    """
    Map a raw distress SCORE to a probability via a validation-fitted
    logistic (Platt). Monotone, so PR-AUC / ROC-AUC / KS are unchanged; it
    only makes the F1 threshold and the Brier score well defined for a model
    that outputs an unbounded score rather than a probability.
    """
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(scores_val, dtype=float).reshape(-1, 1),
           np.asarray(y_val, dtype=int))
    return lr.predict_proba(
        np.asarray(scores_test, dtype=float).reshape(-1, 1))[:, 1]


def run_classical_benchmarks(
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_stem=None,
    period_label: str = "2015--2024",
) -> pd.DataFrame:
    """
    Evaluate the Ohlson (1980) O-score and the Altman (1968) Z-score
    out-of-sample, with the classification threshold selected on validation.

    Both models apply their ORIGINAL published coefficients without
    re-estimation. Ohlson emits a native probability; Altman emits a score
    mapped to a probability by a validation-fitted logistic (monotone, so its
    discrimination is unaffected). Returns one row per benchmark.
    """
    print("\n" + "=" * 60)
    print("  CLASSICAL BENCHMARKS — Ohlson (1980) O-score + Altman (1968) Z")
    print("=" * 60)

    y_val  = val["distress"].astype(int).values
    y_test = test["distress"].astype(int).values
    firm_ids = test["gvkey"].values

    rows = []

    # --- Ohlson (1980): native logit probability --------------------------
    p_val  = compute_ohlson_scores(val)
    p_test = compute_ohlson_scores(test)
    thr = select_threshold(y_val, p_val)
    m = compute_all_metrics(y_test, p_test, thr, model_name="ohlson_1980")
    lo, hi = bootstrap_pr_auc_ci(y_test, p_test, firm_ids)
    m["pr_auc_ci_lower"], m["pr_auc_ci_upper"] = round(lo, 4), round(hi, 4)
    rows.append(m)
    print(f"  Ohlson O-score  — PR-AUC {m['pr_auc']:.4f} [{lo:.4f}, {hi:.4f}] "
          f" ROC-AUC {m['roc_auc']:.4f}  (thr {thr:.4f})")

    # --- Altman (1968): score -> validation-Platt probability -------------
    s_val  = -compute_altman_z(val)     # higher score = higher distress risk
    s_test = -compute_altman_z(test)
    pa_val  = _platt_prob(s_val, y_val, s_val)
    pa_test = _platt_prob(s_val, y_val, s_test)
    thr_a = select_threshold(y_val, pa_val)
    ma = compute_all_metrics(y_test, pa_test, thr_a, model_name="altman_1968")
    loa, hia = bootstrap_pr_auc_ci(y_test, pa_test, firm_ids)
    ma["pr_auc_ci_lower"], ma["pr_auc_ci_upper"] = round(loa, 4), round(hia, 4)
    rows.append(ma)
    print(f"  Altman Z-score  — PR-AUC {ma['pr_auc']:.4f} [{loa:.4f}, {hia:.4f}]"
          f"  ROC-AUC {ma['roc_auc']:.4f}  (thr {thr_a:.4f})")

    results = pd.DataFrame(rows)
    stem = out_stem if out_stem is not None else (
        OUT_TABLES_MODEL / "classical_benchmarks")
    save_table(
        results, stem,
        caption=(
            "Out-of-sample performance of two classical benchmarks — the "
            "Ohlson (1980) O-score and the Altman (1968) Z-score — on the "
            f"held-out test sample ({period_label}). Both apply their original "
            "published coefficients without re-estimation; the threshold is "
            "selected on the validation set to maximise F1. The Altman Z is a "
            "score, mapped to a probability by a validation-fitted logistic "
            "(monotone, so PR-AUC/ROC-AUC are unaffected). PR-AUC 95\\% CIs: "
            "block bootstrap by firm (1{,}000 resamples)."
        ),
        label="tab:classical_benchmarks",
    )
    return results


def main() -> None:
    """Run both classical benchmarks on the final_primary sample."""
    from src.config import DATA_SAMPLES_V2, V2_PROFILE
    val  = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    out_stem = OUT_TABLES_MODEL / V2_PROFILE["spec"] / "classical_benchmarks"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    run_classical_benchmarks(val, test, out_stem=out_stem,
                             period_label="2015--2023")


if __name__ == "__main__":
    main()
