"""
Four-model significance tests — additive extension of significance.py.
======================================================================
Adds the co-primary neural network to the formal pairwise comparison so
that all four headline models receive identical inference: the DeLong
(1988) test for AUC-ROC and the null-recentred paired firm-block
bootstrap for PR-AUC. The frozen ``significance.py`` only compares the
three pre-specified models (LR / RF / XGB); this module closes that gap
for the post-freeze co-primary NN.

READ-ONLY w.r.t. the frozen pipeline. It loads the EXISTING saved models,
scores the EXISTING frozen test sample, and writes ONLY new files:
  outputs/tables/model_results/significance_tests_4models.{csv,tex}

It does NOT retrain anything and does NOT touch the frozen
``significance_tests.csv``. Scoring matches the primary flow exactly
(``run_pipeline.py`` stage 3):
  X      = test[ALL_FEATURES].astype(float).fillna(0)
  y_prob = model.predict_proba(X)[:, 1]
with the saved Platt-scaled logistic regression (Platt is monotonic, so
PR-AUC / ROC-AUC / DeLong / the rank-based bootstrap are identical to the
raw LR). As a safeguard, the three frozen pairs are recomputed here and
checked against the frozen table BEFORE the three new NN pairs are
trusted — if the self-test fails, the saved models no longer match the
frozen results and the NN rows must not be used.

The neural network is the imbalance-matched RC7b model
(``neural_network_balanced.joblib``) — the same specification used in the
four-model headline table — so PR-AUC reproduces the headline value.

Run
---
    python -m src.analysis.significance_4models
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ALL_FEATURES, DATA_SAMPLES, OUT_MODELS_SAVED, OUT_TABLES_MODEL
from src.models.evaluate import delong_test
from src.analysis.significance import _bootstrap_pr_auc_diff, _stars
from src.utils.tables import save_table

LABEL_COL = "distress"

# Saved frozen artifacts (NN = the balanced RC7b headline model).
MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest":       "random_forest.joblib",
    "xgboost":             "xgboost.joblib",
    "neural_network":      "neural_network_balanced.joblib",
}
LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "neural_network":      "Neural Network",
}
# Delta = row model minus column model. Order: the three frozen pairs
# (recomputed for the self-test) first, then the three new NN pairs.
PAIRS = [
    ("random_forest",  "logistic_regression"),
    ("xgboost",        "logistic_regression"),
    ("random_forest",  "xgboost"),
    ("neural_network", "logistic_regression"),
    ("neural_network", "random_forest"),
    ("neural_network", "xgboost"),
]


def _load_probs(test: pd.DataFrame) -> dict:
    """Score every saved model on the frozen test sample (primary-flow scoring)."""
    X = test[ALL_FEATURES].astype(float).fillna(0).values
    probs = {}
    for name, fname in MODEL_FILES.items():
        path = OUT_MODELS_SAVED / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing saved model: {path}")
        probs[name] = joblib.load(path).predict_proba(X)[:, 1]
    return probs


def _selftest(results: pd.DataFrame, allow_missing_reference: bool = False) -> bool:
    """
    Recomputed frozen pairs must match the frozen significance_tests.csv.

    A missing reference file is treated as NOT VERIFIED, not as a pass. The
    reference for this module belongs to a superseded generation and was moved
    to ``outputs/_superseded/`` in the 2026-07-24 provenance quarantine, so the
    absent-file branch is a quarantine symptom rather than a first-run
    bootstrap. Pass ``allow_missing_reference=True`` only when genuinely
    generating the reference for the first time.
    """
    frozen_path = OUT_TABLES_MODEL / "significance_tests.csv"
    if not frozen_path.exists():
        print(f"  [FAIL] reference not found: {frozen_path}")
        print("         Cannot verify against a frozen reference; treating as NOT verified.")
        print("         (Superseded references live under outputs/_superseded/.)")
        return bool(allow_missing_reference)
    frozen = pd.read_csv(frozen_path)
    ok = True
    checked = 0
    for _, fr in frozen.iterrows():
        match = results[results["comparison"] == fr["comparison"]]
        if match.empty:
            continue
        checked += 1
        mr = match.iloc[0]
        for col in ["delta_pr_auc", "p_value_pr_auc", "delong_z", "p_value_roc_auc"]:
            a, b = float(fr[col]), float(mr[col])
            if abs(a - b) > 1e-4:
                print(f"  [FAIL] {fr['comparison']} {col}: frozen {a} vs recomputed {b}")
                ok = False
    print(f"  [ OK ] frozen {checked} pair(s) reproduced exactly" if ok
          else "  [FAIL] self-test mismatch — DO NOT trust the NN rows")
    return ok


def main() -> None:
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    y_true = test[LABEL_COL].astype(int).values
    firm_ids = test["gvkey"].values
    probs = _load_probs(test)

    print("Per-model test metrics (sanity vs headline table):")
    for name in MODEL_FILES:
        print(f"  {LABELS[name]:22s} PR-AUC={average_precision_score(y_true, probs[name]):.4f}"
              f"  ROC-AUC={roc_auc_score(y_true, probs[name]):.4f}")

    rows = []
    for a, b in PAIRS:
        ya, yb = probs[a], probs[b]
        delta_pr = (average_precision_score(y_true, ya)
                    - average_precision_score(y_true, yb))
        p_pr, ci_lo, ci_hi = _bootstrap_pr_auc_diff(y_true, ya, yb, firm_ids)
        z, p_roc = delong_test(y_true, ya, yb)
        rows.append({
            "comparison":            f"{LABELS[a]} vs {LABELS[b]}",
            "delta_pr_auc":          round(delta_pr, 4),
            "delta_pr_auc_ci_lower": round(ci_lo, 4),
            "delta_pr_auc_ci_upper": round(ci_hi, 4),
            "p_value_pr_auc":        round(p_pr, 4),
            "sig_pr_auc":            _stars(p_pr),
            "delong_z":              round(z, 4),
            "p_value_roc_auc":       round(p_roc, 4),
            "sig_roc_auc":           _stars(p_roc),
        })
    results = pd.DataFrame(rows)

    print("\nFour-model pairwise significance (Delta = row minus column):")
    print(results.to_string(index=False))
    print()
    if not _selftest(results):
        raise SystemExit(
            "SELF-TEST FAILED: recomputed frozen pairs do not match "
            "significance_tests.csv. The frozen 3-model table is stale relative to "
            "the saved models — regenerate it first (scripts/regen_significance.py "
            "or run_pipeline.py --stages 10). No output written."
        )

    save_table(
        results,
        OUT_TABLES_MODEL / "significance_tests_4models",
        caption=(
            "Pairwise comparison across the four co-primary models. DeLong "
            "(1988) test for AUC-ROC and a null-recentred block-bootstrap test "
            "for PR-AUC (1{,}000 resamples by firm; two-sided $p$ with the 95\\% "
            "percentile CI of the difference). $\\Delta$ = row model minus column "
            "model. The two-sided recentred $p$-value is the decision criterion for "
            "PR-AUC; the percentile CI describes the sampling spread of the raw "
            "difference and, being a different procedure, can disagree with the "
            "$p$-value near the 5\\% boundary. "
            "The neural network is the imbalance-matched (RC7b) specification used "
            "in the headline table. Significance: $\\dagger$ $p<0.10$, * $p<0.05$, "
            "** $p<0.01$, *** $p<0.001$."
        ),
        label="tab:significance_4models",
    )
    print(f"\nSaved -> {OUT_TABLES_MODEL / 'significance_tests_4models.csv'} (+ .tex)")


if __name__ == "__main__":
    main()
