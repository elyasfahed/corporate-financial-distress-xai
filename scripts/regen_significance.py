"""
Surgical regeneration of significance_tests.{csv,tex} from the FINAL saved models.
=================================================================================
Recomputes ONLY the pairwise significance table after the null-recentred
bootstrap fix in `src/analysis/significance.py` (the previous version returned
~0.49-0.51 p-values because its bootstrap distribution was centred on the
observed delta, not under the null).

What it does (and does NOT do)
------------------------------
* Loads the frozen test split (`data/processed/samples/test.parquet`).
* Loads the three FINAL saved models (`.joblib`) — **NO re-training, NO Optuna**.
* Reproduces the headline test predictions EXACTLY as run_pipeline Stage 3 does
  (`X_test = test[ALL_FEATURES].astype(float).fillna(0)`), then prints each
  model's PR-AUC/ROC-AUC so you can confirm they match the corrected headline
  (LR 0.1657, RF 0.1354, XGB 0.1447 after the lag-fix adoption) before
  trusting the new table.
* Rewrites `outputs/tables/model_results/significance_tests.{csv,tex}`.

This recomputes a *derived* statistic from already-final predictions. It does
NOT produce a new headline test evaluation, so it does not breach the
"evaluate test once" rule — it is exactly what pipeline Stage 10 already does.

Run (PyCharm or terminal, repo root):
    python -m scripts.regen_significance
On a plain cp1252 console, prefix:  PYTHONIOENCODING=utf-8
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import DATA_SAMPLES, OUT_MODELS_SAVED, ALL_FEATURES
from src.analysis.significance import run_significance_tests

MODELS = ["logistic_regression", "random_forest", "xgboost"]
# Expected headline PR-AUCs (sanity check that saved models reproduce).
# Updated 2026-06-29 after the lagged-feature fix was ADOPTED (Route 2 full
# re-run on corrected data). Previous pre-fix targets were LR 0.1660 /
# RF 0.1368 / XGB 0.1364.
EXPECTED_PR_AUC = {
    "logistic_regression": 0.1657,
    "random_forest":       0.1354,
    "xgboost":             0.1447,
}


def main() -> None:
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X_test = test[ALL_FEATURES].astype(float).fillna(0).values
    y_true = test["distress"].astype(int).values

    trained_models: dict = {}
    print("Reproducing headline test predictions from saved models:")
    all_ok = True
    for name in MODELS:
        model = joblib.load(OUT_MODELS_SAVED / f"{name}.joblib")
        y_prob = model.predict_proba(X_test)[:, 1]
        pr = average_precision_score(y_true, y_prob)
        roc = roc_auc_score(y_true, y_prob)
        ok = abs(pr - EXPECTED_PR_AUC[name]) < 1e-3
        all_ok &= ok
        flag = "OK" if ok else "MISMATCH"
        print(f"  {name:22s} PR-AUC={pr:.4f} (exp {EXPECTED_PR_AUC[name]:.4f}) "
              f"ROC-AUC={roc:.4f}  [{flag}]")
        trained_models[name] = {"model": model, "y_prob": y_prob}

    if not all_ok:
        raise SystemExit(
            "Saved models did NOT reproduce the frozen headline PR-AUCs — "
            "aborting so the significance table is not written from wrong "
            "predictions. Investigate before re-running."
        )

    print(f"\n  Test sample: n={len(test):,}  distress={int(y_true.sum()):,}  "
          f"prevalence={y_true.mean():.4f}")
    run_significance_tests(trained_models, test, ALL_FEATURES)
    print("\nRewrote outputs/tables/model_results/significance_tests.{csv,tex}")


if __name__ == "__main__":
    from src.utils.superseded import abort_if_superseded
    abort_if_superseded(__file__, "python scripts/run_v2_secondary.py")
    main()
