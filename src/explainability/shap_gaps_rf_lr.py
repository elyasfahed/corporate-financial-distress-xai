"""
Fill the SHAP gaps: Random Forest global SHAP + Logistic Regression theory table.
================================================================================
The frozen pipeline computes SHAP only for XGBoost (full suite) and logistic
regression (beeswarm + ranking). This additive module closes two gaps flagged
in the critical review:

  (1) Random forest has NO explanation at all -> add its global SHAP
      (beeswarm + mean|SHAP| bar + ranking table) via TreeExplainer.
  (2) The deep theory-consistency table is built on XGBoost, not on the
      best-performing model -> add a theory-consistency table for the
      logistic regression (the headline best model) using the SAME
      pre-specified signs and verdict logic as the XGBoost table.

READ-ONLY / additive. Loads the existing frozen models, scores the frozen
test sample, and writes ONLY new files:
  outputs/figures/shap/beeswarm_random_forest.{png,pdf}
  outputs/figures/shap/mean_abs_shap_random_forest.{png,pdf}
  outputs/tables/shap/shap_ranking_random_forest.{csv,tex}
  outputs/tables/shap/theory_consistency_table_logistic_regression.{csv,tex}
It does NOT touch the frozen XGBoost theory table (theory_consistency_table.*)
or any saved model.

Self-test gate: the LR SHAP top-10 ranking is checked against the frozen
shap_ranking_logistic_regression.csv before the LR theory table is trusted.

RF SHAP is computed on a fixed random sample of the test set (seed 42) to keep
the TreeExplainer runtime to a minute or two; this is standard for a beeswarm
and does not affect the sign/ranking conclusions.

Run:  python -m src.explainability.shap_gaps_rf_lr
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import ALL_FEATURES, DATA_SAMPLES, OUT_MODELS_SAVED, OUT_TABLES_SHAP
from src.explainability.shap_global import (
    compute_shap_values, plot_beeswarm, plot_mean_abs_shap,
)
from src.explainability.theory_consistency import (
    THEORETICAL_SIGNS, compute_observed_shap_directions, assign_verdict,
    _topk_collinearity, adjust_verdict_for_collinearity,
)
from src.utils.tables import save_table

LABEL = "distress"
RF_SAMPLE = 800    # rows for the RF TreeExplainer; 800 trees => ~19s/100 rows,
                   # so 800 rows ~= 2.5 min. Ranking/signs are stable at this size.


# ---------------------------------------------------------------------------
# (1) Random forest global SHAP
# ---------------------------------------------------------------------------

def _rf_shap_values(rf, X: np.ndarray) -> np.ndarray:
    """TreeExplainer SHAP for the positive class, robust to shap's return shape."""
    sv = shap.TreeExplainer(rf).shap_values(X, check_additivity=False)
    if isinstance(sv, list):                       # [neg, pos]
        sv = sv[1]
    else:
        sv = np.asarray(sv)
        if sv.ndim == 3:                           # (n, feat, classes) or (classes, n, feat)
            sv = sv[:, :, 1] if sv.shape[-1] == 2 else sv[1]
    if sv.shape != X.shape:
        raise ValueError(f"Unexpected RF SHAP shape {sv.shape} vs X {X.shape}")
    return sv


def run_rf_shap(test: pd.DataFrame) -> pd.DataFrame:
    rf = joblib.load(OUT_MODELS_SAVED / "random_forest.joblib")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test), size=min(len(test), RF_SAMPLE), replace=False)
    X_df = test.iloc[idx][ALL_FEATURES]
    X = X_df.astype(float).fillna(0).values

    print(f"  Random forest TreeExplainer on {len(idx):,} sampled test firm-years ...")
    shap_rf = _rf_shap_values(rf, X)

    plot_beeswarm(shap_rf, X_df, "random_forest")
    ranking = plot_mean_abs_shap(shap_rf, ALL_FEATURES, "random_forest")
    save_table(ranking, OUT_TABLES_SHAP / "shap_ranking_random_forest")
    top = ", ".join(ranking.head(6)["feature"])
    print(f"  RF top-6 by mean|SHAP|: {top}")
    return ranking


# ---------------------------------------------------------------------------
# (2) Logistic regression theory-consistency table (best model)
# ---------------------------------------------------------------------------

def _build_lr_theory_table(shap_values: np.ndarray, X_df: pd.DataFrame,
                           top_n: int = 10) -> pd.DataFrame:
    """Mirror build_theory_consistency_table, but for LR and a custom output."""
    observed = compute_observed_shap_directions(shap_values, X_df, top_n)
    coll = _topk_collinearity(X_df, observed["feature"].tolist(), 0.7).set_index("feature")
    rows = []
    for _, r in observed.iterrows():
        feat, theory = r["feature"], THEORETICAL_SIGNS.get(r["feature"], {})
        t_sign, o_sign, corr = theory.get("sign", "?"), r["observed_sign"], r["corr_feature_shap"]
        v_raw = assign_verdict(t_sign, o_sign, abs(corr))
        is_col = bool(coll.loc[feat, "is_collinear_in_topk"]) if feat in coll.index else False
        rows.append({
            "rank": int(r["rank"]), "feature": feat, "theoretical_sign": t_sign,
            "theoretical_direction": theory.get("direction", ""),
            "reference": theory.get("reference", ""), "observed_sign": o_sign,
            "corr_feature_shap": round(corr, 3),
            "max_corr_with_other_top": round(float(coll.loc[feat, "max_corr_with_other_top"]), 3)
                if feat in coll.index else 0.0,
            "collinear_partner": str(coll.loc[feat, "collinear_partner"]) if feat in coll.index else "",
            "verdict_raw": v_raw,
            "verdict": adjust_verdict_for_collinearity(v_raw, is_col),
        })
    return pd.DataFrame(rows).sort_values("rank")


def _selftest_lr_ranking(shap_lr: np.ndarray,
                         allow_missing_reference: bool = False) -> bool:
    """
    Recomputed LR SHAP top-10 must match the frozen ranking.

    A missing reference file is treated as NOT VERIFIED, not as a pass -- see
    the note in ``src/analysis/significance_4models._selftest``.
    """
    ref_path = OUT_TABLES_SHAP / "shap_ranking_logistic_regression.csv"
    if not ref_path.exists():
        print(f"  [FAIL] reference not found: {ref_path}")
        print("         Cannot verify against a frozen reference; treating as NOT verified.")
        return bool(allow_missing_reference)
    ref_top = list(pd.read_csv(ref_path).sort_values("mean_abs_shap", ascending=False)
                   ["feature"].head(10))
    mine = pd.DataFrame({"feature": ALL_FEATURES, "m": np.abs(shap_lr).mean(0)})
    mine_top = list(mine.sort_values("m", ascending=False)["feature"].head(10))
    ok = ref_top == mine_top
    print(f"  [ OK ] LR SHAP top-10 ranking reproduces the frozen ranking" if ok
          else f"  [FAIL] LR ranking mismatch\n    frozen: {ref_top}\n    mine:   {mine_top}")
    return ok


def run_lr_theory(test: pd.DataFrame) -> pd.DataFrame:
    lr = joblib.load(OUT_MODELS_SAVED / "logistic_regression.joblib")
    X_df = test[ALL_FEATURES]
    X = X_df.astype(float).fillna(0).values
    print("  Logistic-regression LinearExplainer on the full test sample ...")
    shap_lr = compute_shap_values(lr, X, "logistic_regression")

    if not _selftest_lr_ranking(shap_lr):
        raise RuntimeError(
            "LR SHAP ranking self-test did not verify against the frozen reference — "
            "refusing to publish the theory table"
        )

    table = _build_lr_theory_table(shap_lr, X_df, top_n=10)
    n_ok = (table["verdict"] == "Consistent").sum()
    n_bad = (table["verdict"] == "Inconsistent").sum()
    print(f"  LR theory-consistency: {n_ok} Consistent | {n_bad} Inconsistent "
          f"| {len(table) - n_ok - n_bad} other (of top 10)")
    save_table(
        table, OUT_TABLES_SHAP / "theory_consistency_table_logistic_regression",
        caption=(
            "Theory-consistency validation for the best-performing model "
            "(ridge logistic regression). For each of its ten leading "
            "mean-$|$SHAP$|$ features, the observed feature--SHAP correlation sign "
            "is compared with the pre-specified theoretical sign (same locked "
            "predictions and verdict rule as the XGBoost table). For a linear "
            "model the SHAP sign coincides with the standardised-coefficient sign, "
            "so this is the direct economic reading of the headline model."),
        label="tab:theory_consistency_lr",
    )
    return table


def main() -> None:
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    print("Filling SHAP gaps (additive; frozen XGBoost SHAP untouched)\n")
    print("(1) Random forest global SHAP")
    run_rf_shap(test)
    print("\n(2) Logistic-regression theory-consistency table")
    run_lr_theory(test)
    print("\nSaved -> beeswarm_random_forest.*, mean_abs_shap_random_forest.*, "
          "shap_ranking_random_forest.*, theory_consistency_table_logistic_regression.*")


if __name__ == "__main__":
    main()
