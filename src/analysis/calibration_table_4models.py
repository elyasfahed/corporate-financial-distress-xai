"""
4-model calibration TABLE (balanced NN) — companion to calibration_figures_4models.
===================================================================================
The figures `calibration_4models_{raw,calibrated}` already use the BALANCED
neural network (the headline NN). The only matching TABLE on disk
(`calibration_ece_brier_all_models.csv`, June 14) instead uses the RAW RC7 NN,
so table and figure disagree on the NN. This additive script produces a TABLE
consistent with the figures: ECE + Brier (raw and Platt-calibrated) for the four
HEADLINE models (LR, RF, XGB, balanced NN), using the SAME quantile-ECE
definition as the figures.

READ-ONLY / additive. Loads saved models + val/test splits, fits Platt scaling
on the VALIDATION set only (never the test set), writes ONLY a new file:
  outputs/tables/model_results/calibration_ece_brier_4models_balanced.{csv,tex}
No saved model is re-fit or overwritten.

Self-test gate: LR/RF/XGB raw ECE must reproduce the frozen
calibration_ece_brier_all_models.csv (those three rows are unchanged; only the
NN switches from raw to balanced). Platt rescaling is monotone, so PR-AUC and
ROC-AUC are unaffected and are not re-reported here.

Run:  python -m src.analysis.calibration_table_4models
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import ALL_FEATURES, DATA_SAMPLES, OUT_TABLES_MODEL
from src.analysis.lr_calibration import apply_platt_scaling
from src.analysis.calibration_figures_4models import (
    ece_quantile, _raw_base, MODEL_ORDER, DISPLAY,
)
from src.utils.io import load_model
from src.utils.tables import save_table

LABEL = "distress"


def _selftest(df: pd.DataFrame, allow_missing_reference: bool = False) -> bool:
    """
    Recomputed raw ECE must match the frozen calibration reference.

    A missing reference file is treated as NOT VERIFIED, not as a pass -- see
    the note in ``src/analysis/significance_4models._selftest``.
    """
    ref_path = OUT_TABLES_MODEL / "calibration_ece_brier_all_models.csv"
    if not ref_path.exists():
        print(f"  [FAIL] reference not found: {ref_path}")
        print("         Cannot verify against a frozen reference; treating as NOT verified.")
        return bool(allow_missing_reference)
    ref = pd.read_csv(ref_path)
    ref_map = {"Logistic Regression": "logistic_regression",
               "Random Forest": "random_forest", "XGBoost": "xgboost"}
    ok, checked = True, 0
    for disp, name in ref_map.items():
        r = ref[ref["model"] == disp]
        m = df[df["model"] == name]
        if r.empty or m.empty:
            continue
        checked += 1
        a, b = float(r["ece_raw"].iloc[0]), float(m["ece_raw"].iloc[0])
        if abs(a - b) > 1e-3:
            print(f"  [FAIL] {name} ece_raw: frozen {a} vs recomputed {b}")
            ok = False
    print(f"  [ OK ] frozen LR/RF/XGB raw-ECE reproduced ({checked} models)" if ok
          else "  [FAIL] self-test mismatch — do not trust the table")
    return ok


def main() -> None:
    val  = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X = test[ALL_FEATURES].astype(float).fillna(0).values
    y = test[LABEL].astype(int).values

    rows = []
    for name in MODEL_ORDER:                       # ends with neural_network_balanced
        base  = _raw_base(load_model(name))
        platt = apply_platt_scaling(base, val)     # validation-fit only
        y_raw = base.predict_proba(X)[:, 1]
        y_cal = platt.predict_proba(X)[:, 1]
        rows.append({
            "model":            name,
            "display":          DISPLAY[name],
            "ece_raw":          round(ece_quantile(y, y_raw), 4),
            "ece_calibrated":   round(ece_quantile(y, y_cal), 4),
            "brier_raw":        round(brier_score_loss(y, y_raw), 4),
            "brier_calibrated": round(brier_score_loss(y, y_cal), 4),
        })
    df = pd.DataFrame(rows)

    print("Calibration (balanced NN), test sample:")
    print(df[["display", "ece_raw", "ece_calibrated", "brier_raw", "brier_calibrated"]]
          .to_string(index=False))
    print()
    if not _selftest(df):
        raise RuntimeError(
            "calibration self-test did not verify against the frozen reference — "
            "refusing to publish the table"
        )

    save_table(
        df, OUT_TABLES_MODEL / "calibration_ece_brier_4models_balanced",
        caption=(
            "Calibration of the four headline models on the test sample, before and "
            "after Platt scaling (fit on the validation set only). ECE is the "
            "equal-count (quantile) expected calibration error (10 bins), matching the "
            "calibration figures; the neural network is the balanced (RC7b) headline "
            "model. Raw probabilities are poorly calibrated because each model's "
            "imbalance treatment inflates them; Platt scaling brings all four into "
            "agreement. The rescaling is monotone, so PR-AUC and ROC-AUC are unchanged."),
        label="tab:calibration_4models",
    )
    print("Saved -> calibration_ece_brier_4models_balanced.{csv,tex}")


if __name__ == "__main__":
    main()
