"""
Out-of-time stability of the four co-primary models (additive extension).
=========================================================================
Responds to the "single test period is fragile" critique: it re-scores the
four SAVED models on sub-windows of the held-out test decade (2015--2024) to
show whether the LR-over-ML ordering is stable across time or an artefact of
one pooled test period.

READ-ONLY / additive. Loads the existing frozen models, scores frozen test
slices, and writes ONLY new files:
  outputs/tables/model_results/out_of_time_stability.{csv,tex}
  outputs/figures/model/out_of_time_stability.{png,pdf}
No model is retrained. Scoring matches the primary flow exactly
(X = test[ALL_FEATURES].astype(float).fillna(0); predict_proba[:, 1]);
PR-AUC and ROC-AUC are threshold-free, so they reproduce regardless of the
Platt rescaling of the saved logistic regression.

Self-test gate: the pre-COVID / COVID+ windows are recomputed for LR/RF/XGB
and checked against the frozen subperiod_performance.csv BEFORE the finer
windows (and the neural network) are trusted.

Honest caveat (surfaced in the table caption and the x-axis labels): 2-year
windows contain few distress events (as few as ~26), so an individual window
is itself noisy. The point is the *pattern* across the decade, read with the
event counts in view -- not any single window.

Run:  python -m src.analysis.out_of_time_stability
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from src.config import (
    ALL_FEATURES,
    DATA_SAMPLES,
    OUT_MODELS_SAVED,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    OUT_FIGURES_MODEL,
)
from src.utils.tables import save_table

LABEL = "distress"

# name -> (joblib file, config file).  NN = the balanced RC7b headline model.
MODELS = {
    "logistic_regression": ("logistic_regression.joblib",      "logistic_regression_config.yaml"),
    "random_forest":       ("random_forest.joblib",            "random_forest_config.yaml"),
    "xgboost":             ("xgboost.joblib",                  "xgboost_config.yaml"),
    "neural_network":      ("neural_network_balanced.joblib",  "neural_network_balanced_config.yaml"),
}
DISPLAY = {"logistic_regression": "LR", "random_forest": "RF",
           "xgboost": "XGB", "neural_network": "NN"}
COLORS = {"logistic_regression": "#1f77b4", "random_forest": "#ff7f0e",
          "xgboost": "#2ca02c", "neural_network": "#d62728"}

# Full period + finer 2-year windows (the reported analysis).
WINDOWS = [
    ("2015–2024 (full)", 2015, 2024),
    ("2015–2016", 2015, 2016),
    ("2017–2018", 2017, 2018),
    ("2019–2020", 2019, 2020),
    ("2021–2022", 2021, 2022),
    ("2023–2024", 2023, 2024),
]
# Self-test windows — must reproduce subperiod_performance.csv exactly.
SELFTEST = [("2015–2019 (Pre-COVID)", 2015, 2019), ("2020–2024 (COVID+)", 2020, 2024)]


def _load():
    models, thr = {}, {}
    for name, (mf, cf) in MODELS.items():
        path = OUT_MODELS_SAVED / mf
        if not path.exists():
            raise FileNotFoundError(f"Missing saved model: {path}")
        models[name] = joblib.load(path)
        cfg = yaml.safe_load((OUT_MODELS_CONFIGS / cf).read_text()) or {}
        thr[name] = float(cfg.get("threshold", 0.5))
    return models, thr


def _metrics(model, sub: pd.DataFrame, threshold: float) -> dict:
    X = sub[ALL_FEATURES].astype(float).fillna(0).values
    y = sub[LABEL].astype(int).values
    n, nd = len(y), int(y.sum())
    base = float(y.mean()) if n else float("nan")
    if nd == 0 or y.var() == 0:
        return dict(n_obs=n, n_distressed=nd, prevalence_pct=round(100 * base, 3),
                    baseline_pr_auc=round(base, 4), pr_auc=np.nan, roc_auc=np.nan, f1=np.nan)
    p = model.predict_proba(X)[:, 1]
    return dict(n_obs=n, n_distressed=nd, prevalence_pct=round(100 * base, 3),
                baseline_pr_auc=round(base, 4),
                pr_auc=round(average_precision_score(y, p), 4),
                roc_auc=round(roc_auc_score(y, p), 4),
                f1=round(f1_score(y, (p >= threshold).astype(int), zero_division=0), 4))


def _rows(test, models, thr, windows) -> pd.DataFrame:
    out = []
    for label, a, b in windows:
        sub = test[test["fyear"].between(a, b)]
        for name in MODELS:
            out.append({"window": label, "model": name, **_metrics(models[name], sub, thr[name])})
    return pd.DataFrame(out)


def _selftest(test, models, thr, allow_missing_reference: bool = False) -> bool:
    """
    Recomputed sub-period rows must match the frozen reference.

    A missing reference file is treated as NOT VERIFIED, not as a pass -- see
    the note in ``src/analysis/significance_4models._selftest``.
    """
    ref_path = OUT_TABLES_MODEL / "subperiod_performance.csv"
    if not ref_path.exists():
        print(f"  [FAIL] reference not found: {ref_path}")
        print("         Cannot verify against a frozen reference; treating as NOT verified.")
        return bool(allow_missing_reference)
    ref = pd.read_csv(ref_path)
    mine = _rows(test, models, thr, SELFTEST)
    ok, checked = True, 0
    for label, _, _ in SELFTEST:
        for name in ("logistic_regression", "random_forest", "xgboost"):
            r = ref[(ref["period"] == label) & (ref["model"] == name)]
            m = mine[(mine["window"] == label) & (mine["model"] == name)]
            if r.empty or m.empty:
                continue
            checked += 1
            for col in ("pr_auc", "roc_auc"):
                a, b = float(r[col].iloc[0]), float(m[col].iloc[0])
                if abs(a - b) > 1e-4:
                    print(f"  [FAIL] {label}/{name} {col}: ref {a} vs recomputed {b}")
                    ok = False
    print(f"  [ OK ] frozen subperiod {checked} cell(s) reproduced exactly" if ok
          else "  [FAIL] self-test mismatch — DO NOT trust the new windows")
    return ok


def _figure(df: pd.DataFrame, stem: str) -> None:
    fine = [w[0] for w in WINDOWS if "full" not in w[0]]
    counts = [int(df[(df.window == l) & (df.model == "logistic_regression")]["n_distressed"].iloc[0])
              for l in fine]
    xlab = [f"{l}\n(n={c})" for l, c in zip(fine, counts)]
    x = np.arange(len(fine))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in MODELS:
        ys = [df[(df.window == l) & (df.model == name)]["pr_auc"].iloc[0] for l in fine]
        ax.plot(x, ys, marker="o", label=DISPLAY[name], color=COLORS[name])
    full_lr = df[(df.window == "2015–2024 (full)") & (df.model == "logistic_regression")]["pr_auc"].iloc[0]
    ax.axhline(full_lr, ls="--", lw=1, color="#1f77b4", alpha=0.5,
               label="LR full-period")
    ax.set_xticks(x); ax.set_xticklabels(xlab, fontsize=8)
    ax.set_ylabel("PR-AUC"); ax.set_xlabel("Test sub-window (fiscal years; n = distress events)")
    ax.set_title("Out-of-time stability of PR-AUC across the test decade (2015–2024)")
    ax.legend(title="Model", ncol=5, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    models, thr = _load()

    print("Self-test against the frozen subperiod table:")
    if not _selftest(test, models, thr):
        raise SystemExit(
            "SELF-TEST FAILED: recomputed pre-COVID/COVID+ metrics do not match "
            "subperiod_performance.csv. The subperiod table is stale relative to the "
            "saved models — regenerate it first (run_pipeline.py --stages 11). "
            "No output written."
        )

    df = _rows(test, models, thr, WINDOWS)
    show = df.copy(); show["model"] = show["model"].map(DISPLAY)
    print("\nOut-of-time stability (PR-AUC / ROC-AUC by window):")
    print(show[["window", "model", "n_distressed", "pr_auc", "roc_auc", "baseline_pr_auc"]]
          .to_string(index=False))

    # Does LR keep the top PR-AUC in each finer window?
    fine = [w[0] for w in WINDOWS if "full" not in w[0]]
    wins = sum(df[df.window == l].sort_values("pr_auc").iloc[-1]["model"] == "logistic_regression"
               for l in fine)
    print(f"\nLR has the highest PR-AUC in {wins} of {len(fine)} two-year windows.")

    save_table(
        df, OUT_TABLES_MODEL / "out_of_time_stability",
        caption=(
            "Out-of-time stability across the held-out test decade. Each of the four "
            "co-primary models is the frozen saved model, re-scored on the indicated "
            "fiscal-year sub-window (no retraining). PR-AUC and ROC-AUC are threshold-free. "
            "Two-year windows contain few distress events ($n$ shown), so individual "
            "windows are noisy; the relevant signal is the ordering pattern across the "
            "decade. The pre-COVID/COVID+ split reproduces the sub-period table exactly."),
        label="tab:out_of_time_stability",
    )
    OUT_FIGURES_MODEL.mkdir(parents=True, exist_ok=True)
    _figure(df, str(OUT_FIGURES_MODEL / "out_of_time_stability"))
    print(f"\nSaved -> {OUT_TABLES_MODEL / 'out_of_time_stability.csv'} (+ .tex) and "
          f"{OUT_FIGURES_MODEL / 'out_of_time_stability.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
