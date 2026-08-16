"""
v2 secondary evidence — inference-only regeneration from the final v2 models.
==============================================================================
Companion to scripts/run_v2_rebuild.py + scripts/run_v2_nn.py.
Scores the four saved v2 models on the v2 test split ONCE (re-inference on
frozen joblibs — no model is fitted, no v1 artifact is touched) and produces
the v2 counterparts of the headline secondary evidence:

  outputs/tables/model_results/model_performance_test_v2_4models.{csv,tex}
      merged 4-model headline table (LR / RF / XGB / balanced NN)
  outputs/tables/model_results/v2_significance_tests.{csv,tex}
      all 6 pairwise comparisons: DeLong ROC-AUC z/p + firm-block
      bootstrap PR-AUC delta with recentred-under-H0 p and 95% CI
  outputs/figures/model/v2_roc_curves_4models.{png,pdf}
  outputs/figures/model/v2_pr_curves_4models.{png,pdf}
  outputs/figures/shap/v2_shap_beeswarm_{xgboost,logistic_regression,
      random_forest}.{png,pdf} and v2_shap_bar_*.{png,pdf}
  outputs/tables/shap/v2_shap_importance.{csv,tex}
      mean |SHAP| per feature per model (rank + value)

The NN is included in performance/significance evidence but not SHAP
(consistent with v1: no KernelExplainer run; SHAP covers XGB/LR/RF).
Expensive robustness re-runs (RC1-corrected, RC3, H2) are separate scripts —
they require re-tuning, not re-inference.

Run with the project .venv:
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_secondary
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import (
    DATA_SAMPLES_V2,
    FIG_DPI,
    OUT_FIGURES_MODEL,
    OUT_FIGURES_SHAP,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    OUT_TABLES_SHAP,
    V2_PROFILE,
)
from src.analysis.significance import _bootstrap_pr_auc_diff, _stars
from src.models.evaluate import delong_test
from src.models.train import resolve_artifact_dirs
from src.utils.fig_style import MODEL_COLORS, MODEL_LINESTYLES, apply_thesis_style
from src.utils.tables import save_table

apply_thesis_style()

MODELS = ["logistic_regression", "random_forest", "xgboost",
          "neural_network_balanced"]
DISPLAY = {
    "logistic_regression":     "Logistic Regression (ridge)",
    "random_forest":           "Random Forest",
    "xgboost":                 "XGBoost",
    "neural_network_balanced": "Neural Network",
}
# The "(balanced)" qualifier was removed from the model label on the
# author's instruction, in figures, tables and prose together. It is
# unambiguous because final_primary contains no raw neural network; the
# imbalance treatment this network receives is still stated in ch05 and
# ch06, where it is methodological content rather than a name.
COLORS = dict(MODEL_COLORS)   # shared model palette (fig_style)
SHAP_MODELS = ["xgboost", "logistic_regression", "random_forest"]
RUN_FIGURES_MODEL = OUT_FIGURES_MODEL / V2_PROFILE["spec"]
RUN_FIGURES_SHAP = OUT_FIGURES_SHAP / V2_PROFILE["spec"]
RUN_TABLES_MODEL = OUT_TABLES_MODEL / V2_PROFILE["spec"]
RUN_TABLES_ROBUSTNESS = OUT_TABLES_ROBUSTNESS / V2_PROFILE["spec"]
RUN_TABLES_SHAP = OUT_TABLES_SHAP / V2_PROFILE["spec"]


def _save_fig(fig, stem: Path) -> None:
    fig.set_facecolor("white")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  Figure -> {stem}.{{png,pdf}}")


def _holm(pvals: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjusted p-values for a family of tests.

    Returned in the original input order and enforced monotone-nondecreasing
    along the sorted sequence (the standard Holm guarantee). Controls the
    family-wise error rate across the pairwise comparisons within one metric.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def _load_predictions() -> tuple[pd.DataFrame, dict, list[str]]:
    features = list(V2_PROFILE["feature_set"])
    saved_dir, _ = resolve_artifact_dirs(V2_PROFILE["spec"])
    test = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    X = test[features].astype(float).values
    if np.isnan(X).any():
        raise AssertionError("v2 test split carries NaN in model features")
    probs = {}
    for name in MODELS:
        model = joblib.load(saved_dir / f"{name}.joblib")
        probs[name] = model.predict_proba(X)[:, 1]
        print(f"  Scored {name} on {len(test):,} test rows")
    return test, probs, features


# ---------------------------------------------------------------------------
# 1. Merged 4-model headline table (assemble from the two saved v2 CSVs)
# ---------------------------------------------------------------------------

def build_4model_table() -> pd.DataFrame:
    prim = pd.read_csv(RUN_TABLES_MODEL / "model_performance_test_3models.csv")
    test = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    n_test = len(test)
    n_events = int(test["distress"].sum())
    prevalence = float(test["distress"].mean())
    nn = pd.read_csv(
        RUN_TABLES_ROBUSTNESS / "neural_network_balanced_results.csv"
    ).iloc[0]

    cols = ["model", "prevalence_baseline_pr_auc", "pr_auc",
            "pr_auc_ci_lower", "pr_auc_ci_upper", "roc_auc",
            "precision", "recall", "f1", "ks_stat", "threshold"]
    rows = []
    for _, r in prim.iterrows():
        row = {c: r[c] for c in cols}
        row["model"] = DISPLAY.get(r["model"], r["model"])
        rows.append(row)
    nn_row = {c: nn[c] for c in cols if c in nn.index}
    nn_row["model"] = DISPLAY["neural_network_balanced"]
    rows.append(nn_row)

    out = pd.DataFrame(rows)[cols]
    save_table(
        out, RUN_TABLES_MODEL / "model_performance_test_4models",
        caption=(
            "Out-of-sample performance of the four co-primary models under "
            "the primary specification (FY1990--2023; corrected fiscal-"
            "year dating and delisting mapping; date-ranged NYSE/AMEX/NASDAQ "
            "universe; 18 predictors incl. MB\\_MISSING; market-feature "
            "imputation; fold-safe purged tuning CV). Test period 2015--2023, "
            f"{n_test:,} firm-years, {n_events:,} distress events "
            f"(prevalence-baseline average precision = {prevalence:.4f}). "
            "Bootstrap average-precision CIs use block resampling by "
            "firm (1{,}000 resamples). Brier scores are reported separately "
            "(calibration is not cross-model comparable in this table)."),
        label="tab:model_performance_test_v2_4models",
    )
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 1b. Clean the reported 3-model table — drop the non-comparable raw Brier
# ---------------------------------------------------------------------------

def clean_3model_report() -> None:
    """
    A5 fix — the reported 3-model performance table carried a raw Brier
    column (LR ≈ 0.016 vs RF / XGB ≈ 0.088 / 0.120) that is NOT cross-model
    comparable: the uncalibrated tree scores are not probabilities, so their
    raw Brier is inflated for reasons unrelated to discrimination. All
    calibration is reported from calibration_comparison_4models (validation
    Platt scaling, comparable). Drop the raw Brier column so the performance
    table cannot be misread as a calibration comparison. Idempotent.
    """
    path = RUN_TABLES_MODEL / "model_performance_test_3models.csv"
    df = pd.read_csv(path)
    if "brier_score" not in df.columns:
        print("  3-model table already has no brier_score column.")
        return
    df = df.drop(columns=["brier_score"])
    save_table(
        df, RUN_TABLES_MODEL / "model_performance_test_3models",
        caption=(
            "Out-of-sample discrimination of the three pre-specified models "
            "under the primary specification (test 2015--2023). Calibration "
            "is reported separately in the validation-Platt-scaled calibration "
            "comparison, where scores are cross-model comparable; raw Brier is "
            "omitted here because the uncalibrated tree scores are not "
            "probabilities and their raw Brier is not comparable to the "
            "logistic regression's."),
        label="tab:model_performance_test_v2_3models",
    )
    print("  Dropped non-comparable raw brier_score from the 3-model table.")


# ---------------------------------------------------------------------------
# 2. Pairwise significance (DeLong ROC + recentred firm-block bootstrap PR)
# ---------------------------------------------------------------------------

def build_significance(test: pd.DataFrame, probs: dict) -> pd.DataFrame:
    y = test["distress"].astype(int).values
    firms = test["gvkey"].values
    from itertools import combinations
    from sklearn.metrics import average_precision_score

    rows = []
    for a, b in combinations(MODELS, 2):
        delta = (average_precision_score(y, probs[a])
                 - average_precision_score(y, probs[b]))
        p_pr, ci_lo, ci_hi = _bootstrap_pr_auc_diff(y, probs[a], probs[b], firms)
        z, p_roc = delong_test(y, probs[a], probs[b])
        rows.append({
            "comparison": f"{DISPLAY[a]} vs {DISPLAY[b]}",
            "pr_auc_delta": delta,
            "pr_delta_ci_lower": ci_lo,
            "pr_delta_ci_upper": ci_hi,
            # Retain full precision until multiplicity adjustment.  Rounding
            # first can change a borderline Holm decision (for example,
            # 10/1001 = 0.009990... becomes 0.0100 before multiplication).
            "pr_p_value": p_pr,
            "pr_sig": _stars(p_pr),
            "delong_z": z,
            "roc_p_value": p_roc,
            "roc_sig": _stars(p_roc),
        })
    out = pd.DataFrame(rows)

    # Holm-Bonferroni family-wise correction across the six pairwise tests,
    # applied separately within each metric family (PR-AUC and ROC-AUC).
    out["pr_p_holm"] = _holm(out["pr_p_value"].tolist())
    out["pr_sig_holm"] = [_stars(p) for p in out["pr_p_holm"]]
    out["roc_p_holm"] = _holm(out["roc_p_value"].tolist())
    out["roc_sig_holm"] = [_stars(p) for p in out["roc_p_holm"]]

    # Round only for presentation, after all statistical decisions have been
    # computed from the full-precision values.
    for column in (
        "pr_auc_delta", "pr_delta_ci_lower", "pr_delta_ci_upper",
        "pr_p_value", "pr_p_holm", "roc_p_value", "roc_p_holm",
    ):
        out[column] = out[column].round(4)
    out["delong_z"] = out["delong_z"].round(3)
    out = out[[
        "comparison", "pr_auc_delta", "pr_delta_ci_lower", "pr_delta_ci_upper",
        "pr_p_value", "pr_sig", "pr_p_holm", "pr_sig_holm",
        "delong_z", "roc_p_value", "roc_sig", "roc_p_holm", "roc_sig_holm",
    ]]

    save_table(
        out, RUN_TABLES_MODEL / "significance_tests",
        caption=(
            "Pairwise model comparisons under the primary specification. "
            "PR-AUC differences: firm-block bootstrap (1{,}000 resamples), "
            "two-sided p-values from the null-recentred resampling "
            "distribution, percentile 95\\% CIs. ROC-AUC differences: DeLong "
            "test. Six pairwise tests per metric; both the raw two-sided "
            "p-value and the Holm--Bonferroni family-wise-corrected p-value "
            "(\\texttt{\\_holm}) are reported, corrected separately within "
            "each metric family. Signs follow the first-named model."),
        label="tab:v2_significance_tests",
    )
    print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# 3. ROC / PR curve figures (4 models)
# ---------------------------------------------------------------------------

def build_curves(test: pd.DataFrame, probs: dict) -> None:
    y = test["distress"].astype(int).values
    prevalence = y.mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    for name in MODELS:
        fpr, tpr, _ = roc_curve(y, probs[name])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.75,
                ls=MODEL_LINESTYLES[name],
                label=f"{DISPLAY[name]} (AUC = {roc_auc_score(y, probs[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    # Short label inside the figure; the descriptive sentence belongs to the
    # "Fig. N." caption below it (Elsevier convention).
    ax.set_title("ROC Curves (Test Sample)")
    ax.legend(loc="lower right")
    _save_fig(fig, RUN_FIGURES_MODEL / "roc_curves_4models")

    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    for name in MODELS:
        prec, rec, _ = precision_recall_curve(y, probs[name])
        from sklearn.metrics import average_precision_score
        ax.plot(rec, prec, color=COLORS[name], lw=1.75,
                ls=MODEL_LINESTYLES[name],
                label=f"{DISPLAY[name]} (PR-AUC = {average_precision_score(y, probs[name]):.3f})")
    ax.axhline(prevalence, color="k", ls="--", lw=0.8,
               label=f"Prevalence baseline ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.0)
    ax.set_title("Precision–Recall Curves (Test Sample)")
    ax.legend(loc="upper right")
    _save_fig(fig, RUN_FIGURES_MODEL / "pr_curves_4models")


# ---------------------------------------------------------------------------
# 4. Global SHAP (XGB / LR / RF) — beeswarm + bar + importance table
# ---------------------------------------------------------------------------

def build_shap(test: pd.DataFrame, features: list[str]) -> None:
    import shap
    from src.explainability.shap_global import compute_shap_values

    saved_dir, _ = resolve_artifact_dirs(V2_PROFILE["spec"])
    X = test[features].astype(float).values
    importance_rows = []

    for name in SHAP_MODELS:
        print(f"\n  SHAP: {name} (full v2 test, {len(test):,} rows) ...")
        model = joblib.load(saved_dir / f"{name}.joblib")
        sv = compute_shap_values(model, X, model_name=name)

        fig = plt.figure(figsize=(6.8, 5.3))
        shap.summary_plot(sv, X, feature_names=features, show=False,
                          max_display=15, plot_size=None)
        plt.gca().set_title(f"SHAP Value Distribution — {DISPLAY[name]}")
        _save_fig(fig, RUN_FIGURES_SHAP / f"shap_beeswarm_{name}")

        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        top = order[:15][::-1]
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        ax.barh(np.asarray(features)[top], mean_abs[top],
                color=COLORS[name], alpha=0.92)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title(f"Mean Absolute SHAP — {DISPLAY[name]}")
        ax.grid(axis="y", visible=False)
        _save_fig(fig, RUN_FIGURES_SHAP / f"shap_bar_{name}")

        for rank, j in enumerate(order, start=1):
            importance_rows.append({
                "model": DISPLAY[name],
                "rank": rank,
                "feature": features[j],
                "mean_abs_shap": round(float(mean_abs[j]), 5),
            })

    imp = pd.DataFrame(importance_rows)
    save_table(
        imp, RUN_TABLES_SHAP / "shap_importance",
        caption=(
            "Global SHAP feature importance (mean $|$SHAP$|$ over the full "
            "test sample) for the three primary models under the primary "
            "specification (18 predictors incl. MB\\_MISSING). "
            "TreeExplainer for XGBoost/Random Forest; LinearExplainer on the "
            "underlying (pre-Platt) logistic regression."),
        label="tab:v2_shap_importance",
    )
    print(f"\n  Top-5 per model:\n"
          f"{imp[imp['rank'] <= 5].to_string(index=False)}")


def main() -> None:
    if ".venv" not in sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv "
              f"(sys.prefix={sys.prefix}).")
    for d in (RUN_FIGURES_MODEL, RUN_FIGURES_SHAP, RUN_TABLES_MODEL,
              RUN_TABLES_SHAP):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  V2 SECONDARY EVIDENCE — re-inference on saved v2 models")
    print("=" * 68)
    test, probs, features = _load_predictions()

    if "--curves-only" in sys.argv:
        print("\n[Curve-only] ROC / PR figures")
        build_curves(test, probs)
        print("\nCurve-only presentation refresh complete.")
        return

    print("\n[1/4] Merged 4-model headline table")
    clean_3model_report()
    build_4model_table()

    print("\n[2/4] Pairwise significance tests")
    build_significance(test, probs)

    print("\n[3/4] ROC / PR curve figures")
    build_curves(test, probs)

    print("\n[4/4] Global SHAP (XGB / LR / RF)")
    build_shap(test, features)

    print("\nV2 secondary evidence complete.")


if __name__ == "__main__":
    main()
