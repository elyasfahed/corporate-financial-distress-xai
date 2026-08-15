"""
Results comparison figures — thesis figures (Chapters 6 and 7).
================================================================
Two clean, publication-quality figures built ONLY from already-saved result
tables. Nothing is re-run: this reads CSVs and draws bars, so the frozen
models/SHAP/robustness pipeline is never touched.

  Figure 1 (Ch 6):  model_performance_comparison_4models
      PR-AUC (with bootstrap CIs) and ROC-AUC for all four models side by side.
      Source: outputs/tables/model_results/model_performance_test_4models.csv

  Figure 2 (Ch 7):  robustness_pr_auc_comparison
      PR-AUC of each model across the primary spec and RC1–RC5, grouped, with
      the prevalence baseline marked per check.
      Source: outputs/tables/robustness/robustness_consolidated.csv

Both saved as .pdf (LaTeX) + .png (draft) at 300 DPI, academic serif style.

NOTE (transparent deviation): these are post-freeze presentation figures of the
frozen results. They add no new analysis — they only visualise saved numbers.
Disclose them as such. RC1 (bankruptcy-only) shows the trees edging out LR; that
ranking flip is shown faithfully, not suppressed.

Run
---
    python -m src.analysis.results_figures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    OUT_FIGURES_MODEL,
    OUT_FIGURES_ROBUSTNESS,
)
from src.utils.plots import set_academic_style, save_figure


# One consistent colour per model, reused across both figures.
_PAL = sns.color_palette("colorblind")
MODEL_COLOR = {"LR": _PAL[0], "RF": _PAL[1], "XGB": _PAL[2], "NN": _PAL[3]}

# CSV model names → short labels used on axes/legends.
_SHORT = {
    "Logistic Regression (ridge)": "LR",
    "Logistic Regression": "LR",
    "logistic_regression": "LR",
    "Random Forest": "RF",
    "random_forest": "RF",
    "XGBoost": "XGB",
    "xgboost": "XGB",
    "Neural Network (MLP)": "NN",
    "neural_network": "NN",
    "neural_network_balanced": "NN",
}


# ---------------------------------------------------------------------------
# Figure 1 — model performance comparison (Chapter 6)
# ---------------------------------------------------------------------------

def plot_model_comparison(output_path=None) -> None:
    df = pd.read_csv(OUT_TABLES_MODEL / "model_performance_test_4models.csv")
    df["short"] = df["model"].map(_SHORT)
    colours = [MODEL_COLOR[s] for s in df["short"]]
    baseline = float(df["prevalence_baseline_pr_auc"].iloc[0])

    set_academic_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: PR-AUC with bootstrap confidence intervals ──
    yerr = np.vstack([
        df["pr_auc"] - df["pr_auc_ci_lower"],
        df["pr_auc_ci_upper"] - df["pr_auc"],
    ])
    bars1 = ax1.bar(df["short"], df["pr_auc"], color=colours, alpha=0.85,
                    yerr=yerr, capsize=5, error_kw=dict(lw=1.3))
    ax1.axhline(baseline, color="black", lw=1.2, ls="--",
                label=f"Prevalence baseline = {baseline:.4f}")
    for bar, v in zip(bars1, df["pr_auc"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12)
    ax1.set_ylabel("PR-AUC", fontsize=14)
    ax1.set_title("Precision–Recall AUC (primary metric)", fontsize=14)
    ax1.set_ylim(0, df["pr_auc_ci_upper"].max() * 1.18)
    ax1.tick_params(labelsize=12)
    ax1.legend(frameon=False, fontsize=11, loc="upper right")

    # ── Right: ROC-AUC (axis from 0.5 = random classifier) ──
    bars2 = ax2.bar(df["short"], df["roc_auc"], color=colours, alpha=0.85)
    ax2.axhline(0.5, color="black", lw=1.0, ls=":", label="Random = 0.50")
    for bar, v in zip(bars2, df["roc_auc"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=14)
    ax2.set_title("ROC AUC (co-primary metric)", fontsize=14)
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(labelsize=12)
    ax2.legend(frameon=False, fontsize=11, loc="lower right")

    fig.suptitle("Out-of-Sample Model Performance — Test Sample (2015–2024)",
                 fontsize=17, y=1.03)

    if output_path is None:
        output_path = OUT_FIGURES_MODEL / "model_performance_comparison_4models"
    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2 — robustness comparison (Chapter 7)
# ---------------------------------------------------------------------------

CHECK_LABELS = {
    "primary": "Primary",
    "RC1": "RC1\nBankruptcy\nonly",
    "RC2": "RC2\n6-month\nhorizon",
    "RC3": "RC3\nSMOTE",
    "RC4": "RC4\nAccounting\nonly",
    "RC5": "RC5\n+Utilities",
}


def plot_robustness_comparison(output_path=None) -> None:
    df = pd.read_csv(OUT_TABLES_ROBUSTNESS / "robustness_consolidated.csv")
    checks = [c for c in CHECK_LABELS if c in df["check"].unique()]
    models = ["logistic_regression", "random_forest", "xgboost"]

    pr = df.pivot_table(index="check", values="pr_auc", columns="model")
    base = df.groupby("check")["prevalence_baseline_pr_auc"].first()

    set_academic_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(checks))
    width = 0.26
    offsets = {"logistic_regression": -width, "random_forest": 0.0, "xgboost": width}

    for m in models:
        short = _SHORT[m]
        vals = [pr.loc[c, m] for c in checks]
        ax.bar(x + offsets[m], vals, width=width, color=MODEL_COLOR[short],
               alpha=0.85, label=short)

    # Prevalence baseline marker per check (baselines differ across checks).
    for i, c in enumerate(checks):
        b = base.loc[c]
        ax.hlines(b, x[i] - 1.6 * width, x[i] + 1.6 * width,
                  color="black", lw=1.4, ls="--",
                  label="Prevalence baseline" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([CHECK_LABELS[c] for c in checks], fontsize=11)
    ax.set_ylabel("PR-AUC", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(
        "PR-AUC by Model Across Robustness Checks (RC1–RC5) — Test Sample (2015–2024)",
        fontsize=15, y=1.02,
    )
    ax.legend(frameon=False, fontsize=12, ncol=4, loc="upper right")

    if output_path is None:
        output_path = OUT_FIGURES_ROBUSTNESS / "robustness_pr_auc_comparison"
    save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3 — full consolidated view: all checks + all models (Chapter 7)
# ---------------------------------------------------------------------------
# Combines the pre-registered RC1–RC5 with the post-freeze supplementary checks
# (RC6 drop-RSIZE) and the neural network. The NN was evaluated on the PRIMARY
# spec only, so it appears as a 4th bar in the "Primary" group and nowhere else
# (missing NN bars in RC1–RC6 = not run; nothing is imputed).

FULL_CHECK_LABELS = {
    "primary": "Primary",
    "RC1": "RC1\nBankruptcy\nonly",
    "RC2": "RC2\n6-month\nhorizon",
    "RC3": "RC3\nSMOTE",
    "RC4": "RC4\nAccounting\nonly",
    "RC5": "RC5\n+Utilities",
    "RC6": "RC6\nDrop RSIZE",
}


def plot_robustness_comparison_full(output_path=None) -> None:
    cons = pd.read_csv(OUT_TABLES_ROBUSTNESS / "robustness_consolidated.csv")
    rc6 = pd.read_csv(OUT_TABLES_ROBUSTNESS / "rc6_no_rsize_results.csv")
    nn = pd.read_csv(OUT_TABLES_ROBUSTNESS / "rc7b_neural_network_balanced_results.csv")

    # Assemble pr[check][short_model] and the per-check prevalence baseline.
    pr: dict[str, dict[str, float]] = {}
    base: dict[str, float] = {}

    for _, r in cons.iterrows():
        pr.setdefault(r["check"], {})[_SHORT[r["model"]]] = r["pr_auc"]
        base[r["check"]] = r["prevalence_baseline_pr_auc"]
    for _, r in rc6.iterrows():                       # RC6 = "no_rsize"
        pr.setdefault("RC6", {})[_SHORT[r["model"]]] = r["pr_auc"]
        base["RC6"] = r["prevalence_baseline_pr_auc"]
    nn_primary = nn.iloc[0]                           # NN on the primary spec
    pr["primary"]["NN"] = nn_primary["pr_auc"]

    # NN across robustness checks — auto-detect the re-tune job's output, so the
    # figure completes itself once src/robustness/rc_nn_all_checks.py has run.
    nn_checks_file = OUT_TABLES_ROBUSTNESS / "rc_nn_all_checks_results.csv"
    if nn_checks_file.exists():
        for _, r in pd.read_csv(nn_checks_file).iterrows():
            if r["check"] in FULL_CHECK_LABELS:
                pr.setdefault(r["check"], {})["NN"] = r["pr_auc"]
    else:                                             # fall back to per-check checkpoints
        for ckpt in sorted(OUT_TABLES_ROBUSTNESS.glob("rc_nn_RC*_results.csv")):
            r = pd.read_csv(ckpt).iloc[0]
            if r["check"] in FULL_CHECK_LABELS:
                pr.setdefault(r["check"], {})["NN"] = r["pr_auc"]

    checks = [c for c in FULL_CHECK_LABELS if c in pr]
    shorts = ["LR", "RF", "XGB", "NN"]

    set_academic_style()
    fig, ax = plt.subplots(figsize=(15, 6.5))

    x = np.arange(len(checks))
    width = 0.20
    offset = {"LR": -1.5 * width, "RF": -0.5 * width,
              "XGB": 0.5 * width, "NN": 1.5 * width}

    for s in shorts:
        xs = [x[i] + offset[s] for i, c in enumerate(checks) if s in pr[c]]
        ys = [pr[c][s] for c in checks if s in pr[c]]
        ax.bar(xs, ys, width=width, color=MODEL_COLOR[s], alpha=0.85, label=s)

    # Mark "RC7" only while the NN is primary-only; once the re-tune job gives it
    # a bar in every check it is a full model and needs no special tag.
    nn_only_primary = all("NN" not in pr[c] for c in checks if c != "primary")
    if nn_only_primary and "NN" in pr.get("primary", {}):
        ax.annotate("RC7", xy=(x[0] + offset["NN"], pr["primary"]["NN"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, color=MODEL_COLOR["NN"])

    # Per-check prevalence baseline (baselines differ across checks).
    for i, c in enumerate(checks):
        ax.hlines(base[c], x[i] - 2 * width, x[i] + 2 * width,
                  color="black", lw=1.3, ls="--",
                  label="Prevalence baseline" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([FULL_CHECK_LABELS[c] for c in checks], fontsize=11)
    ax.set_ylabel("PR-AUC", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(
        "PR-AUC by Model Across All Checks — Test Sample (2015–2024)",
        fontsize=15, y=1.02,
    )
    ax.legend(frameon=False, fontsize=12, ncol=5, loc="upper right")

    if output_path is None:
        output_path = OUT_FIGURES_ROBUSTNESS / "robustness_pr_auc_comparison_full"
    save_figure(fig, output_path)


def main() -> None:
    plot_model_comparison()
    plot_robustness_comparison()
    plot_robustness_comparison_full()
    print("\nThesis figures written to outputs/figures/model/ and outputs/figures/robustness/")


if __name__ == "__main__":
    main()
