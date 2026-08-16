"""
Figure utilities — academic-quality plot saving and styling.
=============================================================
All figures are saved as both PDF (for LaTeX inclusion) and PNG
(for draft review) at 300 DPI.

Style conventions:
  - No background gridlines on academic plots
  - Font: serif (Times New Roman or Computer Modern compatible)
  - Figure size: default 8×5 inches (landscape) for wide plots
  - Colour scheme: colourblind-safe palette (seaborn 'colorblind')
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — prevents Tcl/Tk threading crash
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

from src.config import FIG_DPI, FIG_FORMATS, TRAIN_END_YEAR, VAL_END_YEAR
from src.utils.fig_style import (
    MODEL_COLORS, MODEL_LINESTYLES, apply_thesis_style,
)


# ---------------------------------------------------------------------------
# Global matplotlib style settings
# ---------------------------------------------------------------------------

def set_academic_style() -> None:
    """
    Apply academic-quality matplotlib style settings.
    Call once at the top of any plotting script.
    """
    apply_thesis_style()
    mpl.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "figure.constrained_layout.use": True,
    })


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    path_stem: Path,
    formats: list[str] = FIG_FORMATS,
    dpi: int = FIG_DPI,
    close: bool = True,
) -> None:
    """
    Save a matplotlib figure in all required formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_stem : Path
        File path WITHOUT extension.
    formats : list[str]
        Default ['pdf', 'png'].
    dpi : int
    close : bool
        If True, close the figure after saving to free memory.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = path_stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  Figure saved -> {out}")
    if close:
        plt.close(fig)


# ---------------------------------------------------------------------------
# ROC and PR curve plots
# ---------------------------------------------------------------------------

_PALETTE = [MODEL_COLORS[name] for name in (
    "logistic_regression", "random_forest", "xgboost"
)]
_MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}


def plot_roc_curves(
    results: dict,
    y_test: "np.ndarray",
    output_path: Path,
) -> None:
    """
    Plot ROC curves for all three models on a single axes.

    Parameters
    ----------
    results : dict
        {model_name: {'model': fitted_model, 'threshold': float, ...}}
        models must have a predict_proba method.
    y_test : np.ndarray
    output_path : Path
        Stem path for saving (no extension).
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, info), colour in zip(results.items(), _PALETTE):
        model  = info["model"]
        X_test = info.get("X_test")   # caller may pre-attach X_test for convenience
        y_prob = info.get("y_prob")
        if y_prob is None and X_test is not None:
            y_prob = model.predict_proba(X_test)[:, 1]
        if y_prob is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.8, color=colour,
                ls=MODEL_LINESTYLES.get(name, "-"),
                label=f"{_MODEL_LABELS.get(name, name)} (AUC = {roc_auc:.3f})")

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_figure(fig, output_path)


def plot_pr_curves(
    results: dict,
    y_test: "np.ndarray",
    output_path: Path,
) -> None:
    """
    Plot Precision-Recall curves for all three models.

    Includes the no-skill baseline (horizontal line at prevalence rate).

    Parameters
    ----------
    results : dict
    y_test : np.ndarray
    output_path : Path
    """
    set_academic_style()
    prevalence = float(y_test.mean())
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, info), colour in zip(results.items(), _PALETTE):
        y_prob = info.get("y_prob")
        if y_prob is None:
            continue
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, lw=1.8, color=colour,
                ls=MODEL_LINESTYLES.get(name, "-"),
                label=f"{_MODEL_LABELS.get(name, name)} (PR-AUC = {pr_auc:.3f})")

    # No-skill baseline
    ax.axhline(y=prevalence, color="gray", linestyle="--", lw=0.8,
               label=f"No-skill baseline (prevalence = {prevalence:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_figure(fig, output_path)


def plot_distress_rate_timeseries(
    annual_stats: "pd.DataFrame",
    output_path: Path,
) -> None:
    """
    Time-series plot of annual distress rate 1990–2024.

    Required by §8.4 of the pre-specified study design. Must show visible spikes in
    2001–02 (dot-com), 2008–09 (GFC), and 2020 (COVID).
    Vertical shaded bands mark the train / val / test periods.

    Parameters
    ----------
    annual_stats : pd.DataFrame
        Output of src.utils.metrics.annual_distress_rate().
        Columns: fyear, n_total, n_distressed, distress_rate_pct.
    output_path : Path
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(annual_stats["fyear"], annual_stats["distress_rate_pct"],
            color="steelblue", lw=1.8, marker="o", ms=4, label="Annual distress rate (%)")

    # Shaded sample period bands
    ax.axvspan(annual_stats["fyear"].min(), TRAIN_END_YEAR + 0.5,
               alpha=0.08, color="blue",   label="Training (1990–2009)")
    ax.axvspan(TRAIN_END_YEAR + 0.5,    VAL_END_YEAR + 0.5,
               alpha=0.08, color="orange", label="Validation (2010–2014)")
    ax.axvspan(VAL_END_YEAR + 0.5, annual_stats["fyear"].max() + 0.5,
               alpha=0.08, color="green",  label="Test (2015–2024)")

    # Crisis annotations
    for crisis_year, label in [(2001, "Dot-com"), (2008, "GFC"), (2020, "COVID")]:
        ax.axvline(x=crisis_year, color="crimson", lw=0.8, linestyle=":")
        ax.text(crisis_year + 0.1,
                annual_stats["distress_rate_pct"].max() * 0.95,
                label, fontsize=8, color="crimson")

    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Annual Distress Rate (%)")
    ax.set_title("Annual Corporate Financial Distress Rate: US CRSP/Compustat Panel, 1990–2024")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    save_figure(fig, output_path)


def plot_calibration(
    results: dict,
    y_test: "np.ndarray",
    output_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Calibration plots (reliability diagrams) for all three models.

    Compares mean predicted probability against observed distress frequency
    within probability bins. A well-calibrated model lies on the diagonal.

    Parameters
    ----------
    results : dict
    y_test : np.ndarray
    output_path : Path
    n_bins : int
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    for (name, info), colour in zip(results.items(), _PALETTE):
        y_prob = info.get("y_prob")
        if y_prob is None:
            continue
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="s", lw=1.5, color=colour,
                label=_MODEL_LABELS.get(name, name))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Observed)")
    ax.set_title("Calibration Plots: Corporate Financial Distress Prediction — Test Sample (2015–2024)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_figure(fig, output_path)
