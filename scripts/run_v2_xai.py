"""
v2 XAI + remaining figures — steps 1-5 of the v2 figure parity plan.
=====================================================================
Companion to scripts/run_v2_secondary.py. Re-inference on
the saved v2 models only; writes NEW v2_-prefixed files, never touches a
v1 artifact.

  1. SHAP dependence plots for the top-5 XGBoost features
       outputs/figures/shap/v2_shap_dependence_{feature}.{png,pdf}
  2. Leverage x Profitability interaction heatmap (TLTA x NITA tertiles,
     mean raw XGB distress score)
       outputs/figures/shap/v2_lev_roa_heatmap.{png,pdf}
  3. Local SHAP waterfalls for three firms (TP / FN / TN on XGBoost)
       outputs/figures/shap/v2_shap_waterfall_{tp,fn,tn}.{png,pdf}
  4. LIME local explanations for the same three firms + SHAP-LIME
     agreement table (binary indicators treated as categorical)
       outputs/figures/shap/v2_lime_{tp,fn,tn}.{png,pdf}
       outputs/tables/shap/v2_lime_shap_agreement.{csv,tex}
  5. Calibration figure (4 models; one validation-only Platt fit per native
     model, with reliability intervals and score distributions) + descriptive distress-rate
     time series and sample-composition table
       outputs/figures/model/v2_calibration_4models.{png,pdf}
       outputs/figures/descriptive/v2_distress_rate_timeseries.{png,pdf}
       outputs/tables/descriptive/v2_sample_composition_by_year.{csv,tex}

Run with the project .venv:
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_xai

Presentation-only regeneration (no SHAP/LIME recomputation):
    PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_xai --presentation-only
"""
from __future__ import annotations

import argparse
import re
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
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import (
    DATA_SAMPLES_V2,
    FIG_DPI,
    OUT_FIGURES_DESCRIPTIVE,
    OUT_FIGURES_MODEL,
    OUT_FIGURES_SHAP,
    OUT_MODELS_CONFIGS,
    OUT_MODELS_SAVED,
    OUT_TABLES_DESCRIPTIVE,
    OUT_TABLES_MODEL,
    OUT_TABLES_SHAP,
    RANDOM_SEED,
    V2_PROFILE,
)
from src.utils.fig_style import (
    MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS, apply_thesis_style,
)
from src.utils.tables import save_table

apply_thesis_style()

MODELS = ["logistic_regression", "random_forest", "xgboost",
          "neural_network_balanced"]
DISPLAY = {
    "logistic_regression":     "Logistic Regression",
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
BINARY_FEATURES = ["OENEG", "INTWO", "MB_MISSING"]
RUN_FIGURES_SHAP = OUT_FIGURES_SHAP / V2_PROFILE["spec"]
RUN_FIGURES_MODEL = OUT_FIGURES_MODEL / V2_PROFILE["spec"]
RUN_FIGURES_DESCRIPTIVE = OUT_FIGURES_DESCRIPTIVE / V2_PROFILE["spec"]
RUN_TABLES_SHAP = OUT_TABLES_SHAP / V2_PROFILE["spec"]
RUN_TABLES_DESCRIPTIVE = OUT_TABLES_DESCRIPTIVE / V2_PROFILE["spec"]
RUN_TABLES_MODEL = OUT_TABLES_MODEL / V2_PROFILE["spec"]

# The SHAP waterfalls and the LIME panels are typeset three-up at
# 0.47\textwidth. With \textwidth = 15.5 cm that is 2.87 in on the page
# against a 7 in canvas — a reduction to 41%, which renders the shared 9 pt
# figure font at under 4 pt. Scaling the type on the unchanged canvas (rather
# than shrinking the canvas) keeps SHAP's internal row layout intact while
# putting the printed size back at roughly 8 pt.
SMALL_MULTIPLE_SCALE = 2.2
SMALL_MULTIPLE_RC = {
    key: value * SMALL_MULTIPLE_SCALE
    for key, value in {
        "font.size": 9.0, "axes.titlesize": 10.5, "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0, "ytick.labelsize": 9.0, "legend.fontsize": 8.5,
    }.items()
}


def _save_fig(fig, stem: Path) -> None:
    fig.set_facecolor("white")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=FIG_DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  Figure -> {stem}.{{png,pdf}}")


def _typeset_rule(rule: str) -> str:
    """Typeset a LIME discretisation rule for print.

    LIME emits rules as source text: ``RSIZE <= -12.52``. Rendered in the
    thesis serif face those ASCII operators read as code output rather than as
    typeset mathematics, so the relations become the proper glyphs and the
    hyphen-minus in front of a numeral becomes U+2212. Feature names carry
    underscores but never hyphens, so keying the minus on a following digit
    cannot corrupt a name. Only the label text changes; the rule, its
    threshold, and its weight are untouched.
    """
    out = rule.replace("<=", "≤").replace(">=", "≥")
    return re.sub(r"-(?=\d)", "−", out)


def _save_distress_rate_figure(comp: pd.DataFrame) -> None:
    """Plot annual rates without connecting across purged boundary years."""
    annual = (
        comp.set_index("fyear")["distress_rate_pct"]
        .reindex(range(int(comp["fyear"].min()), int(comp["fyear"].max()) + 1))
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        annual.index,
        annual.values,
        "o-",
        ms=3.8,
        lw=1.55,
        color="#1f3864",
        zorder=3,
    )
    # The three split periods are shaded at an opacity that is actually legible
    # in print. The previous alpha of 0.035 rendered all three bands as bare
    # white, so their legend swatches were empty boxes — an encoding the reader
    # could not see. Each band is now labelled in place, at the top of its own
    # region, which removes the need for those swatches in the legend at all.
    ax.set_xlim(1989.5, 2023.5)
    ax.set_ylim(0, float(annual.max()) * 1.22)
    top = ax.get_ylim()[1]
    for x0, x1, label, color in [
        (1989.5, 2008.5, "Training", "#4C72B0"),
        (2009.5, 2013.5, "Validation", "#E69F00"),
        (2014.5, 2023.5, "Test", "#009E73"),
    ]:
        ax.axvspan(x0, x1, alpha=0.10, color=color, lw=0, zorder=0)
        ax.text(
            (x0 + x1) / 2.0, top * 0.965, label,
            ha="center", va="top", fontsize=8.5, color="#333333",
        )
    # The two purged boundary years (2009, 2014) carry no observation, so the
    # series is already broken there; the rules make the break explicit.
    for year in (2009, 2014):
        ax.axvline(year, color="#7F8C8D", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Distress rate (%)")
    # Short label inside the figure; the descriptive sentence belongs to the
    # "Fig. N." caption below it (Elsevier convention).
    ax.set_title("Annual Distress Rate")
    # No legend: one series, and the three periods are named on the plot.
    _save_fig(fig, RUN_FIGURES_DESCRIPTIVE / "distress_rate_timeseries")


def _quantile_reliability(y_true: np.ndarray, y_prob: np.ndarray,
                          n_bins: int = 10) -> tuple[np.ndarray, ...]:
    """Equal-count reliability bins with 95% Wilson event-rate intervals."""
    groups = np.array_split(np.argsort(y_prob), n_bins)
    mean_pred, observed, lower, upper, counts = [], [], [], [], []
    z = 1.959963984540054
    for idx in groups:
        n = len(idx)
        p = float(np.mean(y_true[idx]))
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2.0 * n)) / denom
        half = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
        mean_pred.append(float(np.mean(y_prob[idx])))
        observed.append(p)
        lower.append(max(0.0, centre - half))
        upper.append(min(1.0, centre + half))
        counts.append(n)
    return tuple(np.asarray(x) for x in
                 (mean_pred, observed, lower, upper, counts))


def _save_calibration_figure(models: dict, X_val: np.ndarray,
                             y_val: np.ndarray, X_test: np.ndarray,
                             y_test: np.ndarray, write_table: bool = True) -> None:
    """Like-for-like reliability curves after one validation-only Platt fit.

    A single panel, on the author's instruction. The zoom inset and the lower
    score-distribution panel were removed because together with the legend they
    made the exhibit hard to read. What they carried is not lost, only moved to
    the ch07 caption and text: that eight of the ten equal-count bins fall below
    4.5% predicted risk and therefore crowd near the origin, and that the
    calibrated scores are strongly right-skewed at a 1.58% event rate. Dropping
    the distribution panel also removes the axis cap's only real cost, since the
    reliability points all lie inside the frame by construction.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rows, plotted, calibrated, reliability = [], [], {}, {}
    for name in MODELS:
        # The persisted headline LR is already Platt-wrapped.  Unwrap it here
        # so every model receives exactly one calibration map in this audit.
        model = models[name]
        native_model = model._base if name == "logistic_regression" \
            and hasattr(model, "_base") else model
        raw_val = native_model.predict_proba(X_val)[:, 1]
        raw_test = native_model.predict_proba(X_test)[:, 1]
        calibrator = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        calibrator.fit(raw_val.reshape(-1, 1), y_val)
        prob = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]
        mean_pred, frac_pos, ci_low, ci_high, counts = \
            _quantile_reliability(y_test, prob, n_bins=10)
        plotted.extend([mean_pred, frac_pos, ci_high])
        calibrated[name] = prob
        reliability[name] = (mean_pred, frac_pos, ci_low, ci_high)
        ax.errorbar(
            mean_pred, frac_pos,
            yerr=np.vstack([
                np.maximum(0.0, frac_pos - ci_low),
                np.maximum(0.0, ci_high - frac_pos),
            ]),
            fmt=MODEL_MARKERS[name], ls=MODEL_LINESTYLES[name], ms=3.5,
            lw=1.35, elinewidth=0.75, capsize=1.8,
            color=COLORS[name], label=DISPLAY[name], alpha=0.95,
        )
        ece = float(np.average(np.abs(frac_pos - mean_pred), weights=counts))
        rows.append({
            "model": DISPLAY[name],
            "brier_raw": round(float(brier_score_loss(y_test, raw_test)), 5),
            "brier_platt": round(float(brier_score_loss(y_test, prob)), 5),
            "mean_probability_platt": round(float(prob.mean()), 5),
            "ece_platt_10bin": round(ece, 5),
        })
    # The limit is set by the RELIABILITY points, not by the maximum calibrated
    # score. Spanning the full score range put the axis at ~0.35 while the
    # highest bin sits near 0.19, so roughly two thirds of the panel was empty
    # and the curves were compressed into one corner. With the distribution
    # panel gone nothing is clipped: every plotted point lies inside the frame.
    upper = 1.05 * max(np.max(v) for v in plotted)
    ax.plot([0, upper], [0, upper], color="#555555", ls="--", lw=1.0,
            label="Perfect calibration")
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Calibrated predicted probability")
    ax.set_ylabel("Observed distress frequency")
    ax.set_title("Calibration (Test Sample)")
    # Every model lies below the diagonal, so the upper-left triangle is empty
    # and takes the legend without covering any data. Kept compact so it does
    # not dominate the panel.
    ax.legend(loc="upper left", fontsize=9.5, borderpad=0.5,
              labelspacing=0.35, handlelength=2.4)

    _save_fig(fig, RUN_FIGURES_MODEL / "calibration_4models")
    if not write_table:
        return
    save_table(
        pd.DataFrame(rows), RUN_TABLES_MODEL / "calibration_comparison_4models",
        caption=("Calibration comparison for all four primary models. A "
                 "single two-parameter Platt (sigmoid) mapping is fitted exclusively on "
                 "each model's native validation predictions and applied once "
                 "to the test predictions. The persisted headline logistic "
                 "regression is unwrapped for this like-for-like comparison. "
                 "ECE uses 10 equal-count test bins; discrimination metrics "
                 "continue to use the original ranking scores."),
        label="tab:final_calibration_comparison",
    )


def _save_lev_roa_heatmap(test: pd.DataFrame, prob_xgb: np.ndarray,
                          write_table: bool = True) -> None:
    """Save the raw-score heatmap and additive-interaction contrast."""
    d = pd.DataFrame({
        "TLTA": test["TLTA"].values,
        "NITA": test["NITA"].values,
        "score": prob_xgb,
    })
    d["lev_ter"] = pd.qcut(d["TLTA"], 3, labels=["Low", "Mid", "High"])
    d["roa_ter"] = pd.qcut(d["NITA"], 3, labels=["Low", "Mid", "High"])
    grid = d.pivot_table(index="lev_ter", columns="roa_ter", values="score",
                         aggfunc="mean", observed=False)
    grid = grid.reindex(index=["High", "Mid", "Low"],
                        columns=["Low", "Mid", "High"])

    fig, ax = plt.subplots(figsize=(5.7, 4.25))
    im = ax.imshow(grid.values, cmap="YlOrRd")
    ax.set_xticks(range(3), list(grid.columns))
    ax.set_yticks(range(3), list(grid.index))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{grid.values[i, j]:.3f}", ha="center",
                    va="center",
                    color="white" if grid.values[i, j] > grid.values.max()/2
                    else "black")
    ax.set_xlabel("Profitability (NITA) tertile")
    ax.set_ylabel("Leverage (TLTA) tertile")
    ax.set_title("Leverage × Profitability Interaction", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean raw score")
    _save_fig(fig, RUN_FIGURES_SHAP / "lev_roa_heatmap")

    grand = float(grid.to_numpy().mean())
    contrast = grid.sub(grid.mean(axis=1), axis=0) \
                   .sub(grid.mean(axis=0), axis=1) + grand
    if not write_table:
        return
    save_table(
        contrast.reset_index(),
        RUN_TABLES_SHAP / "lev_roa_interaction_contrast",
        caption=(
            "Leverage $\\times$ profitability interaction contrast (XGBoost, "
            "primary specification): observed cell mean minus the additive "
            "expectation (row margin $+$ column margin $-$ grand mean). A "
            "positive value in the high-leverage / low-profitability corner "
            "is descriptively consistent with non-additivity (H$_4$). The "
            "contrast is not a ceteris-paribus interaction estimator."),
        label="tab:v2_lev_roa_contrast",
    )
    print("  Interaction contrast (observed - additive):")
    print(contrast.round(4).to_string())


def _shap_explanation(model, model_name: str, X: np.ndarray,
                      features: list[str]):
    """Return an exact positive-class SHAP Explanation in model-score units."""
    if model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1] if values.shape[-1] == 2 else values[1]
        expected = np.asarray(explainer.expected_value)
        base = float(expected.ravel()[1] if expected.size == 2 else expected.ravel()[0])
    elif model_name == "logistic_regression":
        pipeline = model._base if hasattr(model, "_base") else model
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["clf"]
        X_scaled = scaler.transform(X)
        explainer = shap.LinearExplainer(clf, X_scaled)
        values = explainer.shap_values(X_scaled)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        expected = np.asarray(explainer.expected_value)
        base = float(expected.ravel()[-1])
    else:
        raise ValueError(f"No validated exact-SHAP route for {model_name!r}")
    if values.shape != X.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {X.shape}")
    return shap.Explanation(
        values=values,
        base_values=np.repeat(base, len(X)),
        data=X,
        feature_names=features,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--presentation-only", action="store_true",
        help="Regenerate the calibration and leverage-profitability figures "
             "without recomputing SHAP or LIME outputs.",
    )
    args = parser.parse_args()
    if ".venv" not in sys.prefix.lower():
        print(f"  WARNING: not running under the project .venv ({sys.prefix})")
    for d in (RUN_FIGURES_SHAP, RUN_FIGURES_MODEL,
              RUN_FIGURES_DESCRIPTIVE, RUN_TABLES_SHAP,
              RUN_TABLES_DESCRIPTIVE, RUN_TABLES_MODEL):
        d.mkdir(parents=True, exist_ok=True)

    features = list(V2_PROFILE["feature_set"])
    saved_dir = OUT_MODELS_SAVED / V2_PROFILE["spec"]
    configs_dir = OUT_MODELS_CONFIGS / V2_PROFILE["spec"]

    print("Loading v2 splits + models ...")
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    X_test = test[features].astype(float).values
    y_test = test["distress"].astype(int).values
    X_val = val[features].astype(float).values
    y_val = val["distress"].astype(int).values

    models = {n: joblib.load(saved_dir / f"{n}.joblib") for n in MODELS}
    xgb = models["xgboost"]
    prob_xgb = xgb.predict_proba(X_test)[:, 1]

    scores = {
        name: average_precision_score(
            y_test, model.predict_proba(X_test)[:, 1]
        )
        for name, model in models.items()
    }
    print("  Test PR-AUC by model: "
          + ", ".join(f"{n}={s:.4f}" for n, s in scores.items()))

    if args.presentation_only:
        _save_calibration_figure(
            models, X_val, y_val, X_test, y_test, write_table=False
        )
        _save_lev_roa_heatmap(test, prob_xgb, write_table=False)
        return

    global shap
    import shap
    import yaml

    # XAI decision (final_primary co-lead framing): all SHAP/LIME
    # explanations are reported on XGBoost — the co-leading ML model on which
    # SHAP is non-circular and the H4 non-linearities are testable. LR (the
    # nominal PR-AUC leader) is read through its standardised coefficients as
    # the linear benchmark, not via SHAP. The `best_*` names below therefore
    # denote the *explanation* model (XGBoost), not the top-PR-AUC model.
    best_name = "xgboost"
    best_model = models[best_name]
    best_label = DISPLAY[best_name]
    prob_best = best_model.predict_proba(X_test)[:, 1]

    # ── SHAP Explanation for XGBoost (used by steps 1-4) ───────────────────
    print(f"Computing exact SHAP explanations for {best_label} ...")
    ex_xgb = _shap_explanation(xgb, "xgboost", X_test, features)
    sv_xgb = np.asarray(ex_xgb.values)
    ex_best = ex_xgb
    sv_best = np.asarray(ex_best.values)

    # ── 1. Dependence plots: empirical top-5 ∪ the pre-registered H4 trio ──
    print("\n[1/5] SHAP dependence plots (top-5 XGBoost features ∪ H4 trio)")
    from src.explainability.shap_nonlinear import select_dependence_features
    dep_feats = select_dependence_features(sv_xgb, features, k=5)
    print(f"  Dependence features: {dep_feats}")
    # shap.dependence_plot sets no title of its own, so these were the only
    # thesis figures left unlabelled. They get the same short bold label as
    # every other exhibit.
    #
    # Four of the six are typeset FOUR-UP at 0.47\textwidth in the appendix,
    # i.e. 2.87 in on the page against a 6.4 in canvas — a reduction to 45%,
    # which would print the 11.5 pt base font at about 5 pt. Those four get the
    # small-multiple type scaling for the same reason the waterfalls do. TLTA
    # and SIGMA are shown singly at 0.72\textwidth and need no scaling.
    four_up = {"RSIZE", "PRICE", "NITA", "OCF_TA"}
    for feat in dep_feats:
        rc = SMALL_MULTIPLE_RC if feat in four_up else {}
        with matplotlib.rc_context(rc):
            plt.figure(figsize=(6.4, 4.35))
            shap.dependence_plot(features.index(feat), sv_xgb, X_test,
                                 feature_names=features, show=False)
            plt.gca().set_title(f"SHAP Dependence — {feat}")
            _save_fig(plt.gcf(), RUN_FIGURES_SHAP / f"shap_dependence_{feat}")

    # ── 2. LEV x ROA interaction heatmap ────────────────────────────────────
    print("\n[2/5] Leverage x Profitability heatmap (raw XGB score)")
    _save_lev_roa_heatmap(test, prob_xgb)

    # ── 3. Waterfalls for TP / FN / TN ──────────────────────────────────────
    print("\n[3/5] Local SHAP waterfalls (TP / FN / TN)")
    cfg = yaml.safe_load(
        (configs_dir / f"{best_name}_config.yaml").read_text(encoding="utf-8")
    )
    threshold = float(cfg["threshold"])
    ev_idx = np.where(y_test == 1)[0]
    tp_idx = np.where((y_test == 1) & (prob_best >= threshold))[0]
    fn_idx = np.where((y_test == 1) & (prob_best < threshold))[0]
    tn_idx = np.where((y_test == 0) & (prob_best < threshold))[0]
    if not len(tp_idx) or not len(fn_idx) or not len(tn_idx):
        raise RuntimeError(
            f"Cannot select exact TP/FN/TN cases for {best_name} at the "
            f"validation-locked threshold {threshold:.6f}."
        )
    picks = {
        "tp": tp_idx[np.argmax(prob_best[tp_idx])],
        "fn": fn_idx[np.argmin(prob_best[fn_idx])],
        "tn": tn_idx[np.argmin(prob_best[tn_idx])],
    }
    # The three waterfall/LIME panels are typeset three-up under a single
    # caption, so each keeps a short label. It is spelled out rather than
    # abbreviated: "TP case" reads as a script's internal tag, not as a thesis
    # exhibit. The same spelled-out label is used for the ``case`` column of
    # the two generated tables, so figure and table agree and neither reprints
    # the tag. It stays machine-readable, and no code keys on the abbreviation
    # (both verifiers only assert the row count).
    CASE_LABELS = {
        "tp": "True positive",
        "fn": "False negative",
        "tn": "True negative",
    }
    firm_rows = []
    for tag, idx in picks.items():
        r = test.iloc[idx]
        firm_rows.append({"case": CASE_LABELS[tag], "model": best_label,
                          "gvkey": r["gvkey"],
                          "fyear": int(r["fyear"]),
                          "predicted_probability": round(float(prob_best[idx]), 4),
                          "validation_threshold": round(threshold, 4),
                          "distress": int(y_test[idx])})
        with matplotlib.rc_context(SMALL_MULTIPLE_RC):
            plt.figure(figsize=(7.0, 4.9))
            shap.plots.waterfall(ex_best[int(idx)], max_display=12, show=False)
            plt.gca().set_title(f"{CASE_LABELS[tag]} — {best_label}")
            _save_fig(plt.gcf(), RUN_FIGURES_SHAP / f"shap_waterfall_{tag}")
    local_cases = pd.DataFrame(firm_rows)
    save_table(
        local_cases,
        RUN_TABLES_SHAP / "local_explanation_cases",
        caption=("Deterministically selected held-out test cases used for the "
                 "paired SHAP and LIME local explanations. The classification "
                 "threshold was selected on validation data and then locked."),
        label="tab:local_explanation_cases",
        column_labels={
            "case": "Case", "model": "Model", "gvkey": "GVKEY",
            "fyear": "Fiscal year", "predicted_probability": "Score",
            "validation_threshold": "Threshold", "distress": "Distress",
        },
        column_format="ll" + "r" * 5,
    )
    print(local_cases.to_string(index=False))

    # ── 4. LIME for the same firms + SHAP-LIME agreement ────────────────────
    print("\n[4/5] LIME local explanations (binary indicators categorical)")
    from lime.lime_tabular import LimeTabularExplainer
    X_train = train[features].astype(float).values
    cat_idx = [features.index(f) for f in BINARY_FEATURES if f in features]
    lime_exp = LimeTabularExplainer(
        X_train, feature_names=features, class_names=["healthy", "distress"],
        categorical_features=cat_idx, discretize_continuous=True,
        random_state=RANDOM_SEED, mode="classification",
    )
    agreement_rows = []
    for tag, idx in picks.items():
        e = lime_exp.explain_instance(X_test[int(idx)], best_model.predict_proba,
                                      num_features=10, num_samples=5000)
        # Hand-drawn rather than lime's `as_pyplot_figure`, whose saturated
        # green/red default sits outside the thesis palette and is the one
        # pairing deuteranopes cannot separate. Signed colour carries the
        # direction, so the sign is also readable in grayscale from the axis.
        pairs = e.as_list(label=1)                 # [(rule, weight)], |w| desc
        rules = [_typeset_rule(p[0]) for p in pairs][::-1]  # small at top of barh
        weights = [p[1] for p in pairs][::-1]
        with matplotlib.rc_context(SMALL_MULTIPLE_RC):
            fig, ax = plt.subplots(figsize=(7.0, 4.6))
            ax.barh(
                range(len(weights)), weights, height=0.66,
                color=[COLORS["xgboost"] if w > 0
                       else COLORS["logistic_regression"] for w in weights],
            )
            ax.set_yticks(range(len(weights)))
            ax.set_yticklabels(rules)
            ax.axvline(0.0, color="black", lw=0.8)
            # Kept to the width of the label this replaced (4.87 in vs 4.88 in
            # at the 2.2x small-multiple scale). The axes sit right of centre
            # because the rule tick labels are wide, so a longer label runs off
            # the saved bbox: "(positive raises predicted risk)" printed flush
            # against the right edge with the closing bracket cut off.
            ax.set_xlabel("LIME local weight (positive raises risk)")
            ax.set_title(f"{CASE_LABELS[tag]} — {best_label}")
            _save_fig(fig, RUN_FIGURES_SHAP / f"lime_{tag}")

        lime_map = dict(e.as_map()[1])           # {feature_idx: weight}
        shap_row = sv_best[int(idx)]
        common = sorted(lime_map.keys())
        lime_w = np.array([lime_map[j] for j in common])
        shap_w = np.array([shap_row[j] for j in common])
        tau, _ = kendalltau(np.abs(lime_w), np.abs(shap_w))
        sign_agree = float((np.sign(lime_w) == np.sign(shap_w)).mean())
        agreement_rows.append({
            "case": CASE_LABELS[tag], "model": best_label,
            "n_lime_features": len(common),
            "kendall_tau_abs_rank": round(float(tau), 3),
            "sign_agreement": round(sign_agree, 3),
            "top_lime_feature": features[common[int(np.argmax(np.abs(lime_w)))]],
            "top_shap_feature": features[int(np.argmax(np.abs(shap_row)))],
        })
    agree = pd.DataFrame(agreement_rows)
    save_table(
        agree, RUN_TABLES_SHAP / "lime_shap_agreement",
        caption=(
            "SHAP--LIME agreement for the three local-explanation firms "
            f"({best_label}, the co-leading ML model used for the SHAP/LIME "
            "analysis). LIME uses 5{,}000 "
            "perturbation samples with the binary indicators (OENEG, INTWO, "
            "MB\\_MISSING) treated as categorical. Kendall's $\\tau$ compares "
            "absolute-importance ranks over LIME's top-10 features; sign "
            "agreement is the share of those features with equal attribution "
            "direction."),
        label="tab:v2_lime_shap_agreement",
        # LaTeX headers only; the CSV keeps the machine-readable names. The
        # printed table had been showing the raw snake_case column names.
        column_labels={
            "case": "Case", "model": "Model",
            "n_lime_features": "LIME features",
            "kendall_tau_abs_rank": "Kendall $\\tau$",
            "sign_agreement": "Sign agreement",
            "top_lime_feature": "Top LIME feature",
            "top_shap_feature": "Top SHAP feature",
        },
        column_format="ll" + "r" * 5,
    )
    print(agree.to_string(index=False))

    # ── 5a. Calibration figure (4 models) ───────────────────────────────────
    print("\n[5/5] Calibration figure + v2 descriptive outputs")
    _save_calibration_figure(models, X_val, y_val, X_test, y_test)

    # ── 5b. Descriptive: distress-rate time series + composition table ─────
    panel = pd.concat([train, val, test], ignore_index=True)
    n_panel = len(panel)
    comp = (panel.groupby("fyear")
            .agg(n_obs=("distress", "size"), n_distress=("distress", "sum"))
            .reset_index())
    comp["distress_rate_pct"] = (100 * comp["n_distress"] / comp["n_obs"]).round(3)
    save_table(
        comp, RUN_TABLES_DESCRIPTIVE / "sample_composition_by_year",
        caption=(
            "Sample composition by fiscal year under the primary "
            f"specification ({n_panel:,} modeling firm-years, FY1990--2023; "
            "corrected delisting label; date-ranged NYSE/AMEX/NASDAQ "
            "universe). FY2009 and FY2014 are excluded as purged boundary "
            "years to ensure outcome-label maturity between chronological "
            "splits."),
        label="tab:sample_annual",
    )
    _save_distress_rate_figure(comp)

    print("\nV2 XAI + figures complete.")


if __name__ == "__main__":
    main()
