"""
Descriptive statistics — 4 required outputs for Blueprint v4 §8.
=================================================================
Produces all descriptive statistics and plots required by the blueprint:

  1. Sample composition table
     - Annual counts: N, N_distressed, distress_rate_pct (1990–2024)
     - Must show spikes in 2001–02, 2008–09, 2020 as sanity check
     - By SIC major division (for cross-industry coverage check)

  2. Predictor summary statistics
     - mean, median, std, p10, p90 for all 17 predictors
     - Separately for distressed vs non-distressed
     - Welch t-statistic and Wilcoxon z-statistic for group differences

  3. Pairwise correlation matrix
     - Pearson correlations for all 17 predictors
     - Flags pairs with |r| > 0.70 (relevant for SHAP interpretation)

  4. Distress rate time-series plot
     - Annual distress rate 1990–2024
     - Shaded bands for train / val / test periods
     - Vertical lines for crisis years (2001, 2008, 2020)

Sanity checks
-------------
These outputs serve as data quality checks BEFORE any model is estimated:
  - Distress spikes in crisis years confirm the label is well-constructed
  - Between-group differences confirm predictors carry economic signal
  - High correlations (NITA/OCF_TA; TLTA/LNMK) are expected and must be
    acknowledged in the SHAP interpretation (conservatively interpret pairs)

Blueprint v4 reference: §8.1–8.4

Run
---
    python -m src.analysis.descriptive
"""

from __future__ import annotations

import re
import pandas as pd

from src.config import (
    ALL_FEATURES,
    OUT_TABLES_DESCRIPTIVE,
    OUT_FIGURES_DESCRIPTIVE,
    DATA_SAMPLES,
    DATA_FEATURES,
    DATA_MERGED,
)
from src.utils.metrics import (
    annual_distress_rate,
    summary_stats_by_group,
    correlation_matrix,
)
from src.utils.plots import (
    plot_distress_rate_timeseries,
    set_academic_style,
    save_figure,
)
from src.utils.tables import save_table

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# §4.1  Sample attrition table  (Blueprint v4 §4.1)
# ---------------------------------------------------------------------------

def run_sample_attrition() -> None:
    """
    Produce Table 4.1: Sample attrition at each construction step.

    Reads the attrition log written by merge_crsp_compustat.py and
    formats it as an academic-quality table with step labels, remaining
    firm-years, unique firms, and the count dropped at each step.

    Required by Blueprint v4 §4.1 (Universe and exclusions).
    """
    print("\n--- §4.1  Sample attrition table ---")
    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)

    attrition_path = DATA_MERGED / "attrition_table.csv"
    if not attrition_path.exists():
        print(f"  WARNING:attrition_table.csv not found at {attrition_path} — "
              "re-run src.data.merge_crsp_compustat to generate it.")
        return

    raw = pd.read_csv(attrition_path)

    # ── Extend the merge-layer log with the two post-merge construction steps ──
    # The AttritionTracker in merge_crsp_compustat.py stops at the lenient
    # >=6-of-11 raw-item pre-screen. Two binding steps are applied afterwards and
    # are NOT captured by that tracker:
    #   (a) the strict >=8-of-11 accounting *features* filter (post-winsorisation,
    #       in build_features.py — the Blueprint v4 §6.4 design rule);
    #   (b) the chronological train/val/test split, which drops firm-years that
    #       cannot form a valid split record.
    # Both are appended here, computed dynamically from the saved parquets so the
    # table always reconciles to the actual modeling sample (no hard-coded N).
    raw = raw[raw["step"].astype(str).str.strip() != "FINAL analysis sample"].copy()

    # Relabel the merge-layer accounting pre-screen to flag it as lenient.
    _pre = raw["step"].astype(str).str.contains("of-11 acctg", regex=False)
    raw.loc[_pre, "step"] = "12  ≥6-of-11 raw accounting items (pre-screen)"

    feat_path = DATA_FEATURES / "features_all.parquet"
    if feat_path.exists():
        _feat = pd.read_parquet(feat_path, columns=["gvkey"])
        raw = pd.concat([raw, pd.DataFrame([{
            "step": "13  ≥8-of-11 accounting features (binding)",
            "firm_years": len(_feat),
            "unique_firms": int(_feat["gvkey"].nunique()),
        }])], ignore_index=True)

    _splits = []
    for _name in ("train", "val", "test"):
        _p = DATA_SAMPLES / f"{_name}.parquet"
        if _p.exists():
            _splits.append(pd.read_parquet(_p, columns=["gvkey"]))
    if _splits:
        _model = pd.concat(_splits, ignore_index=True)
        n_model, firms_model = len(_model), int(_model["gvkey"].nunique())
        raw = pd.concat([raw, pd.DataFrame([{
            "step": "14  Valid record for chronological split",
            "firm_years": n_model, "unique_firms": firms_model,
        }])], ignore_index=True)
    else:
        n_model = int(raw["firm_years"].iloc[-1])
        firms_model = int(raw["unique_firms"].iloc[-1])

    # Corrected FINAL = the actual modeling sample (sum of the three splits).
    raw = pd.concat([raw, pd.DataFrame([{
        "step": "FINAL analysis sample",
        "firm_years": n_model, "unique_firms": firms_model,
    }])], ignore_index=True)

    # Clean step labels: strip leading "NN  " prefix and LaTeX-escape symbols.
    def _clean_label(s: str) -> str:
        s = re.sub(r"^\d+\s+", "", str(s))       # remove "01  " prefix
        s = s.replace("at <= 0", r"$AT \leq 0$")
        s = s.replace("≥", r"$\geq$")
        return s

    raw["step_label"] = raw["step"].apply(_clean_label)

    # Add "dropped at this step" column (difference from previous row)
    raw["dropped"] = raw["firm_years"].shift(1) - raw["firm_years"]
    raw["dropped"] = raw["dropped"].fillna(0).astype(int)

    # Build formatted table for thesis
    attrition_df = pd.DataFrame({
        "Step":                  raw["step_label"],
        "Firm-years remaining":  raw["firm_years"].map("{:,}".format),
        "Unique firms":          raw["unique_firms"].map("{:,}".format),
        "Dropped (firm-years)":  raw["dropped"].map(lambda x: f"({x:,})" if x > 0 else "—"),
    })

    save_table(
        attrition_df,
        OUT_TABLES_DESCRIPTIVE / "sample_attrition",
        caption=(
            "Sample construction and attrition. Starting from the raw Compustat "
            "universe (1990--2024), firm-years are sequentially dropped according "
            "to the exclusion criteria defined in §4.1. The final analysis sample "
            f"covers \\num{{{n_model}}} firm-years across \\num{{{firms_model}}} "
            "unique firms."
        ),
        label="tab:sample_attrition",
    )
    print(f"  Sample attrition table saved ({len(attrition_df)} steps).")
    print(attrition_df.to_string(index=False))


# ---------------------------------------------------------------------------
# §8.1  Sample composition table
# ---------------------------------------------------------------------------

def run_sample_composition(panel: pd.DataFrame) -> None:
    """
    Produce Table 4.1: Annual sample composition 1990–2024.

    Columns: fyear, n_total, n_distressed, distress_rate_pct.
    Sanity check: rates should spike in 2001–02, 2008–09, and 2020.
    Also produces a by-SIC-division breakdown (secondary table).
    """
    print("\n--- §8.1  Sample composition ---")
    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)

    # ── Annual by fiscal year ─────────────────────────────────────────────
    annual = annual_distress_rate(panel)
    save_table(
        annual,
        OUT_TABLES_DESCRIPTIVE / "sample_composition_annual",
        caption=(
            "Annual sample composition for the corporate financial distress "
            "prediction study: firm-years, distressed firms, "
            "and distress rate (\\%), 1990--2024. "
            "Distress is defined as CRSP delisting codes 400--499 "
            "within a 12-month window following the 10-K filing date."
        ),
        label="tab:sample_annual",
    )

    # Sanity check printout
    spikes = annual[annual["fyear"].isin([2001, 2002, 2008, 2009, 2020])]
    print("  Crisis-year distress rates (sanity check):")
    print(spikes[["fyear", "n_distressed", "distress_rate_pct"]].to_string(index=False))

    # ── By SIC major division ─────────────────────────────────────────────
    # Prefer the harmonised `_sic` column built in apply_compustat_filters
    # (historical SIC from companyhistory.HSIC with sich/companydescription
    # fallbacks). Fall back to `sic` or `sich` if `_sic` was not preserved.
    sic_source = None
    for cand in ("_sic", "sic", "sich"):
        if cand in panel.columns:
            sic_source = panel[cand]
            break
    if sic_source is not None:
        panel = panel.copy()
        panel["sic_div"] = _sic_division(sic_source)
        sic_table = (
            panel.groupby("sic_div", observed=False)
                 .agg(
                     n_total      =("distress", "count"),
                     n_distressed =("distress", "sum"),
                 )
                 .reset_index()
        )
        sic_table["distress_rate_pct"] = (
            sic_table["n_distressed"] / sic_table["n_total"] * 100
        ).round(3)
        save_table(
            sic_table,
            OUT_TABLES_DESCRIPTIVE / "sample_composition_by_sic",
            caption=(
                "Sample composition by SIC major industry division: "
                "firm-years, distressed observations, and distress rate (\\%) "
                "for the corporate financial distress prediction study (1990--2024). "
                "Financial firms (SIC 6000--6999) and utilities (SIC 4900--4999) "
                "are excluded from the primary sample."
            ),
            label="tab:sample_sic",
        )

    # ── Time-series plot ──────────────────────────────────────────────────
    OUT_FIGURES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)
    plot_distress_rate_timeseries(annual, OUT_FIGURES_DESCRIPTIVE / "distress_rate_timeseries")
    print("  Distress rate time-series plot saved.")


def _sic_division(sic: pd.Series) -> pd.Series:
    """Map SIC codes to major industrial divisions (letter labels)."""
    bins   = [0, 999, 1499, 1999, 3999, 4999, 5199, 5999, 7999, 8999, 9999]
    labels = [
        "A: Agriculture (0–999)",
        "B: Mining (1000–1499)",
        "C: Construction (1500–1999)",
        "D: Manufacturing (2000–3999)",
        "E: Transp/Comm (4000–4999, util. excluded)",
        "F: Wholesale (5000–5199)",
        "G: Retail (5200–5999)",
        "H: Services I (7000–7999)",
        "I: Services II (8000–8999)",
        "J: Other (9000–9999)",
    ]
    return pd.cut(
        pd.to_numeric(sic, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )


# ---------------------------------------------------------------------------
# §8.2  Predictor summary statistics
# ---------------------------------------------------------------------------

def run_predictor_summary(panel: pd.DataFrame, features: list[str] = ALL_FEATURES) -> None:
    """
    Produce Table 4.2: Summary statistics for all 17 predictors by group.

    For each predictor: mean, median, std, p10, p90 for distressed and
    non-distressed firms, plus Welch t-stat and Wilcoxon z-stat.
    """
    print("\n--- §8.2  Predictor summary statistics ---")
    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)

    stats_df = summary_stats_by_group(panel, features, group_col="distress")

    save_table(
        stats_df,
        OUT_TABLES_DESCRIPTIVE / "predictor_summary_stats",
        caption=(
            "Descriptive statistics for the 17 predictors, separately for "
            "non-distressed (D=0) and distressed (D=1) firm-years. "
            "Welch \\textit{t}-statistic and Wilcoxon \\textit{z}-statistic "
            "test the null hypothesis of equal group means / distributions. "
            "Stars: * $p<0.05$, ** $p<0.01$, *** $p<0.001$."
        ),
        label="tab:predictor_stats",
    )

    # Print features with non-significant group difference (potential concern)
    insig = stats_df[stats_df["t_pvalue"] > 0.10]
    if len(insig) > 0:
        print(f"  WARNING:{len(insig)} features with t-test p > 0.10 "
              f"(weak group separation): {insig['feature'].tolist()}")
    else:
        print(f"  All {len(stats_df)} features show significant group differences (p < 0.10).")


# ---------------------------------------------------------------------------
# §8.3  Pairwise correlation matrix
# ---------------------------------------------------------------------------

def _save_correlation_matrix_tex(corr_mat: pd.DataFrame) -> None:
    """
    Save the full 17×17 Pearson correlation matrix as a landscape LaTeX table.

    Uses \\sidewaystable (rotating package) and \\tiny font so all 17 columns
    fit on one page. Intended for the thesis appendix.
    """
    tex_path = OUT_TABLES_DESCRIPTIVE / "correlation_matrix.tex"

    # Build tabular body: index column + 17 feature columns
    features = list(corr_mat.columns)
    col_format = "l" + "r" * len(features)

    # Header row
    header = " & ".join([""] + features) + " \\\\"

    # Data rows
    body_lines = []
    for feat in features:
        row_vals = [f"{corr_mat.loc[feat, col]:.2f}" for col in features]
        body_lines.append(feat + " & " + " & ".join(row_vals) + " \\\\")

    body = "\n".join(body_lines)

    tex = (
        "\\begin{sidewaystable}[htbp]\n"
        "\\centering\n"
        "\\caption{Pairwise Pearson correlation matrix for all 17 predictors "
        "(full sample, 1990--2024). Pairs with $|r| > 0.70$ are discussed "
        "in Section~\\ref{sec:corr}. Correlations rounded to two decimal places.}\n"
        "\\label{tab:correlation_matrix}\n"
        "\\tiny\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{sidewaystable}\n"
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Table saved -> {tex_path}")


def run_correlation_analysis(
    panel: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> None:
    """
    Produce Table 4.3 and Figure 4.2: Pearson correlation matrix.

    Flags pairs with |r| > 0.70. These pairs must be interpreted
    conservatively in the SHAP analysis (SHAP arbitrarily distributes
    importance between correlated features).
    """
    print("\n--- §8.3  Pairwise correlation matrix ---")
    OUT_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)

    corr_mat, high_corr = correlation_matrix(panel, features, high_corr_threshold=0.70)

    # ── Save full correlation matrix as CSV ───────────────────────────────
    corr_mat.round(4).to_csv(
        OUT_TABLES_DESCRIPTIVE / "correlation_matrix.csv"
    )
    print(f"  Full correlation matrix saved (CSV).")

    # ── Save full correlation matrix as LaTeX (landscape, appendix) ───────
    _save_correlation_matrix_tex(corr_mat.round(2))

    # ── Save high-correlation pairs as LaTeX table ────────────────────────
    if len(high_corr) > 0:
        save_table(
            high_corr,
            OUT_TABLES_DESCRIPTIVE / "high_correlation_pairs",
            caption=(
                "Predictor pairs with Pearson $|r| > 0.70$. "
                "These pairs should be interpreted conservatively in SHAP analysis "
                "as importance may be arbitrarily distributed between them."
            ),
            label="tab:high_corr",
        )
        print(f"  {len(high_corr)} high-correlation pairs (|r| > 0.70):")
        print(high_corr.to_string(index=False))
    else:
        print("  No feature pairs with |r| > 0.70.")

    # ── Heatmap figure ────────────────────────────────────────────────────
    set_academic_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = corr_mat.isna()

    sns.heatmap(
        corr_mat,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.3,
        linecolor="lightgray",
        annot=corr_mat.round(2),
        fmt=".2f",
        annot_kws={"size": 7},
        mask=mask,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title("Pairwise Pearson Correlation Matrix — 17 Predictors: Corporate Financial Distress Prediction (Full Sample, 1990–2024)", fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, OUT_FIGURES_DESCRIPTIVE / "correlation_heatmap")
    print("  Correlation heatmap figure saved.")


# ---------------------------------------------------------------------------
# Full descriptive pipeline
# ---------------------------------------------------------------------------

def run_all_descriptive(
    panel: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
) -> None:
    """
    Run all 4 required descriptive outputs in sequence.

    Parameters
    ----------
    panel : pd.DataFrame
        Full feature panel (all years, all splits).
    features : list[str]
    """
    print("\n" + "="*60)
    print("  DESCRIPTIVE STATISTICS  (Blueprint v4 §8)")
    print("="*60)

    run_sample_attrition()
    run_sample_composition(panel)
    run_predictor_summary(panel, features)
    run_correlation_analysis(panel, features)

    print("\n  All descriptive outputs complete.")
    print(f"  Tables -> {OUT_TABLES_DESCRIPTIVE}")
    print(f"  Figures -> {OUT_FIGURES_DESCRIPTIVE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    panel = pd.concat(
        [pd.read_parquet(DATA_SAMPLES / f"{s}.parquet") for s in ("train", "val", "test")],
        ignore_index=True,
    )
    run_all_descriptive(panel)
