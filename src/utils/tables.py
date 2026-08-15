"""
Table utilities — save results as LaTeX and CSV.
=================================================
All output tables are saved in two formats:
  - .tex  : for direct inclusion in the LaTeX thesis
  - .csv  : for inspection and further analysis in Excel/Python

LaTeX tables follow academic conventions:
  - \\booktabs formatting (\\toprule, \\midrule, \\bottomrule)
  - Caption and label placeholders included
  - No vertical lines (standard in academic finance tables)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(
    df: pd.DataFrame,
    path_stem: Path,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
    index: bool = False,
    column_labels: dict[str, str] | None = None,
    column_format: str | None = None,
) -> None:
    """
    Save a DataFrame as both LaTeX (.tex) and CSV (.csv).

    Parameters
    ----------
    df : pd.DataFrame
    path_stem : Path
        File path WITHOUT extension.
    caption : str
        LaTeX caption text (can be filled in later).
    label : str
        LaTeX label for \\ref{} (e.g., 'tab:model_performance').
    float_format : str
        Format string for floating-point values in LaTeX.
    index : bool
        Whether to include the DataFrame index in the output.
    column_labels : dict[str, str] | None
        Optional ``{dataframe_column: printed_header}`` map applied to the
        LaTeX output only; the CSV keeps the machine-readable names so
        downstream analysis is unaffected. Headers must be plain text — no
        math and no macros — because they are written with ``escape=False``.
    column_format : str | None
        Optional LaTeX column specification (e.g. ``"llrrrrr"``). Defaults to
        one ``l`` followed by ``r`` for every remaining column.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = path_stem.with_suffix(".csv")
    df.to_csv(csv_path, index=index)
    print(f"  Table saved -> {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────
    tex_path = path_stem.with_suffix(".tex")
    labels = column_labels or {}
    df_tex = df.rename(
        columns=lambda c: str(labels.get(c, c)).replace("_", r"\_")
    )
    df_tex = df_tex.apply(
        lambda col: col.map(lambda v: v.replace("_", r"\_") if isinstance(v, str) else v)
    )
    latex_str = df_tex.to_latex(
        index=index,
        float_format=float_format,
        escape=False,
        na_rep="—",
        column_format=(
            column_format
            or "l" + "r" * (len(df.columns) - (0 if index else 1))
        ),
    )

    # Wrap with booktabs-compatible environment
    header = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
    )
    footer = "\\end{table}\n"

    # Replace \toprule / \midrule / \bottomrule (pandas uses \hline by default)
    latex_str = (
        latex_str
        .replace("\\begin{tabular}", "\\begin{tabular}")
        .replace("\\hline", "")
        .replace("\\toprule",    "\\toprule")
        .replace("\\midrule",    "\\midrule")
        .replace("\\bottomrule", "\\bottomrule")
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(header + latex_str + footer)
    print(f"  Table saved -> {tex_path}")


def format_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply display formatting to a model performance table.

    Rounds metrics to 4 decimal places; adds prevalence baseline
    as a reference row at the top.

    Parameters
    ----------
    df : pd.DataFrame
        Output of src.models.evaluate.compute_all_metrics() for multiple models.

    Returns
    -------
    pd.DataFrame
        Formatted for LaTeX output.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    out = df.copy()
    out[numeric_cols] = out[numeric_cols].round(4)

    # Prevalence baseline row: PR-AUC equals the prevalence rate;
    # all discrimination metrics are at their random-classifier values.
    if "prevalence_baseline_pr_auc" in out.columns:
        prevalence = float(out["prevalence_baseline_pr_auc"].iloc[0])
    else:
        prevalence = float("nan")

    baseline_row: dict = {col: "" for col in out.columns}
    baseline_row["model"]                    = "No-skill baseline"
    baseline_row["pr_auc"]                   = round(prevalence, 4)
    baseline_row["roc_auc"]                  = 0.5
    baseline_row["precision"]                = round(prevalence, 4)
    baseline_row["recall"]                   = 1.0
    baseline_row["f1"]                       = round(
        2 * prevalence / (prevalence + 1) if prevalence > 0 else 0.0, 4
    )
    baseline_row["ks_stat"]                  = 0.0
    baseline_row["brier_score"]              = round(prevalence * (1 - prevalence), 4)
    baseline_row["prevalence_baseline_pr_auc"] = round(prevalence, 4)

    baseline_df = pd.DataFrame([baseline_row])
    out = pd.concat([baseline_df, out], ignore_index=True)

    return out
