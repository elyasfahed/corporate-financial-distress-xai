"""
Filing-date matched-vs-unmatched selection analysis — audit item 3.
====================================================================
The design drops, rather than imputes, firm-years without a real 10-K filing
date (``attach_filing_dates(drop_missing_fdates=True)``, faithful to Blueprint
v4 §5.2). On the reported sample this removes a large minority of firm-years, so
the honest question is not *how many* were dropped but **whether the dropped
firm-years differ systematically from the retained ones**.

``src/analysis/filing_date_qc.py`` answers a different question — which *source*
the filing date should come from (`perioddescriptorannual.FDATE` vs the
dedicated `filingdates.rds`). It contains no matched-vs-unmatched comparison,
and its outputs currently live under ``outputs/_superseded/``. This module
supplies the missing selection evidence on the reported construction.

Three outputs
-------------
1. ``filing_date_selection_overview``  — attrition and match rate overall.
2. ``filing_date_selection_by_year``   — match rate by fiscal year (is the
   selection stable, or does it trend?).
3. ``filing_date_selection_covariates``— matched vs unmatched on observable
   accounting characteristics, with standardised differences.

Standardised difference is reported instead of a t-statistic because with tens
of thousands of firm-years every trivial difference is "significant"; |d| > 0.10
is the conventional imbalance threshold.

Choice of scale (2026-07-30 correction)
---------------------------------------
Cohen's d on RAW dollar levels is not a usable imbalance diagnostic for Compustat
items. Pooled skewness on these variables runs 17-35, so the pooled SD is
dominated by a handful of mega-caps and the ratio (mean_a - mean_b)/SD is driven
toward zero regardless of how different the two groups are in the body of the
distribution. Measured on this sample the levels-based statistic reports |d| <
0.10 -- "balanced" -- for all eight dollar covariates, while matched firm-years
have ~1.9x the median total assets and ~2.2x the median sales of unmatched ones.

Each covariate is therefore reported on three scales: the raw level (retained for
continuity and comparability with the earlier table), log1p, and a scale-free
pooled-rank version. The rank statistic is the one the imbalance flag is now
based on, because it is invariant to the monotone transform and cannot be
deflated by tail mass. Ratio and loss-indicator covariates are also included: for
a distress-prediction label the relevant question is whether selection operates
on financial weakness, which dollar levels cannot answer.

Run
---
    PYTHONUTF8=1 PYTHONPATH=. .venv/Scripts/python -m src.analysis.filing_date_selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MAX_FILING_LAG_DAYS, OUT_TABLES_DESCRIPTIVE
from src.utils.tables import save_table

#: Observable characteristics available BEFORE the filing-date merge.
_COVARIATES = [
    ("at", "Total assets"),
    ("lt", "Total liabilities"),
    ("sale", "Sales"),
    ("ni", "Net income"),
    ("ib", "Income before extraordinary items"),
    ("che", "Cash and short-term investments"),
    ("act", "Current assets"),
    ("lct", "Current liabilities"),
]

#: Scale-free characteristics that speak directly to financial weakness. Dollar
#: levels cannot answer whether the selection operates on distress risk; these
#: can. Built from the same pre-merge Compustat items, so they introduce no new
#: data dependency. (2026-07-30 addition.)
_DERIVED_COVARIATES = [
    ("_roa",  "Return on assets (ib/at)"),
    ("_lev",  "Leverage (lt/at)"),
    ("_loss", "Loss indicator (ib < 0)"),
]

#: Conventional imbalance threshold on a standardised difference.
IMBALANCE_THRESHOLD = 0.10


def standardised_difference(a: pd.Series, b: pd.Series) -> float:
    """
    Cohen-style standardised difference between two groups.

    Uses the pooled standard deviation. Returns NaN when either group has no
    variation or no observations.

    NOTE ON SCALE: applied to raw Compustat dollar levels this statistic is
    dominated by tail mass and understates imbalance -- see the module docstring.
    Prefer `standardised_difference_rank` for the imbalance verdict.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def standardised_difference_log(a: pd.Series, b: pd.Series) -> float:
    """
    Standardised difference on the log1p scale.

    Compresses the right tail that deflates the levels-based statistic. Values
    are clipped at zero first, so this is meaningful only for covariates that
    are non-negative in the main (assets, sales, cash); for signed variables the
    rank version is the appropriate diagnostic.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return standardised_difference(np.log1p(a.clip(lower=0)),
                                   np.log1p(b.clip(lower=0)))


def standardised_difference_rank(a: pd.Series, b: pd.Series) -> float:
    """
    Scale-free standardised difference on pooled percentile ranks.

    The two groups are pooled, ranked jointly (percentile ranks, ties averaged),
    and the standardised difference is taken on the ranks. Invariant to any
    monotone transform of the covariate, so unlike the levels version it cannot
    be driven to zero by a heavy right tail. This is the statistic the imbalance
    flag is based on.
    """
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_ranks = pd.concat([a, b], ignore_index=True).rank(pct=True)
    return standardised_difference(pooled_ranks.iloc[:len(a)],
                                   pooled_ranks.iloc[len(a):])


def add_derived_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the scale-free financial-weakness covariates.

    Returns a copy; division by a non-positive or missing asset base yields NaN
    rather than an infinity, so the standardised differences stay finite.
    """
    out = df.copy()
    ib = pd.to_numeric(out.get("ib"), errors="coerce")
    at = pd.to_numeric(out.get("at"), errors="coerce")
    lt = pd.to_numeric(out.get("lt"), errors="coerce")
    at_pos = at.where(at > 0)
    out["_roa"] = ib / at_pos
    out["_lev"] = lt / at_pos
    out["_loss"] = (ib < 0).astype("float64").where(ib.notna())
    return out


def classify_match(comp: pd.DataFrame, filing: pd.DataFrame) -> pd.DataFrame:
    """
    Label each Compustat firm-year as filing-date matched or unmatched.

    Reproduces the merge in ``attach_filing_dates`` without applying the drop,
    so both groups remain observable.

    Returns
    -------
    pd.DataFrame
        ``comp`` plus boolean 'fdate_matched' and the merged 'fdate'.
    """
    out = comp.drop(columns=["fdate", "fdate_source"], errors="ignore").merge(
        filing[["gvkey", "datadate", "fdate", "fdate_source"]],
        on=["gvkey", "datadate"], how="left",
    )
    src = out["fdate_source"].astype(str)
    unmatched = src.str.startswith("fallback") | out["fdate"].isna()
    out["fdate_matched"] = ~unmatched
    return out


def selection_overview(labelled: pd.DataFrame) -> pd.DataFrame:
    """Attrition, match rate, and the additional filing-lag cap loss."""
    n = len(labelled)
    matched = labelled["fdate_matched"]
    lag = (pd.to_datetime(labelled["fdate"]) - pd.to_datetime(labelled["datadate"])).dt.days
    over_cap = matched & (lag > MAX_FILING_LAG_DAYS)
    rows = [
        ("Compustat firm-years entering the filing-date merge", n, 1.0),
        ("Matched to an actual 10-K filing date", int(matched.sum()),
         float(matched.mean())),
        ("Unmatched — dropped, never imputed", int((~matched).sum()),
         float((~matched).mean())),
        (f"Additionally dropped: filing lag > {MAX_FILING_LAG_DAYS}d",
         int(over_cap.sum()), float(over_cap.mean())),
        ("Retained after both filters",
         int((matched & ~over_cap).sum()), float((matched & ~over_cap).mean())),
    ]
    return pd.DataFrame(rows, columns=["step", "firm_years", "share"])


def selection_by_year(labelled: pd.DataFrame) -> pd.DataFrame:
    """Match rate by fiscal year — detects trend in the selection."""
    g = labelled.groupby("fyear")["fdate_matched"]
    out = pd.DataFrame({
        "fyear": g.size().index,
        "firm_years": g.size().values,
        "matched": g.sum().values.astype(int),
    })
    out["match_rate"] = (out["matched"] / out["firm_years"]).round(4)
    return out.reset_index(drop=True)


def selection_covariates(labelled: pd.DataFrame) -> pd.DataFrame:
    """
    Matched vs unmatched on pre-merge observables, with standardised differences
    on three scales.

    Medians are reported alongside the standardised differences because the raw
    Compustat levels are heavily right-skewed. `std_diff_levels` is retained for
    continuity with the earlier version of this table but is NOT the basis of the
    imbalance verdict: on skewness of 17-35 it is deflated toward zero (see the
    module docstring). `std_diff_rank` is scale-free and carries the flag.

    Ratio and loss covariates are appended because a distress label makes
    selection on financial weakness the material question, and dollar levels
    cannot speak to it.
    """
    labelled = add_derived_covariates(labelled)
    m = labelled[labelled["fdate_matched"]]
    u = labelled[~labelled["fdate_matched"]]
    rows = []
    for col, label in list(_COVARIATES) + list(_DERIVED_COVARIATES):
        if col not in labelled.columns:
            continue
        a = pd.to_numeric(m[col], errors="coerce")
        b = pd.to_numeric(u[col], errors="coerce")
        is_dollar = any(col == c for c, _ in _COVARIATES)
        rows.append({
            "variable": col,
            "description": label,
            "matched_median": a.median(),
            "unmatched_median": b.median(),
            "std_diff_levels": standardised_difference(a, b),
            # log1p is only interpretable for the non-negative dollar items
            "std_diff_log": (standardised_difference_log(a, b)
                             if is_dollar else float("nan")),
            "std_diff_rank": standardised_difference_rank(a, b),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["imbalanced"] = out["std_diff_rank"].abs() > IMBALANCE_THRESHOLD
        for c in ("matched_median", "unmatched_median", "std_diff_levels",
                  "std_diff_log", "std_diff_rank"):
            out[c] = out[c].round(4)
    return out


def _esc(text: object) -> str:
    """Escape the few LaTeX-active characters that occur in these labels."""
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("<", r"$<$"), (">", r"$>$")):
        s = s.replace(a, b)
    return s


def _fmt(v: object, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "---"
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (int, float, np.floating, np.integer)):
        return f"{float(v):,.{nd}f}"
    return _esc(v)


def covariates_table_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Render the matched-vs-unmatched covariate table as **two stacked panels**.

    The single wide eight-column layout this replaces ran past the right text
    margin and clipped its final columns --- including ``std_diff_rank``, the
    column that actually carries the imbalance verdict. Splitting the table by
    content rather than shrinking the type keeps every column visible at the
    document's normal ``\\small`` size:

    * **Panel A** — group medians, with the long description text in a fixed-width
      wrapping column.
    * **Panel B** — the three standardised differences and the flag. The
      description is not repeated; the rows are the same variables in the same
      order.
    """
    a_rows = "\n".join(
        f"{_esc(r.variable)} & {_esc(r.description)} & "
        f"{_fmt(r.matched_median)} & {_fmt(r.unmatched_median)} \\\\"
        for r in df.itertuples()
    )
    b_rows = "\n".join(
        f"{_esc(r.variable)} & {_fmt(r.std_diff_levels)} & {_fmt(r.std_diff_log)} & "
        f"{_fmt(r.std_diff_rank)} & {_fmt(r.imbalanced)} \\\\"
        for r in df.itertuples()
    )
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        "\\begin{tabular}{@{}l p{5.6cm} r r@{}}\n"
        "\\multicolumn{4}{@{}l}{\\textbf{Panel A: group medians}}\\\\[2pt]\n"
        "\\toprule\n"
        "Variable & Description & Matched & Unmatched \\\\\n"
        "\\midrule\n"
        f"{a_rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n\n"
        "\\vspace{1.1em}\n\n"
        "\\begin{tabular}{@{}l r r r c@{}}\n"
        "\\multicolumn{5}{@{}l}{\\textbf{Panel B: standardised differences "
        "(matched $-$ unmatched)}}\\\\[2pt]\n"
        "\\toprule\n"
        "Variable & Levels & $\\log(1+x)$ & Rank & Flagged \\\\\n"
        "\\midrule\n"
        f"{b_rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def run_filing_date_selection(comp: pd.DataFrame, filing: pd.DataFrame,
                              write: bool = True) -> dict[str, pd.DataFrame]:
    """Build all three tables; optionally persist them."""
    labelled = classify_match(comp, filing)
    tables = {
        "filing_date_selection_overview": selection_overview(labelled),
        "filing_date_selection_by_year": selection_by_year(labelled),
        "filing_date_selection_covariates": selection_covariates(labelled),
    }
    captions = {
        "filing_date_selection_overview":
            "Filing-date attrition: firm-years entering the 10-K filing-date "
            "merge, matched to an actual filing date, and dropped. Unmatched "
            "firm-years are dropped rather than imputed, faithful to the "
            "frozen design.",
        "filing_date_selection_by_year":
            "Filing-date match rate by fiscal year. A monotone trend indicates "
            "the selection is not stationary and must be read alongside the "
            "chronological split.",
        "filing_date_selection_covariates":
            "Matched versus unmatched firm-years on observables available "
            "before the merge. Standardised differences are reported instead "
            "of t-statistics because at this sample size negligible "
            "differences attain significance. Three scales are shown: raw "
            "levels, $\\log(1+x)$, and scale-free pooled percentile ranks. "
            "The levels statistic is deflated toward zero on these variables "
            "because pooled skewness runs 17--35, so the flag ($|d|>0.10$) is "
            "based on the rank version, which is invariant to monotone "
            "transformation. The last three rows are scale-free measures of "
            "financial weakness, which dollar levels cannot capture. "
            "Panel A reports group medians and Panel B the standardised "
            "differences; the two panels cover the same variables in the same "
            "order, and the split is a layout device so that every column "
            "remains legible.",
    }
    if write:
        for name, df in tables.items():
            save_table(df, OUT_TABLES_DESCRIPTIVE / name, caption=captions[name],
                       label=f"tab:{name}")
        # The covariate table needs a two-panel layout; the generic one-tabular
        # writer above produced an eight-column table that overran the text
        # width and clipped its rightmost columns. The CSV written by
        # save_table is unaffected and remains the machine-readable record.
        cov = "filing_date_selection_covariates"
        (OUT_TABLES_DESCRIPTIVE / f"{cov}.tex").write_text(
            covariates_table_tex(tables[cov], captions[cov], f"tab:{cov}"),
            encoding="utf-8")
        print(f"  Table re-rendered as two panels -> "
              f"{OUT_TABLES_DESCRIPTIVE / cov}.tex")
    return tables


def main() -> None:
    """
    Build the selection tables against the REPORTED run's own inputs.

    The Compustat panel is read from the persisted
    ``raw_final_primary/compustat_annual_raw.parquet`` rather than rebuilt from
    the .rds sources: rebuilding takes minutes, writes to the legacy raw path as
    a side effect, and — verified 2026-07-29 by SHA-256 — reproduces this file
    byte-for-byte anyway.
    """
    import pandas as pd

    from src.config import DATA_RAW_V2, FILING_DATE_UNMATCHED_POLICY
    from src.data.load_local_rds import _build_filing_dates_from_dedicated

    print("Filing-date matched-vs-unmatched selection analysis")
    comp = pd.read_parquet(DATA_RAW_V2 / "compustat_annual_raw.parquet")
    print(f"  Compustat firm-years loaded: {len(comp):,}")
    filing = _build_filing_dates_from_dedicated(FILING_DATE_UNMATCHED_POLICY)
    print(f"  Filing-date records:         {len(filing):,}")
    tables = run_filing_date_selection(comp, filing)
    for name, df in tables.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
