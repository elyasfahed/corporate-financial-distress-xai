"""
Shared loaders for the 2026-08 supplementary analyses.
======================================================
**Classification: post-hoc supplementary.** Every consumer of this module scores
the *frozen* ``final_primary`` estimators on the frozen test split. Nothing here
fits, tunes, or writes a model.

All supplementary output lands under ``outputs/tables/supplementary/final_primary/``
so it can never collide with a primary table.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C

SPEC = "final_primary"
SAMPLES = C.DATA_ROOT / "processed_final_primary" / "samples"
MODELS = Path("outputs/models/saved") / SPEC
CONFIGS = Path("outputs/models/configs") / SPEC
SUPP_TABLES = Path("outputs/tables/supplementary") / SPEC
SUPP_FIGURES = Path("outputs/figures/supplementary") / SPEC

FEATS = C.ALL_FEATURES_V2

#: Display order used in every supplementary table.
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost",
               "neural_network_balanced"]
DISPLAY = {
    "logistic_regression": "Logistic regression (ridge)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "neural_network_balanced": "Neural network (balanced)",
}

#: Reported headline PR-AUCs — asserted before any supplementary table is written.
HEADLINE_PR_AUC = {
    "logistic_regression": 0.1751,
    "random_forest": 0.1608,
    "xgboost": 0.1729,
    "neural_network_balanced": 0.1338,
}


def ensure_dirs() -> None:
    SUPP_TABLES.mkdir(parents=True, exist_ok=True)
    SUPP_FIGURES.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(SAMPLES / f"{name}.parquet")


def load_frozen_scores(split: str = "test") -> tuple[pd.DataFrame, np.ndarray,
                                                     dict[str, np.ndarray]]:
    """
    Score every frozen estimator on ``split``.

    Returns
    -------
    (frame, y, scores)
        ``frame`` is the split (for gvkey / fyear / permno grouping), ``y`` the
        binary outcome, ``scores`` a model -> probability-vector mapping.
    """
    df = load_split(split)
    X = df[FEATS].to_numpy(dtype=float)
    y = df["distress"].to_numpy(dtype=int)
    scores = {}
    for name in MODEL_ORDER:
        model = joblib.load(MODELS / f"{name}.joblib")
        scores[name] = model.predict_proba(X)[:, 1]
    return df, y, scores


def assert_headline_reproduces(y: np.ndarray, scores: dict[str, np.ndarray],
                               tol: float = 5e-5) -> dict[str, float]:
    """
    Abort before writing anything if the frozen models no longer reproduce the
    reported test PR-AUCs. Guards every supplementary analysis against silently
    describing a different run.
    """
    from sklearn.metrics import average_precision_score
    got = {}
    for name, ref in HEADLINE_PR_AUC.items():
        ap = float(average_precision_score(y, scores[name]))
        if abs(ap - ref) > tol:
            raise AssertionError(
                f"{name}: test PR-AUC {ap:.6f} does not reproduce the reported "
                f"{ref} (tolerance {tol}). Refusing to write supplementary output."
            )
        got[name] = ap
    return got


def firm_blocks(df: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row indices grouped by firm (GVKEY) for block resampling."""
    codes, _ = pd.factorize(df["gvkey"].to_numpy())
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
    return codes, np.split(order, bounds)


def year_blocks(df: pd.DataFrame) -> list[np.ndarray]:
    """Row indices grouped by fiscal year."""
    return [g.to_numpy() for _, g in df.reset_index(drop=True).groupby("fyear").groups.items()] \
        if False else [np.flatnonzero((df["fyear"] == y).to_numpy())
                       for y in sorted(df["fyear"].unique())]


def bootstrap_indices(blocks: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One block-bootstrap resample: draw len(blocks) blocks with replacement."""
    pick = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[i] for i in pick])


def write_table(df: pd.DataFrame, stem: str, caption: str, label: str,
                float_format: str = "%.4f", column_format: str | None = None,
                resize: bool = True) -> None:
    """
    Write a supplementary table as both CSV and LaTeX.

    Parameters
    ----------
    resize
        Constrain the tabular to the text width with
        ``\\adjustbox{max width=\\textwidth}{...}``. This **shrinks a table that
        is too wide and leaves everything else at its natural size**.

        It deliberately does *not* use ``\\resizebox{\\textwidth}{!}{...}``,
        which forces the tabular to exactly ``\\textwidth`` in both directions
        and therefore scales a narrow table *up*. Several supplementary tables
        have only three or four short numeric columns, and under ``\\resizebox``
        they were typeset at roughly twice the body type size — visually jarring
        and, worse, implying an emphasis the content does not carry.

        Pass ``resize=False`` when ``column_format`` already fixes ``p{}``
        widths in units of ``\\textwidth``: those widths are correct by
        construction and any further scaling only distorts the type size.
    """
    ensure_dirs()
    df.to_csv(SUPP_TABLES / f"{stem}.csv", index=False)
    tex = df.to_latex(index=False, escape=True, float_format=float_format,
                      column_format=column_format
                      or ("l" + "r" * (len(df.columns) - 1)))
    inner = ("\\adjustbox{max width=\\textwidth}{%\n" + tex + "}\n") if resize else tex
    body = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n"
        + inner + "\\end{table}\n"
    )
    (SUPP_TABLES / f"{stem}.tex").write_text(body, encoding="utf-8")
    print(f"  wrote {SUPP_TABLES / stem}.{{csv,tex}}")
