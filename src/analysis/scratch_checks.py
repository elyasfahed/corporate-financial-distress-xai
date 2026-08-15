"""
Scratch checks — your sandbox for quick, throwaway plots.
=========================================================
This is the file you OPEN IN PYCHARM AND RUN (green ▶ button, or right-click →
"Run 'scratch_checks'"). Nothing produced here is thesis material — every plot
lands in the git-ignored  scratch/  folder, dated and logged. See
src/utils/scratch.py for the full idea.

HOW TO USE
----------
1. Open this file in PyCharm.
2. Click the green ▶ at the top (or right-click in the editor → Run).
3. Look at the PNGs it printed under  scratch/<today>/ .
4. Edit the "YOUR CHECKS" section below to ask your own questions, then Run again.

The three building blocks (add as many as you like):

    check_corr(df, "NITA", "TLTA", note="why am I looking?")
        scatter/hexbin of two columns + Pearson & Spearman correlation

    check_scaling(df, "MB", note="why am I looking?")
        raw vs asinh distribution + skew — does this var need a transform?

    scratch_save(fig, "my_slug", note="...")
        save ANY matplotlib figure you built yourself

Housekeeping (run occasionally, or uncomment at the bottom):
    scratch_clean(days=14)   # delete scratch folders older than two weeks
"""

from __future__ import annotations

import pandas as pd

from src.config import DATA_SAMPLES, DATA_FEATURES
from src.utils.scratch import (
    check_corr,          # scatter of two vars + correlation
    check_scaling,       # does a var need a transform?
    check_dist,          # distribution of one var
    check_by_label,      # predictor distribution: distressed vs healthy
    check_missingness,   # % missing per variable
    check_corr_matrix,   # correlation heatmap for many vars
    check_outliers,      # where winsorisation would cut
    check_over_time,     # a variable's mean/median across fiscal years
    scratch_save,        # save any figure you built yourself
    scratch_clean,       # delete old scratch folders
)


def load_data() -> pd.DataFrame:
    """
    Load a sample to plot from. Defaults to the full feature panel; falls back
    to the training split. Change the file here if you want test/val instead.
    """
    candidates = [
        DATA_FEATURES / "features_all.parquet",   # full feature panel
        DATA_SAMPLES / "train.parquet",           # fallback: training split
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"Loaded {path.name}  ({len(df):,} rows, {df.shape[1]} cols)")
            return df
    raise FileNotFoundError(
        "No data file found. Make sure your Seafile data folder is synced and "
        "THESIS_DATA_ROOT is set (see src/config.py). Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def main() -> None:
    df = load_data()

    # ======================================================================
    # YOUR CHECKS — edit freely. Each call writes one PNG to scratch/<today>/.
    # ======================================================================

    # 1. Are two predictors correlated?
    check_corr(df, "SIGMA", "EXRET", note="my check")

    # 2. Does a variable need scaling/transforming?
    check_scaling(df, "MB", note="market-to-book is heavy-tailed — transform?")

    # 3. What does one variable's distribution look like?
    check_dist(df, "SIGMA", note="range and shape of return volatility")

    # 4. Does a predictor separate distressed from healthy firms?
    check_by_label(df, "TLTA", note="is leverage higher for distressed firms?")

    # 5. Which variables have missing values, and how much?
    check_missingness(df, note="missingness across the 17 predictors")

    # 6. Which predictors are highly correlated (multicollinearity)?
    check_corr_matrix(df, note="spot |r|>0.7 pairs before reading SHAP")

    # 7. Where would winsorisation cut a variable?
    check_outliers(df, "NITA", note="sanity-check winsorisation thresholds")

    # 8. How does a variable behave across fiscal years?
    check_over_time(df, "TLTA", note="leverage trend over the panel")

    # 9. Build your OWN figure and save it (uncomment to try):
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # df["PRICE"].plot(kind="hist", bins=60, ax=ax)
    # scratch_save(fig, "price_hist", note="custom check")

    # ======================================================================

    print("\nDone. Open the newest folder under  scratch/  to view the PNGs.")
    print("Tip: scratch/INDEX.md lists every check with its note.")

    # Optional cleanup — uncomment to delete scratch folders older than 14 days:
    # scratch_clean(days=14)


if __name__ == "__main__":
    main()
