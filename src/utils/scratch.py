"""
Scratch / throwaway plotting — for ad-hoc, internal checks ONLY.
================================================================
NOTHING saved through this module is thesis material. The thesis figure
namespace is  outputs/figures/<chapter>/ , produced reproducibly by committed
src/ scripts via  src.utils.plots.save_figure()  (PDF + PNG, 300 DPI, academic
style). This module is the deliberate opposite: fast, dated, git-ignored,
auto-expiring PNGs you look at once to make a decision and then discard.

The boundary (memorise this one rule)
-------------------------------------
A figure is thesis material  IFF  it lives under  outputs/figures/<chapter>/
AND is produced by a committed src/ script. Everything under  scratch/  is
disposable by definition. To *promote* a scratch check, RE-DERIVE it with a
real script that writes to outputs/figures/ via save_figure() — never move the
scratch PNG. That keeps every thesis figure traceable to code.

What you get
------------
- scratch_save(fig, slug, note=...)   save any matplotlib fig as scratch
- check_corr(df, x, y, note=...)      scatter + Pearson/Spearman for two vars
- check_scaling(df, col, note=...)    raw vs asinh distribution + skew, to
                                      decide whether a var benefits from scaling
- scratch_clean(days=14)              delete dated folders older than N days
- scratch_open_latest()               path of the most recent scratch dir

Layout
------
    scratch/
        INDEX.md                 append-only log: when / what / why / path
        2026-06-21/
            143002_corr_nita_vs_tlta.png
            143515_scaling_mb.png

Every PNG carries a small grey stamp (timestamp + your note) burned into the
corner, so even a plot that escapes into a slide or email still announces that
it is scratch and when it was made.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — matches src/utils/plots.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUT_SCRATCH, ALL_FEATURES

# Scratch plots are for eyeballing, not for LaTeX — keep them light and fast.
SCRATCH_DPI = 110
_MAX_SCATTER_POINTS = 5_000     # subsample large panels so scatter stays legible
_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.lower()]
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "plot"


def _today_dir(when: _dt.datetime | None = None) -> Path:
    when = when or _dt.datetime.now()
    d = OUT_SCRATCH / when.strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_index(timestamp: str, kind: str, rel_path: str, note: str) -> None:
    """Append a one-line record to scratch/INDEX.md (created on first use)."""
    OUT_SCRATCH.mkdir(parents=True, exist_ok=True)
    index = OUT_SCRATCH / "INDEX.md"
    if not index.exists():
        index.write_text(
            "# Scratch plot index\n\n"
            "Append-only log of ad-hoc checks. NONE of these are thesis figures.\n"
            "Purge anytime with `scratch_clean()` or by deleting the scratch/ folder.\n\n",
            encoding="utf-8",
        )
    line = f"- {timestamp} | {kind} | `{rel_path}` | {note or '(no note)'}\n"
    with index.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _stamp(fig: plt.Figure, note: str, timestamp: str) -> None:
    label = f"SCRATCH · {timestamp}" + (f" · {note}" if note else "")
    fig.text(0.995, 0.005, label, ha="right", va="bottom",
             fontsize=6, color="0.55", style="italic")


# ---------------------------------------------------------------------------
# Core save
# ---------------------------------------------------------------------------

def scratch_save(fig: plt.Figure, slug: str, note: str = "",
                 kind: str = "plot", close: bool = True) -> Path:
    """
    Save a matplotlib figure as a dated, stamped, logged scratch PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    slug : str
        Short descriptive name, e.g. 'corr_nita_vs_tlta'. Sanitised to a
        filesystem-safe stem; a HHMMSS prefix is added automatically.
    note : str
        The question/reason this check exists. Burned onto the image and
        written to scratch/INDEX.md. Strongly recommended.
    kind : str
        Free-text category for the index log (e.g. 'corr', 'scaling').
    close : bool
        Close the figure after saving (default True) to free memory.

    Returns
    -------
    Path to the written PNG.
    """
    now = _dt.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    folder = _today_dir(now)
    fname = f"{now.strftime('%H%M%S')}_{_slugify(slug)}.png"
    out = folder / fname

    _stamp(fig, note, timestamp)
    fig.savefig(out, dpi=SCRATCH_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)

    rel = out.relative_to(OUT_SCRATCH.parent).as_posix()
    _log_index(timestamp, kind, rel, note)
    print(f"  [scratch] {rel}")
    return out


# ---------------------------------------------------------------------------
# Ready-made checks (the common cases)
# ---------------------------------------------------------------------------

def _clean_pair(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    return sub


def check_corr(df: pd.DataFrame, x: str, y: str, note: str = "",
               hexbin: bool | None = None) -> Path:
    """
    Quick bivariate check: scatter (or hexbin) of two columns plus Pearson and
    Spearman correlations in the title. Large samples are subsampled for the
    scatter; correlations always use the full cleaned data.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Column names.
    note : str
        Why you're looking (logged + stamped).
    hexbin : bool | None
        Force hexbin density (True) or scatter (False). Default None = auto:
        hexbin when >20k points.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    sub = _clean_pair(df, x, y)
    n = len(sub)
    pearson = sub[x].corr(sub[y], method="pearson")
    spearman = sub[x].corr(sub[y], method="spearman")

    use_hex = hexbin if hexbin is not None else (n > 20_000)
    fig, ax = plt.subplots(figsize=(6, 5))
    if use_hex:
        hb = ax.hexbin(sub[x], sub[y], gridsize=45, mincnt=1, cmap="viridis")
        fig.colorbar(hb, ax=ax, label="count")
    else:
        plot_df = sub
        if n > _MAX_SCATTER_POINTS:
            idx = _RNG.choice(n, _MAX_SCATTER_POINTS, replace=False)
            plot_df = sub.iloc[idx]
        ax.scatter(plot_df[x], plot_df[y], s=8, alpha=0.3, edgecolors="none")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    flag = "  ⚠ |r|>0.70" if abs(pearson) > 0.70 else ""
    ax.set_title(f"{x} vs {y}   (n={n:,})\n"
                 f"Pearson r={pearson:.3f}{flag}   Spearman ρ={spearman:.3f}")
    return scratch_save(fig, f"corr_{x}_vs_{y}", note=note, kind="corr")


def check_scaling(df: pd.DataFrame, col: str, note: str = "") -> Path:
    """
    Decide whether a variable benefits from scaling/transformation.

    Three panels: raw histogram, asinh-transformed histogram (handles negatives
    and compresses heavy tails — a good general-purpose probe), and a boxplot.
    Skewness and excess kurtosis are reported raw vs asinh; a large drop in |skew|
    suggests a transform would help. asinh is used (not log) because finance
    ratios such as NITA, EXRET and CHIN take negative values.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
    note : str
    """
    from scipy import stats
    from src.utils.plots import set_academic_style
    set_academic_style()

    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    raw = s.to_numpy()
    asinh = np.arcsinh(raw)

    skew_raw, skew_t = stats.skew(raw), stats.skew(asinh)
    kurt_raw, kurt_t = stats.kurtosis(raw), stats.kurtosis(asinh)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(raw, bins=60, color="steelblue")
    axes[0].set_title(f"raw\nskew={skew_raw:.2f}  exkurt={kurt_raw:.2f}")
    axes[0].set_xlabel(col)

    axes[1].hist(asinh, bins=60, color="seagreen")
    axes[1].set_title(f"asinh({col})\nskew={skew_t:.2f}  exkurt={kurt_t:.2f}")
    axes[1].set_xlabel(f"asinh({col})")

    axes[2].boxplot(raw, vert=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3))
    axes[2].set_title("raw spread (outliers)")
    axes[2].set_xticks([])

    verdict = ("transform likely helps" if abs(skew_t) < abs(skew_raw) - 0.5
               else "scaling unlikely to change shape much")
    fig.suptitle(f"{col}  (n={len(s):,})  —  {verdict}", fontsize=12)
    return scratch_save(fig, f"scaling_{col}", note=note, kind="scaling")


def check_dist(df: pd.DataFrame, col: str, note: str = "") -> Path:
    """
    Distribution of a single variable: histogram + boxplot + key percentiles.
    Use to eyeball shape, spread and where the bulk of the data sits.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    q = s.quantile([0.01, 0.10, 0.50, 0.90, 0.99])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                   gridspec_kw={"height_ratios": [4, 1]})
    ax1.hist(s, bins=70, color="steelblue")
    for p, v in q.items():
        ax1.axvline(v, color="crimson", lw=0.6, ls=":")
    ax1.set_ylabel("count")
    ax1.set_title(f"{col}  (n={len(s):,})   "
                  f"median={q[0.50]:.3g}  p1={q[0.01]:.3g}  p99={q[0.99]:.3g}")
    ax2.boxplot(s, vert=False, showfliers=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax2.set_yticks([])
    ax2.set_xlabel(col)
    return scratch_save(fig, f"dist_{col}", note=note, kind="dist")


def check_by_label(df: pd.DataFrame, col: str, label: str = "distress",
                   note: str = "") -> Path:
    """
    Compare a predictor's distribution for distressed vs healthy firms.
    Overlaid density histograms + group medians. This is the single most useful
    'does this variable separate the classes?' check for distress prediction.

    Parameters
    ----------
    label : str
        Binary 0/1 column. Defaults to 'distress'.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    sub = df[[col, label]].replace([np.inf, -np.inf], np.nan).dropna()
    healthy = sub.loc[sub[label] == 0, col]
    distressed = sub.loc[sub[label] == 1, col]

    # Clip the view to 1–99th pct of the pooled data so tails don't hide the bulk.
    lo, hi = sub[col].quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(healthy, bins=bins, density=True, alpha=0.5,
            label=f"healthy (n={len(healthy):,})", color="steelblue")
    ax.hist(distressed, bins=bins, density=True, alpha=0.5,
            label=f"distressed (n={len(distressed):,})", color="crimson")
    ax.axvline(healthy.median(), color="steelblue", lw=1.2, ls="--")
    ax.axvline(distressed.median(), color="crimson", lw=1.2, ls="--")
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    gap = distressed.median() - healthy.median()
    ax.set_title(f"{col} by {label}   median gap (distressed−healthy) = {gap:.3g}")
    ax.legend(frameon=False)
    return scratch_save(fig, f"bylabel_{col}", note=note, kind="by_label")


def check_missingness(df: pd.DataFrame, cols: list[str] | None = None,
                      note: str = "") -> Path:
    """
    Percentage of missing values per column (inf counted as missing).
    Defaults to the 17 thesis predictors. Bar chart sorted worst-first.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    cols = cols or [c for c in ALL_FEATURES if c in df.columns]
    clean = df[cols].replace([np.inf, -np.inf], np.nan)
    pct = (clean.isna().mean() * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(cols))))
    ax.barh(pct.index[::-1], pct.values[::-1], color="darkorange")
    ax.set_xlabel("% missing")
    ax.set_title(f"Missingness across {len(cols)} variables  (n={len(df):,} rows)")
    for i, v in enumerate(pct.values[::-1]):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8)
    return scratch_save(fig, "missingness", note=note, kind="missing")


def check_corr_matrix(df: pd.DataFrame, cols: list[str] | None = None,
                      note: str = "", threshold: float = 0.70) -> Path:
    """
    Pearson correlation heatmap for many variables at once, with pairs above
    `threshold` printed to the console. Defaults to the 17 thesis predictors.
    A fast way to spot multicollinearity before trusting SHAP splits.
    """
    import seaborn as sns
    from src.utils.plots import set_academic_style
    set_academic_style()

    cols = cols or [c for c in ALL_FEATURES if c in df.columns]
    corr = df[cols].replace([np.inf, -np.inf], np.nan).corr()

    flagged = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                flagged.append((cols[i], cols[j], r))
    if flagged:
        print(f"  [scratch] |r|>{threshold} pairs:")
        for a, b, r in flagged:
            print(f"            {a:>8} ~ {b:<8}  r={r:+.3f}")

    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 2, 0.6 * len(cols) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title(f"Pearson correlations ({len(cols)} vars) — "
                 f"{len(flagged)} pair(s) with |r|>{threshold}")
    return scratch_save(fig, "corr_matrix", note=note, kind="corr_matrix")


def check_outliers(df: pd.DataFrame, col: str, note: str = "",
                   lower: float = 0.01, upper: float = 0.99) -> Path:
    """
    Show where winsorisation would cut a variable. Histogram with the chosen
    lower/upper percentile cut lines, and how many rows sit beyond them.
    Helps sanity-check the training-sample winsorisation thresholds.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    s = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = s.quantile([lower, upper])
    n_low = int((s < lo).sum())
    n_high = int((s > hi).sum())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(s, bins=70, color="steelblue")
    ax.axvline(lo, color="crimson", lw=1.0, ls="--",
               label=f"p{lower*100:.0f} = {lo:.3g}  ({n_low:,} below)")
    ax.axvline(hi, color="crimson", lw=1.0, ls="--",
               label=f"p{upper*100:.0f} = {hi:.3g}  ({n_high:,} above)")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(f"{col}: winsorisation cut points  (n={len(s):,})")
    ax.legend(frameon=False, fontsize=9)
    return scratch_save(fig, f"outliers_{col}", note=note, kind="outliers")


def check_over_time(df: pd.DataFrame, col: str, year: str = "fyear",
                    note: str = "") -> Path:
    """
    Track a variable's cross-sectional mean and median over fiscal years.
    Use to spot regime shifts, trends or data-quality breaks across the panel.
    """
    from src.utils.plots import set_academic_style
    set_academic_style()

    sub = df[[year, col]].replace([np.inf, -np.inf], np.nan).dropna()
    grp = sub.groupby(year)[col].agg(["mean", "median"]).sort_index()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grp.index, grp["mean"], marker="o", ms=4, lw=1.6, label="mean")
    ax.plot(grp.index, grp["median"], marker="s", ms=4, lw=1.6, label="median")
    ax.set_xlabel(year)
    ax.set_ylabel(col)
    ax.set_title(f"{col} over time")
    ax.legend(frameon=False)
    return scratch_save(fig, f"overtime_{col}", note=note, kind="over_time")


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def scratch_clean(days: int = 14, dry_run: bool = False) -> list[Path]:
    """
    Delete dated scratch folders older than `days`. INDEX.md is left intact as
    a record. Returns the list of folders removed (or that would be removed).

    Parameters
    ----------
    days : int
        Keep folders dated within the last `days` days; remove older ones.
    dry_run : bool
        If True, only report what would be deleted.
    """
    if not OUT_SCRATCH.exists():
        print("  [scratch] nothing to clean (no scratch/ dir)")
        return []

    cutoff = _dt.date.today() - _dt.timedelta(days=days)
    removed: list[Path] = []
    for child in sorted(OUT_SCRATCH.iterdir()):
        if not child.is_dir():
            continue
        try:
            folder_date = _dt.datetime.strptime(child.name, "%Y-%m-%d").date()
        except ValueError:
            continue  # not a dated folder — leave it alone
        if folder_date < cutoff:
            removed.append(child)
            if not dry_run:
                shutil.rmtree(child)

    verb = "would remove" if dry_run else "removed"
    print(f"  [scratch] {verb} {len(removed)} folder(s) older than {days} days")
    return removed


def scratch_open_latest() -> Path | None:
    """Return the most recent dated scratch folder, or None if empty."""
    if not OUT_SCRATCH.exists():
        return None
    dated = []
    for child in OUT_SCRATCH.iterdir():
        if child.is_dir():
            try:
                _dt.datetime.strptime(child.name, "%Y-%m-%d")
                dated.append(child)
            except ValueError:
                pass
    return max(dated, default=None)
