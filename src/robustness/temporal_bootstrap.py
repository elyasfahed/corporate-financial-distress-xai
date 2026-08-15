"""
Supplementary robustness: temporal (two-way cluster) bootstrap uncertainty.
===========================================================================
Methodological motivation: the frozen
PR-AUC confidence intervals and the paired PR-AUC significance test resample
firms with replacement (a one-way *firm*-block bootstrap). That scheme captures
cross-sectional (cross-firm) sampling uncertainty but holds the temporal
composition of the test panel FIXED -- the same ten calendar years (2015-2024)
enter every resample. Financial distress is not i.i.d. across time: it arrives
in macro-driven waves (the dot-com / GFC / COVID pattern discussed in Chapter 4),
so the finite number of test years is itself a source of estimation uncertainty
that a pure firm-block bootstrap does not represent. A firm-only bootstrap can
therefore understate the true sampling variability of a metric evaluated over one
particular decade.

The academically standard remedy is a MULTIWAY (two-way) cluster-robust
bootstrap that resamples clusters along BOTH the firm and the calendar-year
dimension, giving inference that is valid under arbitrary within-firm AND
within-year dependence (Cameron, Gelbach & Miller 2011; Cameron & Miller 2015).
This module quantifies how much the test-set PR-AUC confidence intervals widen
once temporal clustering is added, and whether the H1 conclusion --- ridge
logistic regression significantly beats the tree ensembles on PR-AUC --- survives
the wider, temporally-robust intervals.

Three resampling schemes are compared for every headline model, all evaluated on
the SAME frozen test predictions:

    FIRM  : resample the 4,906 test firms with replacement (the frozen scheme;
            reproduces the published firm-block CI).
    YEAR  : resample the 10 test calendar years with replacement (temporal
            uncertainty only; coarse because T = 10).
    TWOWAY: Cameron-Gelbach-Miller two-way cluster-robust variance,
            Var_2way = Var_firm + Var_year - Var_iid, where the firm x year
            intersection cluster is the individual firm-year observation (each
            (gvkey, fyear) cell is unique in the annual panel). The standard
            error is sqrt(max(Var_2way, 0)) and the reported interval is the
            normal-approximation point +/- 1.96 * SE_2way.

SAFETY -- READ-ONLY with respect to the frozen pipeline. This script loads the
frozen .joblib models and the frozen test parquet, reproduces the headline test
predictions EXACTLY as run_pipeline does (X_test = test[ALL_FEATURES].astype(
float).fillna(0)), sanity-checks each PR-AUC against the frozen headline, and then
recomputes a *derived* statistic (confidence intervals of already-final
predictions). It re-fits NOTHING and overwrites NO model. It writes ONLY:
    outputs/tables/robustness/temporal_bootstrap_ci.{csv,tex}
    outputs/tables/robustness/temporal_bootstrap_significance.{csv,tex}

Run from the project root:
    python -m src.robustness.temporal_bootstrap
On a plain cp1252 console:  PYTHONIOENCODING=utf-8 python -m src.robustness.temporal_bootstrap
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from src.config import (
    DATA_SAMPLES, OUT_MODELS_SAVED, OUT_TABLES_ROBUSTNESS,
    ALL_FEATURES, BOOTSTRAP_REPS, RANDOM_SEED,
)
from src.utils.tables import save_table

# Frozen headline test PR-AUCs (post lag-fix adoption, 2026-06-29). The balanced
# neural network is the co-primary fourth model (0.0994). These are the same
# targets used by scripts/regen_significance.py and verify_outputs.py.
MODELS = {
    "logistic_regression":      ("Logistic Regression",      0.1657),
    "random_forest":            ("Random Forest",            0.1354),
    "xgboost":                  ("XGBoost",                  0.1447),
    "neural_network_balanced":  ("Neural Network (bal.)",    0.0994),
}

# Paired comparisons for the H1 significance table (row minus column).
PAIRS = [
    ("random_forest",      "logistic_regression"),
    ("xgboost",            "logistic_regression"),
    ("random_forest",      "xgboost"),
]


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"      # dagger
    return ""


# ---------------------------------------------------------------------------
# Pure, testable resampling primitives
# ---------------------------------------------------------------------------
def group_index_map(group_ids: np.ndarray) -> dict:
    """Map each unique group id to the array of observation indices it owns."""
    uniq = np.unique(group_ids)
    return {g: np.where(group_ids == g)[0] for g in uniq}


def resample_cluster_indices(index_map: dict, rng: np.random.Generator) -> np.ndarray:
    """
    One draw of a cluster (block) bootstrap: sample len(index_map) groups with
    replacement and concatenate the observation indices of the drawn groups.
    A group drawn k times contributes k copies of its observations.
    """
    groups = np.fromiter(index_map.keys(), dtype=object, count=len(index_map))
    sampled = rng.choice(groups, size=len(groups), replace=True)
    return np.concatenate([index_map[g] for g in sampled])


def resample_iid_indices(n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """One draw of an ordinary (observation-level, i.i.d.) bootstrap."""
    return rng.integers(0, n_obs, size=n_obs)


def twoway_variance(var_firm: float, var_year: float, var_iid: float) -> float:
    """
    Cameron-Gelbach-Miller (2011) two-way cluster-robust variance:
        Var_2way = Var_firm + Var_year - Var_intersection,
    where the firm x year intersection cluster is the single firm-year
    observation, so Var_intersection is the i.i.d. (observation) variance.
    Truncated at zero (the CGM estimator is not guaranteed PSD in finite
    samples; the standard fix is to floor a negative combined variance at 0).
    """
    return max(var_firm + var_year - var_iid, 0.0)


# ---------------------------------------------------------------------------
# Bootstrap distributions of a metric / a paired metric difference
# ---------------------------------------------------------------------------
def _metric_distribution(
    y_true: np.ndarray,
    probs: dict[str, np.ndarray],
    diffs: list[tuple[str, str]],
    index_sampler,
    n_reps: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[tuple, np.ndarray]]:
    """
    Draw ``n_reps`` bootstrap resamples using ``index_sampler(rng) -> idx`` and,
    on each resample, evaluate every model's PR-AUC and every requested paired
    PR-AUC difference. Resamples with no positive case are skipped (PR-AUC
    undefined). One index draw is shared across all models/pairs per rep so the
    schemes are internally consistent and cheap.
    """
    model_boot = {name: [] for name in probs}
    diff_boot = {pair: [] for pair in diffs}
    for _ in range(n_reps):
        idx = index_sampler(rng)
        yb = y_true[idx]
        if yb.sum() == 0:
            continue
        ap = {name: average_precision_score(yb, p[idx]) for name, p in probs.items()}
        for name, v in ap.items():
            model_boot[name].append(v)
        for a, b in diffs:
            diff_boot[(a, b)].append(ap[a] - ap[b])
    return (
        {k: np.asarray(v) for k, v in model_boot.items()},
        {k: np.asarray(v) for k, v in diff_boot.items()},
    )


def _percentile_ci(dist: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (float(np.percentile(dist, 100 * alpha / 2)),
            float(np.percentile(dist, 100 * (1 - alpha / 2))))


def _recentred_p(dist: np.ndarray, observed: float) -> float:
    """Two-sided p-value from the null-recentred bootstrap distribution
    (same construction as src/analysis/significance.py)."""
    centred = dist - dist.mean()
    return float((np.sum(np.abs(centred) >= abs(observed)) + 1) / (len(dist) + 1))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_frozen_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load frozen test parquet + saved models; reproduce headline predictions."""
    test = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    X = test[list(ALL_FEATURES)].astype(float).fillna(0).values
    y = test["distress"].astype(int).values
    firm_ids = test["gvkey"].values
    year_ids = test["fyear"].astype(int).values

    # Each (gvkey, fyear) cell must be unique for the CGM intersection = i.i.d.
    n_cells = pd.MultiIndex.from_arrays([firm_ids, year_ids]).nunique()
    assert n_cells == len(test), (
        f"firm x year cells not unique ({n_cells} vs {len(test)}); the two-way "
        "intersection cluster is not the individual observation."
    )

    probs, ok = {}, True
    print("Reproducing frozen headline test predictions:")
    for name, (_label, expected) in MODELS.items():
        model = joblib.load(OUT_MODELS_SAVED / f"{name}.joblib")
        p = model.predict_proba(X)[:, 1]
        pr = average_precision_score(y, p)
        good = abs(pr - expected) < 1.5e-3
        ok &= good
        print(f"  {name:26s} PR-AUC={pr:.4f} (exp {expected:.4f})  "
              f"[{'OK' if good else 'MISMATCH'}]")
        probs[name] = p
    if not ok:
        raise SystemExit(
            "Saved models did NOT reproduce the frozen headline PR-AUCs -- aborting "
            "so the temporal-bootstrap tables are not written from wrong predictions."
        )
    return y, firm_ids, year_ids, probs


def run(n_reps: int = BOOTSTRAP_REPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    y, firm_ids, year_ids, probs = load_frozen_predictions()
    n_obs = len(y)
    print(f"\n  Test sample: n={n_obs:,}  firms={len(np.unique(firm_ids)):,}  "
          f"years={len(np.unique(year_ids))}  events={int(y.sum())}  "
          f"prevalence={y.mean():.4f}")

    firm_map = group_index_map(firm_ids)
    year_map = group_index_map(year_ids)

    # Independent RNG per scheme so schemes do not share draws.
    rng_firm = np.random.default_rng(RANDOM_SEED)
    rng_year = np.random.default_rng(RANDOM_SEED + 1)
    rng_iid  = np.random.default_rng(RANDOM_SEED + 2)

    print(f"\n  Drawing {n_reps} resamples under three schemes "
          "(firm cluster / year cluster / i.i.d.)...")
    m_firm, d_firm = _metric_distribution(
        y, probs, PAIRS, lambda r: resample_cluster_indices(firm_map, r), n_reps, rng_firm)
    m_year, d_year = _metric_distribution(
        y, probs, PAIRS, lambda r: resample_cluster_indices(year_map, r), n_reps, rng_year)
    m_iid, d_iid = _metric_distribution(
        y, probs, PAIRS, lambda r: resample_iid_indices(n_obs, r), n_reps, rng_iid)

    # ---- Per-model CI table -------------------------------------------------
    ci_rows = []
    for name, (label, _exp) in MODELS.items():
        point = average_precision_score(y, probs[name])
        fw_lo, fw_hi = _percentile_ci(m_firm[name])
        yr_lo, yr_hi = _percentile_ci(m_year[name])
        v_firm = float(np.var(m_firm[name], ddof=1))
        v_year = float(np.var(m_year[name], ddof=1))
        v_iid  = float(np.var(m_iid[name], ddof=1))
        se2 = np.sqrt(twoway_variance(v_firm, v_year, v_iid))
        z = stats.norm.ppf(0.975)
        tw_lo, tw_hi = point - z * se2, point + z * se2
        firm_width = fw_hi - fw_lo
        tw_width = tw_hi - tw_lo
        ci_rows.append({
            "model": label,
            "pr_auc": round(point, 4),
            "firm_ci_lower": round(fw_lo, 4), "firm_ci_upper": round(fw_hi, 4),
            "year_ci_lower": round(yr_lo, 4), "year_ci_upper": round(yr_hi, 4),
            "se_firm": round(np.sqrt(v_firm), 4),
            "se_year": round(np.sqrt(v_year), 4),
            "se_twoway": round(se2, 4),
            "twoway_ci_lower": round(tw_lo, 4), "twoway_ci_upper": round(tw_hi, 4),
            "width_ratio_twoway_firm": round(tw_width / firm_width, 2) if firm_width > 0 else np.nan,
        })
    ci_df = pd.DataFrame(ci_rows)

    # ---- Paired-difference significance table -------------------------------
    sig_rows = []
    labels = {k: v[0] for k, v in MODELS.items()}
    z975 = stats.norm.ppf(0.975)
    for a, b in PAIRS:
        delta = (average_precision_score(y, probs[a])
                 - average_precision_score(y, probs[b]))
        # Firm-block (frozen scheme): percentile CI + null-recentred p-value
        f_lo, f_hi = _percentile_ci(d_firm[(a, b)])
        p_firm = _recentred_p(d_firm[(a, b)], delta)
        # Year-block: percentile CI + null-recentred p-value
        y_lo, y_hi = _percentile_ci(d_year[(a, b)])
        p_year = _recentred_p(d_year[(a, b)], delta)
        # Two-way CGM SE of the difference -> normal CI + p-value
        v_firm = float(np.var(d_firm[(a, b)], ddof=1))
        v_year = float(np.var(d_year[(a, b)], ddof=1))
        v_iid  = float(np.var(d_iid[(a, b)], ddof=1))
        se2 = np.sqrt(twoway_variance(v_firm, v_year, v_iid))
        tw_lo, tw_hi = delta - z975 * se2, delta + z975 * se2
        p_two = float(2 * (1 - stats.norm.cdf(abs(delta) / se2))) if se2 > 0 else 0.0
        sig_rows.append({
            "comparison": f"{labels[a]} vs {labels[b]}",
            "delta_pr_auc": round(delta, 4),
            "firm_ci_lower": round(f_lo, 4), "firm_ci_upper": round(f_hi, 4),
            "p_firm": round(p_firm, 4), "sig_firm": _stars(p_firm),
            "year_ci_lower": round(y_lo, 4), "year_ci_upper": round(y_hi, 4),
            "p_year": round(p_year, 4), "sig_year": _stars(p_year),
            "se_twoway": round(se2, 4),
            "twoway_ci_lower": round(tw_lo, 4), "twoway_ci_upper": round(tw_hi, 4),
            "p_twoway": round(p_two, 4), "sig_twoway": _stars(p_two),
        })
    sig_df = pd.DataFrame(sig_rows)

    # ---- Persist ------------------------------------------------------------
    print("\nPer-model PR-AUC confidence intervals by resampling scheme:")
    print(ci_df.to_string(index=False))
    print("\nPaired PR-AUC differences by resampling scheme:")
    print(sig_df.to_string(index=False))

    save_table(
        ci_df, OUT_TABLES_ROBUSTNESS / "temporal_bootstrap_ci",
        caption=(
            "Test-set PR-AUC 95\\% confidence intervals under three bootstrap "
            "resampling schemes (1,000 resamples each): the frozen firm-cluster "
            "block bootstrap, a year-cluster (temporal) block bootstrap, and the "
            "Cameron-Gelbach-Miller two-way (firm $\\times$ year) cluster-robust "
            "interval "
            "($\\mathrm{Var}_{2\\mathrm{way}}=\\mathrm{Var}_{\\text{firm}}+"
            "\\mathrm{Var}_{\\text{year}}-\\mathrm{Var}_{\\text{iid}}$; "
            "normal-approximation interval). The final column is the ratio of the "
            "two-way to the firm-only interval width."),
        label="tab:temporal_bootstrap_ci",
    )
    save_table(
        sig_df, OUT_TABLES_ROBUSTNESS / "temporal_bootstrap_significance",
        caption=(
            "Paired PR-AUC difference tests ($\\Delta$ = row model minus column "
            "model) under the firm-cluster, year-cluster, and two-way "
            "cluster-robust bootstrap. Firm- and year-cluster $p$-values come from "
            "the null-recentred bootstrap distribution; the two-way $p$-value uses "
            "the Cameron-Gelbach-Miller standard error and a normal approximation. "
            "A confidence interval that excludes zero indicates a significant "
            "difference. Significance: $\\dagger$ $p<0.10$, * $p<0.05$, "
            "** $p<0.01$, *** $p<0.001$."),
        label="tab:temporal_bootstrap_significance",
    )
    print(f"\nSaved -> {OUT_TABLES_ROBUSTNESS / 'temporal_bootstrap_ci.{csv,tex}'}")
    print(f"Saved -> {OUT_TABLES_ROBUSTNESS / 'temporal_bootstrap_significance.{csv,tex}'}")
    return ci_df, sig_df


if __name__ == "__main__":
    run()
