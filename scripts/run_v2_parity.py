"""
v2 parity batch — the six remaining re-inference items.
=========================================================================
Pure re-inference on the saved v2 models (nothing re-fitted, no v1 file
touched; all outputs v2_-prefixed):

  1. Theory-consistency table (H3) — pre-specified sign test, top-10
     XGBoost SHAP features -> v2_theory_consistency.{csv,tex}
  2. SHAP-vs-LR-coefficient Kendall-tau concordance (XAI strategy §global)
     -> v2_shap_lr_concordance.{csv,tex}
  3. Temporal (two-way firm x year) bootstrap on the v2 predictions (§14
     machinery) -> v2_temporal_bootstrap_ci / v2_temporal_bootstrap_significance
  4. Subperiod stability (2015-2019 vs 2020-2023, 0.7x-of-full flag)
     -> v2_subperiod_performance.{csv,tex}
  5. Predictor summary stats by outcome group (Welch t) + correlation
     matrix (|r|>0.7 flags) -> v2_predictor_summary_stats / v2_correlation_matrix
  6. SHAP stability — firm-block bootstrap of the precomputed XGB SHAP
     matrix, rank CIs for the top-10 -> v2_shap_stability.{csv,tex}

Run: PYTHONUTF8=1 ./.venv/Scripts/python.exe -m scripts.run_v2_parity
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    BOOTSTRAP_REPS,
    DATA_SAMPLES_V2,
    OUT_TABLES_DESCRIPTIVE,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
    OUT_TABLES_SHAP,
    RANDOM_SEED,
    V2_PROFILE,
)
from src.explainability.shap_global import compute_shap_values
from src.explainability.theory_consistency import build_theory_consistency_table
from src.models.train import resolve_artifact_dirs
from src.robustness.temporal_bootstrap import (
    _metric_distribution,
    _percentile_ci,
    _recentred_p,
    group_index_map,
    resample_cluster_indices,
    resample_iid_indices,
    twoway_variance,
)
from src.utils.tables import save_table

MODELS = ["logistic_regression", "random_forest", "xgboost",
          "neural_network_balanced"]
FEATURES = list(V2_PROFILE["feature_set"])
RUN_TABLES_SHAP = OUT_TABLES_SHAP / V2_PROFILE["spec"]
RUN_TABLES_ROBUSTNESS = OUT_TABLES_ROBUSTNESS / V2_PROFILE["spec"]
RUN_TABLES_MODEL = OUT_TABLES_MODEL / V2_PROFILE["spec"]
RUN_TABLES_DESCRIPTIVE = OUT_TABLES_DESCRIPTIVE / V2_PROFILE["spec"]


def main() -> None:
    saved_dir, _ = resolve_artifact_dirs(V2_PROFILE["spec"])
    train = pd.read_parquet(DATA_SAMPLES_V2 / "train.parquet")
    test = pd.read_parquet(DATA_SAMPLES_V2 / "test.parquet")
    X = test[FEATURES].astype(float).values
    y = test["distress"].astype(int).values
    firms = test["gvkey"].values
    years = test["fyear"].astype(int).values
    models = {n: joblib.load(saved_dir / f"{n}.joblib") for n in MODELS}
    probs = {n: m.predict_proba(X)[:, 1] for n, m in models.items()}
    print(f"v2 test scored: {len(test):,} rows, {int(y.sum())} events")

    pr = {name: average_precision_score(y, probs[name]) for name in MODELS}
    print("  Test PR-AUC by model: "
          + ", ".join(f"{n}={s:.4f}" for n, s in pr.items()))

    # XAI decision (final_primary co-lead framing): H3 theory-consistency and
    # SHAP-rank stability are reported on XGBoost — the co-leading ML model on
    # which SHAP is non-circular and the H4 non-linearities are testable. LR
    # (the nominal PR-AUC leader) is read via its standardised coefficients as
    # the linear benchmark (the concordance table below), not through SHAP.
    best_name = "xgboost"
    best_label = "XGBoost"
    xgb_sv = compute_shap_values(models["xgboost"], X, model_name="xgboost")
    best_sv = xgb_sv
    X_df = pd.DataFrame(X, columns=FEATURES)

    # ── 1. Theory-consistency table (H3) ────────────────────────────────
    print(f"\n[1/6] Theory-consistency table (top-10 {best_label} SHAP)")
    theory = build_theory_consistency_table(best_sv, X_df, top_n=10,
                                            write_table=False)
    theory.insert(1, "model", best_label)
    save_table(
        theory, RUN_TABLES_SHAP / "theory_consistency",
        caption=(
            "Pre-specified theory-consistency validation (H$_3$) under the "
            "primary specification: theoretical sign prediction vs observed "
            "mean SHAP direction for the ten highest-importance features of "
            f"{best_label} (the co-leading ML model reported for explanation; "
            "full test sample). The headline verdict is the pre-specified raw "
            "sign test; features collinear ($|r|>0.7$) with another top-10 "
            "feature are flagged (is\\_collinear) and additionally shown under "
            "a conservative symmetric collinearity annotation that is never "
            "used to inflate the consistency count."),
        label="tab:v2_theory_consistency",
    )
    print(theory[["rank", "feature", "theoretical_sign", "observed_sign",
                  "verdict"]].to_string(index=False))

    # ── 2. SHAP vs LR-coefficient concordance ───────────────────────────
    print("\n[2/6] Kendall-tau concordance (SHAP vs LR standardised coefficients)")
    lr = models["logistic_regression"]
    pipe = lr._base if hasattr(lr, "_base") else lr
    coefs = np.abs(pipe.named_steps["clf"].coef_.ravel())
    lr_sv = compute_shap_values(lr, X, model_name="logistic_regression")
    rows = []
    for tag, sv in (("XGBoost", xgb_sv), ("Logistic Regression", lr_sv)):
        imp = np.abs(sv).mean(axis=0)
        tau, p = stats.kendalltau(imp, coefs)
        rows.append({"shap_model": tag,
                     "vs": "LR standardised |coefficient|",
                     "kendall_tau": round(float(tau), 3),
                     "p_value": round(float(p), 4)})
    conc = pd.DataFrame(rows)
    save_table(
        conc, RUN_TABLES_SHAP / "shap_lr_concordance",
        caption=(
            "Kendall's $\\tau$ concordance between mean-$|$SHAP$|$ feature "
            "rankings and the ridge logistic regression's standardised "
            "coefficient magnitudes (18 predictors, primary "
            "specification). High concordance indicates the models exploit "
            "the same economic signals."),
        label="tab:v2_shap_lr_concordance",
    )
    print(conc.to_string(index=False))

    # ── 3. Temporal (two-way) bootstrap ─────────────────────────────────
    print("\n[3/6] Temporal two-way bootstrap (firm / year / firm x year)")
    rng = np.random.default_rng(RANDOM_SEED)
    fmap, ymap = group_index_map(firms), group_index_map(years)
    n = len(y)
    pairs = [("logistic_regression", m) for m in MODELS[1:]]
    dists = {}
    for scheme, sampler in (
        ("firm", lambda r: resample_cluster_indices(fmap, r)),
        ("year", lambda r: resample_cluster_indices(ymap, r)),
        ("iid",  lambda r: resample_iid_indices(n, r)),
    ):
        dists[scheme] = _metric_distribution(y, probs, pairs, sampler,
                                             BOOTSTRAP_REPS, rng)
    ci_rows, sig_rows = [], []
    for name in MODELS:
        obs = average_precision_score(y, probs[name])
        var = {s: float(np.var(dists[s][0][name])) for s in dists}
        se2 = np.sqrt(twoway_variance(var["firm"], var["year"], var["iid"]))
        ci_rows.append({
            "model": name, "pr_auc": round(obs, 4),
            "ci_firm": str(tuple(round(v, 4) for v in _percentile_ci(dists["firm"][0][name]))),
            "ci_year": str(tuple(round(v, 4) for v in _percentile_ci(dists["year"][0][name]))),
            "ci_twoway_normal": f"({obs - 1.96 * se2:.4f}, {obs + 1.96 * se2:.4f})",
            "se_firm": round(float(np.std(dists["firm"][0][name])), 4),
            "se_twoway": round(float(se2), 4),
        })
    for a, b in pairs:
        d_obs = (average_precision_score(y, probs[a])
                 - average_precision_score(y, probs[b]))
        var = {s: float(np.var(dists[s][1][(a, b)])) for s in dists}
        se2 = np.sqrt(twoway_variance(var["firm"], var["year"], var["iid"]))
        z = d_obs / se2 if se2 > 0 else np.nan
        sig_rows.append({
            "comparison": f"{a} - {b}", "delta_pr_auc": round(d_obs, 4),
            "p_firm_block": round(_recentred_p(dists["firm"][1][(a, b)], d_obs), 4),
            "p_twoway_normal": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 4),
        })
    tci, tsig = pd.DataFrame(ci_rows), pd.DataFrame(sig_rows)
    save_table(tci, RUN_TABLES_ROBUSTNESS / "temporal_bootstrap_ci",
               caption=("PR-AUC confidence intervals under firm-cluster, "
                        "year-cluster, and two-way (Cameron--Gelbach--Miller) "
                        "resampling, primary specification (9 test "
                        "years, 2015--2023)."),
               label="tab:v2_temporal_ci")
    save_table(tsig, RUN_TABLES_ROBUSTNESS / "temporal_bootstrap_significance",
               caption=("Pairwise PR-AUC differences vs the logistic "
                        "benchmark under firm-block vs temporal-robust "
                        "two-way inference. Few year clusters (9) make "
                        "the temporal CIs coarse — disclosed upper bound."),
               label="tab:v2_temporal_sig")
    print(tci.to_string(index=False))
    print(tsig.to_string(index=False))

    # ── 4. Subperiod stability ──────────────────────────────────────────
    print("\n[4/6] Subperiod stability (2015-2019 vs 2020-2023)")
    full_baseline = float(y.mean())
    sub_rows = []
    for name in MODELS:
        full = average_precision_score(y, probs[name])
        full_lift = full / full_baseline if full_baseline > 0 else float("nan")
        for lo, hi in ((2015, 2019), (2020, 2023)):
            m = (years >= lo) & (years <= hi)
            if y[m].sum() == 0:
                continue
            sub = average_precision_score(y[m], probs[name][m])
            base = float(y[m].mean())
            lift = sub / base if base > 0 else float("nan")
            sub_rows.append({
                "model": name, "period": f"{lo}-{hi}",
                "n": int(m.sum()), "events": int(y[m].sum()),
                "baseline_pr_auc": round(base, 4),
                "pr_auc": round(sub, 4),
                "lift": round(lift, 2),
                "roc_auc": round(roc_auc_score(y[m], probs[name][m]), 4),
                "pr_auc_full": round(full, 4),
                "lift_full": round(full_lift, 2),
                # The raw-PR-AUC flag is confounded by the subperiod prevalence
                # shift (pre-COVID prevalence is materially lower); the headline
                # stability flag normalises by the subperiod baseline (lift).
                "stable_raw_0.7x": bool(sub >= 0.7 * full),
                "stable_lift_0.7x": bool(lift >= 0.7 * full_lift),
            })
    subp = pd.DataFrame(sub_rows)
    save_table(subp, RUN_TABLES_MODEL / "subperiod_performance",
               caption=("Subperiod stability under the primary specification. "
                        "PR-AUC is bounded below by prevalence, and the "
                        "pre-COVID window (2015--2019) has materially lower "
                        "prevalence than 2020--2023, so a raw-PR-AUC "
                        "comparison across subperiods conflates model decay "
                        "with the prevalence shift. The headline flag "
                        "therefore normalises by the subperiod baseline "
                        "(lift $=$ PR-AUC $/$ prevalence): a model is flagged "
                        "unstable when its subperiod lift falls below 70\\% of "
                        "its full-test lift. The raw-PR-AUC flag "
                        "(\\texttt{stable\\_raw\\_0.7x}) and the "
                        "prevalence-invariant ROC-AUC are shown alongside."),
               label="tab:v2_subperiod")
    print(subp.to_string(index=False))

    # ── 5. Predictor summary stats + correlation matrix ─────────────────
    print("\n[5/6] Predictor summary statistics + correlation matrix (v2 sample)")
    panel = pd.concat([train, pd.read_parquet(DATA_SAMPLES_V2 / "val.parquet"),
                       test], ignore_index=True)
    n_panel = len(panel)
    stat_rows = []
    g1 = panel[panel["distress"] == 1]
    g0 = panel[panel["distress"] == 0]
    for f in FEATURES:
        a, b = g1[f].dropna().astype(float), g0[f].dropna().astype(float)
        t, p = stats.ttest_ind(a, b, equal_var=False)
        stat_rows.append({
            "feature": f,
            "mean_distressed": round(a.mean(), 4), "mean_healthy": round(b.mean(), 4),
            "median_distressed": round(a.median(), 4), "median_healthy": round(b.median(), 4),
            "std_distressed": round(a.std(), 4), "std_healthy": round(b.std(), 4),
            "p10_all": round(panel[f].quantile(.1), 4), "p90_all": round(panel[f].quantile(.9), 4),
            "welch_t": round(float(t), 2), "p_value": round(float(p), 4),
        })
    sumstats = pd.DataFrame(stat_rows)
    save_table(sumstats, RUN_TABLES_DESCRIPTIVE / "predictor_summary_stats",
               caption=("Predictor summary statistics by outcome group, "
                        f"primary modeling sample ({n_panel:,} firm-years; "
                        "18 predictors incl. MB\\_MISSING). Welch $t$ for the "
                        "between-group mean difference."),
               label="tab:predictor_stats")
    corr = panel[FEATURES].corr()
    RUN_TABLES_DESCRIPTIVE.mkdir(parents=True, exist_ok=True)
    corr.round(3).to_csv(RUN_TABLES_DESCRIPTIVE / "correlation_matrix.csv")
    high = [(FEATURES[i], FEATURES[j], round(corr.iloc[i, j], 3))
            for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))
            if abs(corr.iloc[i, j]) > 0.7]
    pd.DataFrame(high, columns=["feature_a", "feature_b", "r"]).to_csv(
        RUN_TABLES_DESCRIPTIVE / "correlation_high_pairs.csv", index=False)
    print(f"  correlation matrix saved; |r|>0.7 pairs: {high}")

    # ── 6. SHAP stability (firm-block bootstrap of the SHAP matrix) ─────
    print(f"\n[6/6] SHAP stability (500 firm-block resamples; {best_label})")
    rng = np.random.default_rng(RANDOM_SEED)
    base_rank = pd.Series(np.abs(best_sv).mean(axis=0), index=FEATURES) \
        .rank(ascending=False).astype(int)
    top10 = base_rank[base_rank <= 10].sort_values().index.tolist()
    ranks = {f: [] for f in top10}
    for _ in range(500):
        idx = resample_cluster_indices(fmap, rng)
        r = pd.Series(np.abs(best_sv[idx]).mean(axis=0), index=FEATURES) \
            .rank(ascending=False)
        for f in top10:
            ranks[f].append(r[f])
    stab = pd.DataFrame([{
        "model": best_label, "feature": f,
        "rank_full_test": int(base_rank[f]),
        "rank_ci_2.5": float(np.percentile(ranks[f], 2.5)),
        "rank_ci_97.5": float(np.percentile(ranks[f], 97.5)),
        "rank_sd": round(float(np.std(ranks[f])), 2),
    } for f in top10])
    save_table(stab, RUN_TABLES_SHAP / "shap_stability",
               caption=("SHAP importance-rank stability under firm-block "
                        "bootstrap of the test sample (500 resamples of "
                        f"the precomputed {best_label} SHAP matrix): 95\\% rank "
                        "intervals for the ten highest-importance features."),
               label="tab:v2_shap_stability")
    print(stab.to_string(index=False))

    print("\nV2 parity batch complete.")


if __name__ == "__main__":
    main()
