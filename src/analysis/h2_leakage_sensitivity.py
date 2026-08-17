"""
Leakage Sensitivity Analysis — chronological vs random split designs.
=====================================================================
**Thesis mapping (2026-08-12). The hypothesis this module was built to test
has been WITHDRAWN. This module has NOT.**

The hypothesis fixed in design §3.2 read:

  H2: The ML advantage documented in H1 is *smaller* than comparable
  estimates from studies using random train-test splits, reflecting
  design optimism in the prior literature.

It is a conditional on an ML advantage that H1 does not produce — the best
machine-learning model finishes *below* the benchmark — so its antecedent is
false, it is vacuously true, and it is not testable on this sample. The thesis
withdraws it rather than restating it around the quantity that survived, because
rewriting a pre-registered hypothesis into the shape of the surviving result is
the design optimism the thesis documents in others.

**What this module therefore is:** a designed comparison of split regimes,
reported on its own terms (thesis Section ``sec:h2_results``). Its central
output — the advantage differential, i.e. how far the ML-over-benchmark margin
moves between chronological and random splits — remains valid and is a headline
result. Only its status as a *hypothesis test* is retracted.

Read every "tests H2" / "H2 estimand" below as "the split-design experiment".
The module name, table stems and section labels keep the frozen ``h2`` spelling
deliberately: they are provenance identifiers recorded in the run manifest and
checked by ``verify_final_outputs.py``. No behaviour depends on the label.

Why this matters
----------------
Studies that use random (non-chronological) splits allow future
observations to enter the training set. Even a firm-level block random
split (where all observations from a firm are either in train or test)
suffers from "time contamination": a firm sampled into training may have
FY2020 observations, teaching the model patterns that are not yet
observable at the moment of prediction for FY2005 test observations.

Experiment
----------
Both designs use IDENTICAL models (same hyperparameter search space,
same rolling-origin CV objective, same Optuna sampler seed). Only the
DATA SPLIT differs:

  Chronological (primary):
      Train : 1990–2009 (20 years)
      Val   : 2010–2014 (threshold selection)
      Test  : 2015–2024 (evaluation)

  Random (leaky comparison):
      Firm-level block split — 80 % of unique firms → train,
      20 % → test. No chronological ordering.
      This is the MOST FAVOURABLE version of the random split
      (firm-level blocks avoid leaking a single firm's series
      across folds). Even this conservative version is expected
      to inflate PR-AUC relative to the chronological design.

  Observation-level random split:
      Individual firm-years randomly assigned 80/20.
      Most extreme form of leakage (the same firm appears in
      train and test in different years). Included to bound the
      full range of design optimism.

Key output
----------
A table comparing PR-AUC (and optionally AUC-ROC) across the three
designs. The difference  Δ = PR-AUC(random) − PR-AUC(chronological)
quantifies the magnitude of design optimism and directly tests H2.

Design reference: §3.2, §6.3, §10.1 (H2)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    ALL_FEATURES,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    RANDOM_SEED,
    OPTUNA_TRIALS,
    OUT_TABLES_MODEL,
    OUT_TABLES_ROBUSTNESS,
)
from src.models.tune import tune_model
from src.models.train import build_model_with_params
from src.models.evaluate import select_threshold, compute_all_metrics
from src.utils.tables import save_table

LABEL_COL  = "distress"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def chronological_split(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Primary specification split (identical to build_features.split_panel).

    Train : fyear <= TRAIN_END_YEAR (1990–2009)
    Val   : TRAIN_END_YEAR < fyear <= VAL_END_YEAR (2010–2014)
    Test  : fyear > VAL_END_YEAR (2015–2024)
    """
    train = panel[panel["fyear"] <= TRAIN_END_YEAR].copy()
    val   = panel[(panel["fyear"] > TRAIN_END_YEAR) & (panel["fyear"] <= VAL_END_YEAR)].copy()
    test  = panel[panel["fyear"] > VAL_END_YEAR].copy()
    return train, val, test


def firm_level_random_split(
    panel: pd.DataFrame,
    train_frac: float = 0.80,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Firm-level block random split — most conservative (least leaky) random design.

    All observations from a given firm go entirely into train or test.
    Avoids splitting a single firm's time-series across folds, but
    DOES allow future information into training (e.g., FY2020 obs
    from a "train firm" inform predictions for FY2005 "test firms").

    Parameters
    ----------
    panel : pd.DataFrame
    train_frac : float
        Proportion of unique firms assigned to training.

    Returns
    -------
    train_r, test_r : pd.DataFrame
    """
    rng          = np.random.default_rng(seed=seed)
    unique_firms = panel["gvkey"].unique()
    n_train      = int(train_frac * len(unique_firms))
    train_firms  = set(rng.choice(unique_firms, size=n_train, replace=False))

    train_r = panel[panel["gvkey"].isin(train_firms)].copy()
    test_r  = panel[~panel["gvkey"].isin(train_firms)].copy()
    return train_r, test_r


def observation_level_random_split(
    panel: pd.DataFrame,
    train_frac: float = 0.80,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Observation-level random split — most leaky design.

    Each firm-year is randomly assigned to train or test independently.
    This is the design used in many early ML-in-finance papers that
    failed to account for the panel structure. The same firm appears
    in both train and test in different years, creating severe leakage.

    Parameters
    ----------
    panel : pd.DataFrame
    train_frac : float

    Returns
    -------
    train_o, test_o : pd.DataFrame
    """
    rng   = np.random.default_rng(seed=seed)
    mask  = rng.random(len(panel)) < train_frac
    return panel[mask].copy(), panel[~mask].copy()


def chronological_test_contaminated_train(
    panel: pd.DataFrame,
    train_frac: float = 0.80,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hybrid design that ISOLATES the leakage effect from the prevalence effect.

    The original H2 comparison (firm_random vs chronological) confounds two
    distinct sources of design optimism:
      (a) Time leakage — training on firm-years that postdate the test
          period (e.g., FY2020 obs of a "train firm" help predict FY2005).
      (b) Population shift — random splits draw test obs from the full
                    1990-2024 panel, which has higher unconditional distress
          prevalence (~2.4%) than the chronological 2015-2024 test
          (~1.42%). PR-AUC is not directly comparable across populations
          with different prevalence.

    This design controls for (b):

      Test sample  : identical to primary chronological design (fyear > VAL_END_YEAR)
      Train pool   : randomly drawn firm-years from firms NOT in the test
                     set, across ALL years (1990-2024). The training data
                     therefore contains post-VAL_END_YEAR firm-years
                     (the "leakage"), but the test sample's distress
                     prevalence is held constant.
      Validation   : 20% random subset of the training pool.

    Any PR-AUC difference vs the chronological baseline under this design
    is attributable to (a) time leakage alone, not to population shift.

    Parameters
    ----------
    panel : pd.DataFrame
    train_frac : float
        Proportion of non-test firms assigned to the training pool.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    rng = np.random.default_rng(seed=seed)

    # Test: identical to the primary chronological design
    test = panel[panel["fyear"] > VAL_END_YEAR].copy()
    test_firms = set(test["gvkey"].unique())

    # Eligible train firms: those NOT in test (avoids firm-level leakage;
    # this design tests TIME-leakage only)
    train_eligible_pool = panel[~panel["gvkey"].isin(test_firms)].copy()
    eligible_firms = train_eligible_pool["gvkey"].unique()

    # Sample train_frac of eligible firms
    n_train_firms  = int(train_frac * len(eligible_firms))
    chosen_firms   = set(rng.choice(eligible_firms, size=n_train_firms, replace=False))
    train_pool     = train_eligible_pool[
        train_eligible_pool["gvkey"].isin(chosen_firms)
    ].copy()

    # Carve a firm-level validation slice (20% of training firms)
    pool_firms      = list(chosen_firms)
    n_val_firms     = int(0.2 * len(pool_firms))
    val_firms_set   = set(rng.choice(pool_firms, size=n_val_firms, replace=False))
    val   = train_pool[train_pool["gvkey"].isin(val_firms_set)].copy()
    train = train_pool[~train_pool["gvkey"].isin(val_firms_set)].copy()

    return train, val, test


# ---------------------------------------------------------------------------
# Training for a single split design (no tuning — use primary best params)
# ---------------------------------------------------------------------------

def _train_and_eval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    design_label: str,
    tune: bool = False,
) -> pd.DataFrame:
    """
    Train all three models on the given split and evaluate on test.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    features : list[str]
    design_label : str
        Label for the design ('chronological', 'firm_random', 'obs_random').
    tune : bool
        If True, re-run Optuna tuning (slow). If False, use default params.

    Returns
    -------
    pd.DataFrame
        Metrics for all three models under this design.
    """
    X_train = train[features].astype(float).fillna(0).values
    y_train = train[LABEL_COL].astype(int).values
    X_val   = val[features].astype(float).fillna(0).values   if val is not None else X_train
    y_val   = val[LABEL_COL].astype(int).values  if val is not None else y_train
    X_test  = test[features].astype(float).fillna(0).values
    y_test  = test[LABEL_COL].astype(int).values
    firm_ids = test["gvkey"].values

    rows = []
    for name in MODEL_NAMES:
        if tune:
            best_params, _ = tune_model(name, train, features)
        else:
            # Use minimal default parameters (avoid re-tuning to save time)
            # Tuning would be fairer but computationally prohibitive for H2.
            # Stated as a limitation in the thesis.
            best_params = _default_params(name)

        model = build_model_with_params(name, best_params, y_train)
        model.fit(X_train, y_train)

        y_prob_val  = model.predict_proba(X_val)[:, 1]
        threshold   = select_threshold(y_val, y_prob_val)
        y_prob_test = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_prob_test, threshold, model_name=name)
        metrics["design"] = design_label
        rows.append(metrics)

        print(f"  [{design_label}] {name:25s}  "
              f"PR-AUC={metrics['pr_auc']:.4f}  AUC-ROC={metrics['roc_auc']:.4f}")

    return pd.DataFrame(rows)


def _default_params(model_name: str) -> dict:
    """
    Return fixed default hyperparameters for the H2 comparison.

    These are not tuned on the random splits (which would require
    a time-consistent inner CV — not meaningful for random splits).
    The same defaults are used across all three designs to ensure
    fair comparison. This is documented as a design choice in the thesis.
    """
    defaults = {
        "logistic_regression": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        "random_forest":       {"n_estimators": 500, "max_depth": None,
                                "min_samples_leaf": 20, "max_features": "sqrt",
                                "class_weight": "balanced"},
        "xgboost":             {"n_estimators": 500, "max_depth": 5,
                                "learning_rate": 0.05, "subsample": 0.8,
                                "colsample_bytree": 0.8, "min_child_weight": 10,
                                "reg_alpha": 0.1, "reg_lambda": 1.0},
    }
    return defaults[model_name]


# ---------------------------------------------------------------------------
# Within-design preprocessing
# ---------------------------------------------------------------------------

def preprocess_within_design(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    sic_col: str = "sich",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit winsorisation + imputation on the DESIGN's training split only and
    apply to its val/test — mirroring what the primary pipeline does at
    split level. The frozen H2 experiment consumed the raw feature panel
    (pre-winsorisation, pre-imputation) with a zero fill, which is not
    what any headline model saw.
    """
    from src.features.winsorize import apply_thresholds, compute_thresholds
    from src.features.impute import apply_imputation, compute_imputation_medians

    thr = compute_thresholds(train)
    train = apply_thresholds(train, thr)
    val = apply_thresholds(val, thr)
    test = apply_thresholds(test, thr)

    med = compute_imputation_medians(train, sic_col=sic_col)
    train = apply_imputation(train, *med, sic_col=sic_col)
    val = apply_imputation(val, *med, sic_col=sic_col)
    test = apply_imputation(test, *med, sic_col=sic_col)
    return train, val, test


# ---------------------------------------------------------------------------
# Advantage differential (the H2 estimand)
# ---------------------------------------------------------------------------

def compute_advantage_differential(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the statistic H2 actually asserts something about: the
    ML-over-LR ADVANTAGE per design, and its change relative to the
    chronological design.

    For each tree model M and design d:
        adv(M, d)       = PR-AUC(M, d) − PR-AUC(LR, d)
        lift_adv(M, d)  = lift(M, d) − lift(LR, d)      [prevalence-controlled]
        Δadv(M, d)      = adv(M, d) − adv(M, chronological)

    H2 predicts Δadv > 0: random splits flatter the ML models MORE than
    the linear benchmark. The per-model Δ of compute_design_optimism()
    conflates this with design effects common to all models.

    When `results` carries a 'seed' column with several seeds, the random
    designs are aggregated as mean ± sd across seeds (the chronological
    baseline is deterministic).
    """
    df = results.copy()
    if "seed" not in df.columns:
        df["seed"] = 0
    df["lift"] = df["pr_auc"] / df["prevalence_baseline_pr_auc"]

    designs = [d for d in df["design"].unique() if d != "chronological"]
    lr = df[df["model"] == "logistic_regression"].set_index(["design", "seed"])

    rows = []
    for ml in ("random_forest", "xgboost"):
        mdf = df[df["model"] == ml].set_index(["design", "seed"])
        adv = (mdf["pr_auc"] - lr["pr_auc"]).rename("adv")
        lift_adv = (mdf["lift"] - lr["lift"]).rename("lift_adv")

        # Chronological baseline (deterministic; take the first seed present)
        chrono_keys = [k for k in adv.index if k[0] == "chronological"]
        adv_c = float(adv.loc[chrono_keys[0]])
        lift_adv_c = float(lift_adv.loc[chrono_keys[0]])

        row = {
            "ml_model": ml,
            "adv_chronological": round(adv_c, 4),
            "lift_adv_chronological": round(lift_adv_c, 2),
        }
        for d in designs:
            d_adv = adv.loc[d] if d in adv.index.get_level_values(0) else None
            if d_adv is None:
                continue
            d_lift = lift_adv.loc[d]
            row[f"adv_{d}"] = round(float(d_adv.mean()), 4)
            row[f"delta_adv_{d}"] = round(float(d_adv.mean()) - adv_c, 4)
            row[f"lift_adv_{d}"] = round(float(d_lift.mean()), 2)
            row[f"delta_lift_adv_{d}"] = round(float(d_lift.mean()) - lift_adv_c, 2)
            if d_adv.shape[0] > 1:
                row[f"sd_adv_{d}"] = round(float(d_adv.std(ddof=1)), 4)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\nAdvantage differential (H2 estimand — Δ of [ML − LR] advantage "
          "vs chronological):")
    print(table.to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Compute delta (design optimism)
# ---------------------------------------------------------------------------

def compute_design_optimism(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δ PR-AUC = PR-AUC(random) − PR-AUC(chronological) per model.

    Positive Δ = design optimism (random split inflates performance).
    This directly operationalises H2.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_leakage_comparison(). Must have columns
        'design', 'model', 'pr_auc', 'roc_auc'.

    Returns
    -------
    pd.DataFrame
        Columns: model, pr_auc_chrono, pr_auc_firm_random,
                 pr_auc_obs_random, delta_firm_random, delta_obs_random.
    """
    # "mean" aggregates across seeds when the multi-seed replication is
    # used (identical to the old "first" for the single-seed frozen run).
    pivot = results.pivot_table(
        index="model",
        columns="design",
        values=["pr_auc", "roc_auc", "prevalence_baseline_pr_auc"],
        aggfunc="mean",
    )

    def _safe_lift(pr, base):
        return round(pr / base, 2) if base and base > 0 else float("nan")

    rows = []
    for model in MODEL_NAMES:
        if model not in pivot.index:
            continue
        pr_c   = pivot.loc[model, ("pr_auc", "chronological")]
        pr_fr  = pivot.loc[model, ("pr_auc", "firm_random")]
        pr_or  = pivot.loc[model, ("pr_auc", "obs_random")]
        roc_c  = pivot.loc[model, ("roc_auc", "chronological")]
        roc_fr = pivot.loc[model, ("roc_auc", "firm_random")]
        roc_or = pivot.loc[model, ("roc_auc", "obs_random")]

        # Design 4 may be absent if older results are read back from disk
        try:
            pr_c2  = pivot.loc[model, ("pr_auc", "chronological_leaky_train")]
            roc_c2 = pivot.loc[model, ("roc_auc", "chronological_leaky_train")]
        except KeyError:
            pr_c2, roc_c2 = float("nan"), float("nan")

        # Baseline prevalences (needed for lift)
        base_c   = pivot.loc[model, ("prevalence_baseline_pr_auc", "chronological")]
        base_fr  = pivot.loc[model, ("prevalence_baseline_pr_auc", "firm_random")]
        base_or  = pivot.loc[model, ("prevalence_baseline_pr_auc", "obs_random")]
        try:
            base_c2 = pivot.loc[model, ("prevalence_baseline_pr_auc", "chronological_leaky_train")]
        except KeyError:
            base_c2 = float("nan")

        rows.append({
            "model":               model,
            # PR-AUC across all 4 designs
            "pr_auc_chrono":       round(pr_c,  4),
            "pr_auc_firm_random":  round(pr_fr, 4),
            "pr_auc_obs_random":   round(pr_or, 4),
            "pr_auc_leaky_train":  round(pr_c2, 4) if pr_c2 == pr_c2 else float("nan"),
            # Δ vs primary chronological (raw PR-AUC)
            "delta_pr_firm_rand":  round(pr_fr - pr_c, 4),
            "delta_pr_obs_rand":   round(pr_or - pr_c, 4),
            "delta_pr_leaky_train":(round(pr_c2 - pr_c, 4) if pr_c2 == pr_c2
                                    else float("nan")),
            # Lift = PR-AUC / baseline_prevalence (cross-design comparable)
            "lift_chrono":         _safe_lift(pr_c,  base_c),
            "lift_firm_random":    _safe_lift(pr_fr, base_fr),
            "lift_obs_random":     _safe_lift(pr_or, base_or),
            "lift_leaky_train":    _safe_lift(pr_c2, base_c2),
            # ROC-AUC across all 4 designs
            "roc_auc_chrono":       round(roc_c,  4),
            "roc_auc_firm_random":  round(roc_fr, 4),
            "roc_auc_obs_random":   round(roc_or, 4),
            "roc_auc_leaky_train":  (round(roc_c2, 4) if roc_c2 == roc_c2
                                     else float("nan")),
            "delta_roc_firm_rand":  round(roc_fr - roc_c, 4),
            "delta_roc_obs_rand":   round(roc_or - roc_c, 4),
            "delta_roc_leaky_train":(round(roc_c2 - roc_c, 4) if roc_c2 == roc_c2
                                     else float("nan")),
        })

    optimism_table = pd.DataFrame(rows)
    print("\nDesign optimism table (Δ = design − chronological):")
    print(optimism_table[
        ["model", "pr_auc_chrono", "pr_auc_firm_random", "pr_auc_obs_random",
         "pr_auc_leaky_train", "delta_pr_firm_rand", "delta_pr_obs_rand",
         "delta_pr_leaky_train"]
    ].to_string(index=False))
    print("\nLift table (PR-AUC / baseline_prevalence — comparable across designs):")
    print(optimism_table[
        ["model", "lift_chrono", "lift_firm_random", "lift_obs_random",
         "lift_leaky_train"]
    ].to_string(index=False))
    return optimism_table


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_leakage_comparison(
    panel: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_seeds: int = 1,
    within_design_preprocessing: bool = False,
    sic_col: str = "sich",
    write_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full H2 leakage sensitivity experiment.

    Compares PR-AUC and AUC-ROC across four split designs:
      1. Chronological (primary specification)
      2. Firm-level block random split (conservative leaky design)
      3. Observation-level random split (extreme leaky design)
      4. Chronological test + contaminated train (prevalence-controlled)

    Parameters
    ----------
    panel : pd.DataFrame
        FULL feature panel.
    features : list[str]
    n_seeds : int, default 1
        Number of independent replications of the RANDOM designs (2-4).
        1 (frozen) reproduces the original single-draw experiment; the
        corrected regeneration uses several seeds so the advantage
        deltas carry sampling uncertainty. The chronological design is
        deterministic and runs once.
    within_design_preprocessing : bool, default False
        False (frozen) consumes the panel as passed — historically the
        RAW pre-winsorisation/pre-imputation panel with a zero fill.
        True fits winsorisation + imputation on each design's training
        split (preprocess_within_design), matching what the headline
        models actually saw.
    sic_col : str
        Imputation SIC column when within_design_preprocessing=True.

    Returns
    -------
    results : pd.DataFrame
        All metrics per model × design (× seed).
    optimism_table : pd.DataFrame
        Δ PR-AUC and Δ AUC-ROC per model (means across seeds).
        The advantage-differential table (the H2 estimand) is saved to
        h2_advantage_differential.{csv,tex} as a side effect.
    """
    print("\n" + "="*60)
    print("  H2 LEAKAGE SENSITIVITY ANALYSIS")
    print("="*60)
    print(
        "  Comparing chronological vs random split designs.\n"
        f"  Seeds: {n_seeds} | within-design preprocessing: "
        f"{within_design_preprocessing}"
    )

    def _prep(tr, va, te):
        if within_design_preprocessing:
            return preprocess_within_design(tr, va, te, sic_col=sic_col)
        return tr, va, te

    all_results = []

    # ── 1. Chronological (primary specification) — deterministic ──────────
    print("\n--- Design 1: Chronological split (primary specification) ---")
    train_c, val_c, test_c = _prep(*chronological_split(panel))
    print(f"  Train: {len(train_c):,}  |  Val: {len(val_c):,}  |  Test: {len(test_c):,}")
    res_c = _train_and_eval(train_c, val_c, test_c, features, "chronological")
    res_c["seed"] = 0
    all_results.append(res_c)

    for k in range(n_seeds):
        # k=0 uses the original seeds so the frozen single-draw run is
        # reproduced exactly; further seeds are independent draws.
        base = RANDOM_SEED + 1000 * k
        seed_note = f" [seed draw {k}]" if n_seeds > 1 else ""

        # ── 2. Firm-level block random split ──────────────────────────────
        print(f"\n--- Design 2: Firm-level block random split (80/20){seed_note} ---")
        train_fr, test_fr = firm_level_random_split(panel, seed=base)
        # No separate validation split — use a random 20% of train as pseudo-val
        rng = np.random.default_rng(seed=base + 1)
        train_firms_fr = train_fr["gvkey"].unique()
        n_val_f = int(0.2 * len(train_firms_fr))
        val_firms_fr  = set(rng.choice(train_firms_fr, size=n_val_f, replace=False))
        val_fr   = train_fr[train_fr["gvkey"].isin(val_firms_fr)].copy()
        train_fr = train_fr[~train_fr["gvkey"].isin(val_firms_fr)].copy()
        train_fr, val_fr, test_fr = _prep(train_fr, val_fr, test_fr)
        print(f"  Train: {len(train_fr):,}  |  Val: {len(val_fr):,}  |  Test: {len(test_fr):,}")
        res_fr = _train_and_eval(train_fr, val_fr, test_fr, features, "firm_random")
        res_fr["seed"] = k
        all_results.append(res_fr)

        # ── 3. Observation-level random split ─────────────────────────────
        print(f"\n--- Design 3: Observation-level random split (80/20){seed_note} ---")
        train_or_full, test_or = observation_level_random_split(panel, seed=base)
        # Pseudo-validation: 20% of the training obs
        rng2 = np.random.default_rng(seed=base + 2)
        val_mask_or = rng2.random(len(train_or_full)) < 0.2
        val_or   = train_or_full[val_mask_or].copy()
        train_or = train_or_full[~val_mask_or].copy()
        train_or, val_or, test_or = _prep(train_or, val_or, test_or)
        print(f"  Train: {len(train_or):,}  |  Val: {len(val_or):,}  |  Test: {len(test_or):,}")
        res_or = _train_and_eval(train_or, val_or, test_or, features, "obs_random")
        res_or["seed"] = k
        all_results.append(res_or)

        # ── 4. Chronological test + contaminated train ─────────────────────
        print(f"\n--- Design 4: Chronological test + contaminated train{seed_note} ---")
        train_c2, val_c2, test_c2 = chronological_test_contaminated_train(
            panel, seed=base)
        train_c2, val_c2, test_c2 = _prep(train_c2, val_c2, test_c2)
        print(f"  Train: {len(train_c2):,}  |  Val: {len(val_c2):,}  |  Test: {len(test_c2):,}")
        print(f"  Train fyear range: {int(train_c2['fyear'].min())}–{int(train_c2['fyear'].max())}")
        print(f"  Test  fyear range: {int(test_c2['fyear'].min())}–{int(test_c2['fyear'].max())}"
              f"  (matches primary chronological)")
        res_c2 = _train_and_eval(train_c2, val_c2, test_c2, features,
                                 "chronological_leaky_train")
        res_c2["seed"] = k
        all_results.append(res_c2)

    # ── Consolidate ───────────────────────────────────────────────────────
    results = pd.concat(all_results, ignore_index=True)
    optimism_table = compute_design_optimism(results)
    advantage_table = compute_advantage_differential(results)

    # ── Save ──────────────────────────────────────────────────────────────
    # write_tables=False lets an alternative-specification caller (the v2
    # robustness run) save its own prefixed copies WITHOUT overwriting the
    # frozen v1 h2_* tables (2026-07-14: a v2 run silently rewrote them;
    # restored from git).
    if not write_tables:
        print("\n  H2 analysis complete (write_tables=False — caller saves).")
        return results, optimism_table

    OUT_TABLES_ROBUSTNESS.mkdir(parents=True, exist_ok=True)
    save_table(
        results,
        OUT_TABLES_ROBUSTNESS / "h2_leakage_all_results",
        caption=(
            "H2 Leakage Sensitivity: Performance by Split Design. "
            "All models are retrained under a unified protocol within this "
            "experiment so the four split designs are directly comparable; "
            "the chronological rows therefore differ slightly from the "
            "primary specification table, which remains the canonical result."
        ),
        label="tab:h2_leakage",
    )
    save_table(
        optimism_table,
        OUT_TABLES_ROBUSTNESS / "h2_design_optimism",
        caption=(
            "Design Optimism: $\\Delta$ PR-AUC and $\\Delta$ AUC-ROC "
            "(Random $-$ Chronological Split)"
        ),
        label="tab:h2_optimism",
    )
    save_table(
        advantage_table,
        OUT_TABLES_ROBUSTNESS / "h2_advantage_differential",
        caption=(
            "H2 Advantage Differential: change in the ML-over-LR PR-AUC "
            "advantage (raw and prevalence-controlled lift units) under "
            "leaky split designs relative to the chronological design. "
            "This is the statistic H2 asserts something about; positive "
            "values indicate that leaky designs flatter the ML models "
            "more than the linear benchmark."
        ),
        label="tab:h2_advantage",
    )

    print(f"\n  H2 analysis complete. Tables saved → {OUT_TABLES_ROBUSTNESS}")
    return results, optimism_table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.config import DATA_SAMPLES, DATA_FEATURES

    # NOTE: features_all.parquet is the PRE-winsorisation, PRE-imputation
    # panel (build_features saves it before stage 6/7) — the frozen H2 run
    # consumed it as-is with a zero fill. Pass
    # within_design_preprocessing=True (corrected regeneration) to fit
    # winsorisation/imputation per design instead.
    panel = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    results, optimism = run_leakage_comparison(panel)
