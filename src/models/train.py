"""
Model training pipeline — orchestrates tuning, fitting, and evaluation.
========================================================================
Implements the full training protocol specified in the Pre-Specified Empirical
Design (§9):

  1. Load train / val / test splits from data/processed/samples/
  2. For each of the 3 models (LogReg, RF, XGBoost):
     a. Tune hyperparameters via Optuna on rolling-origin CV (train only)
     b. Re-fit the winning configuration on the FULL training set
     c. Select classification threshold on validation set (maximise F1)
     d. Save the fitted model to outputs/models/saved/
  3. Evaluate ALL models on the test set (ONCE, at the end)
  4. Save the results table to outputs/tables/model_results/

Critical rule: The test set is evaluated EXACTLY ONCE after all models
are trained and thresholds are locked. No design decisions may be
revisited after test-set results are seen.

Design reference: §9
"""

from __future__ import annotations

import joblib
import pandas as pd
import yaml

from src.config import (
    DATA_SAMPLES,
    DISTRESS_HORIZON_DAYS,
    OUT_MODELS_SAVED,
    OUT_MODELS_CONFIGS,
    OUT_TABLES_MODEL,
    ALL_FEATURES,
    OPTUNA_TRIALS,
    RANDOM_SEED,
    V2_PROFILE,
)
from src.models.tune import tune_model
from src.models.evaluate import compute_all_metrics, select_threshold, bootstrap_pr_auc_ci
from src.models.logistic_regression import build_logistic_regression
from src.models.random_forest import build_random_forest
from src.models.xgboost_model import build_xgboost, compute_scale_pos_weight
from src.utils.tables import save_table

LABEL_COL = "distress"
MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


# ---------------------------------------------------------------------------
# Specification-aware artifact locations (F7 fix — provenance protection)
# ---------------------------------------------------------------------------
# Historically every caller of train_all_models() — including robustness
# checks — wrote models and config YAMLs to the SAME shared primary paths,
# which is how the 2026-06-29 primary re-run silently reused a
# robustness-run config. From now on artifacts are
# namespaced by specification: spec="primary" keeps the frozen layout,
# any other spec writes to a subdirectory, and overwriting existing
# primary artifacts requires an explicit opt-in.

def resolve_artifact_dirs(spec: str):
    """
    Return (saved_dir, configs_dir) for a training specification.

    spec="primary" -> the frozen top-level dirs (byte-identical layout);
    any other spec -> outputs/models/{saved,configs}/{spec}/ subdirs.
    """
    if not spec or "/" in spec or "\\" in spec or spec == "..":
        raise ValueError(f"Invalid spec name: {spec!r}")
    if spec == "primary":
        return OUT_MODELS_SAVED, OUT_MODELS_CONFIGS
    return OUT_MODELS_SAVED / spec, OUT_MODELS_CONFIGS / spec


def _guard_primary_artifacts(saved_dir, configs_dir, spec: str,
                             allow_primary_overwrite: bool) -> None:
    """
    Refuse to overwrite existing primary model artifacts unless the caller
    explicitly opts in. Non-primary specs are never blocked (they own
    their own subdirectories).
    """
    if spec != "primary" or allow_primary_overwrite:
        return
    existing = [
        p for name in MODEL_NAMES
        for p in (saved_dir / f"{name}.joblib", configs_dir / f"{name}_config.yaml")
        if p.exists()
    ]
    if existing:
        raise RuntimeError(
            "train_all_models(spec='primary') would overwrite existing primary "
            f"artifacts ({existing[0]} …). If this is a deliberate primary "
            "re-run, pass allow_primary_overwrite=True (run_pipeline stage 3 "
            "does this). Robustness/experimental callers must instead pass a "
            "spec tag, e.g. spec='rc6', which writes to its own subdirectory."
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / validation / test splits from parquet files.

    Returns
    -------
    train, val, test : pd.DataFrame
    """
    print("Loading train / val / test splits ...")
    train = pd.read_parquet(DATA_SAMPLES / "train.parquet")
    val   = pd.read_parquet(DATA_SAMPLES / "val.parquet")
    test  = pd.read_parquet(DATA_SAMPLES / "test.parquet")
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
    return train, val, test


# ---------------------------------------------------------------------------
# Model building with tuned hyperparameters
# ---------------------------------------------------------------------------

def build_model_with_params(
    model_name: str,
    params: dict,
    y_train=None,
):
    """
    Instantiate a model using the tuned hyperparameters.

    Parameters
    ----------
    model_name : str
    params : dict
        Best hyperparameters from Optuna.
    y_train : array-like or None
        Required for XGBoost scale_pos_weight computation.

    Returns
    -------
    Fitted-ready estimator.
    """
    if model_name == "logistic_regression":
        return build_logistic_regression(**params)
    elif model_name == "random_forest":
        return build_random_forest(**params)
    elif model_name == "xgboost":
        spw = compute_scale_pos_weight(y_train) if y_train is not None else 1.0
        return build_xgboost(**params, scale_pos_weight=spw)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train_all_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    n_trials: int | None = None,
    reuse_configs: bool = False,
    spec: str = "primary",
    allow_primary_overwrite: bool = False,
    tune_raw_train: pd.DataFrame | None = None,
    tune_fold_safe: bool = False,
    tune_purge_horizon_days: int | None = None,
    tune_sic_col: str = "sich",
    tune_peer_rule: str = "rows",
    tune_storage_dir=None,
    skip_if_saved: bool = False,
    tune_impute_features: list[str] | None = None,
    samples_dir=None,
    manifest_input_files=None,
    manifest_extra: dict | None = None,
) -> dict:
    """
    Tune, train, and threshold-select all three models.

    Parameters
    ----------
    train : pd.DataFrame
    val : pd.DataFrame
    features : list[str]
    n_trials : int or None
        Override Optuna n_trials (None = config default).
    reuse_configs : bool
        If True, load best_params from the spec's config directory
        (`{configs_dir}/{name}_config.yaml`) and skip the Optuna search.
        Used to re-fit existing tuned models on a corrected sample
        without paying the ~2-hour re-tune cost. Threshold and model
        fit are still recomputed from scratch.
    spec : str
        Artifact namespace. "primary" (default) writes to the frozen
        top-level paths; any other value ("rc1" … "rc6", "v2", …) writes
        to outputs/models/{saved,configs}/{spec}/ so no experimental run
        can ever overwrite the primary artifacts again.
    allow_primary_overwrite : bool
        Required opt-in for spec="primary" when primary artifacts
        already exist on disk. run_pipeline stage 3 — the designated
        owner of the primary artifacts — passes True.
    tune_raw_train : pd.DataFrame or None
        RAW (pre-winsorisation, pre-imputation) training frame used for
        hyperparameter tuning when fold-safe CV is requested. The final
        model is still fitted on the (outer-preprocessed) `train`.
        None (frozen) tunes on `train` itself.
    tune_fold_safe : bool
        Re-fit winsorisation/imputation inside each CV fold during
        tuning (v2 profile; src/models/fold_safe_cv.py).
    tune_purge_horizon_days : int or None
        Label-maturity purging horizon for the tuning CV (v2 passes
        DISTRESS_HORIZON_DAYS; None = frozen, no purging).
    tune_sic_col, tune_peer_rule : str
        In-fold imputation options for fold-safe tuning.
    tune_impute_features : list[str] or None
        Features routed through the in-fold imputation hierarchy when
        fold-safe tuning is active (None = frozen accounting-only; the
        v2 profile passes ACCOUNTING + EXRET/SIGMA/MB — §8).
    samples_dir : Path or None
        Directory whose train/val/test parquets the run manifest should
        hash. None = the frozen primary DATA_SAMPLES. The v2 rebuild
        MUST pass DATA_SAMPLES_V2, otherwise the manifest records v1
        data hashes for a v2 run (2026-07-12 second-audit fix).
    manifest_input_files : iterable[Path] or None
        Additional immutable inputs whose hashes are required to reproduce
        the run (for example the delisting extract used by an RC relabel).
    manifest_extra : dict or None
        Specification-specific provenance fields merged into the standard
        training metadata.

    Returns
    -------
    dict
        {model_name: {'model': fitted_model, 'threshold': float, 'best_params': dict}}
    """
    saved_dir, configs_dir = resolve_artifact_dirs(spec)
    _guard_primary_artifacts(saved_dir, configs_dir, spec, allow_primary_overwrite)

    X_train = train[features].astype(float).fillna(0).values
    y_train = train[LABEL_COL].astype(int).values
    X_val   = val[features].astype(float).fillna(0).values
    y_val   = val[LABEL_COL].astype(int).values

    results = {}
    for name in MODEL_NAMES:
        print(f"\n{'-'*50}")
        print(f"  MODEL: {name.upper()}  (spec={spec})")
        print(f"{'-'*50}")

        # Resume support: a model fully finished by an interrupted run
        # (joblib + config both saved) is loaded, not re-tuned.
        model_path = saved_dir / f"{name}.joblib"
        cfg_path_existing = configs_dir / f"{name}_config.yaml"
        if skip_if_saved and model_path.exists() and cfg_path_existing.exists():
            with open(cfg_path_existing, "r") as f:
                cfg = yaml.safe_load(f) or {}
            results[name] = {
                "model": joblib.load(model_path),
                "threshold": float(cfg.get("threshold", 0.5)),
                "best_params": cfg.get("best_params", {}),
            }
            print(f"  [resume] loaded finished model from {model_path.name}")
            continue

        # Step 1: Hyperparameter selection — either reuse a saved config
        # or run Optuna's rolling-origin CV search.
        study = None
        if reuse_configs:
            cfg_path = configs_dir / f"{name}_config.yaml"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"reuse_configs=True but {cfg_path} does not exist. "
                    "Run a full tuning pass first (without --reuse-configs) "
                    "to create the config, or unset reuse_configs."
                )
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            best_params = cfg.get("best_params", {})
            print(f"  Reusing saved best_params from {cfg_path.name}: {best_params}")
        else:
            tuning_frame = tune_raw_train if tune_raw_train is not None else train
            best_params, study = tune_model(
                name, tuning_frame, features,
                fold_safe=tune_fold_safe,
                purge_horizon_days=tune_purge_horizon_days,
                sic_col=tune_sic_col,
                peer_rule=tune_peer_rule,
                storage_dir=tune_storage_dir,
                impute_features=tune_impute_features,
                **({} if n_trials is None else {"n_trials": n_trials}),
            )

        # Step 2: Re-fit on FULL training set with best hyperparameters
        model = build_model_with_params(name, best_params, y_train)
        model.fit(X_train, y_train)

        # Step 3: Select threshold on validation set (maximise F1)
        y_prob_val = model.predict_proba(X_val)[:, 1]
        threshold  = select_threshold(y_val, y_prob_val)
        print(f"  Optimal threshold (val F1): {threshold:.4f}")

        # Step 4: Save model and metadata
        saved_dir.mkdir(parents=True, exist_ok=True)
        configs_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, saved_dir / f"{name}.joblib")

        meta = {"model": name, "best_params": best_params, "threshold": float(threshold)}
        if study is not None:
            complete_trials = sum(
                t.state.name == "COMPLETE" for t in study.trials
            )
            meta.update({
                "optuna_trials_target": int(n_trials or OPTUNA_TRIALS),
                "optuna_trials_complete": int(complete_trials),
                "optuna_trials_total": int(len(study.trials)),
                "best_cv_pr_auc": float(study.best_value),
            })
        elif reuse_configs:
            # Contained refit that reused frozen hyperparameters (no re-tune):
            # carry the tuning provenance (trial counts, CV score) forward from
            # the loaded config so the re-fit does not strip it.
            for k in ("optuna_trials_target", "optuna_trials_complete",
                      "optuna_trials_total", "best_cv_pr_auc"):
                if k in cfg:
                    meta[k] = cfg[k]
        with open(configs_dir / f"{name}_config.yaml", "w") as f:
            yaml.dump(meta, f)

        results[name] = {"model": model, "threshold": threshold, "best_params": best_params}
        print(f"  Saved model -> {saved_dir / name}.joblib")

    # Provenance record: git state, package versions, data hashes, model
    # checksums. Never allowed to kill a multi-hour training run.
    try:
        from src.utils.run_manifest import write_run_manifest
        write_run_manifest(
            configs_dir=configs_dir,
            saved_dir=saved_dir,
            spec=spec,
            features=list(features),
            samples_dir=samples_dir,
            input_files=manifest_input_files,
            extra={
                "reuse_configs": reuse_configs,
                "n_trials": n_trials,
                **(manifest_extra or {}),
            },
        )
    except Exception as exc:  # pragma: no cover — best-effort provenance
        print(f"  WARNING: run manifest could not be written: {exc}")

    return results


# ---------------------------------------------------------------------------
# Final test-set evaluation  (ONCE — do not call multiple times)
# ---------------------------------------------------------------------------

def apply_lr_platt_calibration(
    trained_models: dict,
    val: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    spec: str = "primary",
) -> dict:
    """
    Fix D8 — wire Platt-scaled LR into the headline evaluation flow.

    The primary LR uses class_weight='balanced' which inflates probabilities
    by ~1/prevalence (mean predicted prob ~0.36 vs true rate ~0.015). This
    is harmless for PR-AUC and ROC-AUC (rank-only metrics), but inflates
    the Brier score and produces meaningless calibration. Platt scaling
    fits a 2-parameter sigmoid on validation predictions to remap the
    output probabilities into a properly-calibrated 0-1 scale.

    This helper applies Platt scaling in-place: it replaces
    `trained_models["logistic_regression"]["model"]` with the calibrated
    wrapper, re-selects the F1-maximising threshold on calibrated
    validation predictions, and re-saves the joblib + config YAML so that
    downstream stages (SHAP, robustness, subperiod) see the calibrated
    model.

    The underlying LR coefficients are NOT changed — only the output
    probability scale. SHAP attribution remains unaffected (LinearExplainer
    reads raw coefficients, not probabilities).

    Parameters
    ----------
    trained_models : dict
        Output of train_all_models(). Must contain 'logistic_regression'.
    val : pd.DataFrame
        Validation set (2010–2014) — used for calibration fitting only.
    features : list[str]

    Returns
    -------
    dict
        The same trained_models dict, with the LR entry replaced in-place.
    """
    from src.analysis.lr_calibration import apply_platt_scaling

    print("\n" + "="*60)
    print("  LR Platt calibration (D8 — headline metric fix)")
    print("="*60)

    if "logistic_regression" not in trained_models:
        print("  No logistic_regression model found — skipping calibration.")
        return trained_models

    raw_lr = trained_models["logistic_regression"]["model"]
    calibrated = apply_platt_scaling(raw_lr, val, features=features)

    # Re-select threshold on calibrated validation predictions.
    # The calibrated scale is ~prevalence, so the new optimal threshold
    # will be much smaller (~0.05–0.10) than the uncalibrated one (~0.85).
    X_val = val[features].astype(float).fillna(0).values
    y_val = val[LABEL_COL].astype(int).values
    y_prob_val = calibrated.predict_proba(X_val)[:, 1]
    new_threshold = float(select_threshold(y_val, y_prob_val))
    print(f"  Re-selected threshold on calibrated probabilities: "
          f"{new_threshold:.4f} (was "
          f"{trained_models['logistic_regression']['threshold']:.4f})")

    # Replace in-place
    trained_models["logistic_regression"]["model"]     = calibrated
    trained_models["logistic_regression"]["threshold"] = new_threshold

    # Re-save the joblib so downstream stages (SHAP, robustness, subperiod)
    # load the calibrated wrapper. Spec-aware: non-primary specs re-save
    # into their own subdirectory for provenance protection.
    saved_dir, configs_dir = resolve_artifact_dirs(spec)
    saved_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, saved_dir / "logistic_regression.joblib")

    # Update config YAML to reflect the new threshold
    cfg_path = configs_dir / "logistic_regression_config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["threshold"] = new_threshold
        cfg["calibration"] = "platt"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

    print("  Logistic regression model replaced with Platt-scaled wrapper.")
    return trained_models


def _append_evaluation_manifest(
    run_label: str,
    n_test: int,
    n_distress: int,
    features: list[str],
    output_stem,
    identity: dict | None = None,
    override_reason: str | None = None,
) -> None:
    """
    Append one row to outputs/tables/model_results/evaluation_manifest.csv
    each time evaluate_on_test() is called.

    Provides an auditable record of every test-set evaluation, addressing
    the evaluate-once rule in design §6.3.

    Each row records: ISO timestamp (UTC), run label (primary / RC1-RC6 /
    ad_hoc), output CSV path, test sample size, distress count, feature
    count, and an 8-char hash of the sorted feature list.
    """
    import hashlib
    from datetime import datetime, timezone

    from src.models.evaluation_guard import GUARD_COLUMNS

    manifest_path = OUT_TABLES_MODEL / "evaluation_manifest.csv"
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_label": run_label,
        "output_path": (str(output_stem) + ".csv") if output_stem is not None else "in-memory",
        "n_test": int(n_test),
        "n_distress": int(n_distress),
        "n_features": len(features),
        "features_hash": hashlib.md5(",".join(sorted(features)).encode()).hexdigest()[:8],
    }
    if identity is not None:
        row.update({k: identity.get(k) for k in GUARD_COLUMNS if k in identity})
        row["override_reason"] = override_reason or ""

    OUT_TABLES_MODEL.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if manifest_path.exists():
        # The guard added columns to the schema. Legacy rows predate them, so
        # rewrite the whole file on a column union rather than blind-appending,
        # which would silently misalign the new fields against the old header.
        old = pd.read_csv(manifest_path)
        if set(df.columns) <= set(old.columns):
            df.reindex(columns=old.columns).to_csv(
                manifest_path, mode="a", header=False, index=False)
        else:
            pd.concat([old, df], ignore_index=True).to_csv(manifest_path, index=False)
    else:
        df.to_csv(manifest_path, index=False)


def evaluate_on_test(
    trained_models: dict,
    test: pd.DataFrame,
    features: list[str] = ALL_FEATURES,
    output_stem=None,
    run_label: str = "ad_hoc",
    spec: str | None = None,
    outcome_label: str | None = None,
    horizon_days: int | None = None,
    override_reason: str | None = None,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the held-out test set.

    *** THIS FUNCTION MUST BE CALLED EXACTLY ONCE PER (config, label, horizon). ***
    The test set is reserved for final evaluation under design §6.3. Robustness
    checks call this function under DIFFERENT specifications, so each call is
    a distinct evaluation, not a re-evaluation of the same model.

    Parameters
    ----------
    trained_models : dict
        Output of train_all_models().
    test : pd.DataFrame
    features : list[str]
    output_stem : Path | None, default None
        Destination for the results CSV (without extension). If None, the
        DataFrame is returned in memory only and NOT written to disk. This
        prevents robustness-check evaluations from silently overwriting the
        primary results table (resolves review issues B and C).
    run_label : str, default "ad_hoc"
        Identifier recorded in evaluation_manifest.csv ("primary", "RC1",
        "RC2", ..., "RC6"). Provides an audit trail for the "test set
        evaluated exactly once" design rule.
    spec, outcome_label, horizon_days : str / str / int, optional
        Together with the feature list and a content hash of ``test`` these
        form the *evaluation identity* enforced by
        :mod:`src.models.evaluation_guard`. Defaults are taken from the frozen
        configuration. Supplying them explicitly is preferred for robustness
        checks, whose specification differs from the primary one.
    override_reason : str, optional
        Required to repeat an evaluation whose identity is already recorded.
        Must be a substantive explanation; it is written to the manifest.

    Returns
    -------
    pd.DataFrame
        Performance table: one row per model, all required metrics.

    Raises
    ------
    src.models.evaluation_guard.RepeatedEvaluationError
        If this exact (spec, outcome label, horizon, feature set, test data)
        tuple has already been evaluated and no ``override_reason`` is given.
    """
    from src.models.evaluation_guard import (check_evaluation_permitted,
                                             evaluation_identity)

    identity = evaluation_identity(
        spec=spec if spec is not None else V2_PROFILE.get("spec", "primary"),
        outcome_label=(outcome_label if outcome_label is not None
                       else V2_PROFILE.get("distress_event_col", LABEL_COL)),
        horizon_days=(horizon_days if horizon_days is not None
                      else DISTRESS_HORIZON_DAYS),
        features=features,
        test=test,
        label_col=LABEL_COL,
    )
    recorded_reason = check_evaluation_permitted(
        identity, OUT_TABLES_MODEL / "evaluation_manifest.csv",
        override_reason=override_reason,
    )

    print("\n" + "="*60)
    print(f"  FINAL TEST-SET EVALUATION  [run_label={run_label}]")
    print(f"  evaluation identity: {identity['evaluation_identity']}"
          f"  (spec={identity['spec']}, label={identity['outcome_label']}, "
          f"horizon={identity['horizon_days']}d)")
    if recorded_reason:
        print(f"  OVERRIDE: {recorded_reason}")
    print("="*60)

    X_test = test[features].astype(float).fillna(0).values
    y_test = test[LABEL_COL].astype(int).values
    firm_ids = test["gvkey"].values

    rows = []
    for name, info in trained_models.items():
        model     = info["model"]
        threshold = info["threshold"]
        y_prob    = model.predict_proba(X_test)[:, 1]

        metrics = compute_all_metrics(y_test, y_prob, threshold, model_name=name)

        # Bootstrap CI for PR-AUC
        ci_lo, ci_hi = bootstrap_pr_auc_ci(y_test, y_prob, firm_ids)
        metrics["pr_auc_ci_lower"] = round(ci_lo, 4)
        metrics["pr_auc_ci_upper"] = round(ci_hi, 4)

        rows.append(metrics)
        print(f"  {name:25s}  PR-AUC={metrics['pr_auc']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"
              f"  ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}")

    results_df = pd.DataFrame(rows)

    # Save results — only if an explicit output_stem is provided.
    # Robustness scripts pass output_stem=None and persist results via
    # their own checkpoint mechanism in run_all_robustness.py.
    if output_stem is not None:
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        n_test_fmt = f"{len(test):,}".replace(",", "{,}")
        n_dist     = int(y_test.sum())
        prev_pct   = 100.0 * y_test.mean()
        test_start = int(test["fyear"].min())
        test_end = int(test["fyear"].max())
        save_table(
            results_df,
            output_stem,
            caption=(
                f"Out-of-sample test-set performance ({test_start}--{test_end}), "
                f"$n={n_test_fmt}$ firm-years, "
                f"{n_dist} distressed (prevalence {prev_pct:.2f}\\%). PR-AUC is the primary metric; baseline "
                "PR-AUC equals the sample prevalence rate. Confidence intervals are 95\\% "
                "bootstrap with block resampling by firm (1{,}000 resamples). Classification "
                "threshold selected on the validation sample to maximise F1 and "
                "locked before test evaluation."
            ),
            label="tab:model_performance_test",
        )
    # Always append to the evaluation manifest for audit purposes.
    _append_evaluation_manifest(
        run_label=run_label,
        n_test=len(test),
        n_distress=int(y_test.sum()),
        features=features,
        output_stem=output_stem,
        identity=identity,
        override_reason=recorded_reason,
    )

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    train, val, test = load_splits()

    # Deliberate primary training run — the one caller allowed to
    # overwrite existing primary artifacts (guard in train_all_models).
    trained_models = train_all_models(train, val, allow_primary_overwrite=True)
    # D8 — apply Platt scaling to LR before headline evaluation.
    # This corrects the inflated Brier score from class_weight='balanced'
    # without changing rank-based metrics (PR-AUC, ROC-AUC).
    trained_models = apply_lr_platt_calibration(trained_models, val)
    results_df     = evaluate_on_test(
        trained_models,
        test,
        output_stem=OUT_TABLES_MODEL / "model_performance_test",
        run_label="primary",
    )

    print("\nFinal results table:")
    print(results_df.to_string(index=False))
    print("\nNext step: python -m src.explainability.shap_global")


if __name__ == "__main__":
    main()
