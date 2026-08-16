"""
Winsorisation convention: realised censoring and its effect on the ordering.
============================================================================
**Classification: post-hoc supplementary (robustness).**

The pre-specified design computes 1st/99th-percentile winsorisation bounds on the
1990--2008 training sample and applies them unchanged to validation and test.
That is the correct leakage-free rule and it is not in question here. What was
never measured is its *realised* effect on the evaluation sample: because the
predictor distributions drift over the 25 years separating the two windows, a
bound calibrated to cut 1% of the training distribution does not cut 1% of the
test distribution.

Two things are produced.

**A. Realised censoring.** For each winsorised feature, the share of surviving
firm-years pinned to the lower and upper training bound, separately by split.
Train is 1%/1% by construction; val and test are the quantities of interest.

**B. Convention sensitivity.** Whether the convention affects the reported
ordering, measured rather than argued. Three counterfactual cleaning regimes are
compared against the frozen pipeline:

* ``S0`` -- the frozen pipeline (control). It must reproduce the published
  headline exactly or the module refuses to write anything.
* ``P``  -- ``PRICE`` added to the winsorised set. ``PRICE`` is the only
  continuous predictor the frozen pipeline exempts, on the ground that it is
  "already bounded"; that is true above (the $\\log(15)$ cap) and false below.
* ``Y``  -- per-fiscal-year cross-sectional winsorisation. Each year's
  cross-section is clipped at its own 1st/99th percentile. This is leakage-free
  in the point-in-time sense that matters -- it uses contemporaneous predictor
  values only, never an outcome and never a future period, and a forecaster
  standing at fiscal year-end $t$ observes the year-$t$ cross-section -- and it
  is a common convention in the accounting literature. It is nonetheless a
  *deviation* from the pre-specified design, reported as a sensitivity and not as a
  correction.
* ``YP`` -- ``Y`` with ``PRICE`` included.

Each regime rebuilds the three splits from the raw (pre-winsorisation,
pre-imputation) checkpoints through the pipeline's own
winsorise -> >=8/11 coverage filter -> impute chain, then refits all four models
with the **frozen hyperparameters** (no re-tuning, so no search is repeated on
altered data) and evaluates once. The primary sample is untouched and the frozen
``.joblib`` files are never written.

Run::

    PYTHONPATH=. PYTHONUTF8=1 python -m src.analysis.supp_winsorisation_convention
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.analysis.supplementary_common import (CONFIGS, DISPLAY, FEATS,
                                               HEADLINE_PR_AUC, MODEL_ORDER,
                                               SAMPLES, write_table)
from src.features.build_features import apply_post_winsor_missingness_filter
from src.features.impute import apply_imputation, compute_imputation_medians
from src.features.winsorize import (NO_WINSORISE, apply_thresholds,
                                    compute_thresholds)
from src.models.train import build_model_with_params
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn

SPLITS = ("train", "val", "test")
KEY = ["gvkey", "fyear"]

#: Features routed through the imputation hierarchy in the reported run.
IMPUTE_FEATURES = C.ACCOUNTING_FEATURES + C.MARKET_IMPUTE_FEATURES

#: The frozen winsorisation set: every continuous predictor except NO_WINSORISE.
FROZEN_WINSOR = [f for f in C.ALL_FEATURES if f not in NO_WINSORISE]

#: Realised censoring above this share of a split is reported as drifted. The
#: design intent is 1% per tail, so 2% is a doubling -- a deliberately lax
#: trigger, chosen so that it flags only unambiguous drift.
CENSORING_FLAG_PCT = 2.0


def _quiet(fn, *args, **kwargs):
    """Run a chatty pipeline stage without its progress output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def load_raw() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(SAMPLES / f"{s}_raw.parquet") for s in SPLITS}


def load_processed() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(SAMPLES / f"{s}.parquet") for s in SPLITS}


# ---------------------------------------------------------------------------
# A. realised censoring
# ---------------------------------------------------------------------------

def load_thresholds() -> dict[str, dict[str, float]]:
    """The winsorisation bounds actually used by the reported run."""
    path = (C.ROOT / "outputs" / "models" / "configs" / "final_primary"
            / "winsor_thresholds.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def realised_censoring(proc: dict[str, pd.DataFrame],
                       raw: dict[str, pd.DataFrame],
                       thresholds: dict[str, dict[str, float]] | None = None
                       ) -> pd.DataFrame:
    """
    Share of surviving firm-years whose value the winsorisation actually moved.

    Measured against the *raw* value and using a **strict** comparison, so a row
    counts as censored only if its pre-winsorisation value lay strictly beyond
    the bound and was therefore altered. This matters for ``CHIN``, whose
    1st/99th percentiles are exactly $-1$ and $+1$: those are the natural bounds
    of Ohlson's ratio, which saturates whenever net income changes sign, so a
    fifth of the sample sits *at* the bound while nothing at all is clipped. A
    non-strict comparison would report that mass point as censoring and would be
    wrong. Restricted to rows that survive into the model, since censoring of a
    dropped row is moot.

    ``thresholds`` defaults to the bounds the reported run used; it is a
    parameter so the measure can be exercised on constructed cases.
    """
    saved = load_thresholds() if thresholds is None else thresholds
    rows = []
    for feat, bounds in sorted(saved.items()):
        lo, hi = bounds["lower"], bounds["upper"]
        rec = {"feature": feat,
               "train_lower_bound": lo,
               "train_upper_bound": hi}
        for split in SPLITS:
            merged = proc[split][KEY].merge(raw[split][KEY + [feat]],
                                            on=KEY, how="left")[feat]
            n = int(merged.notna().sum())
            rec[f"{split}_lower_pct"] = round(100 * int((merged < lo).sum()) / n, 2)
            rec[f"{split}_upper_pct"] = round(100 * int((merged > hi).sum()) / n, 2)
        rec["test_total_pct"] = round(rec["test_lower_pct"] + rec["test_upper_pct"], 2)
        rec["drifted"] = bool(max(rec["test_lower_pct"], rec["test_upper_pct"])
                              > CENSORING_FLAG_PCT)
        rows.append(rec)
    return (pd.DataFrame(rows)
            .sort_values("test_total_pct", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# B. convention sensitivity
# ---------------------------------------------------------------------------

def build_splits(raw: dict[str, pd.DataFrame], winsor_features: list[str],
                 per_year: bool = False) -> dict[str, pd.DataFrame]:
    """
    Rebuild the three processed splits under one winsorisation convention.

    Mirrors the outer pipeline exactly: winsorise -> >=8/11 coverage filter ->
    impute, with imputation medians fitted on the regime's own training split.
    With ``per_year=False`` and the frozen feature list this reproduces the
    stored splits byte-for-byte, which is what makes the counterfactuals
    interpretable.
    """
    thresholds = _quiet(compute_thresholds, raw["train"], features=winsor_features)
    out = {}
    for split in SPLITS:
        if per_year:
            frame = raw[split].copy()
            grouped = frame.groupby("fyear")
            for feat in winsor_features:
                lower = grouped[feat].transform(lambda s: s.quantile(0.01))
                upper = grouped[feat].transform(lambda s: s.quantile(0.99))
                frame[feat] = frame[feat].clip(lower=lower, upper=upper)
        else:
            frame = apply_thresholds(raw[split], thresholds)
        out[split] = _quiet(apply_post_winsor_missingness_filter, frame, split)

    medians = _quiet(compute_imputation_medians, out["train"],
                     features=IMPUTE_FEATURES,
                     sic_col=C.V2_PROFILE["sic_col"],
                     peer_rule=C.V2_PROFILE["impute_peer_rule"])
    for split in SPLITS:
        out[split] = _quiet(apply_imputation, out[split], *medians,
                            features=IMPUTE_FEATURES,
                            sic_col=C.V2_PROFILE["sic_col"])
    return out


def fit_and_score(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """Refit all four models with frozen hyperparameters; evaluate once."""
    X_tr = splits["train"][FEATS].to_numpy(float)
    y_tr = splits["train"]["distress"].to_numpy(int)
    X_te = splits["test"][FEATS].to_numpy(float)
    y_te = splits["test"]["distress"].to_numpy(int)

    out = {}
    for name in MODEL_ORDER:
        cfg = yaml.safe_load(open(CONFIGS / f"{name}_config.yaml", encoding="utf-8"))
        params = dict(cfg["best_params"])
        model = (build_balanced_nn(**params) if name == "neural_network_balanced"
                 else build_model_with_params(name, params, y_train=y_tr))
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        out[name] = {"pr_auc": float(average_precision_score(y_te, p)),
                     "roc_auc": float(roc_auc_score(y_te, p))}
    return out


REGIMES = {
    "S0_frozen": dict(winsor_features=FROZEN_WINSOR, per_year=False),
    "P_price_winsorised": dict(winsor_features=FROZEN_WINSOR + ["PRICE"], per_year=False),
    "Y_per_year": dict(winsor_features=FROZEN_WINSOR, per_year=True),
    "YP_per_year_with_price": dict(winsor_features=FROZEN_WINSOR + ["PRICE"], per_year=True),
}

REGIME_NOTE = {
    "S0_frozen": "frozen pipeline (control)",
    "P_price_winsorised": "PRICE winsorised at the training 1st/99th percentile",
    "Y_per_year": "per-fiscal-year cross-sectional 1st/99th winsorisation",
    "YP_per_year_with_price": "per-fiscal-year winsorisation, PRICE included",
}


def main() -> None:
    raw, proc = load_raw(), load_processed()

    print("A. realised censoring at the training-derived bounds\n")
    censoring = realised_censoring(proc, raw)
    print(censoring.to_string(index=False), "\n")

    print("B. convention sensitivity (frozen hyperparameters, evaluate once)\n")
    results, control = [], None
    for tag, kwargs in REGIMES.items():
        splits = build_splits(raw, **kwargs)
        scored = fit_and_score(splits)

        if tag == "S0_frozen":
            # The control must reproduce the stored splits and the published
            # headline. If it does not, every counterfactual below is
            # uninterpretable, so refuse to write anything.
            for split in SPLITS:
                a = splits[split].sort_values(KEY).reset_index(drop=True)
                b = proc[split].sort_values(KEY).reset_index(drop=True)
                if len(a) != len(b):
                    raise AssertionError(
                        f"control rebuild of {split!r} has {len(a)} rows against "
                        f"the stored {len(b)}; refusing to write."
                    )
                worst = max(float(np.nanmax(np.abs(a[f].astype(float)
                                                   - b[f].astype(float))))
                            for f in C.ALL_FEATURES_V2)
                if worst > 1e-12:
                    raise AssertionError(
                        f"control rebuild of {split!r} differs from the stored "
                        f"split by {worst:.3g}; refusing to write."
                    )
            for name, ref in HEADLINE_PR_AUC.items():
                if abs(scored[name]["pr_auc"] - ref) > 5e-5:
                    raise AssertionError(
                        f"{name}: control PR-AUC {scored[name]['pr_auc']:.6f} "
                        f"does not reproduce the reported {ref}; refusing to write."
                    )
            control = {k: v["pr_auc"] for k, v in scored.items()}
            print("  control reproduces the stored splits and the headline exactly\n")

        for name in MODEL_ORDER:
            results.append({
                "regime": tag,
                "convention": REGIME_NOTE[tag],
                "model": DISPLAY[name],
                "pr_auc": round(scored[name]["pr_auc"], 4),
                "roc_auc": round(scored[name]["roc_auc"], 4),
                "pr_auc_delta": round(scored[name]["pr_auc"] - control[name], 4),
            })
        best = max(MODEL_ORDER, key=lambda n: scored[n]["pr_auc"])
        print(f"  {tag:24s} best on PR-AUC: {DISPLAY[best]} "
              f"({scored[best]['pr_auc']:.4f})")

    sensitivity = pd.DataFrame(results)
    print("\n" + sensitivity.to_string(index=False))

    display_cols = ["feature", "train_lower_bound", "train_upper_bound",
                    "val_lower_pct", "val_upper_pct",
                    "test_lower_pct", "test_upper_pct", "drifted"]
    write_table(
        censoring[display_cols],
        "supp_winsorisation_realised_censoring",
        "Realised winsorisation censoring by split (post-hoc supplementary). "
        "Bounds are the 1st/99th percentiles of the 1990--2008 training sample "
        "and are applied unchanged to validation and test, which is the "
        "leakage-free rule the design specifies. The columns report the share "
        "of surviving firm-years whose raw value lay strictly beyond each bound "
        "and was therefore altered. Training shares are 1\\% by construction "
        "(marginally below, since a value exactly at the percentile is not "
        "clipped); validation and test shares "
        "are not, because the predictor distributions drift over the 25 years "
        "separating the estimation and evaluation windows. A feature is marked "
        "drifted when either test tail exceeds "
        f"{CENSORING_FLAG_PCT:.0f}\\%, twice the design intent.",
        "tab:supp_winsor_censoring", float_format="%.4g")

    write_table(
        sensitivity[["convention", "model", "pr_auc", "roc_auc", "pr_auc_delta"]],
        "supp_winsorisation_convention_sensitivity",
        "Effect of the winsorisation convention on test performance "
        "(post-hoc supplementary). Each regime rebuilds all three splits from "
        "the raw checkpoints through the pipeline's own winsorise, coverage-"
        "filter and impute chain, then refits the four models on the frozen "
        "hyperparameters and evaluates once; no model is re-tuned and the "
        "frozen artifacts are never overwritten. The control reproduces the "
        "stored splits to machine precision and the published PR-AUCs exactly, "
        "which is asserted before this table is written.",
        "tab:supp_winsor_sensitivity", float_format="%.4f")


if __name__ == "__main__":
    main()
