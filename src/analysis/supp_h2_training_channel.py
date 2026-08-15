"""
Does contaminating the TRAINING sample with future data inflate performance?
============================================================================
**Thesis mapping (2026-08-12).** The frozen hypothesis numbered H2 has been
withdrawn (see ``h2_leakage_sensitivity`` for why); this module is unaffected.
It measures the training-side channel of the split-design experiment and is
reported on its own terms in thesis Section ``sec:supp_h2_training_channel``.
The ``h2`` in the module name is a frozen provenance identifier, not a live
hypothesis reference.

**Classification: post-hoc supplementary.** Read-only with respect to every
frozen artefact: no saved estimator, Optuna study, configuration or split is
read for reuse or written. Models are fitted in memory with fixed default
hyperparameters, exactly as the main H2 split-design experiment does, and are
discarded.

Why this module replaces the earlier fourth design
--------------------------------------------------
``h2_leakage_sensitivity.chronological_test_contaminated_train`` was written to
isolate the training-side leakage channel: hold the chronological evaluation
sample fixed, and let the training sample draw from all fiscal years. Its
implementation defines the test block as *every* firm-year after
``VAL_END_YEAR`` and then draws the training pool from firms **absent** from that
block. A firm owning any post-2014 observation is therefore excluded in its
entirety, and the eligible pool contains no post-2014 firm-years at all --- zero
under every seed. The near-zero effect it reported was a property of the
construction, not a measurement, and the mechanism claim that rested on it has
been withdrawn from the thesis.

The obstacle is real and has to be designed around rather than patched: if the
evaluation block is *all* post-2014 firm-years, then no post-2014 row can enter
training without being an evaluation row. This module resolves it by splitting
the post-2014 period in time.

Design
------
Four fiscal-year blocks of the feature panel::

    T  = FY <= 2009          base training block
    V  = FY 2010-2014        validation block (threshold selection only)
    E  = FY 2015-2019        EVALUATION block, identical in every arm
    R  = FY 2020-2023        future block, strictly after E

Every arm trains on ``T`` plus some addition, selects its threshold on ``V``, and
is scored on ``E``. Because ``R`` lies entirely *after* ``E``, training on ``R``
is genuine future-information contamination of exactly the kind random splitting
induces: a model that has seen FY2022 predicting FY2016.

``R`` is partitioned by whether the firm is also evaluated::

    R_same = rows of R whose gvkey appears in E
    R_diff = rows of R whose gvkey does NOT appear in E

Five arms, with ``n_add = |R_diff|`` fixing the size of the matched arms:

=========================  ===============================================
arm                        training sample
=========================  ===============================================
chronological              T                       (baseline, no addition)
future_all                 T + R                   (variant A)
future_firm_disjoint       T + R_diff              (variant B)
future_matched             T + subsample(R,     n_add)
recent_past_matched        T + subsample(V,     n_add)
=========================  ===============================================

The contrasts the design supports:

* ``future_all - chronological`` --- the total training-side effect, mixing
  future information, firm overlap and a larger training sample.
* ``future_firm_disjoint - chronological`` --- future information carried only
  by firms that are never evaluated, so no evaluated firm contributes a future
  observation about itself.
* ``future_matched - recent_past_matched`` --- adding ``n_add`` future rows
  against adding ``n_add`` recent-past rows. This is the control the earlier
  design lacked entirely: it holds the *quantity* of added training data fixed
  and varies only whether that data postdates the evaluation block, so a
  positive contrast cannot be attributed to training-set size.
* ``future_matched - future_firm_disjoint`` --- the increment from allowing
  evaluated firms to contribute their own future rows, at equal added row count.

What the design still does not identify
---------------------------------------
``R_diff`` is not a random subset of ``R``: firms present in FY2020-2023 but
absent from FY2015-2019 entered the panel late and carry a materially higher
event rate. The firm-disjoint arm therefore changes the composition of the added
data as well as its firm membership, and its contrast against the baseline
should be read with that in mind. Preprocessing is fitted within each arm's own
training sample, so the contaminated arms' winsorisation thresholds and
imputation medians also see the future; that is deliberate --- it is part of the
same training-side channel --- but it means the measured effect bundles
contaminated preprocessing statistics with contaminated training rows rather
than separating them.

Inference
---------
Every arm is scored on the same rows of ``E``, so arm-to-arm PR-AUC differences
are paired. They are tested with the firm-block paired bootstrap already used
for the headline model comparisons
(``src.analysis.significance._bootstrap_pr_auc_diff``), which recentres under the
null. Arms 1-3 are deterministic given the year partition and the fixed
estimator seed; the two matched arms resample and are replicated over several
seeds, whose spread is reported.

Run
---
    PYTHONPATH=. python -m src.analysis.supp_h2_training_channel
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analysis.h2_leakage_sensitivity import (LABEL_COL, MODEL_NAMES,
                                                 _default_params,
                                                 preprocess_within_design)
from src.analysis.significance import _bootstrap_pr_auc_diff
from src.analysis.supplementary_common import FEATS, ensure_dirs, write_table
from src.config import (ALL_FEATURES_V2, DATA_ROOT, RANDOM_SEED, V2_PROFILE)
from src.models.evaluate import compute_all_metrics, select_threshold
from src.models.train import build_model_with_params

PANEL = DATA_ROOT / "processed_final_primary" / "features" / "features_all.parquet"

#: Fiscal-year block boundaries. E is the evaluation block in every arm.
TRAIN_MAX = 2009
VAL_RANGE = (2010, 2014)
EVAL_RANGE = (2015, 2019)
FUTURE_MIN = 2020

N_SEEDS = 5

#: Long-form arm names, used in console output and prose.
DISPLAY_ARM = {
    "chronological": "Chronological (baseline)",
    "future_all": "+ all future rows",
    "future_firm_disjoint": "+ future rows, firms not evaluated",
    "future_matched": "+ future rows (size-matched)",
    "recent_past_matched": "+ recent-past rows (size-matched)",
}
#: Short arm names for the typeset tables. Long labels in a text column force
#: the width-constrained wrapper to shrink the whole tabular until the numbers
#: are unreadable, so the printed tables carry these and the caption carries the
#: definitions.
SHORT_ARM = {
    "chronological": "Baseline",
    "future_all": "+ all future",
    "future_firm_disjoint": "+ future, unevaluated firms",
    "future_matched": "+ future (matched)",
    "recent_past_matched": "+ recent past (matched)",
}
DISPLAY_MODEL = {
    "logistic_regression": "Logistic regression (ridge)",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
SHORT_MODEL = {
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
}
#: Short contrast names, keyed by (arm_a, arm_b).
#: Plain text only: ``write_table`` writes with ``escape=True``, so any LaTeX
#: markup in a header or cell is printed literally rather than typeset.
SHORT_CONTRAST = {
    ("future_all", "chronological"): "All future minus baseline",
    ("future_firm_disjoint", "chronological"): "Unevaluated-firm future minus baseline",
    ("future_matched", "recent_past_matched"): "Future minus recent past (matched)",
    ("future_matched", "future_firm_disjoint"): "Own-firm future (matched)",
}


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def load_blocks() -> dict[str, pd.DataFrame]:
    """Partition the feature panel into the four fiscal-year blocks."""
    p = pd.read_parquet(PANEL)
    blocks = {
        "T": p[p["fyear"] <= TRAIN_MAX],
        "V": p[p["fyear"].between(*VAL_RANGE)],
        "E": p[p["fyear"].between(*EVAL_RANGE)],
        "R": p[p["fyear"] >= FUTURE_MIN],
    }
    eval_firms = set(blocks["E"]["gvkey"].unique())
    blocks["R_same"] = blocks["R"][blocks["R"]["gvkey"].isin(eval_firms)]
    blocks["R_diff"] = blocks["R"][~blocks["R"]["gvkey"].isin(eval_firms)]
    return {k: v.copy() for k, v in blocks.items()}


def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw n rows without replacement (all rows if n exceeds the pool)."""
    if n >= len(df):
        return df.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)].copy()


def build_arm(blocks: dict[str, pd.DataFrame], arm: str,
              n_add: int, seed: int) -> pd.DataFrame:
    """Assemble one arm's training sample."""
    T = blocks["T"]
    if arm == "chronological":
        return T.copy()
    if arm == "future_all":
        return pd.concat([T, blocks["R"]], ignore_index=True)
    if arm == "future_firm_disjoint":
        return pd.concat([T, blocks["R_diff"]], ignore_index=True)
    if arm == "future_matched":
        return pd.concat([T, _subsample(blocks["R"], n_add, seed)],
                         ignore_index=True)
    if arm == "recent_past_matched":
        return pd.concat([T, _subsample(blocks["V"], n_add, seed)],
                         ignore_index=True)
    raise ValueError(f"unknown arm {arm!r}")


#: Arms whose training sample depends on a random draw.
SEEDED_ARMS = {"future_matched", "recent_past_matched"}
ARMS = ["chronological", "future_all", "future_firm_disjoint",
        "future_matched", "recent_past_matched"]


# ---------------------------------------------------------------------------
# Fit and score one arm
# ---------------------------------------------------------------------------

def run_arm(blocks: dict[str, pd.DataFrame], arm: str, n_add: int,
            seed: int, features: list[str]) -> tuple[pd.DataFrame,
                                                     dict[str, np.ndarray]]:
    """
    Fit the three models on one arm and score the fixed evaluation block.

    Returns the metric rows and the raw test-score vectors, the latter so the
    paired bootstrap can compare arms on identical evaluation rows.
    """
    train = build_arm(blocks, arm, n_add, seed)
    val = blocks["V"]
    test = blocks["E"]

    train, val, test = preprocess_within_design(
        train, val, test, sic_col=V2_PROFILE["sic_col"])

    X_tr = train[features].astype(float).fillna(0).values
    y_tr = train[LABEL_COL].astype(int).values
    X_va = val[features].astype(float).fillna(0).values
    y_va = val[LABEL_COL].astype(int).values
    X_te = test[features].astype(float).fillna(0).values
    y_te = test[LABEL_COL].astype(int).values

    rows, scores = [], {}
    for name in MODEL_NAMES:
        model = build_model_with_params(name, _default_params(name), y_tr)
        model.fit(X_tr, y_tr)
        thr = select_threshold(y_va, model.predict_proba(X_va)[:, 1])
        s = model.predict_proba(X_te)[:, 1]
        scores[name] = s

        m = compute_all_metrics(y_te, s, thr, model_name=name)
        m["arm"] = arm
        m["seed"] = seed
        m["n_train"] = len(train)
        m["train_events"] = int(y_tr.sum())
        rows.append(m)
        print(f"    [{arm:22s} seed {seed}] {name:20s} "
              f"PR-AUC={m['pr_auc']:.4f}  ROC={m['roc_auc']:.4f}")
    return pd.DataFrame(rows), scores


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(n_seeds: int = N_SEEDS) -> None:
    ensure_dirs()
    features = list(ALL_FEATURES_V2)
    blocks = load_blocks()
    n_add = len(blocks["R_diff"])

    print("\n" + "=" * 66)
    print("  H2 TRAINING-SIDE CHANNEL (corrected design)")
    print("=" * 66)
    for k in ["T", "V", "E", "R", "R_same", "R_diff"]:
        b = blocks[k]
        print(f"  {k:7s} rows={len(b):6,d}  events={int(b[LABEL_COL].sum()):4d}  "
              f"prevalence={b[LABEL_COL].mean():.4f}  firms={b['gvkey'].nunique():5,d}")
    print(f"  matched arms add n_add = {n_add:,} rows\n")

    all_rows, score_store = [], {}
    for arm in ARMS:
        seeds = range(n_seeds) if arm in SEEDED_ARMS else [0]
        for seed in seeds:
            df, sc = run_arm(blocks, arm, n_add, seed, features)
            all_rows.append(df)
            if seed == 0:
                score_store[arm] = sc

    results = pd.concat(all_rows, ignore_index=True)

    # ---- arm-level summary (mean across seeds where an arm is seeded) ------
    summary = (results
               .groupby(["arm", "model"], as_index=False)
               .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"),
                    roc_auc=("roc_auc", "mean"), n_train=("n_train", "mean"),
                    train_events=("train_events", "mean"),
                    n_seeds=("pr_auc", "size")))
    summary["pr_auc_sd"] = summary["pr_auc_sd"].fillna(0.0)
    base = (summary[summary.arm == "chronological"]
            .set_index("model")["pr_auc"])
    summary["delta_vs_chronological"] = (summary["pr_auc"]
                                         - summary["model"].map(base))
    summary["arm"] = pd.Categorical(summary["arm"], ARMS, ordered=True)
    summary = summary.sort_values(["arm", "model"]).reset_index(drop=True)
    summary["n_train"] = summary["n_train"].round().astype(int)
    summary["train_events"] = summary["train_events"].round().astype(int)

    # Compact display frame: short labels and merged columns, so the printed
    # tabular is legible at body type instead of being scaled down to fit.
    disp = pd.DataFrame({
        "Arm": summary["arm"].map(SHORT_ARM),
        "Model": summary["model"].map(SHORT_MODEL),
        "Train rows": summary["n_train"].map("{:,}".format),
        "Train events": summary["train_events"],
        "PR-AUC": summary["pr_auc"].round(4),
        "SD": summary["pr_auc_sd"].round(4),
        "ROC-AUC": summary["roc_auc"].round(4),
        "Diff. vs baseline": summary["delta_vs_chronological"].round(4),
    })

    # ---- paired contrasts on the fixed evaluation block --------------------
    y_eval = blocks["E"][LABEL_COL].astype(int).values
    firm_ids = blocks["E"]["gvkey"].values
    contrasts = [
        ("future_all", "chronological",
         "Total training-side effect"),
        ("future_firm_disjoint", "chronological",
         "Future information, evaluated firms excluded"),
        ("future_matched", "recent_past_matched",
         "Future vs recent past at equal added rows"),
        ("future_matched", "future_firm_disjoint",
         "Own-firm future rows, at equal added rows"),
    ]
    crows = []
    for a, b, meaning in contrasts:
        for name in MODEL_NAMES:
            sa, sb = score_store[a][name], score_store[b][name]
            delta = (average_precision_score(y_eval, sa)
                     - average_precision_score(y_eval, sb))
            p, lo, hi = _bootstrap_pr_auc_diff(y_eval, sa, sb, firm_ids)
            crows.append({
                "Contrast": SHORT_CONTRAST[(a, b)],
                "Model": SHORT_MODEL[name],
                "PR-AUC diff.": round(delta, 4),
                "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "p-value": round(p, 4),
                "ROC-AUC diff.": round(roc_auc_score(y_eval, sa)
                                       - roc_auc_score(y_eval, sb), 4),
                "interpretation": meaning,
            })
    contrast_tbl = pd.DataFrame(crows)
    # `interpretation` is carried in the CSV for the reader of the archive but
    # dropped from the typeset table, where it belongs in the caption.
    contrast_disp = contrast_tbl.drop(columns=["interpretation"])

    print("\n" + "-" * 66)
    print(summary.to_string(index=False))
    print("\n" + "-" * 66)
    print(contrast_tbl.to_string(index=False))

    ev = int(y_eval.sum())
    write_table(
        disp, "supp_h2_training_channel",
        caption=(
            "Training-side contamination under a corrected design (post-hoc "
            "supplementary). Every arm is scored on the same evaluation block "
            f"--- fiscal years 2015--2019, {len(y_eval):,} firm-years and {ev} "
            "distress events --- and differs only in what is added to the "
            "FY$\\leq$2009 training block. The future block (FY2020--2023) lies "
            "entirely after the evaluation block, so training on it is genuine "
            "future-information contamination. The two size-matched arms add "
            f"the same {n_add:,} rows, drawn from the future block and from the "
            "recent past (FY2010--2014) respectively, which separates the "
            "effect of future information from the effect of simply having more "
            "training data. Hyperparameters are held at the same defaults across "
            "all arms, as in the main split-design experiment; preprocessing is "
            "fitted within each arm's own training sample. The predecessor "
            "design this replaces admitted no future observations at all and "
            "its mechanism claim has been withdrawn. Arms: \\emph{baseline} adds "
            "nothing; \\emph{all future} adds the whole FY2020--2023 block; "
            "\\emph{unevaluated firms} adds only its rows belonging to firms "
            "absent from the evaluation block; the two \\emph{matched} arms add "
            f"{n_add:,} rows drawn from the future block and from FY2010--2014 "
            "respectively. The matched arms are replicated over five draws and "
            "report the across-draw standard deviation; the other three are "
            "deterministic given the year partition and the fixed estimator "
            "seed. LR, RF and XGB denote ridge logistic regression, the random "
            "forest and XGBoost."),
        label="tab:supp_h2_training_channel")

    write_table(
        contrast_disp, "supp_h2_training_channel_contrasts",
        caption=(
            "Paired contrasts between arms of the corrected training-side "
            "experiment (post-hoc supplementary). All arms score identical "
            "evaluation rows, so each difference is paired; confidence "
            "intervals and $p$-values come from the same firm-block bootstrap, "
            "recentred under the null, that is used for the headline model "
            "comparisons. Reading the four contrasts: the first is the total "
            "training-side effect; the second carries future information only "
            "through firms that are never evaluated; the fourth isolates the "
            "increment from letting evaluated firms contribute their own later "
            "rows. \\textbf{The third is the decisive one}---it holds the "
            "number of added training rows fixed and varies only whether they "
            "postdate the evaluation block, so it separates future information "
            "from the effect of simply having more training data."),
        label="tab:supp_h2_training_channel_contrasts")

    # The typeset tables carry short labels and merged columns for legibility.
    # The full-detail frames are archived alongside them so nothing printed in
    # the thesis is unrecoverable from the outputs.
    out = "outputs/tables/supplementary/final_primary/supp_h2_training_channel"
    results.to_csv(f"{out}_raw.csv", index=False)
    summary.to_csv(f"{out}_summary_full.csv", index=False)
    contrast_tbl.to_csv(f"{out}_contrasts_full.csv", index=False)
    print("\n  wrote supp_h2_training_channel{,_contrasts}.{csv,tex}")
    print("  archived _raw.csv, _summary_full.csv, _contrasts_full.csv")


if __name__ == "__main__":
    main()
