"""
CCM / CUSIP linkage sensitivity.
================================
**Classification: post-hoc supplementary (robustness).**

Layer 2 of the merge flags 2,699 firm-years where the CCM link and an exact
eight-character CUSIP comparison disagree. The reported design **retains** them,
for three stated reasons: CCM is the WRDS-maintained authoritative link with
date-range validation; the flagged set is dominated by identifier churn rather
than by wrong firms; and flagged firm-years carry roughly twice the distress
prevalence of matched ones (3.20% vs 1.48% on test), so dropping them removes
outcome-relevant observations non-randomly.

That decision is defensible but it was never tested. This module tests it, in
two variants of increasing severity:

* **U** — exclude only the 631 firm-years whose mismatch the disposition rules
  leave *unresolved*. This is the targeted variant: everything with a documented
  benign explanation stays.
* **A** — exclude all 2,699 flagged firm-years. A deliberately conservative
  bound, not a preferred specification: it discards a large block of
  outcome-relevant data on weak evidence.

Protocol for each variant: filter train, validation and test consistently;
**reuse the frozen hyperparameters** (no re-tuning, so no search is repeated on
altered data); refit on the filtered training split; select the threshold on the
filtered validation split; evaluate once on the filtered test split. The primary
sample is untouched and the frozen ``.joblib`` files are never written.

Run::

    PYTHONPATH=. python -m src.analysis.supp_cusip_linkage_sensitivity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config as C
from src.analysis.supplementary_common import (CONFIGS, DISPLAY, FEATS,
                                               MODEL_ORDER, load_split,
                                               write_table)
from src.data.cusip_disposition import classify_cusip_mismatches
from src.models.evaluate import bootstrap_pr_auc_ci
from src.models.train import build_model_with_params
from src.robustness.rc7b_neural_network_balanced import build_balanced_nn

SEED = 42


def _flagged_keys() -> tuple[set, set]:
    panel = pd.read_parquet(C.DATA_ROOT / "processed_final_primary" / "merged"
                            / "panel_raw.parquet")
    sn = pd.read_parquet(C.DATA_ROOT / "raw_final_primary" / "crsp_security_names.parquet")
    flagged = panel.loc[panel.cusip_mismatch.fillna(False).astype(bool),
                        ["gvkey", "fyear", "permno", "cusip8_comp", "ncusip8"]].copy()
    cl = classify_cusip_mismatches(flagged, sn)
    all_keys = set(zip(cl.gvkey, cl.fyear))
    unres = cl[cl.disposition == "unresolved"]
    return set(zip(unres.gvkey, unres.fyear)), all_keys


def _filter(df: pd.DataFrame, drop: set) -> pd.DataFrame:
    keys = pd.MultiIndex.from_arrays([df.gvkey, df.fyear])
    return df[~keys.isin(drop)].copy()


def _fit_and_score(train, val, test):
    """Refit all four models with frozen hyperparameters; evaluate once."""
    Xtr, ytr = train[FEATS].to_numpy(float), train["distress"].to_numpy(int)
    Xva, yva = val[FEATS].to_numpy(float), val["distress"].to_numpy(int)
    Xte, yte = test[FEATS].to_numpy(float), test["distress"].to_numpy(int)
    out = {}
    for name in MODEL_ORDER:
        cfg = yaml.safe_load(open(CONFIGS / f"{name}_config.yaml"))
        params = dict(cfg["best_params"])
        if name == "neural_network_balanced":
            model = build_balanced_nn(**params)
        else:
            model = build_model_with_params(name, params, y_train=ytr)
        model.fit(Xtr, ytr)
        # threshold selected on the FILTERED validation split only
        pv = model.predict_proba(Xva)[:, 1]
        grid = np.unique(np.quantile(pv, np.linspace(0.5, 0.9995, 400)))
        f1s = []
        for t in grid:
            tp = int(((pv >= t) & (yva == 1)).sum())
            fp = int(((pv >= t) & (yva == 0)).sum())
            fn = int(((pv < t) & (yva == 1)).sum())
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        thr = float(grid[int(np.argmax(f1s))])
        pt = model.predict_proba(Xte)[:, 1]
        lo, hi = bootstrap_pr_auc_ci(yte, pt, firm_ids=test["gvkey"].to_numpy(),
                                     n_reps=1000)
        out[name] = {
            "pr_auc": float(average_precision_score(yte, pt)),
            "roc_auc": float(roc_auc_score(yte, pt)),
            "ci_lower": float(lo), "ci_upper": float(hi),
            "threshold": thr,
        }
    return out


def main() -> None:
    unres, allf = _flagged_keys()
    print(f"flagged firm-years: {len(allf):,}; unresolved: {len(unres):,}\n")

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    base_n, base_e = len(te), int(te.distress.sum())

    variants = {
        "P_primary_retain_flagged (reported)": set(),
        "U_drop_unresolved": unres,
        "A_drop_all_flagged": allf,
    }

    rows = []
    for vname, drop in variants.items():
        f_tr, f_va, f_te = _filter(tr, drop), _filter(va, drop), _filter(te, drop)
        print(f"--- {vname}: train {len(f_tr):,} ({int(f_tr.distress.sum())} ev) "
              f"| val {len(f_va):,} ({int(f_va.distress.sum())} ev) "
              f"| test {len(f_te):,} ({int(f_te.distress.sum())} ev, "
              f"prev {f_te.distress.mean():.5f})")
        res = _fit_and_score(f_tr, f_va, f_te)
        order = sorted(res, key=lambda k: -res[k]["pr_auc"])
        for name in MODEL_ORDER:
            r = res[name]
            rows.append({
                "variant": vname,
                "model": DISPLAY[name],
                "test_n": len(f_te),
                "test_events": int(f_te.distress.sum()),
                "test_prevalence": round(float(f_te.distress.mean()), 5),
                "pr_auc": round(r["pr_auc"], 4),
                "pr_auc_ci_lower": round(r["ci_lower"], 4),
                "pr_auc_ci_upper": round(r["ci_upper"], 4),
                "roc_auc": round(r["roc_auc"], 4),
                "rank_pr_auc": order.index(name) + 1,
            })
        print("    ranking:", " > ".join(DISPLAY[k] for k in order))
        print("    PR-AUC :", {DISPLAY[k]: round(res[k]["pr_auc"], 4) for k in order})
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))

    attr = pd.DataFrame([{
        "variant": v,
        "firm_years_dropped_total": len(drop),
        "test_n": out.loc[out.variant == v, "test_n"].iloc[0],
        "test_events": out.loc[out.variant == v, "test_events"].iloc[0],
        "test_n_dropped": base_n - out.loc[out.variant == v, "test_n"].iloc[0],
        "test_events_dropped": base_e - out.loc[out.variant == v, "test_events"].iloc[0],
        "test_prevalence": out.loc[out.variant == v, "test_prevalence"].iloc[0],
    } for v, drop in variants.items()])
    print()
    print(attr.to_string(index=False))

    write_table(out, "supp_cusip_linkage_sensitivity",
                "CCM/CUSIP linkage sensitivity (post-hoc supplementary). "
                "Variant U excludes only the firm-years whose CUSIP mismatch the "
                "disposition rules leave unresolved; variant A excludes every "
                "flagged mismatch and is a deliberately conservative bound "
                "rather than a preferred specification, since it discards "
                "outcome-relevant data on weak evidence. Each variant filters "
                "all three splits consistently, reuses the frozen "
                "hyperparameters without re-tuning, refits on the filtered "
                "training split, selects its threshold on the filtered "
                "validation split, and is evaluated once. The primary sample and "
                "the saved models are untouched.",
                "tab:supp_cusip_sensitivity")
    write_table(attr, "supp_cusip_linkage_attrition",
                "Sample and event attrition under the CCM/CUSIP linkage "
                "sensitivity variants (post-hoc supplementary).",
                "tab:supp_cusip_attrition", float_format="%.5f")


if __name__ == "__main__":
    main()
