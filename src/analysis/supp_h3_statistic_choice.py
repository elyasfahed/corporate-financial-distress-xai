"""
Evidence that the implemented directional statistic is the necessary one.
=========================================================================
**Thesis mapping (2026-08-12).** This module serves the hypothesis the thesis
now states as **H2(a)** (SHAP direction consistency). It was written when that
prediction was numbered H3, and the module name, table stems and section labels
keep the frozen ``h3`` spelling deliberately: they are provenance identifiers
recorded in the run manifest and checked by ``verify_final_outputs.py``.
Read every "H3" below as "H2(a)". No behaviour depends on the label.

**Classification: post-hoc supplementary.** Read-only. Scores the frozen
``final_primary`` XGBoost estimator on the frozen test split to recompute SHAP
attributions. Nothing here fits, tunes, or writes a model.

Why this table exists
---------------------
The frozen hypothesis H3 is worded in terms of "the sign of the mean SHAP
attribution" for each top-ten feature. The implementation instead uses the sign
of the Pearson correlation between the raw feature and its SHAP attribution
(``src/explainability/theory_consistency.py``), and Chapter 5 discloses that
substitution as a transparent deviation, justified on the ground that
unconditional mean SHAP values are baseline-centred.

That justification was asserted but never demonstrated. This module computes the
literal quantity so the deviation can be judged on evidence rather than on the
claim alone. The result is decisive: in a test sample that is 98.4% non-distress,
the mean SHAP attribution is negative for *every* top-ten feature, because the
average firm sits below the model's baseline risk on essentially every
predictor. Applying the literal rule would return "Inconsistent" for the three
features whose theoretical prediction is positive --- leverage (TLTA), volatility
(SIGMA) and the current-liability ratio (CLCA) --- purely as an artefact of class
imbalance, and would do so however well the model had learned those
relationships.

The deviation is therefore required for the test to have any content, not a
weakening of it. That is a stronger statement than "a different quantity was
measured", and it is the statement the table supports.

Run
---
    PYTHONPATH=. python -m src.analysis.supp_h3_statistic_choice
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.supplementary_common import (FEATS, MODELS, ensure_dirs,
                                               load_split, write_table)
from src.explainability.theory_consistency import THEORETICAL_SIGNS

#: Reported model for every SHAP result in the thesis.
SHAP_MODEL = "xgboost"

#: Number of top features H3 is evaluated on (the frozen scope).
TOP_N = 10


def _theoretical_sign(feature: str) -> str:
    """Look up the pre-registered directional prediction for a feature."""
    return str(THEORETICAL_SIGNS.get(feature, {}).get("sign", "?"))


def build_table() -> pd.DataFrame:
    """Recompute SHAP on the frozen model and contrast the two statistics."""
    import joblib
    import shap
    from scipy.stats import pearsonr

    test = load_split("test")
    model = joblib.load(MODELS / f"{SHAP_MODEL}.joblib")

    X = test[FEATS].astype(float).fillna(0)
    sv = shap.TreeExplainer(model).shap_values(X.values)
    if isinstance(sv, list):          # older shap returns one array per class
        sv = sv[1]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:TOP_N]

    rows = []
    for rank, j in enumerate(order, start=1):
        feat = FEATS[j]
        theory = _theoretical_sign(feat)
        mean_shap = float(sv[:, j].mean())
        corr = float(pearsonr(X.iloc[:, j].values, sv[:, j])[0])

        literal_sign = "+" if mean_shap >= 0 else "-"
        implemented_sign = "+" if corr >= 0 else "-"
        rows.append({
            "rank": rank,
            "feature": feat,
            "theoretical_sign": theory,
            "mean_shap": round(mean_shap, 5),
            "sign_of_mean_shap": literal_sign,
            "verdict_literal_rule": ("Consistent" if literal_sign == theory
                                     else "Inconsistent"),
            "corr_feature_shap": round(corr, 4),
            "sign_of_correlation": implemented_sign,
            "verdict_implemented_rule": ("Consistent" if implemented_sign == theory
                                         else "Inconsistent"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    table = build_table()

    n_lit = (table.verdict_literal_rule == "Consistent").sum()
    n_imp = (table.verdict_implemented_rule == "Consistent").sum()
    flipped = table.loc[table.verdict_literal_rule != table.verdict_implemented_rule,
                        "feature"].tolist()

    print(f"  literal (sign of mean SHAP) rule    : {n_lit}/{len(table)} Consistent")
    print(f"  implemented (correlation) rule      : {n_imp}/{len(table)} Consistent")
    print(f"  features the two rules disagree on  : {', '.join(flipped) or 'none'}")
    print(f"  mean SHAP is negative for {(table.mean_shap < 0).sum()}/{len(table)} features")

    # Compact display frame. The two explicit sign columns are redundant --- the
    # sign is legible from the value beside them --- and dropping them keeps the
    # tabular at body type instead of forcing the width wrapper to shrink it.
    # Headers are plain text because ``write_table`` writes with escape=True.
    disp = pd.DataFrame({
        "Rank": table["rank"],
        "Feature": table["feature"],
        "Theory": table["theoretical_sign"],
        "Mean SHAP": table["mean_shap"],
        "Literal verdict": table["verdict_literal_rule"],
        "Feature-SHAP r": table["corr_feature_shap"],
        "Implemented verdict": table["verdict_implemented_rule"],
    })
    table.to_csv("outputs/tables/supplementary/final_primary/"
                 "supp_h3_statistic_choice_full.csv", index=False)

    write_table(
        disp,
        "supp_h3_statistic_choice",
        caption=(
            "Why the H\\textsubscript{3} directional statistic departs from the "
            "frozen wording. The frozen hypothesis refers to the sign of the "
            "mean SHAP attribution; the implementation uses the sign of the "
            "feature--SHAP Pearson correlation. Because the test sample is "
            "98.4\\% non-distress, the mean attribution is negative for every "
            "one of the ten leading features --- the average firm sits below the "
            "model's baseline risk on essentially every predictor --- so the "
            "literal rule would record ``Inconsistent'' for the three features "
            "whose pre-registered prediction is positive (\\texttt{TLTA}, "
            "\\texttt{SIGMA}, \\texttt{CLCA}) regardless of what the model had "
            "learned. The substitution is therefore necessary for the test to "
            "carry content, not a relaxation of it. The two verdict columns "
            "apply the same theoretical sign prediction to the two candidate "
            "statistics; the explicit sign of each statistic is legible from "
            "the value beside it and is retained in "
            "\\texttt{supp\\_h3\\_statistic\\_choice\\_full.csv}. "
            "Computed on the frozen "
            "final-primary XGBoost estimator and the frozen test split; no "
            "estimator is refitted."),
        label="tab:supp_h3_statistic_choice",
        float_format="%.4f",
    )


if __name__ == "__main__":
    main()
