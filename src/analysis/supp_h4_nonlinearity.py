"""
Quantitative diagnostics for non-linear SHAP patterns.
======================================================
**Thesis mapping (2026-08-12).** This module serves the hypothesis the thesis
now states as **H2(b)** (non-linear SHAP patterns). It was written when that
prediction was numbered H4, and the module name, table stems and section labels
keep the frozen ``h4`` spelling deliberately: they are provenance identifiers
recorded in the run manifest and checked by ``verify_final_outputs.py``.
Read every "H4" below as "H2(b)". No behaviour depends on the label.

**Classification: exploratory corroboration.** H₄ was pre-registered as a
*visual* claim about SHAP dependence plots. Everything here was designed after
the results were seen, so it cannot confirm H₄; it can only corroborate or fail
to corroborate the reading the plots suggest. The formal H₄ wording is unchanged
and no result below is described as confirmatory.

Three diagnostics for TLTA (leverage), NITA (profitability) and SIGMA
(volatility), all on XGBoost SHAP values over the frozen test split:

1. **Curvature.** A natural-cubic-spline fit of SHAP on the feature is compared
   with a straight line. Two quantities are reported and kept strictly apart:

   * an **effect size** — the additional variance explained, both in sample and
     under firm-grouped out-of-fold prediction; and
   * a **null test** — a firm-cluster wild bootstrap conducted *under the
     fitted linear model*, which is the only one of the two that licenses an
     inferential statement.

   .. warning::

      An earlier version of this module reported an ordinary percentile
      bootstrap interval for the in-sample R² gain and treated its exclusion of
      zero as evidence of curvature. **That was not a valid test.** The gain is
      the improvement of a nested richer model over a nested poorer one fitted
      to the same data, so it is *mechanically non-negative* in every resample
      whether or not the true conditional mean is curved; its percentile
      interval therefore excludes zero under a perfectly linear
      data-generating process. Those interval columns have been removed rather
      than relabelled, so they cannot be misread. See
      ``tests/test_supp_h4_inference.py`` for the regression test that pins
      this distinction.

2. **Segmented (two-slope) regression.** The breakpoint is estimated by grid
   search over interior quantiles, and the change in slope is reported with a
   firm-block bootstrap interval. **The breakpoint is estimated, not
   pre-specified**; it is reported with its own interval and must not be read as
   a theory-confirmed threshold. In particular the 0.6 leverage figure discussed
   in the literature review is indicative only. The slope change is a *signed*
   quantity, so unlike the curvature gain its interval is not mechanically
   one-sided; it remains subject to the usual Davies problem, the breakpoint
   being unidentified under the no-change null, and is read as a descriptive
   shape summary rather than as a test.
3. **SHAP interaction values** for TLTA x NITA, aggregated to the same 3x3
   tertile grid as the descriptive heatmap, so the model-internal interaction can
   be compared with the descriptive cell contrast.

Run::

    PYTHONPATH=. python -m src.analysis.supp_h4_nonlinearity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.supplementary_common import (FEATS, MODELS,
                                               assert_headline_reproduces,
                                               bootstrap_indices, firm_blocks,
                                               load_frozen_scores, write_table)

TARGETS = ["TLTA", "NITA", "SIGMA"]
N_BOOT = 400               # firm-block resamples for the segmented slope change
N_WILD = 999               # firm-cluster wild-bootstrap replications, curvature test
N_OOF_FOLDS = 5            # firm-grouped folds for the out-of-fold effect size
SEED = 42
N_INTERACTION = 4000       # subsample for the O(p^2) interaction computation

#: Spline complexity and knot placement are **fixed in advance** and identical
#: for every feature, every bootstrap replication and every out-of-fold split,
#: so no aspect of the fit is tuned against the outcome being tested.
KNOT_QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)


def _spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (truncated power, second-derivative free ends)."""
    K = len(knots)
    def d(k):
        num = np.clip(x - knots[k], 0, None) ** 3 - np.clip(x - knots[-1], 0, None) ** 3
        return num / (knots[-1] - knots[k])
    cols = [np.ones_like(x), x] + [d(k) - d(K - 2) for k in range(K - 2)]
    return np.column_stack(cols)


def _knots(x: np.ndarray) -> np.ndarray | None:
    k = np.unique(np.quantile(x, KNOT_QUANTILES))
    return k if len(k) >= 4 else None


def _r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - yhat) ** 2) / ss if ss > 0 else np.nan


def _fit_ls(B, y):
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return B @ coef


def _orthonormal(B: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Orthonormal column basis for the column space of ``B`` (rank-safe).

    Projecting onto this basis gives exactly the least-squares fit, but lets a
    bootstrap replication be evaluated in O(n p) instead of refitting.
    """
    U, sv, _ = np.linalg.svd(B, full_matrices=False)
    if sv.size == 0 or sv.max() <= 0:
        return U[:, :0]
    return U[:, sv > tol * sv.max()]


def curvature(x, s):
    """
    Extra variance explained by a spline over a straight line (in-sample).

    This is an **effect size**, not a test statistic with a usable null. It is
    the improvement of a richer nested model over a poorer one on the same data
    and is therefore non-negative by construction. Use
    :func:`curvature_wild_cluster_test` for the inferential statement and
    :func:`curvature_out_of_fold` for an effect size that can go negative when
    the extra flexibility buys nothing.
    """
    knots = _knots(x)
    if knots is None:
        return np.nan, np.nan, np.nan
    lin = _r2(s, _fit_ls(np.column_stack([np.ones_like(x), x]), s))
    spl = _r2(s, _fit_ls(_spline_basis(x, knots), s))
    return lin, spl, spl - lin


def curvature_wild_cluster_test(x, s, groups, n_boot: int = N_WILD,
                                seed: int = SEED) -> dict:
    """
    Valid test of the linear null against the spline alternative.

    Procedure (firm-cluster wild bootstrap, Cameron--Gelbach--Miller 2008,
    imposing the null):

    1. Fit the **linear** model ``s = a + b x + e`` on the full sample and keep
       the fitted values and residuals. This is the null model.
    2. For each replication draw one Rademacher weight :math:`v_g \\in \\{-1,+1\\}`
       per *firm* and build a synthetic response
       ``s* = a_hat + b_hat x + v_g e_hat``. Every residual within a firm gets
       the same sign flip, so within-firm dependence and arbitrary
       heteroskedasticity are preserved, and the synthetic response has a
       **linear conditional mean by construction**.
    3. Refit both the linear and the (fixed-knot) spline model to ``s*`` and
       record the R² gain.
    4. The p-value is the fraction of replications whose gain reaches the
       observed gain.

    Because step 2 generates data under the null, the reference distribution is
    the distribution of the non-negative gain *when the truth is linear* --- which
    is precisely what the discarded percentile-bootstrap interval failed to
    supply. The knots are held at their full-sample values throughout: ``x`` is
    not resampled, only the response is regenerated.

    Returns a mapping with the observed gain, the bootstrap p-value, and the
    95th percentile of the null gain distribution (reported so the reader can
    see how large a gain a purely linear process produces here).
    """
    x = np.asarray(x, float)
    s = np.asarray(s, float)
    knots = _knots(x)
    if knots is None:
        return {"gain_observed": np.nan, "p_value": np.nan,
                "null_gain_p95": np.nan, "n_boot": 0}

    U_lin = _orthonormal(np.column_stack([np.ones_like(x), x]))
    U_spl = _orthonormal(_spline_basis(x, knots))

    def gain(y):
        y = y - y.mean()
        tss = float(y @ y)
        if tss <= 0:
            return np.nan
        # R2_spl - R2_lin = (||U_spl' y||^2 - ||U_lin' y||^2) / TSS
        return float(((U_spl.T @ y) ** 2).sum() - ((U_lin.T @ y) ** 2).sum()) / tss

    obs = gain(s)

    fitted_null = U_lin @ (U_lin.T @ s)
    resid = s - fitted_null

    codes, _ = pd.factorize(np.asarray(groups))
    n_clusters = int(codes.max()) + 1
    rng = np.random.default_rng(seed)

    null = np.empty(n_boot)
    for b in range(n_boot):
        v = rng.integers(0, 2, size=n_clusters) * 2.0 - 1.0   # Rademacher per firm
        null[b] = gain(fitted_null + v[codes] * resid)

    ok = np.isfinite(null)
    # (1 + #{gain* >= obs}) / (1 + B): never returns an impossible p of exactly 0.
    p = float((1 + np.sum(null[ok] >= obs)) / (1 + ok.sum()))
    return {"gain_observed": obs, "p_value": p,
            "null_gain_p95": float(np.nanpercentile(null[ok], 95)) if ok.any() else np.nan,
            "n_boot": int(ok.sum())}


def curvature_out_of_fold(x, s, groups, n_splits: int = N_OOF_FOLDS) -> float:
    """
    Firm-grouped out-of-fold R² gain of the spline over the straight line.

    An honest effect size: both models are fitted on the fold's training firms
    and scored on firms held out entirely, so extra flexibility that merely
    interpolates the training rows *reduces* the gain. Unlike the in-sample
    gain this quantity can be negative, which is exactly the property that makes
    it informative. Knots follow the same fixed quantile rule, recomputed on
    each fold's training rows so no held-out information enters the basis.
    """
    from sklearn.model_selection import GroupKFold

    x = np.asarray(x, float)
    s = np.asarray(s, float)
    pred_lin = np.full(len(x), np.nan)
    pred_spl = np.full(len(x), np.nan)

    for tr, te in GroupKFold(n_splits=n_splits).split(x, s, groups=np.asarray(groups)):
        knots = _knots(x[tr])
        B_lin_tr = np.column_stack([np.ones_like(x[tr]), x[tr]])
        c_lin, *_ = np.linalg.lstsq(B_lin_tr, s[tr], rcond=None)
        pred_lin[te] = np.column_stack([np.ones_like(x[te]), x[te]]) @ c_lin
        if knots is None:
            pred_spl[te] = pred_lin[te]
            continue
        c_spl, *_ = np.linalg.lstsq(_spline_basis(x[tr], knots), s[tr], rcond=None)
        pred_spl[te] = _spline_basis(x[te], knots) @ c_spl

    m = np.isfinite(pred_lin) & np.isfinite(pred_spl)
    return float(_r2(s[m], pred_spl[m]) - _r2(s[m], pred_lin[m]))


def segmented(x, s, grid_q=np.linspace(0.15, 0.85, 29)):
    """Two-slope fit with the breakpoint chosen by grid search on residual SS."""
    best = (np.inf, np.nan, np.nan, np.nan)
    for q in grid_q:
        c = float(np.quantile(x, q))
        hinge = np.clip(x - c, 0, None)
        B = np.column_stack([np.ones_like(x), x, hinge])
        coef, *_ = np.linalg.lstsq(B, s, rcond=None)
        rss = float(np.sum((s - B @ coef) ** 2))
        if rss < best[0]:
            best = (rss, c, float(coef[1]), float(coef[2]))
    _, bp, slope_lo, delta = best
    return bp, slope_lo, delta, slope_lo + delta


def main() -> None:
    import shap
    import joblib

    df, y, scores = load_frozen_scores("test")
    assert_headline_reproduces(y, scores)
    X = df[FEATS].to_numpy(float)

    model = joblib.load(MODELS / "xgboost.joblib")
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    print(f"SHAP matrix {sv.shape} on the frozen XGBoost model\n")

    blocks = firm_blocks(df)[1]
    rng = np.random.default_rng(SEED)
    draws = [bootstrap_indices(blocks, rng) for _ in range(N_BOOT)]
    gvkeys = df["gvkey"].to_numpy()

    curv_rows, seg_rows = [], []
    for feat in TARGETS:
        j = FEATS.index(feat)
        x, s = X[:, j], sv[:, j]
        lin, spl, gain = curvature(x, s)

        # --- valid null test + honest effect size -------------------------
        test = curvature_wild_cluster_test(x, s, gvkeys)
        oof = curvature_out_of_fold(x, s, gvkeys)
        curv_rows.append({
            "feature": feat,
            "r2_linear": round(lin, 4),
            "r2_spline": round(spl, 4),
            "gain_in_sample": round(gain, 4),
            "gain_out_of_fold": round(oof, 4),
            "null_gain_p95": round(test["null_gain_p95"], 4),
            "p_wild_cluster": round(test["p_value"], 4),
        })

        # --- segmented shape summary (signed slope change) ----------------
        bp, slo, dslope, shi = segmented(x, s)
        d_b, b_b = [], []
        for idx in draws:
            bpb, slb, db, shb = segmented(x[idx], s[idx])
            d_b.append(db)
            b_b.append(bpb)
        seg_rows.append({
            "feature": feat,
            "breakpoint_estimated": round(bp, 4),
            "breakpoint_ci_lower": round(float(np.nanpercentile(b_b, 2.5)), 4),
            "breakpoint_ci_upper": round(float(np.nanpercentile(b_b, 97.5)), 4),
            "slope_below": round(slo, 4),
            "slope_above": round(shi, 4),
            "slope_change": round(dslope, 4),
            "slope_change_ci_lower": round(float(np.nanpercentile(d_b, 2.5)), 4),
            "slope_change_ci_upper": round(float(np.nanpercentile(d_b, 97.5)), 4),
        })

    res = pd.DataFrame(curv_rows)
    seg = pd.DataFrame(seg_rows)
    print(res.to_string(index=False), "\n")
    print(seg.to_string(index=False), "\n")

    # ---- SHAP interaction values for TLTA x NITA -------------------------
    sub = rng.choice(len(X), size=min(N_INTERACTION, len(X)), replace=False)
    iv = expl.shap_interaction_values(X[sub])
    if isinstance(iv, list):
        iv = iv[1]
    iv = np.asarray(iv)
    if iv.ndim == 4:
        iv = iv[:, :, :, 1]
    a, b = FEATS.index("TLTA"), FEATS.index("NITA")
    inter = iv[:, a, b] + iv[:, b, a]      # symmetric off-diagonal pair
    main_a, main_b = iv[:, a, a], iv[:, b, b]
    print(f"interaction values on {len(sub)} sampled test firm-years")
    print(f"  mean |TLTA x NITA interaction| : {np.abs(inter).mean():.5f}")
    print(f"  mean |TLTA main effect|        : {np.abs(main_a).mean():.5f}")
    print(f"  mean |NITA main effect|        : {np.abs(main_b).mean():.5f}")
    share = np.abs(inter).mean() / (np.abs(main_a).mean() + np.abs(main_b).mean())
    print(f"  interaction relative to the two main effects: {share:.3f}\n")

    dsub = df.iloc[sub]
    tl = pd.qcut(dsub["TLTA"], 3, labels=["low", "mid", "high"])
    ni = pd.qcut(dsub["NITA"], 3, labels=["low", "mid", "high"])
    grid = (pd.DataFrame({"TLTA_tertile": tl.to_numpy(),
                          "NITA_tertile": ni.to_numpy(),
                          "interaction": inter})
            .groupby(["TLTA_tertile", "NITA_tertile"], observed=True)["interaction"]
            .agg(["mean", "size"]).reset_index()
            .rename(columns={"mean": "mean_shap_interaction", "size": "n"}))
    grid["mean_shap_interaction"] = grid["mean_shap_interaction"].round(5)
    print(grid.to_string(index=False), "\n")

    write_table(res, "supp_h4_nonlinearity",
                "Curvature diagnostics for H\\textsubscript{4} (exploratory "
                "corroboration): is the relation between an XGBoost SHAP "
                "attribution and its own feature non-linear? Spline complexity "
                "and knot placement are fixed in advance (natural cubic spline, "
                "knots at the "
                f"{', '.join(str(q) for q in KNOT_QUANTILES)} quantiles) and are "
                "identical across features, replications and folds. "
                "\\emph{Effect size and test are reported separately and must "
                "not be conflated.} \\texttt{gain\\_in\\_sample} is the extra "
                "variance explained by the spline over a straight line; it is "
                "the improvement of a richer nested model on the same data and "
                "is therefore \\textbf{non-negative by construction}, so it "
                "carries no evidence against linearity on its own. "
                "\\texttt{gain\\_out\\_of\\_fold} repeats the comparison under "
                f"{N_OOF_FOLDS}-fold firm-grouped out-of-fold prediction, where "
                "flexibility that only interpolates is penalised and the gain "
                "may be negative. \\texttt{p\\_wild\\_cluster} is the inferential "
                "quantity: a firm-cluster wild bootstrap "
                f"({N_WILD} replications, Rademacher weights drawn per firm, "
                f"seed {SEED}) conducted \\emph{{under the fitted linear null}}, "
                "so the reference distribution is the distribution of the gain "
                "when the conditional mean really is linear; "
                "\\texttt{null\\_gain\\_p95} is that distribution's 95th "
                "percentile. These diagnostics were designed after the results "
                "were observed and do not alter the pre-registered "
                "H\\textsubscript{4} wording.",
                "tab:supp_h4_nonlinearity")

    write_table(seg, "supp_h4_segmented",
                "Segmented (two-slope) shape summaries for H\\textsubscript{4} "
                "(exploratory corroboration). A single breakpoint is located by "
                "grid search over interior quantiles of the feature. "
                "\\textbf{The breakpoint is estimated from the data, not "
                "pre-specified}, and is reported with a bootstrap interval "
                "precisely so it is not read as a theory-confirmed threshold. "
                "Unlike the curvature gain the slope change is a \\emph{signed} "
                "quantity, so its interval is not mechanically one-sided; it is "
                "nonetheless a descriptive shape summary rather than a test, "
                "because the breakpoint is unidentified under the no-change "
                "null (the Davies problem) and the grid search selects it on the "
                "same data. A two-segment slope change is also a different "
                "object from the point at which the attribution changes "
                f"\\emph{{sign}}. Intervals are firm-block bootstrap ({N_BOOT} "
                f"resamples, seed {SEED}).",
                "tab:supp_h4_segmented")
    write_table(grid, "supp_h4_tlta_nita_interaction",
                "Mean XGBoost SHAP interaction value for TLTA $\\times$ NITA by "
                "tertile cell (exploratory corroboration), on a random sample of "
                f"{len(sub)} test firm-years. Unlike the descriptive 3$\\times$3 "
                "heatmap, which reports mean predicted probability and therefore "
                "confounds main effects with interaction, this isolates the "
                "model-internal interaction component.",
                "tab:supp_h4_interaction", float_format="%.5f")


if __name__ == "__main__":
    main()
