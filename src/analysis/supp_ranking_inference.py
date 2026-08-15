"""
H1 ranking inference: Romano--Wolf stepdown and a bootstrap model set.
=====================================================================
**Classification: post-hoc supplementary (ranking inference).**

Motivation
----------
H₁ asks whether *the best* machine-learning model outperforms the regularised
logistic benchmark. The reported analysis answers a different question: it runs
six *pairwise* comparisons and Holm-corrects them. Holm controls the family-wise
error rate over those six tests, but it does not address the selection effect
built into "the best" — the maximum of several correlated statistics is biased
upward, and no pairwise test targets the maximum.

This module supplies two inference procedures that do target it, using only the
stored test score vectors. Nothing is refitted or retuned.

1. **Romano--Wolf stepdown** (Romano & Wolf 2005) over the six pairwise PR-AUC
   differences, in the conventional ordered form: hypotheses are ranked by
   descending absolute studentised statistic and removed one at a time, each
   judged against the bootstrap maximum over itself and every smaller
   statistic. The procedure is FWER-controlling like Holm but more powerful,
   because it exploits the correlation between comparisons that share a model
   which Holm must ignore.
2. **A simultaneous bootstrap ("range-based") model set** on PR-AUC: the subset
   of models the data cannot separate on the primary metric, obtained by
   sequential elimination on the bootstrap range. It states the conclusion in
   set form rather than as a list of individually inconclusive pairs. It is
   **not** a Hansen--Lunde--Nason model confidence set — see
   :func:`bootstrap_range_model_set` for why PR-AUC does not admit the
   observation-level loss differentials the HLN guarantee is defined on — and
   nothing here should be read as carrying the formal MCS coverage property.

The **max-statistic test** for H₁ proper is reported alongside: under
H₀: max_m [AP(m) - AP(LR)] <= 0 over m in {RF, XGB, NN}, is the observed maximum
larger than the recentred bootstrap null quantile?

Two resampling schemes are reported, mirroring the thesis's existing practice:
firm-block (the pre-registered primary) and a two-way firm x year scheme that
also admits temporal uncertainty.

Run::

    PYTHONPATH=. python -m src.analysis.supp_ranking_inference
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.analysis.supplementary_common import (DISPLAY, MODEL_ORDER,
                                               assert_headline_reproduces,
                                               bootstrap_indices, firm_blocks,
                                               load_frozen_scores, write_table)

N_BOOT = 2000
SEED = 42
ALPHA = 0.05


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _resample_aps(y, scores, blocks, rng, n_boot):
    """PR-AUC for every model on each bootstrap resample (paired: shared draws)."""
    out = np.empty((n_boot, len(MODEL_ORDER)))
    for b in range(n_boot):
        idx = bootstrap_indices(blocks, rng)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            out[b, :] = np.nan
            continue
        for j, m in enumerate(MODEL_ORDER):
            out[b, j] = _ap(yb, scores[m][idx])
    return out


def _two_way_aps(y, scores, df, rng, n_boot):
    """
    Two-way (firm x year) resample. Firms and years are drawn independently and
    a row enters when both its firm and its year are drawn, which induces
    dependence along both margins.
    """
    gcodes, _ = pd.factorize(df["gvkey"].to_numpy())
    ycodes, _ = pd.factorize(df["fyear"].to_numpy())
    n_g, n_y = gcodes.max() + 1, ycodes.max() + 1
    by_g = [np.flatnonzero(gcodes == i) for i in range(n_g)]
    by_y = [np.flatnonzero(ycodes == i) for i in range(n_y)]
    out = np.empty((n_boot, len(MODEL_ORDER)))
    for b in range(n_boot):
        gp = rng.integers(0, n_g, size=n_g)
        yp = rng.integers(0, n_y, size=n_y)
        ycount = np.bincount(yp, minlength=n_y)
        parts = []
        for gi in gp:
            rows = by_g[gi]
            reps = ycount[ycodes[rows]]
            if reps.sum():
                parts.append(np.repeat(rows, reps))
        idx = np.concatenate(parts) if parts else np.array([], dtype=int)
        yb = y[idx]
        if len(idx) == 0 or yb.sum() == 0 or yb.sum() == len(yb):
            out[b, :] = np.nan
            continue
        for j, m in enumerate(MODEL_ORDER):
            out[b, j] = _ap(yb, scores[m][idx])
    return out


def romano_wolf(obs_diff, boot_diff, labels, alpha=ALPHA):
    """
    Conventional ordered Romano--Wolf stepdown (Romano & Wolf 2005).

    ``boot_diff`` is recentred on the observed differences to impose the null,
    and every statistic is studentised by its bootstrap standard error so the
    maximum is taken over comparable quantities.

    The algorithm is the textbook stepdown, one hypothesis removed per step:

    1. Order the hypotheses by *descending* absolute studentised observed
       statistic, giving :math:`r_1, \\dots, r_k`.
    2. At step :math:`j`, form the bootstrap distribution of the maximum
       absolute studentised statistic over the hypotheses still under
       consideration, which is :math:`\\{r_j, r_{j+1}, \\dots, r_k\\}` --- the
       current hypothesis together with every hypothesis carrying a *smaller*
       observed statistic.
    3. The adjusted p-value for :math:`r_j` is the bootstrap frequency with
       which that maximum reaches :math:`|t_{r_j}|`.
    4. Enforce monotonicity against the *previous* adjusted p-value, so the
       sequence is non-decreasing along the ordering.
    5. Remove :math:`r_j` only, and continue through the full ordering so every
       hypothesis receives an adjusted p-value.

    Step 2 is what makes this a stepdown: the largest statistic is judged
    against the maximum over all :math:`k` comparisons, the second largest
    against the maximum over :math:`k-1`, and so on, so the critical value
    falls as the procedure descends. This is the source of the power gain over
    Holm, which must assume the worst-case dependence throughout.

    An earlier implementation removed *every* hypothesis rejected at a given
    step simultaneously rather than descending one at a time. That variant
    tests each hypothesis against a maximum taken over a set that still
    contains hypotheses with larger statistics, so its adjusted p-values are
    conservative relative to the conventional algorithm. Rejection decisions
    are unchanged here, but the adjusted p-values are not, and the ordered
    algorithm is the one the literature defines.
    """
    obs_diff = np.asarray(obs_diff, dtype=float)
    boot_diff = np.asarray(boot_diff, dtype=float)
    k = len(obs_diff)
    se = np.nanstd(boot_diff, axis=0, ddof=1)
    se = np.where(se > 0, se, np.nan)
    t_obs = np.abs(obs_diff) / se
    centred = np.abs(boot_diff - obs_diff[None, :]) / se[None, :]

    # Descending |t|; NaN statistics (degenerate SE) sort to the end.
    order = np.argsort(-np.nan_to_num(t_obs, nan=-np.inf), kind="stable")

    p_rw = np.empty(k)
    previous = 0.0
    for step, j in enumerate(order):
        active = order[step:]                       # r_j and everything below it
        maxstat = np.nanmax(centred[:, active], axis=1)
        p_raw = float(np.nanmean(maxstat >= t_obs[j]))
        p_adj = max(p_raw, previous)                # monotonicity along the ordering
        p_rw[j] = p_adj
        previous = p_adj

    # With monotone adjusted p-values this reproduces the stepwise rejection
    # rule exactly: the procedure stops rejecting at the first p_adj > alpha.
    reject = p_rw <= alpha
    return pd.DataFrame({
        "comparison": labels,
        "pr_auc_delta": obs_diff,
        "std_error": se,
        "t_statistic": t_obs,
        "p_romano_wolf": p_rw,
        "reject_at_5pct": reject,
    })


def bootstrap_range_model_set(boot_aps, alpha=ALPHA):
    """
    Simultaneous bootstrap ("range-based") model set on test PR-AUC.

    Sequential elimination on the *range* of the paired bootstrap PR-AUC
    distribution: at each round the observed spread between the best and worst
    surviving model is compared with the bootstrap distribution of that same
    spread under equal predictive ability, and the worst model is eliminated
    while the null is rejected.

    **This is not a Hansen--Lunde--Nason model confidence set, and must not be
    described as one.** The HLN construction is defined on *observation-level
    loss differentials*: it requires a loss :math:`L(y_i, \\hat p_i)` that is
    additive over observations, from which the differentials
    :math:`d_{ij,t} = L_i - L_j` and their studentised range statistic are
    built, and the MCS guarantee is stated in terms of that loss.

    PR-AUC (average precision) is not such a loss. It is a rank-based
    functional of the *entire* score vector: an observation's contribution
    depends on where every other observation falls in the ordering, so no
    additive decomposition into per-observation losses exists and no
    observation-level differential can be formed. Applying the range machinery
    to a resampled non-additive functional is an *adaptation* of the idea, not
    the procedure whose formal guarantee HLN prove. It is reported here as a
    descriptive simultaneous-bootstrap statement about which models the data
    cannot separate on the primary metric, and the claim is confined to that.

    A formal MCS is deliberately *not* substituted, because doing so would
    require replacing the primary estimand with an additive loss (log score or
    Brier), which is not the metric H\\textsubscript{1} is posed on and which
    Section~\\ref{sec:supp_calibration_quality} shows is dominated by
    calibration level rather than by ranking quality.
    """
    alive = list(range(boot_aps.shape[1]))
    obs = np.nanmean(boot_aps, axis=0)
    elim_order, elim_p = [], []
    running = 0.0
    while len(alive) > 1:
        sub = boot_aps[:, alive]
        centred = sub - np.nanmean(sub, axis=0)[None, :]
        # range statistic across surviving models
        rng_boot = np.nanmax(centred, axis=1) - np.nanmin(centred, axis=1)
        m = np.nanmean(sub, axis=0)
        rng_obs = m.max() - m.min()
        p = float(np.nanmean(rng_boot >= rng_obs))
        running = max(running, p)
        if running > alpha:
            break
        worst = alive[int(np.argmin(m))]
        elim_order.append(worst)
        elim_p.append(running)
        alive = [a for a in alive if a != worst]
    return alive, elim_order, elim_p, obs


def run_scheme(y, scores, df, scheme: str, rng) -> dict:
    blocks = firm_blocks(df)[1]
    if scheme == "firm":
        boot = _resample_aps(y, scores, blocks, rng, N_BOOT)
    elif scheme == "firm_year":
        boot = _two_way_aps(y, scores, df, rng, N_BOOT)
    else:
        raise ValueError(scheme)

    obs_ap = np.array([_ap(y, scores[m]) for m in MODEL_ORDER])
    pairs, obs_d, boot_d = [], [], []
    for i in range(len(MODEL_ORDER)):
        for j in range(i + 1, len(MODEL_ORDER)):
            pairs.append(f"{DISPLAY[MODEL_ORDER[i]]} vs {DISPLAY[MODEL_ORDER[j]]}")
            obs_d.append(obs_ap[i] - obs_ap[j])
            boot_d.append(boot[:, i] - boot[:, j])
    rw = romano_wolf(np.array(obs_d), np.column_stack(boot_d), pairs)
    rw.insert(0, "scheme", scheme)

    alive, elim, elim_p, _ = bootstrap_range_model_set(boot)
    mcs = pd.DataFrame({
        "scheme": scheme,
        "model": [DISPLAY[m] for m in MODEL_ORDER],
        "test_pr_auc": obs_ap,
        "in_model_set": [i in alive for i in range(len(MODEL_ORDER))],
        "elimination_p": [elim_p[elim.index(i)] if i in elim else np.nan
                          for i in range(len(MODEL_ORDER))],
    })

    # H1 max-statistic: best ML model minus LR
    lr = MODEL_ORDER.index("logistic_regression")
    ml = [k for k in range(len(MODEL_ORDER)) if k != lr]
    obs_max = float(np.max(obs_ap[ml] - obs_ap[lr]))
    boot_gaps = boot[:, ml] - boot[:, [lr]]
    null_gaps = boot_gaps - np.nanmean(boot_gaps, axis=0)[None, :]
    p_max = float(np.nanmean(np.nanmax(null_gaps, axis=1) >= obs_max))
    # The one-sided test can only fail to reject H0: max <= 0. Failing to reject
    # is NOT evidence for the reverse ordering, and the verdict string is worded
    # so that it cannot be read that way.
    h1 = {
        "scheme": scheme,
        "statistic": "max over {RF, XGB, NN} of (PR-AUC minus logistic)",
        "observed": round(obs_max, 4),
        "best_ml_model": DISPLAY[MODEL_ORDER[ml[int(np.argmax(obs_ap[ml] - obs_ap[lr]))]]],
        "p_value_one_sided": round(p_max, 4),
        "verdict": ("no evidence best ML exceeds LR; does not establish LR superiority"
                    if p_max > ALPHA else "H1 supported"),
    }
    return {"rw": rw, "mcs": mcs, "h1": h1}


def main() -> None:
    df, y, scores = load_frozen_scores("test")
    aps = assert_headline_reproduces(y, scores)
    print("frozen PR-AUC reproduced:",
          {k: round(v, 4) for k, v in aps.items()}, "\n")

    rng = np.random.default_rng(SEED)
    res = {s: run_scheme(y, scores, df, s, rng) for s in ("firm", "firm_year")}

    rw = pd.concat([res[s]["rw"] for s in res], ignore_index=True)
    mcs = pd.concat([res[s]["mcs"] for s in res], ignore_index=True)
    h1 = pd.DataFrame([res[s]["h1"] for s in res])

    print(rw.to_string(index=False), "\n")
    print(mcs.to_string(index=False), "\n")
    print(h1.to_string(index=False), "\n")

    write_table(
        rw, "supp_romano_wolf_pr_auc",
        "Romano--Wolf stepdown inference on pairwise test PR-AUC differences "
        "(post-hoc supplementary). The conventional ordered stepdown is used: "
        "hypotheses are ranked by descending absolute studentised statistic and "
        "removed one at a time, each judged against the bootstrap distribution "
        "of the maximum studentised statistic over itself and every "
        "smaller-statistic hypothesis, with adjusted $p$-values made monotone "
        "along that ordering. The procedure controls the family-wise error rate "
        "while exploiting the correlation between comparisons that Holm must "
        "ignore. Scheme \\texttt{firm} resamples firms; \\texttt{firm\\_year} "
        "resamples firms and fiscal years independently. "
        f"{N_BOOT} resamples, seed {SEED}. No model is refitted.",
        "tab:supp_romano_wolf")

    write_table(
        mcs, "supp_model_confidence_set",
        "Simultaneous bootstrap (range-based) model set on test PR-AUC at the "
        "95\\% level (post-hoc supplementary). Models flagged \\texttt{True} "
        "are not separated from the best performer by sequential elimination on "
        "the bootstrap range statistic. \\textbf{This is not a "
        "Hansen--Lunde--Nason model confidence set and carries no formal MCS "
        "coverage guarantee}: that construction is defined on observation-level "
        "loss differentials, whereas PR-AUC is a rank-based functional of the "
        "whole score vector and admits no additive per-observation "
        "decomposition. The statement is a descriptive simultaneous-bootstrap "
        "one about which models the data cannot separate on the primary metric.",
        "tab:supp_mcs")

    write_table(
        h1, "supp_h1_max_statistic",
        "Max-statistic test of H\\textsubscript{1} (post-hoc supplementary). "
        "H\\textsubscript{1} asks whether the \\emph{best} machine-learning "
        "model beats the logistic benchmark, which is a maximum over three "
        "correlated comparisons rather than any single pairwise test. The "
        "null distribution is the recentred bootstrap distribution of that "
        "maximum. The test is one-sided against "
        "H\\textsubscript{0}: $\\max_m [\\mathrm{AP}(m) - \\mathrm{AP}(\\mathrm{LR})] "
        "\\le 0$, so it can only fail to reject; a large $p$-value records the "
        "\\emph{absence} of the predicted machine-learning advantage and does "
        "not establish that the benchmark is superior.",
        "tab:supp_h1_max")


if __name__ == "__main__":
    main()
