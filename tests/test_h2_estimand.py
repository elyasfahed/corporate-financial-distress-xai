"""
Regression tests — H2 advantage-differential estimand (Phase 5 fix)
====================================================================
The frozen H2 table reported per-model Δ(PR-AUC), which conflates
design effects common to all models with the quantity H2 asserts
something about: the change in the ML-over-LR ADVANTAGE. These tests
pin compute_advantage_differential on synthetic results, including the
multi-seed aggregation, and the seed-parameterised split helpers.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.h2_leakage_sensitivity import (
    compute_advantage_differential,
    firm_level_random_split,
    observation_level_random_split,
)


def make_results(with_seeds=False):
    """
    Synthetic experiment: chronological LR 0.16 / RF 0.13; firm_random
    LR 0.20 / RF 0.22. Advantage: chrono −0.03, random +0.02 →
    Δadv = +0.05 (design optimism on the ML side).
    """
    rows = []
    seeds = [0, 1] if with_seeds else [0]
    for seed in seeds:
        bump = 0.01 * seed          # second draw slightly different
        rows += [
            dict(model="logistic_regression", design="chronological",
                 pr_auc=0.16, prevalence_baseline_pr_auc=0.014, seed=0),
            dict(model="random_forest", design="chronological",
                 pr_auc=0.13, prevalence_baseline_pr_auc=0.014, seed=0),
            dict(model="xgboost", design="chronological",
                 pr_auc=0.14, prevalence_baseline_pr_auc=0.014, seed=0),
            dict(model="logistic_regression", design="firm_random",
                 pr_auc=0.20 + bump, prevalence_baseline_pr_auc=0.028, seed=seed),
            dict(model="random_forest", design="firm_random",
                 pr_auc=0.22 + bump, prevalence_baseline_pr_auc=0.028, seed=seed),
            dict(model="xgboost", design="firm_random",
                 pr_auc=0.23 + bump, prevalence_baseline_pr_auc=0.028, seed=seed),
        ]
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["model", "design", "seed"])


def test_advantage_differential_single_seed():
    table = compute_advantage_differential(make_results())
    rf = table[table["ml_model"] == "random_forest"].iloc[0]
    assert rf["adv_chronological"] == pytest.approx(-0.03)
    assert rf["adv_firm_random"] == pytest.approx(0.02)
    assert rf["delta_adv_firm_random"] == pytest.approx(0.05)
    # Lift units are prevalence-controlled: chrono LR lift 0.16/0.014,
    # RF 0.13/0.014 -> lift adv ≈ -2.14
    assert rf["lift_adv_chronological"] == pytest.approx(-2.14, abs=0.01)


def test_advantage_differential_multi_seed_aggregates():
    table = compute_advantage_differential(make_results(with_seeds=True))
    rf = table[table["ml_model"] == "random_forest"].iloc[0]
    # Advantage identical across seeds here (bump cancels in ML−LR), so
    # mean equals the single-seed value and sd = 0
    assert rf["adv_firm_random"] == pytest.approx(0.02)
    assert rf["sd_adv_firm_random"] == pytest.approx(0.0, abs=1e-12)


def test_split_helpers_respect_seed():
    panel = pd.DataFrame({
        "gvkey": [f"g{i//3}" for i in range(300)],
        "fyear": 2000 + (np.arange(300) % 3),
        "x": np.arange(300),
    })
    tr_a, te_a = firm_level_random_split(panel, seed=42)
    tr_b, te_b = firm_level_random_split(panel, seed=42)
    tr_c, te_c = firm_level_random_split(panel, seed=43)
    pd.testing.assert_frame_equal(tr_a, tr_b)
    assert not tr_a.equals(tr_c)

    o_a, _ = observation_level_random_split(panel, seed=7)
    o_b, _ = observation_level_random_split(panel, seed=7)
    o_c, _ = observation_level_random_split(panel, seed=8)
    pd.testing.assert_frame_equal(o_a, o_b)
    assert not o_a.equals(o_c)
