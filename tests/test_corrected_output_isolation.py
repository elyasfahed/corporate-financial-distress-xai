"""Regression tests for corrected-run artifact isolation."""

from src.config import OUT_TABLES_ROBUSTNESS, V2_PROFILE
from scripts.run_v2_robustness import ROBUSTNESS_DIR, _rc_spec
from scripts.run_v2_nn import RESULTS_DIR
from scripts.run_v2_secondary import (
    RUN_FIGURES_MODEL,
    RUN_FIGURES_SHAP,
    RUN_TABLES_MODEL,
    RUN_TABLES_SHAP,
)
from scripts.run_v2_xai import (
    RUN_FIGURES_DESCRIPTIVE,
    RUN_TABLES_DESCRIPTIVE,
)


def test_corrected_robustness_tables_use_dedicated_directory():
    assert V2_PROFILE["spec"] == "final_primary"
    assert ROBUSTNESS_DIR == OUT_TABLES_ROBUSTNESS / "final_primary"


def test_corrected_robustness_models_use_dedicated_namespaces():
    assert _rc_spec(1) == "final_primary_rc1"
    assert _rc_spec(5) == "final_primary_rc5"


def test_corrected_nn_and_explanation_outputs_are_isolated():
    tag = V2_PROFILE["spec"]
    assert RESULTS_DIR.name == tag
    for path in (
        RUN_FIGURES_MODEL,
        RUN_FIGURES_SHAP,
        RUN_FIGURES_DESCRIPTIVE,
        RUN_TABLES_MODEL,
        RUN_TABLES_SHAP,
        RUN_TABLES_DESCRIPTIVE,
    ):
        assert path.name == tag
