"""Regression guard for the final-primary RC5 fold-safe tuning inputs."""

from __future__ import annotations

import ast
from pathlib import Path


def test_rc5_dataset_build_saves_raw_splits() -> None:
    """RC5 must persist pre-processing splits used inside rolling CV folds."""
    source_path = Path(__file__).parents[1] / "scripts" / "run_v2_robustness.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    stage_rc5 = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "stage_rc5"
    )
    feature_calls = [
        node
        for node in ast.walk(stage_rc5)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_feature_pipeline"
    ]

    assert len(feature_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in feature_calls[0].keywords}
    raw_flag = keywords.get("save_raw_splits")
    assert isinstance(raw_flag, ast.Constant) and raw_flag.value is True
