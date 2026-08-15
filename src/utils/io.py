"""
File I/O utilities.
===================
Convenience functions for loading and saving project data files.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with an informative print statement."""
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}  ({len(df):,} rows, {df.shape[1]} cols)")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to parquet with an informative print statement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  Saved  {path.name}  ({len(df):,} rows)")


def _resolve_artifact_dirs(spec: str):
    """
    Return (saved_dir, configs_dir) for a specification.

    Mirrors ``src.models.train.resolve_artifact_dirs`` deliberately rather than
    importing it: that module pulls in scikit-learn and XGBoost, which is far
    too heavy for resolving a path. The two must stay in sync -- see
    ``tests/test_io_spec_dirs.py``, which asserts they agree.
    """
    from src.config import OUT_MODELS_SAVED, OUT_MODELS_CONFIGS
    if not spec or "/" in spec or "\\" in spec or spec == "..":
        raise ValueError(f"Invalid spec name: {spec!r}")
    if spec == "primary":
        return OUT_MODELS_SAVED, OUT_MODELS_CONFIGS
    return OUT_MODELS_SAVED / spec, OUT_MODELS_CONFIGS / spec


def load_model(name: str, spec: str = "primary") -> object:
    """
    Load a trained model from outputs/models/saved/[{spec}/].

    Parameters
    ----------
    name : str
        Model name without extension, e.g. 'xgboost'.
    spec : str
        Training specification namespace. ``"primary"`` reads the top-level
        (v1) directory; any other value reads the per-spec subdirectory, e.g.
        ``spec="final_primary"`` -> ``outputs/models/saved/final_primary/``.

    Returns
    -------
    Fitted estimator.

    Raises
    ------
    FileNotFoundError
        With an explicit pointer to the per-spec layout. The v1 root-level
        joblibs were moved to ``outputs/_superseded/`` in the 2026-07-24
        quarantine, so a bare ``load_model("xgboost")`` no longer resolves.
    """
    saved_dir, _ = _resolve_artifact_dirs(spec)
    path = saved_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Models are namespaced by specification; "
            f"pass spec=... (e.g. spec='final_primary'). Superseded generations "
            f"live under outputs/_superseded/."
        )
    model = joblib.load(path)
    print(f"  Model loaded: {path.name}  (spec={spec})")
    return model


def load_config(name: str, spec: str = "primary") -> dict:
    """
    Load a model config YAML from outputs/models/configs/[{spec}/].

    Parameters
    ----------
    name : str
        Config filename stem, e.g. 'xgboost_config'.
    spec : str
        Training specification namespace; see :func:`load_model`.

    Returns
    -------
    dict
    """
    _, configs_dir = _resolve_artifact_dirs(spec)
    path = configs_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config at {path}. Configs are namespaced by specification; "
            f"pass spec=... (e.g. spec='final_primary'). Superseded generations "
            f"live under outputs/_superseded/."
        )
    with open(path) as f:
        return yaml.safe_load(f)
