"""
Regression tests — run-manifest data provenance (2026-07-12 audit fix)
=======================================================================
The 2026-07-12 audit found that a v2 training run would write a manifest
hashing the V1 splits: write_run_manifest defaulted to DATA_SAMPLES and
train_all_models never passed samples_dir. The manifest now records the
hashed directory explicitly (data_dir) and callers thread samples_dir.

  1. samples_dir is honoured: the hashes come from the requested
     directory and data_dir records it.
  2. Different split content -> different hash (the hash is real).
  3. The default (samples_dir=None) records the primary DATA_SAMPLES
     path in data_dir, making an unthreaded caller visible in the
     manifest instead of silent.
"""

import json

import pandas as pd

from src.config import DATA_SAMPLES
from src.utils.run_manifest import write_run_manifest


def _write_splits(d, seed):
    d.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        pd.DataFrame({"a": [seed, seed + 1]}).to_parquet(d / f"{s}.parquet")


def test_manifest_hashes_requested_samples_dir(tmp_path):
    sdir = tmp_path / "samples"
    _write_splits(sdir, seed=1)
    out = write_run_manifest(
        configs_dir=tmp_path / "configs",
        saved_dir=tmp_path / "saved",
        spec="v2test",
        features=["a"],
        samples_dir=sdir,
    )
    m = json.loads(out.read_text())
    assert m["data_dir"] == str(sdir)
    for s in ("train", "val", "test"):
        assert m["data"][s] is not None
        assert len(m["data"][s]["sha256"]) == 64


def test_manifest_hash_tracks_content(tmp_path):
    sdir1 = tmp_path / "s1"
    sdir2 = tmp_path / "s2"
    _write_splits(sdir1, seed=1)
    _write_splits(sdir2, seed=2)
    m1 = json.loads(write_run_manifest(
        configs_dir=tmp_path / "c1", saved_dir=tmp_path / "m1",
        spec="t", features=["a"], samples_dir=sdir1).read_text())
    m2 = json.loads(write_run_manifest(
        configs_dir=tmp_path / "c2", saved_dir=tmp_path / "m2",
        spec="t", features=["a"], samples_dir=sdir2).read_text())
    assert m1["data"]["train"]["sha256"] != m2["data"]["train"]["sha256"]


def test_manifest_default_records_primary_dir(tmp_path):
    out = write_run_manifest(
        configs_dir=tmp_path / "configs",
        saved_dir=tmp_path / "saved",
        spec="t",
        features=["a"],
    )
    m = json.loads(out.read_text())
    assert m["data_dir"] == str(DATA_SAMPLES)
