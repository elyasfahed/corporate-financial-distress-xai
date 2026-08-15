"""
Artifact-provenance map for the reported thesis.
================================================
**Classification: corrective validation.** Walks every ``\\input`` and
``\\includegraphics`` target in ``writing/chapters/``, resolves it to a file on
disk, and records which specification namespace it belongs to, when it was
generated, and whether it is headline, supplementary, or historical.

The point of the map is to make one failure mode impossible to miss: a chapter
that reads cleaned-label prose while typesetting a table from a superseded
generation.

Run::

    PYTHONPATH=. python -m src.analysis.artifact_provenance_final_primary
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CHAPTERS = Path("writing/chapters")
OUT = Path("outputs/tables/data_validation/final_primary")

#: ``\input``/``\includegraphics`` paths are written relative to ``writing/``.
TEX_ROOT = Path("writing")

PATTERN = re.compile(
    r"\\(input|includegraphics)\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")

SUPERSEDED_MARKERS = ("_superseded", "processed_v2", "corrected_primary",
                      "_stale", "outputs_frozen_backup", "_wide_label")


def _spec_of(path: str) -> str:
    p = path.replace("\\", "/")
    if any(m in p for m in SUPERSEDED_MARKERS):
        return "SUPERSEDED"
    if "/final_primary/" in p:
        return "final_primary"
    if p.startswith("figures/") or "/figures/framework" in p or "ch05" in p:
        return "local-tikz"
    if "/outputs/" in p:
        return "outputs (un-namespaced)"
    return "local"


def _status_of(spec: str, chapter: str) -> str:
    if spec == "SUPERSEDED":
        return "HISTORICAL (must be labelled)"
    if spec == "local-tikz":
        return "authored figure"
    if chapter.startswith(("ch06", "ch04")):
        return "headline"
    if chapter.startswith("ch07"):
        return "supplementary / robustness"
    return "supporting"


def collect() -> pd.DataFrame:
    rows = []
    for tex in sorted(CHAPTERS.glob("*/*.tex")):
        chapter = tex.parent.name
        for i, line in enumerate(tex.read_text(encoding="utf-8",
                                               errors="replace").splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for kind, target in PATTERN.findall(line):
                t = target.strip()
                if not t:
                    continue
                cand = [TEX_ROOT / t]
                if kind == "input" and not t.endswith(".tex"):
                    cand.append(TEX_ROOT / (t + ".tex"))
                resolved, exists = None, False
                for c in cand:
                    if c.exists():
                        resolved, exists = c, True
                        break
                resolved = resolved or cand[0]
                spec = _spec_of(t)
                try:
                    mtime = datetime.fromtimestamp(
                        resolved.stat().st_mtime, tz=timezone.utc
                    ).strftime("%Y-%m-%d") if exists else ""
                except OSError:
                    mtime = ""
                rows.append({
                    "thesis_location": f"{chapter}:{i}",
                    "kind": kind,
                    "artifact_path": t,
                    "resolved": str(resolved).replace("\\", "/"),
                    "exists": exists,
                    "specification": spec,
                    "sample": ("110,837 firm-years / test 25,512 / 404 events"
                               if spec == "final_primary" else
                               "n/a" if spec in ("local-tikz", "local") else
                               "see artifact"),
                    "generated": mtime,
                    "status": _status_of(spec, chapter),
                })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "artifact_provenance_map.csv", index=False)

    print(f"{len(df)} \\input/\\includegraphics targets across "
          f"{df.thesis_location.str.split(':').str[0].nunique()} chapters\n")
    print(df.specification.value_counts().to_string())
    missing = df[~df.exists]
    print(f"\nunresolved targets: {len(missing)}")
    if len(missing):
        print(missing[["thesis_location", "artifact_path"]].to_string(index=False))
    sup = df[df.specification == "SUPERSEDED"]
    print(f"superseded-namespace targets: {len(sup)}")
    if len(sup):
        print(sup[["thesis_location", "artifact_path"]].to_string(index=False))
    other = df[df.specification == "outputs (un-namespaced)"]
    if len(other):
        print(f"\nun-namespaced outputs targets: {len(other)}")
        print(other[["thesis_location", "artifact_path", "generated"]]
              .to_string(index=False))
    print(f"\nWritten to {OUT / 'artifact_provenance_map.csv'}")


if __name__ == "__main__":
    main()
