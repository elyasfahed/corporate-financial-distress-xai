"""
Safe finalisation pass for the frozen thesis results.
====================================================

This command performs only derived-output and consistency work:

1. Rebuild the corrected LR/RF/XGBoost significance table from FINAL saved
   models and the frozen test split.
2. Reassemble the four-model headline table from saved CSV artifacts.
3. Run the read-only output verifier.
4. Check active LaTeX source for common unfinished-result markers.

It does NOT construct data, tune hyperparameters, fit models, or regenerate
SHAP/LIME values.

Run in PyCharm as module:
    scripts.finalize_without_retraining
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from scripts.regen_significance import main as regenerate_significance
from src.analysis.headline_table_4models import main as regenerate_headline

ROOT = Path(__file__).resolve().parents[1]
WRITING = ROOT / "writing"


def _active_latex_text(path: Path) -> str:
    """Return LaTeX with full-line comments removed."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(line.split("%", 1)[0])
    return "\n".join(lines)


def check_writing_markers() -> None:
    patterns = {
        "numeric placeholder": re.compile(r"\bX\.XXX\b"),
        "unresolved bracket choice": re.compile(
            r"\[(?:was supported|was not supported|strong|moderate|weak|X|Y|Z)"
        ),
        "undefined-reference rendering": re.compile(r"Table\s+\?\?|Figure\s+\?\?"),
        "unresolved name placeholder": re.compile(r"\[[A-Za-z ]*Name\]"),
    }
    problems: list[str] = []
    for path in sorted(WRITING.rglob("*.tex")):
        text = _active_latex_text(path)
        for label, pattern in patterns.items():
            if pattern.search(text):
                problems.append(f"{path.relative_to(ROOT)}: {label}")

    if problems:
        print("\nWriting marker check: FAIL")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(
            "Active LaTeX still contains unfinished-result markers. "
            "Fix them before treating the PDF as final."
        )
    print("\nWriting marker check: OK")


def main() -> None:
    print("=" * 72)
    print("SAFE FINALISATION - NO RETRAINING")
    print("=" * 72)

    print("\nRunning statistical regression tests ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_delong.py",
            "tests/test_significance.py",
            "-q",
        ],
        cwd=ROOT,
        check=True,
    )

    regenerate_significance()
    regenerate_headline()

    print("\nRunning read-only output verification ...")
    subprocess.run(
        [sys.executable, str(ROOT / "verify_outputs.py")],
        cwd=ROOT,
        check=True,
    )
    check_writing_markers()

    print("\nSafe finalisation completed.")
    print("Next: compile writing/main.tex in PyCharm.")


if __name__ == "__main__":
    main()
