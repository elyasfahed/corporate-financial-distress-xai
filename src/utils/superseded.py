"""
Supersession guard for legacy pipeline drivers.
==============================================

Background
----------
The reported run is ``final_primary``. Earlier generations (v1,
v2, ``corrected_primary``) were quarantined to ``outputs/_superseded/`` on
2026-07-24, but their *data* namespaces (``data/processed/``,
``data/processed_v2/``, ``data/processed_primary/``) were deliberately left in
place.

That asymmetry creates a specific hazard: a legacy driver reads the superseded
splits **successfully**, runs part-way, and only fails later when it reaches a
quarantined model file. Worst case, if the missing artifact were ever restored,
such a driver would produce v1-data results under a ``final_primary`` filename.

This module makes that failure immediate and explicit instead.

Usage
-----
Call at the top of a legacy driver's ``__main__`` block::

    from src.utils.superseded import abort_if_superseded
    abort_if_superseded(__file__, "scripts/verify_final_outputs.py")

Set ``THESIS_ALLOW_SUPERSEDED_DRIVER=1`` to run anyway (for forensics on an
older generation), which prints a loud banner rather than aborting.
"""

from __future__ import annotations

import os
import sys

ENV_OVERRIDE = "THESIS_ALLOW_SUPERSEDED_DRIVER"


def abort_if_superseded(driver: str, replacement: str) -> None:
    """
    Abort unless the caller explicitly opted in to running a superseded driver.

    Parameters
    ----------
    driver : str
        Usually ``__file__``; only the basename is shown.
    replacement : str
        The current entry point the user should run instead.
    """
    name = os.path.basename(str(driver))

    if os.environ.get(ENV_OVERRIDE) == "1":
        print("=" * 78, file=sys.stderr)
        print(f"  WARNING: running SUPERSEDED driver {name}", file=sys.stderr)
        print(f"  {ENV_OVERRIDE}=1 is set, so execution continues.", file=sys.stderr)
        print("  Its results describe a superseded generation and must NOT be", file=sys.stderr)
        print("  cited or written into the thesis.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        return

    print("=" * 78, file=sys.stderr)
    print(f"  REFUSING TO RUN: {name} is superseded.", file=sys.stderr)
    print("", file=sys.stderr)
    print("  The reported run is 'final_primary'. This", file=sys.stderr)
    print("  driver targets an earlier generation whose outputs were moved to", file=sys.stderr)
    print("  outputs/_superseded/ on 2026-07-24. It can still read the old", file=sys.stderr)
    print("  splits under data/processed*/, so running it would silently mix", file=sys.stderr)
    print("  superseded data with current filenames.", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  Use instead:  {replacement}", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  To override (forensics only):  {ENV_OVERRIDE}=1", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    sys.exit(2)
