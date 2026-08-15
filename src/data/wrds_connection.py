"""
WRDS database connection helper.
=================================
Provides a single authenticated connection object used by all extraction
scripts. Credentials are stored in ~/.pgpass after the first interactive
login (run `python -c "import wrds; wrds.Connection()"` once).

Usage
-----
    from src.data.wrds_connection import get_connection

    db = get_connection()
    df = db.raw_sql("SELECT * FROM comp.funda LIMIT 5")
    db.close()
"""

import sys
import wrds


def get_connection() -> wrds.Connection:
    """
    Open and return an authenticated WRDS database connection.

    Credentials are read from ~/.pgpass if previously saved, or prompted
    interactively on the first run. Call db.close() when done.

    Returns
    -------
    wrds.Connection
        Active WRDS database connection.

    Raises
    ------
    SystemExit
        If the connection cannot be established (wrong credentials,
        no network access to WRDS server, etc.).

    Notes
    -----
    WRDS access requires a valid account. LUH students access WRDS
    through the university library subscription.
    """
    # TODO: Verify WRDS username has access to comp.* and crsp.* libraries
    try:
        db = wrds.Connection()
        print("WRDS connection established successfully.")
        return db
    except Exception as exc:
        print(f"\nERROR: Failed to connect to WRDS.\nDetails: {exc}")
        print(
            "\nTroubleshooting:\n"
            "  1. Ensure you have a valid WRDS account (via LUH library).\n"
            "  2. Run once interactively to save credentials:\n"
            "       python -c \"import wrds; wrds.Connection()\"\n"
            "  3. Check that ~/.pgpass exists and has correct permissions (chmod 600).\n"
            "  4. Confirm network access to wrds-cloud.wharton.upenn.edu:5432."
        )
        sys.exit(1)
