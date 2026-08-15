"""
Explore the structure of local .rds files.
Run this ONCE to show column names — needed to write the full loader.

Usage:
    python -m src.data.explore_rds
"""
import pyreadr
from pathlib import Path

RAW  = Path("data/raw")
COMP = RAW / "compustat"
CRSP = RAW / "crsp"

FILES = {
    "perioddescriptorannual":           COMP / "perioddescriptorannual.rds",
    "balancesheetindustrialannual":     COMP / "balancesheetindustrialannual.rds",
    "incomestatementindustrialannual":  COMP / "incomestatementindustrialannual.rds",
    "cashflowannual":                   COMP / "cashflowannual.rds",
    "filingdates":                      COMP / "filingdates.rds",
    "companydescription":               COMP / "companydescription.rds",
    "linkhistory":                      COMP / "linkhistory.rds",
    "StkMthSecurityData":               CRSP / "StkMthSecurityData.rds",
    "StkDelists":                       CRSP / "StkDelists.rds",
    "StkSecurityInfoHdr":               CRSP / "StkSecurityInfoHdr.rds",
    "StkSecurityInfoHist":              CRSP / "StkSecurityInfoHist.rds",
}

for name, path in FILES.items():
    if not path.exists():
        print(f"\n[MISSING] {name} — not downloaded yet: {path}")
        continue

    print(f"\n{'='*60}")
    print(f"FILE: {name}")
    print(f"{'='*60}")

    try:
        result = pyreadr.read_r(str(path))
        df = result[None]
        print(f"  Shape : {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"  First row:")
        print(df.head(1).to_string())

    except MemoryError:
        # File too large to read into RAM fully — try to get just column names
        print(f"  [MEMORY ERROR] File too large for full load.")
        print(f"  Trying to read column names only via partial parse ...")
        try:
            import rdata as _rdata
            parsed = _rdata.parser.parse_file(str(path))
            converted = _rdata.conversion.convert(parsed)
            if isinstance(converted, dict):
                for k, v in converted.items():
                    if hasattr(v, "columns"):
                        print(f"  Shape : {v.shape}")
                        print(f"  Columns ({len(v.columns)}): {list(v.columns)}")
                        break
            elif hasattr(converted, "columns"):
                print(f"  Shape : {converted.shape}")
                print(f"  Columns ({len(converted.columns)}): {list(converted.columns)}")
        except ImportError:
            print(f"  Install rdata for large-file support:  pip install rdata")
            print(f"  Skipping — column names for this file will be inferred.")
        except Exception as e2:
            print(f"  rdata also failed: {e2}")

    except Exception as e:
        print(f"  [ERROR] {e}")
