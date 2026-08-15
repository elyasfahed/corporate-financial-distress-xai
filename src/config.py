"""
Central configuration for the thesis project.
=========================================================
All paths, constants, and frozen design parameters live here.
Import this module everywhere — never hardcode paths or magic numbers.

Data path resolution order:
  1. Environment variable  THESIS_DATA_ROOT
  2. Project-root .env file:  DATA_ROOT=<path>
  3. Fallback: thesis/data/  (local, git-ignored)

Point either setting to an authorized local copy of the licensed datasets.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root  (thesis/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data root — Seafile synced folder or local fallback
# ---------------------------------------------------------------------------
def _resolve_data_root() -> Path:
    # 1. Environment variable (recommended — set once in PyCharm run config)
    env = os.environ.get("THESIS_DATA_ROOT")
    if env:
        return Path(env)
    # 2. .env file in project root
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATA_ROOT="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return Path(val)
    # 3. Local fallback
    return ROOT / "data"


DATA_ROOT = _resolve_data_root()

# ---------------------------------------------------------------------------
# Raw data paths  (on Seafile)
# ---------------------------------------------------------------------------
DATA_RAW_CRSP        = DATA_ROOT / "raw" / "crsp"
DATA_RAW_COMPUSTAT   = DATA_ROOT / "raw" / "compustat"

# Processed data paths  (on Seafile)
DATA_MERGED          = DATA_ROOT / "processed" / "merged"
DATA_FEATURES        = DATA_ROOT / "processed" / "features"
DATA_SAMPLES         = DATA_ROOT / "processed" / "samples"

# External data paths  (on Seafile)
DATA_FF_FACTORS      = DATA_ROOT / "external" / "ff_factors"
DATA_MACRO           = DATA_ROOT / "external" / "macro"

# ---------------------------------------------------------------------------
# Output paths  (always local — outputs are lightweight text/figures)
# ---------------------------------------------------------------------------
OUT_FIGURES_DESCRIPTIVE = ROOT / "outputs" / "figures" / "descriptive"
OUT_FIGURES_MODEL       = ROOT / "outputs" / "figures" / "model"
OUT_FIGURES_SHAP        = ROOT / "outputs" / "figures" / "shap"
OUT_FIGURES_ROBUSTNESS  = ROOT / "outputs" / "figures" / "robustness"

OUT_TABLES_DESCRIPTIVE  = ROOT / "outputs" / "tables" / "descriptive"
OUT_TABLES_MODEL        = ROOT / "outputs" / "tables" / "model_results"
OUT_TABLES_SHAP         = ROOT / "outputs" / "tables" / "shap"
OUT_TABLES_ROBUSTNESS   = ROOT / "outputs" / "tables" / "robustness"

OUT_MODELS_SAVED        = ROOT / "outputs" / "models" / "saved"
OUT_MODELS_CONFIGS      = ROOT / "outputs" / "models" / "configs"

# ---------------------------------------------------------------------------
# Scratch  (ad-hoc, throwaway plots — NOT thesis material, git-ignored)
# ---------------------------------------------------------------------------
# Anything written here is disposable by definition. Thesis figures live ONLY
# under outputs/figures/<chapter>/ and are produced by committed src/ scripts.
# To promote a scratch check, re-derive it with a real script — never move the
# PNG. See src/utils/scratch.py.
OUT_SCRATCH = ROOT / "scratch"

# ---------------------------------------------------------------------------
# Sample periods  (Blueprint v4 — frozen)
# ---------------------------------------------------------------------------
SAMPLE_START_YEAR = 1990
SAMPLE_END_YEAR   = 2024   # FY2024 distress window fully observable Apr 2026
TRAIN_END_YEAR    = 2009   # Training:   1990–2009
VAL_END_YEAR      = 2014   # Validation: 2010–2014
# Test:                     2015–2024  → evaluate ONCE, at the very end

# ---------------------------------------------------------------------------
# Universe & exclusions  (Blueprint v4 §6.1)
# ---------------------------------------------------------------------------
EXCLUDED_SIC_RANGES = [
    (6000, 6999),   # Financial firms — leverage is regulatory, not operational
    (4900, 4999),   # Regulated utilities — excluded in primary; see RC5
]
VALID_SHRCDS = [10, 11]          # Ordinary common shares of domestic US companies
MAX_FILING_LAG_DAYS  = 180       # Drop obs where 10-K filed >180 days after fiscal year-end
MIN_CONSECUTIVE_YEARS = 2        # Drop firms with <2 consecutive fiscal years

# --- CIZ letter-field equivalent of SHRCD 10/11 (Implementation Status §18e) -
# The local CIZ data drop has no numeric SHRCD; share type / incorporation are
# letter fields, which the frozen extraction nulled via numeric coercion — so
# VALID_SHRCDS never bound. universe_eligibility() (src/data/universe.py)
# implements the corrected filter using the sets below.
# CAVEAT: the official CIZ value dictionary (MetaFlagInfo) is not in the local
# drop; these values follow CRSP's published SHRCD<->CIZ crosswalk and MUST be
# pinned against the flag dictionary before the v2 rebuild is run (same caveat
# as the §17 delisting mnemonics).
# PINNED 2026-07-12 against the actual raw_v2 extract (stage a2 of
# scripts/run_v2_rebuild.py; full counts in
# outputs/tables/robustness/v2_universe_field_value_counts.csv):
#   SecurityType    {EQTY 136,698 | FUND 24,436 | DERV 81 | N/A 29,833}
#   SecuritySubType {COM 136,698 | ETF 17,949 | CEF 6,487 | ATR 81 | UNK}
#   ShareType       {NS 148,851 | AD 6,696 | SB 4,094 | UG 1,528 | CE 46}
#   USIncFlg        {Y 143,627 | N 47,421}
#   IssuerType      {CORP 128,933 | ACOR 58,394 | REIT 3,721}
# IssuerType matters: REITs are EQTY/COM/NS and would otherwise pass, but
# SHRCD 10/11 (and the ch04 design prose) excludes them.
# Exchange (added 2026-07-12, second-audit fix): the design restricts the
# universe to NYSE/AMEX/NASDAQ listings, which the pipeline never enforced.
# Pinned against the raw_v2 exchcd_src letter counts (same a2 dump):
#   {Q 91,584 | N 39,760 | X 31,810 | A 14,492 | R 10,035 | B 3,366}
# N=NYSE, A=NYSE American (AMEX), Q=NASDAQ per CRSP's exchange letter
# coding; X/R/B/I are other/regional venues or non-trading statuses and
# are excluded. Same MetaFlagInfo caveat as the other letter fields.
CIZ_UNIVERSE_ELIGIBLE = {
    "SecurityType":    ["EQTY"],           # equity securities (excl. FUND/DERV)
    "SecuritySubType": ["COM"],            # ordinary common (excl. ETF/CEF/ATR)
    "ShareType":       ["NS"],             # no special type (excl. AD/UG/SB/CE)
    "USIncFlg":        ["Y"],              # US-incorporated issuer
    "IssuerType":      ["CORP", "ACOR"],   # corporates only (excl. REIT)
    "Exchange":        ["N", "A", "Q"],    # NYSE / AMEX / NASDAQ only
}

# ---------------------------------------------------------------------------
# Distress definition  (Blueprint v4 §5)
# ---------------------------------------------------------------------------
# Primary: CRSP DLSTCD codes 400–499
# Follows Shumway (2001) and Campbell, Hilscher & Szilagyi (2008)
DISTRESS_CODES_PRIMARY  = list(range(400, 500))

# RC1 secondary: formal bankruptcy only (Chapter 7 and 11)
DISTRESS_CODES_RC1      = [572, 574, 584]

# --- Corrected CIZ delisting-code mapping (Implementation Status §17) -------
# The local CRSP CIZ v2 file has no numeric DLSTCD; codes are synthesized
# from DelActionType/DelReasonType (src/data/load_local_rds.synthesize_dlstcd).
# Under the frozen mapping, bankruptcies were coded 572 (outside the primary
# 400-499 range -> excluded from the primary label) and liquidations fell to
# the 500 default (also excluded). The corrected mapping places all three
# distress classes inside 400-499 so the primary label captures them and the
# bankruptcy subset is NESTED in the primary label, as the design intends:
#   400 = liquidation (DelActionType 'GLI')
#   450 = dropped by exchange (DelActionType 'GDR')       [unchanged]
#   470 = bankruptcy (DelReasonType 'BKPY')
# DISTRESS_CODES_PRIMARY (400-499) needs no change. Used only by the
# correction-sensitivity harness until/unless the fix is adopted in a full
# rebuild; the frozen constants above are untouched.
DLSTCD_CIZ_LIQUIDATION       = 400
DLSTCD_CIZ_DROPPED           = 450
DLSTCD_CIZ_BANKRUPTCY        = 470
DISTRESS_CODES_RC1_CORRECTED = [470]   # bankruptcy-only, nested in 400-499

DISTRESS_HORIZON_DAYS   = 365    # 12-month forward window
DISTRESS_HORIZON_RC2    = 182    # RC2: 6-month alternative horizon

FALLBACK_FILING_LAG_DAYS = 90    # Applied when filing date is missing (documented deviation)

# ---------------------------------------------------------------------------
# Filing-date source (transparent deviation)
# ---------------------------------------------------------------------------
# F_{i,t} (the 10-K filing date that anchors the distress window) is sourced
# from the dedicated filingdates.rds file (real SEC submission dates,
# SRCTYPE='10K', 1966–2025) rather than perioddescriptorannual.FDATE, which
# is empty pre-2006 and is NOT the filing date where it does exist (it
# disagrees with the true 10-K date ~75% of the time). See the QC artifact
# produced by src/analysis/filing_date_qc.py.
#
#   "filingdates_10k"   → dedicated filingdates.rds, SRCTYPE='10K' (current)
#   "perioddescriptor"  → legacy perioddescriptorannual.FDATE (deprecated)
FILING_DATE_SOURCE = "filingdates_10k"

# Policy for firm-years with no real 10-K filing date after the source merge
# (~16% of the panel under filingdates_10k; ~12k of these are foreign 20-F
# filers that likely fall outside the US-domestic universe anyway).
#   "drop"      → design-faithful: missing dates are dropped, not imputed
#                 (Blueprint v4 §5.2). DEFAULT.
#   "estimate"  → keep, impute datadate + FALLBACK_FILING_LAG_DAYS, tagged
#   "broaden"   → fall back to 20-F / AR filing dates from the same file for
#                 non-10-K filers; drop only what still has no date
FILING_DATE_UNMATCHED_POLICY = "drop"

# ---------------------------------------------------------------------------
# Predictor set  (Blueprint v4 §7 — 17 variables, frozen)
# ---------------------------------------------------------------------------
ACCOUNTING_FEATURES = [
    "NITA",      # Net income / Total assets (ROA)
    "TLTA",      # Total liabilities / Total assets (Leverage)
    "WCTA",      # Working capital / Total assets
    "CLCA",      # Current liabilities / Current assets (inverse liquidity)
    "OENEG",     # 1 if TL > TA (negative book equity)
    "NITA_LAG",  # Lagged NITA (t-1)
    "CHIN",      # Change in net income
    "INTWO",     # 1 if NI < 0 in both of last 2 years
    "CASHTA",    # Cash / Total assets
    "OCF_TA",    # Operating cash flow / Total assets
    "LNTA",      # log(Total assets, 2012 USD)
]

MARKET_FEATURES = [
    "EXRET",     # Cumulative 12m excess return over CRSP value-weighted index
    "SIGMA",     # Std dev of monthly returns over 12 months
    "LNMK",     # log(Market cap, 2012 USD)
    "RSIZE",     # log(Firm market cap / Total CRSP market cap)
    "MB",        # Market cap / Book equity
    "PRICE",     # log(Price per share), capped at log(15)
]

ALL_FEATURES = ACCOUNTING_FEATURES + MARKET_FEATURES   # 17 total

# RC4: accounting-only model (remove all 6 market features)
ACCOUNTING_ONLY_FEATURES = ACCOUNTING_FEATURES         # 11 features

# --- v2 predictor-set extension (Implementation Status §8 principled fix,
# adopted for the corrected rebuild 2026-07-12) ------------------------------
# Under v1 the market features escape imputation and reach the models as
# fillna(0); on the v1 test split the 2,525 MB-missing rows contain 168 of
# 434 distress events (6.65% vs 0.95% distress rate), so the zero is an
# implicit missingness signal. The v2 rebuild (a) routes EXRET/SIGMA/MB
# through the standard train-fitted imputation hierarchy and (b) adds an
# explicit MB_MISSING indicator so the MNAR mechanism (book equity <= 0,
# non-positive book equity) is encoded deliberately rather than accidentally.
# PRICE and market cap are still never imputed (missing -> row dropped
# upstream, unchanged). The frozen 17-variable set is untouched.
MARKET_IMPUTE_FEATURES = ["EXRET", "SIGMA", "MB"]
MB_MISSING_COL = "MB_MISSING"
ALL_FEATURES_V2 = ALL_FEATURES + [MB_MISSING_COL]      # 18 total (v2 only)

# Minimum number of non-missing accounting items required to retain a firm-year
MIN_ACCOUNTING_NONMISSING = 8   # out of 11 accounting predictors

# GDP deflator base year for constant-dollar size variables
DEFLATOR_BASE_YEAR = 2012

# ---------------------------------------------------------------------------
# v2 corrected-rebuild profile  (Implementation Status §17/§18 — Phase 3)
# ---------------------------------------------------------------------------
# One coherent bundle of every adopted data-layer correction, so the corrected
# rebuild is a single documented switch rather than seven ad-hoc flags. The
# frozen v1 pipeline NEVER reads this profile: every parameter below has a
# frozen default at its point of use, and v1 outputs are byte-reproducible
# without it. The v2 rebuild must
# thread these values through run_pipeline and write to the *_V2 paths and the
# spec="v2" artifact namespace — never over the frozen artifacts.
SAMPLE_END_YEAR_V2 = 2023   # FY2024 is right-censored: delisting extract ends
                            # 2025-12-30 while 2,392/2,982 FY2024 test windows
                            # run into 2026 (§18c). Under the corrected FYE
                            # convention full FY2024 coverage needs delisting
                            # data into mid-2026 -> FY2023 is the cutoff unless
                            # the extract is refreshed.

V2_PROFILE = {
    # data layer (wired — fix-ready code paths exist)
    "datadate_convention": "standard",        # §17a Jan–May FYE June rule
    # Direct CIZ classification.  GDR means "Dropped" and is classified by
    # DelReasonType; it is never assigned one synthetic distress code.
    "delisting_mapping":   "academic",
    # Cleaned primary label (Phase B, 2026-07-23): performance delistings
    # EXCLUDING the clearly-voluntary/administrative GDR reasons (CORQ
    # company-request and MVOT venue move). MTMK is retained as a listing-
    # standard failure after dictionary verification. The broad performance label
    # (which retains them) is kept as a documented sensitivity.
    "distress_event_col":  "is_distress_performance_core",
    "broad_performance_event_col": "is_distress_performance",
    "narrow_distress_event_col": "is_distress_narrow_financial",
    "bankruptcy_event_col": "is_bankruptcy_proxy",
    "unique_event_assignment": True,
    "rc1_codes":           DISTRESS_CODES_RC1_CORRECTED,  # nested RC1
    "sic_col":             "_sic",            # §18b industry imputation
    "impute_peer_rule":    "per_feature",     # §18b peer-count correctness
    "universe_policy":     "v2",              # §18e CIZ SHRCD-10/11 filter
    "universe_date_col":   "datadate",        # 2026-07-12: eligibility is
                                              # evaluated as-of the fiscal
                                              # year-end, not any-time
    "eligible_index":      True,              # §17(ii) index/RSIZE denominator
    "market_min_obs":      12,                # §9 full-window EXRET/SIGMA
    "sample_end_year":     SAMPLE_END_YEAR_V2,  # §18c censoring cutoff
    # feature layer (2026-07-12 second-audit blocker fixes)
    "feature_set":         ALL_FEATURES_V2,   # 18 = frozen 17 + MB_MISSING
    "impute_market":       True,              # §8 principled fix: EXRET/
                                              # SIGMA/MB through imputation
    "missing_policy":      "strict",          # OENEG/INTWO -> NA when their
                                              # raw inputs are missing
    "lag_buffer":          True,              # FY1989 rows fed to the lag
                                              # construction as donors
    # training layer
    # Keep the corrected re-estimation isolated from the frozen top-level
    # primary artifacts.  This tag controls storage provenance only.
    "spec":                "final_primary",
    # Figure-title wording for the final reported specification.
    "display_name":        "final primary specification",
    "retune":              True,              # fresh Optuna; never reuse v1
                                              # configs (provenance, §18a)
    "cv_fold_safe":        True,              # in-fold winsor/impute +
                                              # >=8/11 coverage + purge
}

# v2 processed-data paths — sibling tree, so the frozen v1 parquets can never
# be overwritten by a rebuild.
# Canonical corrected run.  Historical processed_v2 artifacts remain in
# place for provenance; new runs use neutral, thesis-facing directory names.
DATA_RAW_V2      = DATA_ROOT / "raw_final_primary"
DATA_MERGED_V2   = DATA_ROOT / "processed_final_primary" / "merged"
DATA_FEATURES_V2 = DATA_ROOT / "processed_final_primary" / "features"
DATA_SAMPLES_V2  = DATA_ROOT / "processed_final_primary" / "samples"

# ---------------------------------------------------------------------------
# Modeling  (Blueprint v4 §9)
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
CV_FOLDS      = 5                # Rolling-origin CV folds within training set
OPTUNA_TRIALS = 100              # Bayesian optimisation trials per model
TUNE_METRIC   = "pr_auc"         # Optuna objective: PR-AUC
THRESHOLD_METRIC = "f1"          # Threshold selection metric on validation set
BOOTSTRAP_REPS   = 1_000         # Bootstrap resamples for PR-AUC CI (block by firm)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
FIG_DPI     = 300
FIG_FORMATS = ["pdf", "png"]     # Save both for LaTeX and draft use

# Correct display names for figures and tables (never use .title() on model names)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "logistic_regression_calibrated": "LR (Platt-scaled)",
    "neural_network":      "Neural Network (MLP)",   # extension model (RC7)
}

# ---------------------------------------------------------------------------
# Sanity check — run this file directly to verify paths
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"ROOT      : {ROOT}")
    print(f"DATA_ROOT : {DATA_ROOT}")
    print(f"  raw/crsp            exists: {DATA_RAW_CRSP.exists()}")
    print(f"  raw/compustat       exists: {DATA_RAW_COMPUSTAT.exists()}")
    print(f"  processed/merged    exists: {DATA_MERGED.exists()}")
    print(f"  processed/features  exists: {DATA_FEATURES.exists()}")
    print(f"  processed/samples   exists: {DATA_SAMPLES.exists()}")
    print(f"Sample period : {SAMPLE_START_YEAR}–{SAMPLE_END_YEAR}")
    print(f"Train / Val / Test : –{TRAIN_END_YEAR} / {TRAIN_END_YEAR+1}–{VAL_END_YEAR} / {VAL_END_YEAR+1}–{SAMPLE_END_YEAR}")
    print(f"Predictors    : {len(ALL_FEATURES)} ({len(ACCOUNTING_FEATURES)} accounting + {len(MARKET_FEATURES)} market)")
