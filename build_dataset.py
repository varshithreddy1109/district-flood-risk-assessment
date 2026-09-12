"""
build_dataset.py
================
Reproducible preprocessing pipeline for the AI-Powered District Flood Risk
Assessment & Decision Support System (India).

Combines four historical flood CSV files into a single district-level dataset:
    data/processed/district_flood_risk_dataset.csv

Scientific note
---------------
DFSI is the *target/reference* variable (used only to derive risk_class).
It must never appear as a model feature under any name.

The eventual ML model classifies district flood risk based on historical flood
characteristics and is honestly described as a district flood-risk assessment
benchmarked against DFSI - not future flood prediction.

Usage
-----
    python build_dataset.py
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("data", "flood")
LGD_PATH = os.path.join("data", "districts", "LGD_Districts.csv")
OUT_DIR  = os.path.join("data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "district_flood_risk_dataset.csv")

FILES = {
    "dfsi":      os.path.join(DATA_DIR, "DFSI.csv"),
    "inventory": os.path.join(DATA_DIR, "India_Flood_Inventory_v3.csv"),
    "area":      os.path.join(DATA_DIR, "District_FloodedArea.csv"),
    "impact":    os.path.join(DATA_DIR, "District_FloodImpact.csv"),
}

ENCODINGS = ["utf-8-sig", "utf-8", "latin1"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_csv_robust(path: str, **kwargs) -> pd.DataFrame:
    """Try multiple encodings; raise a clear error if all fail."""
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            return df
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e
            break  # non-encoding error -> re-raise immediately
    raise IOError(
        f"Cannot read '{path}' with any of {ENCODINGS}.\n"
        f"Last error: {last_err}"
    )


def normalize_name(series: pd.Series) -> pd.Series:
    """
    Lowercase, strip, collapse internal whitespace, remove stray newlines,
    and drop common encoding artefacts (asterisks used as truncation markers).
    """
    s = series.astype(str)
    s = s.str.replace(r"[\r\n\t]+", " ", regex=True)  # newlines -> space
    s = s.str.strip()
    s = s.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)         # collapse whitespace
    s = s.str.replace(r"\*+$", "", regex=True)          # trailing asterisks
    s = s.str.strip()
    return s


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Strip commas then coerce to numeric; leave genuine NaN as NaN."""
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def strip_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------------------------------------------------------------------------
# Step 1  -  Load DFSI.csv
# ---------------------------------------------------------------------------

def load_dfsi(path: str) -> pd.DataFrame:
    """
    Expected header row: (unnamed index), State_Name, DFSI
    The first column is the district name (index exported from a DataFrame).
    """
    print("\n" + "="*60)
    print("STEP 1 - Loading DFSI.csv")
    print("="*60)

    df = read_csv_robust(path)
    df = strip_col_names(df)

    # The file has an unnamed leading column that contains district names.
    # Detect it by looking for the "Unnamed" pattern or positional fallback.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.rename(columns={unnamed_cols[0]: "District_Name"})
    elif "District_Name" not in df.columns and df.columns[0] not in ("State_Name", "DFSI"):
        df = df.rename(columns={df.columns[0]: "District_Name"})

    required = {"District_Name", "State_Name", "DFSI"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DFSI.csv is missing expected columns: {missing}\n"
                         f"Found: {df.columns.tolist()}")

    # Clean text fields
    df["District_Name"] = (
        df["District_Name"]
        .astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    df["State_Name"] = (
        df["State_Name"]
        .astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Normalised keys for matching
    df["district_key"] = normalize_name(df["District_Name"])
    df["state_key"]    = normalize_name(df["State_Name"])

    # Coerce DFSI to numeric
    df["DFSI"] = to_numeric_series(df["DFSI"])

    # ------- Duplicate handling -------
    n_before = len(df)
    # 1. Remove exact duplicates across all columns
    df = df.drop_duplicates()
    n_after_exact = len(df)
    print(f"  Rows before duplicate removal : {n_before}")
    print(f"  Rows after exact-dup removal  : {n_after_exact}")

    # 2. Identify non-identical duplicates (same district+state, different DFSI)
    dup_mask = df.duplicated(subset=["district_key", "state_key"], keep=False)
    dup_groups = df[dup_mask].groupby(["district_key", "state_key"])
    n_dup_groups = dup_groups.ngroups
    print(f"  Non-identical duplicate groups (same district+state): {n_dup_groups}")
    if n_dup_groups > 0:
        print("  Affected groups:")
        for name, grp in dup_groups:
            print(f"    {name}  DFSI values: {grp['DFSI'].tolist()}")
        # Resolve by averaging DFSI within each district+state group
        df["DFSI"] = df.groupby(["district_key", "state_key"])["DFSI"].transform("mean")
        df = df.drop_duplicates(subset=["district_key", "state_key"])
        print(f"  Rows after averaging duplicate DFSI values: {len(df)}")

    df = df.reset_index(drop=True)
    print(f"  Final DFSI districts: {len(df)}")
    return df

# ---------------------------------------------------------------------------
# Step 2  -  Load and aggregate India_Flood_Inventory_v3.csv
# ---------------------------------------------------------------------------

def load_inventory(path: str) -> pd.DataFrame:
    """
    Each row is a flood event.  Districts is a comma-separated list.
    We explode per district, clean, aggregate per district+state.
    """
    print("\n" + "="*60)
    print("STEP 2 - Loading India_Flood_Inventory_v3.csv")
    print("="*60)

    df = read_csv_robust(path)
    df = strip_col_names(df)

    # Drop unnamed index column(s) from CSV export
    unnamed_cols = [c for c in df.columns if re.match(r"^Unnamed", str(c))]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Raw rows: {len(df)}")

    # ------- Inspect Severity -------
    use_severity = False
    if "Severity" in df.columns:
        severity_vals = df["Severity"].dropna().astype(str).str.strip().unique()
        numeric_sev = pd.to_numeric(pd.Series(severity_vals), errors="coerce")
        n_parseable = int(numeric_sev.notna().sum())
        print(f"\n  Severity column: {len(severity_vals)} unique values, "
              f"{n_parseable} numeric-parseable")
        print(f"  Sample severity values: {severity_vals[:10].tolist()}")
        if n_parseable < len(severity_vals) * 0.5:
            print("  WARNING: Severity column cannot be reliably interpreted as numeric. "
                  "Skipping 'severe_event_count' feature.")
            use_severity = False
        else:
            use_severity = True
    else:
        print("  WARNING: 'Severity' column not found. Skipping 'severe_event_count'.")

    # ------- Parse dates -------
    for col in ["Start Date", "End Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # ------- Parse numeric columns -------
    num_cols = {
        "Duration(Days)":   "duration_days",
        "Area Affected":    "area_affected",
        "Human fatality":   "human_fatality",
        "Human injured":    "human_injured",
        "Human Displaced":  "human_displaced",
        "Animal Fatality":  "animal_fatality",
    }
    for raw, clean in num_cols.items():
        if raw in df.columns:
            df[clean] = to_numeric_series(df[raw])

    if use_severity:
        df["severity_val"] = to_numeric_series(df["Severity"])

    # ------- Explode multi-district rows -------
    districts_col = "Districts"
    state_col     = "State"

    if districts_col not in df.columns:
        raise ValueError(f"Expected column '{districts_col}' not found in inventory.")
    if state_col not in df.columns:
        raise ValueError(f"Expected column '{state_col}' not found in inventory.")

    def split_field(val):
        if pd.isna(val) or str(val).strip() == "":
            return [np.nan]
        return [s.strip() for s in str(val).split(",") if s.strip()]

    df["district_list"] = df[districts_col].apply(split_field)
    df["state_list"]    = df[state_col].apply(split_field)

    # Explode districts
    df_exp = df.explode("district_list").rename(columns={"district_list": "ev_district"})

    # For state: if only one state use it; if multiple states the mapping is
    # unknown so use NaN to avoid incorrect state assignments.
    def safe_state(row):
        sl = row["state_list"]
        if isinstance(sl, list) and len(sl) == 1:
            return sl[0]
        return np.nan

    df_exp["ev_state"] = df_exp.apply(safe_state, axis=1)
    df_exp = df_exp.drop(columns=["state_list"])

    # Drop rows with no district
    df_exp = df_exp[
        df_exp["ev_district"].notna() &
        (df_exp["ev_district"].astype(str).str.strip() != "nan") &
        (df_exp["ev_district"].astype(str).str.strip() != "")
    ]
    print(f"  Rows after district-explosion and NaN-district drop: {len(df_exp)}")

    # Normalise keys
    df_exp["district_key"] = normalize_name(df_exp["ev_district"])
    df_exp["state_key"]    = normalize_name(df_exp["ev_state"].fillna(""))

    # ------- Aggregate per district+state -------
    grp = df_exp.groupby(["district_key", "state_key"], sort=False)

    # Build agg kwargs conditionally
    agg_kwargs = {
        "flood_event_count":      pd.NamedAgg(column="duration_days", aggfunc="count"),
        "avg_flood_duration":     pd.NamedAgg(column="duration_days", aggfunc="mean"),
        "max_flood_duration":     pd.NamedAgg(column="duration_days", aggfunc="max"),
        "median_flood_duration":  pd.NamedAgg(column="duration_days", aggfunc="median"),
        "std_flood_duration":     pd.NamedAgg(column="duration_days", aggfunc="std"),
        "total_area_affected":    pd.NamedAgg(column="area_affected",  aggfunc="sum"),
        "avg_area_affected":      pd.NamedAgg(column="area_affected",  aggfunc="mean"),
        "total_human_fatalities": pd.NamedAgg(column="human_fatality", aggfunc="sum"),
        "total_human_injuries":   pd.NamedAgg(column="human_injured",  aggfunc="sum"),
        "total_human_displaced":  pd.NamedAgg(column="human_displaced",aggfunc="sum"),
        "total_animal_fatalities":pd.NamedAgg(column="animal_fatality",aggfunc="sum"),
    }
    if use_severity:
        agg_kwargs["severe_event_count"] = pd.NamedAgg(column="severity_val", aggfunc="count")

    results = grp.agg(**agg_kwargs).reset_index()

    print(f"  Unique district+state combos in inventory: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# Step 3  -  Load District_FloodedArea.csv
# ---------------------------------------------------------------------------

def load_flooded_area(path: str) -> pd.DataFrame:
    """
    Columns: Dist_Name, Percent_Flooded_Area, Parmanent_Water,
             Corrected_Percent_Flooded_Area
    No state column - district-only data.
    """
    print("\n" + "="*60)
    print("STEP 3 - Loading District_FloodedArea.csv")
    print("="*60)

    df = read_csv_robust(path)
    df = strip_col_names(df)

    required = {"Dist_Name", "Percent_Flooded_Area",
                "Parmanent_Water", "Corrected_Percent_Flooded_Area"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"District_FloodedArea.csv missing columns: {missing}\n"
                         f"Found: {df.columns.tolist()}")

    df["district_key"] = normalize_name(df["Dist_Name"])

    df["percent_flooded_area"]           = to_numeric_series(df["Percent_Flooded_Area"])
    df["permanent_water"]                = to_numeric_series(df["Parmanent_Water"])  # sic – original spelling
    df["corrected_percent_flooded_area"] = to_numeric_series(df["Corrected_Percent_Flooded_Area"])

    df = df[["district_key", "percent_flooded_area",
             "permanent_water", "corrected_percent_flooded_area"]].copy()

    print(f"  Rows loaded: {len(df)}")
    dup_keys = df[df.duplicated("district_key", keep=False)]["district_key"].unique()
    if len(dup_keys):
        print(f"  Duplicate district keys (same name, different rows): {len(dup_keys)}")
        print(f"    {list(dup_keys[:10])}")
    return df


# ---------------------------------------------------------------------------
# Step 4  -  Load District_FloodImpact.csv
# ---------------------------------------------------------------------------

def load_flood_impact(path: str) -> pd.DataFrame:
    """
    Columns: Dist_Name, Human_fatality, Human_injured, Population,
             Mean_Flood_Duration
    No state column.
    """
    print("\n" + "="*60)
    print("STEP 4 - Loading District_FloodImpact.csv")
    print("="*60)

    df = read_csv_robust(path)
    df = strip_col_names(df)

    required = {"Dist_Name", "Human_fatality", "Human_injured",
                "Population", "Mean_Flood_Duration"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"District_FloodImpact.csv missing columns: {missing}\n"
                         f"Found: {df.columns.tolist()}")

    df["district_key"] = normalize_name(df["Dist_Name"])

    df["impact_human_fatalities"]    = to_numeric_series(df["Human_fatality"])
    df["impact_human_injuries"]      = to_numeric_series(df["Human_injured"])
    df["population"]                 = to_numeric_series(df["Population"])
    df["impact_mean_flood_duration"] = to_numeric_series(df["Mean_Flood_Duration"])

    df = df[["district_key", "impact_human_fatalities", "impact_human_injuries",
             "population", "impact_mean_flood_duration"]].copy()

    print(f"  Rows loaded: {len(df)}")
    dup_keys = df[df.duplicated("district_key", keep=False)]["district_key"].unique()
    if len(dup_keys):
        print(f"  Duplicate district keys (same name, different rows): {len(dup_keys)}")
        print(f"    {list(dup_keys[:10])}")
    return df


# ---------------------------------------------------------------------------
# Step 5  -  Load LGD reference (for validation/disambiguation)
# ---------------------------------------------------------------------------

def load_lgd(path: str) -> pd.DataFrame:
    """
    Returns a cleaned LGD reference table with normalised district + state keys.
    Used for disambiguation and validation only; LGD codes are not added to output.
    """
    print("\n" + "="*60)
    print("STEP 5 - Loading LGD_Districts.csv (reference)")
    print("="*60)

    df = read_csv_robust(path)
    df = strip_col_names(df)
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Rows: {len(df)}")

    name_col  = "district_name_english"
    state_col = "state_name_english"
    if name_col not in df.columns or state_col not in df.columns:
        print("  WARNING: Expected LGD English-name columns not found. "
              "LGD validation will be skipped.")
        return pd.DataFrame(columns=["district_key", "state_key"])

    # Strip trailing whitespace (LGD file has lots of padding)
    df[name_col]  = df[name_col].astype(str).str.strip()
    df[state_col] = df[state_col].astype(str).str.strip()

    df["district_key"] = normalize_name(df[name_col])
    df["state_key"]    = normalize_name(df[state_col])

    lgd = df[["district_key", "state_key"]].drop_duplicates().reset_index(drop=True)
    print(f"  Unique district+state pairs in LGD: {len(lgd)}")
    return lgd


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def ambiguous_district_keys(dfsi: pd.DataFrame, source_name: str) -> set:
    """
    Return district_keys that appear in >1 state within the DFSI table.
    These cannot be safely joined from a source that has no state column.
    """
    state_counts = dfsi.groupby("district_key")["state_key"].nunique()
    ambiguous = set(state_counts[state_counts > 1].index)
    if ambiguous:
        print(f"\n  [Ambiguity] {len(ambiguous)} district name(s) appear in "
              f"multiple states in DFSI. They cannot be safely matched from "
              f"'{source_name}' (no state column) and will be left as NaN:")
        for k in sorted(ambiguous)[:20]:
            print(f"    '{k}'")
    return ambiguous


def merge_district_only(dfsi_keys: pd.DataFrame, source_df: pd.DataFrame,
                         source_name: str, ambiguous_keys: set) -> pd.DataFrame:
    """
    Merge a district-name-only source onto the DFSI key table.
    Rows whose district_key is ambiguous receive NaN for all source columns.

    dfsi_keys: DataFrame with columns [district_key, state_key] (one row per DFSI district)
    source_df: DataFrame with column [district_key] + feature columns
    """
    src_cols = [c for c in source_df.columns if c != "district_key"]

    # Split into safe (unambiguous) and ambiguous DFSI rows
    safe_mask  = ~dfsi_keys["district_key"].isin(ambiguous_keys)
    safe_rows  = dfsi_keys[safe_mask].copy()
    ambig_rows = dfsi_keys[~safe_mask].copy()

    merged_safe  = safe_rows.merge(source_df, on="district_key", how="left")
    for col in src_cols:
        ambig_rows[col] = np.nan

    merged = pd.concat([merged_safe, ambig_rows], ignore_index=True)

    # Re-join to restore original DFSI row order
    result = dfsi_keys.merge(merged, on=["district_key", "state_key"], how="left")

    n_matched   = result[src_cols[0]].notna().sum() if src_cols else 0
    n_unmatched = result[src_cols[0]].isna().sum()  if src_cols else len(result)
    print(f"  [{source_name}] matched  : {n_matched}")
    print(f"  [{source_name}] unmatched: {n_unmatched}")
    return result


# ---------------------------------------------------------------------------
# Step 6  -  Risk classification
# ---------------------------------------------------------------------------

def create_risk_class(dfsi_series: pd.Series) -> pd.Series:
    """
    Assign four risk classes using rank-based quartiles so that duplicate DFSI
    values do not prevent cut-points from being established.

    Q25  -> Low
    Q25-Q50 -> Moderate
    Q50-Q75 -> High
    Q75+ -> Very High

    Higher DFSI = higher risk (confirmed by the DFSI scoring formula).
    """
    q25 = float(dfsi_series.quantile(0.25))
    q50 = float(dfsi_series.quantile(0.50))
    q75 = float(dfsi_series.quantile(0.75))

    print(f"\n  DFSI quartile thresholds (computed from full distribution):")
    print(f"    Q25 = {q25:.6f}")
    print(f"    Q50 = {q50:.6f}")
    print(f"    Q75 = {q75:.6f}")

    # Rank-based assignment to handle ties robustly
    ranks = dfsi_series.rank(method="first", na_option="keep")
    n     = int(dfsi_series.notna().sum())

    def assign_class(rank):
        if pd.isna(rank):
            return np.nan
        pct = rank / n
        if pct <= 0.25:
            return "Low"
        elif pct <= 0.50:
            return "Moderate"
        elif pct <= 0.75:
            return "High"
        else:
            return "Very High"

    return ranks.apply(assign_class)


# ---------------------------------------------------------------------------
# Step 7  -  Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    """Print data quality summary and assert key invariants."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    print(f"  Dataset shape            : {df.shape}")
    print(f"  Number of districts      : {df['district'].nunique()}")
    print(f"  Number of states         : {df['state'].nunique()}")

    dfsi_vals = df["dfsi"].dropna()
    print(f"  DFSI min                 : {dfsi_vals.min():.4f}")
    print(f"  DFSI max                 : {dfsi_vals.max():.4f}")
    print(f"  DFSI mean                : {dfsi_vals.mean():.4f}")
    print(f"  DFSI median              : {dfsi_vals.median():.4f}")

    print("\n  Risk class counts:")
    rc = df["risk_class"].value_counts()
    for label in ["Low", "Moderate", "High", "Very High"]:
        print(f"    {label:<12}: {rc.get(label, 0)}")

    # Uniqueness
    dup_check = df[df.duplicated(subset=["district", "state"], keep=False)]
    if not dup_check.empty:
        print(f"\n  WARNING: {len(dup_check)} rows are duplicate district+state pairs:")
        print(dup_check[["district", "state"]].to_string(index=False))
    else:
        print(f"\n  district+state uniqueness : OK (all unique)")

    # DFSI validity
    n_invalid_dfsi = int(df["dfsi"].isna().sum())
    if n_invalid_dfsi > 0:
        print(f"  WARNING: {n_invalid_dfsi} rows have missing DFSI values.")
    else:
        print(f"  dfsi numeric validity     : OK (no missing values)")

    # Risk class validity
    valid_classes = {"Low", "Moderate", "High", "Very High"}
    bad_rc = df["risk_class"].dropna()[~df["risk_class"].dropna().isin(valid_classes)]
    if not bad_rc.empty:
        print(f"  WARNING: Unexpected risk_class values: {bad_rc.unique().tolist()}")
    else:
        print(f"  risk_class validity       : OK")

    # Missing-value summary
    print("\n  Missing-value counts per column:")
    mv = df.isna().sum()
    for col, cnt in mv.items():
        pct = 100.0 * int(cnt) / len(df)
        flag = "  <- ALL MISSING" if cnt == len(df) else ""
        print(f"    {col:<42}: {int(cnt):>5}  ({pct:5.1f}%){flag}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#"*60)
    print("  District Flood Risk Dataset Builder")
    print("#"*60)

    # ---- Load all sources ----
    dfsi_df   = load_dfsi(FILES["dfsi"])
    inv_df    = load_inventory(FILES["inventory"])
    area_df   = load_flooded_area(FILES["area"])
    impact_df = load_flood_impact(FILES["impact"])
    lgd_df    = load_lgd(LGD_PATH)

    n_dfsi = len(dfsi_df)
    print(f"\n  Number of DFSI districts (base population): {n_dfsi}")

    # ---- Base frame: one row per DFSI district ----
    base = dfsi_df[["District_Name", "State_Name",
                    "district_key", "state_key", "DFSI"]].copy()

    # ============================================================
    # MERGE 1 – Event inventory (has district + state keys)
    # ============================================================
    print("\n" + "="*60)
    print("MERGE 1 - Inventory onto DFSI")
    print("="*60)

    # Inventory rows with a resolved single state key
    inv_with_state    = inv_df[inv_df["state_key"] != ""].copy()
    # Inventory rows whose state could not be determined (multi-state events)
    inv_without_state = inv_df[inv_df["state_key"] == ""].copy()

    inv_feat_cols = [c for c in inv_df.columns
                     if c not in ("district_key", "state_key")]

    # Primary join on both keys
    merged = base.merge(inv_with_state, on=["district_key", "state_key"],
                        how="left", suffixes=("", "_inv"))

    # Secondary fallback: for still-unmatched DFSI rows, try district-only
    # join against multi-state inventory rows (imprecise, but better than nothing).
    # We only apply this for district_keys that are unambiguous across states.
    unmatched_mask = merged[inv_feat_cols[0]].isna()
    if unmatched_mask.any() and len(inv_without_state) > 0:
        ambig_inv = ambiguous_district_keys(dfsi_df, "Inventory fallback")
        safe_unmatched = unmatched_mask & ~base["district_key"].isin(ambig_inv)
        if safe_unmatched.any():
            fill_base = base[safe_unmatched][["district_key"]].merge(
                inv_without_state.drop(columns=["state_key"]),
                on="district_key", how="left",
            )
            for col in inv_feat_cols:
                if col in fill_base.columns:
                    merged.loc[safe_unmatched, col] = fill_base[col].values

    n_matched_inv   = int(merged[inv_feat_cols[0]].notna().sum())
    n_unmatched_inv = int(merged[inv_feat_cols[0]].isna().sum())
    print(f"  Matched to inventory   : {n_matched_inv}")
    print(f"  Unmatched from inv.    : {n_unmatched_inv}")

    # ============================================================
    # MERGE 2 – Flooded area (district-name only)
    # ============================================================
    print("\n" + "="*60)
    print("MERGE 2 - Flooded Area onto DFSI")
    print("="*60)

    dfsi_keys   = merged[["district_key", "state_key"]].copy()
    ambig_area  = ambiguous_district_keys(dfsi_df, "FloodedArea")
    area_result = merge_district_only(dfsi_keys, area_df, "FloodedArea", ambig_area)

    area_cols = ["percent_flooded_area", "permanent_water",
                 "corrected_percent_flooded_area"]
    for col in area_cols:
        if col in area_result.columns:
            merged[col] = area_result[col].values

    n_matched_area = int(merged["percent_flooded_area"].notna().sum()) \
        if "percent_flooded_area" in merged.columns else 0

    # ============================================================
    # MERGE 3 – Flood impact (district-name only)
    # ============================================================
    print("\n" + "="*60)
    print("MERGE 3 - Flood Impact onto DFSI")
    print("="*60)

    ambig_impact  = ambiguous_district_keys(dfsi_df, "FloodImpact")
    impact_result = merge_district_only(dfsi_keys, impact_df, "FloodImpact", ambig_impact)

    impact_cols = ["impact_human_fatalities", "impact_human_injuries",
                   "population", "impact_mean_flood_duration"]
    for col in impact_cols:
        if col in impact_result.columns:
            merged[col] = impact_result[col].values

    n_matched_impact = int(merged["population"].notna().sum()) \
        if "population" in merged.columns else 0

    # ============================================================
    # RISK CLASSIFICATION
    # ============================================================
    print("\n" + "="*60)
    print("RISK CLASSIFICATION (DFSI quartiles)")
    print("="*60)

    merged["risk_class"] = create_risk_class(merged["DFSI"])

    # ============================================================
    # POST-MERGE CLEANING
    # ============================================================
    # Districts genuinely absent from the inventory get flood_event_count = 0
    # (true absence of records), while aggregated metrics remain NaN.
    if "flood_event_count" in merged.columns:
        merged["flood_event_count"] = (
            merged["flood_event_count"].fillna(0).astype(int)
        )

    # ============================================================
    # BUILD FINAL COLUMN SET
    # ============================================================
    base_output_cols = [
        "District_Name", "State_Name",
        "flood_event_count",
        "avg_flood_duration", "max_flood_duration",
        "median_flood_duration", "std_flood_duration",
        "total_area_affected", "avg_area_affected",
        "total_human_fatalities", "total_human_injuries",
        "total_human_displaced", "total_animal_fatalities",
        "percent_flooded_area", "permanent_water",
        "corrected_percent_flooded_area",
        "population",
        "impact_mean_flood_duration",
        "impact_human_fatalities", "impact_human_injuries",
        "DFSI", "risk_class",
    ]

    # Insert severe_event_count before area columns, if available
    if "severe_event_count" in merged.columns:
        idx = base_output_cols.index("percent_flooded_area")
        base_output_cols.insert(idx, "severe_event_count")

    # Only keep columns that actually exist after merging
    output_cols = [c for c in base_output_cols if c in merged.columns]
    final = merged[output_cols].copy()

    # Rename to consistent lowercase snake_case
    final = final.rename(columns={
        "District_Name": "district",
        "State_Name":    "state",
        "DFSI":          "dfsi",
    })

    # Safety assertion: DFSI must not leak under any alias
    assert "dfsi" in final.columns, "dfsi column is missing from final dataset"
    assert "DFSI" not in final.columns, "Raw 'DFSI' column leaked into final dataset"

    # ============================================================
    # VALIDATION
    # ============================================================
    validate(final)

    # ============================================================
    # MATCH STATISTICS SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("MATCH STATISTICS SUMMARY")
    print("="*60)
    n_zero_events = int((final["flood_event_count"] == 0).sum()) \
        if "flood_event_count" in final.columns else 0
    dup_pairs = int(final.duplicated(subset=["district", "state"]).sum())

    print(f"  DFSI districts (base population)            : {n_dfsi}")
    print(f"  Final dataset districts                     : {len(final)}")
    print(f"  Matched to inventory (event records)        : {n_matched_inv}")
    print(f"  Unmatched from inventory                    : {n_unmatched_inv}")
    print(f"  Districts with 0 flood events               : {n_zero_events}")
    print(f"  Matched to flooded-area data                : {n_matched_area}")
    print(f"  Matched to impact data                      : {n_matched_impact}")
    print(f"  Duplicate district+state pairs in output    : {dup_pairs}")

    # ============================================================
    # SAVE
    # ============================================================
    os.makedirs(OUT_DIR, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    print("\n" + "#"*60)
    print(f"  Dataset successfully created:")
    print(f"  {OUT_PATH}")
    print("#"*60)

    print("\n  First 5 rows:")
    print(final.head(5).to_string(index=False))

    print("\n  Final column names:")
    for col in final.columns:
        print(f"    {col}")


if __name__ == "__main__":
    main()
