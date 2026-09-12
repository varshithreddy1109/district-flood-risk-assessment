"""
predict.py
==========
AI-Powered District Flood Risk Assessment & Decision Support System
-------------------------------------------------------------------
Reusable inference layer for district flood-risk classification.

Scientific framing
------------------
This module classifies districts into historical DFSI-derived risk
quartiles (Low / Moderate / High / Very High). It does NOT predict
future floods, does NOT use rainfall, and does NOT establish causal
relationships between the features and actual flood severity.

The honest project description is:
    "AI-based district flood-risk classification and assessment
     benchmarked against the District Flood Severity Index (DFSI)."

Public API
----------
    train_model()                             -> fitted sklearn Pipeline
    predict_district(district, state=None)    -> prediction dict
    get_district_data(district, state=None)   -> raw district row dict
    get_all_districts()                       -> list of dicts
    get_model_metadata()                      -> metadata dict

Usage (as a module)
-------------------
    from predict import train_model, predict_district, get_model_metadata

Usage (standalone test)
-----------------------
    python predict.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants -- must match train_models.py / explain_model.py exactly
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("data", "processed", "district_flood_risk_dataset.csv")

RANDOM_SEED  = 42
N_ESTIMATORS = 300
CLASS_ORDER  = ["Low", "Moderate", "High", "Very High"]

MODEL_NAME   = "Model 1 -- Historical Flood Characteristics"

FEATURES = [
    "total_human_displaced",
    "total_animal_fatalities",
    "max_flood_duration",
    "median_flood_duration",
    "std_flood_duration",
    "percent_flooded_area",
    "corrected_percent_flooded_area",
    "permanent_water",
]

# Features exposed to the dashboard for contextual display (not model inputs)
DISPLAY_FEATURES = [
    "district",
    "state",
    "flood_event_count",
    "avg_flood_duration",
    "max_flood_duration",
    "median_flood_duration",
    "std_flood_duration",
    "total_human_displaced",
    "total_animal_fatalities",
    "percent_flooded_area",
    "permanent_water",
    "corrected_percent_flooded_area",
    "population",
    "impact_mean_flood_duration",
    "dfsi",
    "risk_class",
]

# Validated cross-validation metrics from evaluate_models_cv.py
CV_METRICS = {
    "cv_macro_f1":        0.5823,
    "cv_macro_f1_std":    0.0334,
    "cv_accuracy":        0.5855,
    "baseline_m2_macro_f1": 0.3162,
}

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except ImportError as e:
    sys.exit(
        f"\nMissing required package: {e}\n"
        "Install with: pip install scikit-learn\n"
    )

# ---------------------------------------------------------------------------
# Module-level cache: dataset and trained model
# ---------------------------------------------------------------------------
_df: pd.DataFrame = None
_pipeline: Pipeline = None


# ===========================================================================
# Internal helpers
# ===========================================================================

def _load_dataset() -> pd.DataFrame:
    """Load and cache the processed dataset. Drop rows with missing risk_class."""
    global _df
    if _df is not None:
        return _df

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Run build_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=["risk_class"])

    # Validate no duplicate district+state pairs
    dups = df.duplicated(subset=["district", "state"]).sum()
    if dups:
        warnings.warn(f"{dups} duplicate district+state pair(s) detected in dataset.")

    # Normalise text columns for case-insensitive lookup
    df["_district_key"] = df["district"].str.strip().str.lower()
    df["_state_key"]    = df["state"].str.strip().str.lower()

    _df = df
    return _df


def _find_district(df: pd.DataFrame,
                   district_name: str,
                   state_name: str = None) -> pd.DataFrame:
    """
    Locate a district row using case-insensitive matching.

    If state_name is provided, match on both district + state.
    If state_name is None and the district name is unambiguous, return it.
    If ambiguous (multiple states), raise a ValueError listing the matches.
    """
    d_key = district_name.strip().lower()

    # Partial-match fallback: first try exact, then startswith, then contains
    mask_exact = df["_district_key"] == d_key
    if mask_exact.any():
        candidates = df[mask_exact]
    else:
        mask_starts = df["_district_key"].str.startswith(d_key)
        if mask_starts.any():
            candidates = df[mask_starts]
        else:
            mask_contains = df["_district_key"].str.contains(d_key, na=False)
            if mask_contains.any():
                candidates = df[mask_contains]
            else:
                raise ValueError(
                    f"District '{district_name}' not found in dataset.\n"
                    "Use get_all_districts() to see available districts."
                )

    if state_name is not None:
        s_key = state_name.strip().lower()
        state_mask = candidates["_state_key"] == s_key
        if not state_mask.any():
            # try contains
            state_mask = candidates["_state_key"].str.contains(s_key, na=False)
        candidates = candidates[state_mask]
        if candidates.empty:
            raise ValueError(
                f"District '{district_name}' not found in state '{state_name}'.\n"
                "Use get_all_districts() to see available combinations."
            )

    if len(candidates) > 1:
        options = candidates[["district", "state"]].to_dict("records")
        opt_str = "\n  ".join(
            f"{r['district']} ({r['state']})" for r in options
        )
        raise ValueError(
            f"'{district_name}' is ambiguous -- found in multiple states:\n"
            f"  {opt_str}\n"
            "Provide state_name= to disambiguate."
        )

    return candidates


# ===========================================================================
# Public API
# ===========================================================================

def train_model() -> Pipeline:
    """
    Train the Model 1 Random Forest pipeline on ALL available cleaned rows.

    Returns
    -------
    pipe : fitted sklearn Pipeline (SimpleImputer -> RandomForestClassifier)

    Notes
    -----
    - Uses all rows with non-missing risk_class (not an 80/20 split).
    - Training on the full dataset improves production classification quality.
    - Validated performance (5-fold CV): Macro F1 = 0.5823 +/- 0.0334.
    """
    global _pipeline

    df = _load_dataset()
    X  = df[FEATURES].copy()
    y  = df["risk_class"].copy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    pipe.fit(X, y)
    _pipeline = pipe
    return pipe


def predict_district(district_name: str,
                     state_name: str = None) -> dict:
    """
    Classify one district's flood risk using the trained Model 1 pipeline.

    Parameters
    ----------
    district_name : str
        Full or partial district name (case-insensitive).
    state_name : str, optional
        State name to disambiguate districts with the same name in
        multiple states. Required when the district name is not unique.

    Returns
    -------
    dict with keys:
        district              str   -- canonical district name
        state                 str   -- state name
        predicted_risk_class  str   -- one of Low / Moderate / High / Very High
        class_probabilities   dict  -- {class_label: probability} for all 4 classes
        feature_values        dict  -- the 8 model input feature values used
        note                  str   -- scientific framing reminder

    Raises
    ------
    ValueError  if district is not found or is ambiguous
    RuntimeError if train_model() has not been called yet
    """
    global _pipeline

    if _pipeline is None:
        raise RuntimeError(
            "Model has not been trained yet. Call train_model() first."
        )

    df  = _load_dataset()
    row = _find_district(df, district_name, state_name).iloc[0]

    X_row = pd.DataFrame([row[FEATURES]])
    pred  = _pipeline.predict(X_row)[0]
    proba = _pipeline.predict_proba(X_row)[0]

    # Map class labels to probabilities (class order from the fitted model)
    classes = _pipeline.named_steps["clf"].classes_
    prob_dict = {cls: float(round(p, 4)) for cls, p in zip(classes, proba)}
    # Ensure all four classes are present even if one had 0 probability
    for cls in CLASS_ORDER:
        prob_dict.setdefault(cls, 0.0)

    feat_vals = {f: (None if pd.isna(row[f]) else float(row[f]))
                 for f in FEATURES}

    return {
        "district":             str(row["district"]),
        "state":                str(row["state"]),
        "predicted_risk_class": str(pred),
        "class_probabilities":  prob_dict,
        "feature_values":       feat_vals,
        "note": (
            "Classification of historical DFSI-derived flood risk. "
            "Does NOT predict future floods."
        ),
    }


def get_district_data(district_name: str,
                      state_name: str = None) -> dict:
    """
    Return a district's full historical data for dashboard display.

    Includes contextual columns beyond the 8 model features, such as
    population, DFSI score, and flood-event count.  DFSI is included
    here for display only -- it is never used as a model input.

    Parameters
    ----------
    district_name : str
    state_name    : str, optional

    Returns
    -------
    dict -- column values for DISPLAY_FEATURES (NaN -> None for JSON safety)
    """
    df  = _load_dataset()
    row = _find_district(df, district_name, state_name).iloc[0]

    data = {}
    for col in DISPLAY_FEATURES:
        if col in row.index:
            val = row[col]
            data[col] = None if pd.isna(val) else val
        else:
            data[col] = None
    return data


def get_all_districts() -> list:
    """
    Return a sorted list of all district/state combinations in the dataset.

    Returns
    -------
    list of dicts, each with keys:
        district    str
        state       str
        risk_class  str   -- actual DFSI-derived class (for reference)
    """
    df = _load_dataset()
    result = (
        df[["district", "state", "risk_class"]]
        .copy()
        .sort_values(["state", "district"])
        .reset_index(drop=True)
        .to_dict("records")
    )
    return result


def get_model_metadata() -> dict:
    """
    Return a metadata dictionary describing the trained model configuration
    and validated performance metrics.

    Returns
    -------
    dict with keys:
        model_name          str
        features            list[str]
        n_estimators        int
        random_seed         int
        training_rows       int
        target              str
        target_definition   str
        class_order         list[str]
        cv_macro_f1         float   -- mean across 5 folds
        cv_macro_f1_std     float
        cv_accuracy         float
        baseline_m2_macro_f1 float  -- physical-only model (Model 2)
        scientific_note     str
    """
    df = _load_dataset()
    return {
        "model_name":           MODEL_NAME,
        "features":             FEATURES.copy(),
        "n_estimators":         N_ESTIMATORS,
        "random_seed":          RANDOM_SEED,
        "training_rows":        len(df),
        "target":               "risk_class",
        "target_definition":    (
            "Quartile of the District Flood Severity Index (DFSI). "
            "Derived from historical flood records, NOT from future observations."
        ),
        "class_order":          CLASS_ORDER.copy(),
        "cv_macro_f1":          CV_METRICS["cv_macro_f1"],
        "cv_macro_f1_std":      CV_METRICS["cv_macro_f1_std"],
        "cv_accuracy":          CV_METRICS["cv_accuracy"],
        "baseline_m2_macro_f1": CV_METRICS["baseline_m2_macro_f1"],
        "scientific_note": (
            "This model classifies Indian districts into historical DFSI-derived "
            "flood risk quartiles. It does NOT predict future flood events, does NOT "
            "use rainfall data, and does NOT establish causal relationships. "
            "Reported CV Macro F1 = 0.5823 vs. a random four-class baseline of ~0.25."
        ),
    }


# ===========================================================================
# Standalone test  (python predict.py)
# ===========================================================================

def _print_separator(char: str = "=", width: int = 60) -> None:
    print(char * width)


def _run_test() -> None:
    print("\nDistrict Flood Risk -- Inference Layer Test")
    _print_separator()
    print(f"Dataset : {DATASET_PATH}")
    print(f"Model   : {MODEL_NAME}")
    print()

    # 1. Train on full dataset
    _print_separator()
    print("  Step 1: Training model on full cleaned dataset ...")
    _print_separator()
    pipe = train_model()
    meta = get_model_metadata()
    print(f"  Training rows      : {meta['training_rows']}")
    print(f"  Features           : {len(meta['features'])}")
    print(f"  Estimators         : {meta['n_estimators']}")
    print(f"  CV Macro F1        : {meta['cv_macro_f1']} +/- {meta['cv_macro_f1_std']}")
    print(f"  CV Accuracy        : {meta['cv_accuracy']}")
    print(f"  Baseline M2 F1     : {meta['baseline_m2_macro_f1']}")

    # 2. List all districts (summary)
    _print_separator()
    print("  Step 2: get_all_districts() -- sample")
    _print_separator()
    districts = get_all_districts()
    print(f"  Total districts available: {len(districts)}")
    print(f"  First 5:")
    for d in districts[:5]:
        print(f"    {d['district']:<35} | {d['state']:<20} | {d['risk_class']}")

    # 3. Predict a well-known high-risk district (Patna, Bihar)
    _print_separator()
    print("  Step 3: predict_district('Patna', 'BIHAR')")
    _print_separator()
    try:
        result = predict_district("Patna", "BIHAR")
        print(f"  District              : {result['district']}")
        print(f"  State                 : {result['state']}")
        print(f"  Predicted risk class  : {result['predicted_risk_class']}")
        print(f"\n  Class probabilities:")
        for cls in CLASS_ORDER:
            prob = result["class_probabilities"].get(cls, 0.0)
            bar  = "#" * int(prob * 30)
            print(f"    {cls:<12} : {prob:.4f}  {bar}")
        print(f"\n  Feature values used:")
        print(f"  {'Feature':<35} {'Value':>12}")
        print(f"  {'-'*50}")
        for feat, val in result["feature_values"].items():
            val_str = f"{val:.2f}" if val is not None else "NaN (imputed)"
            print(f"  {feat:<35} {val_str:>12}")
        print(f"\n  Note: {result['note']}")
    except ValueError as e:
        print(f"  ERROR: {e}")

    # 4. get_district_data for same district
    _print_separator()
    print("  Step 4: get_district_data('Patna', 'BIHAR') -- contextual display data")
    _print_separator()
    try:
        data = get_district_data("Patna", "BIHAR")
        print(f"  {'Field':<35} {'Value':>15}")
        print(f"  {'-'*53}")
        for k, v in data.items():
            v_str = str(round(v, 4)) if isinstance(v, float) else str(v)
            print(f"  {k:<35} {v_str:>15}")
    except ValueError as e:
        print(f"  ERROR: {e}")

    # 5. Model metadata
    _print_separator()
    print("  Step 5: get_model_metadata()")
    _print_separator()
    for k, v in meta.items():
        if k == "features":
            print(f"  {k:<25}: {v}")
        elif k == "scientific_note":
            print(f"  {k:<25}:")
            # Word-wrap at 65 chars
            words = v.split()
            line  = "    "
            for w in words:
                if len(line) + len(w) + 1 > 69:
                    print(line)
                    line = "    " + w + " "
                else:
                    line += w + " "
            if line.strip():
                print(line)
        else:
            print(f"  {k:<25}: {v}")

    _print_separator()
    print("  predict.py inference layer test complete.")
    _print_separator()
    print()


if __name__ == "__main__":
    _run_test()
