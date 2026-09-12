"""
evaluate_models_cv.py
=====================
AI-Powered District Flood Risk Assessment & Decision Support System
-------------------------------------------------------------------
Cross-validation experiment: district flood-risk classification
benchmarked against the District Flood Severity Index (DFSI).

This script does NOT claim to predict future floods.
The target (risk_class) is a historically derived DFSI quartile label.

Performs 5-fold Stratified Cross-Validation for the same two Random
Forest feature sets defined in train_models.py, to assess whether the
80/20 hold-out result is stable across different data partitions.

Usage
-----
    python evaluate_models_cv.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants  (must match train_models.py exactly)
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("data", "processed", "district_flood_risk_dataset.csv")
OUTPUT_DIR   = "outputs"
OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "cross_validation_results.csv")

RANDOM_SEED  = 42
N_SPLITS     = 5
N_ESTIMATORS = 300
CLASS_ORDER  = ["Low", "Moderate", "High", "Very High"]

MODEL1_NAME  = "Model 1 -- Historical Flood Characteristics"
MODEL2_NAME  = "Model 2 -- Physical Exposure Only"

MODEL1_FEATURES = [
    "total_human_displaced",
    "total_animal_fatalities",
    "max_flood_duration",
    "median_flood_duration",
    "std_flood_duration",
    "percent_flooded_area",
    "corrected_percent_flooded_area",
    "permanent_water",
]

MODEL2_FEATURES = [
    "percent_flooded_area",
    "corrected_percent_flooded_area",
    "permanent_water",
]

# Metrics to report (sklearn scoring keys)
METRICS = {
    "accuracy":        "accuracy",
    "macro_precision": "precision_macro",
    "macro_recall":    "recall_macro",
    "macro_f1":        "f1_macro",
    "weighted_f1":     "f1_weighted",
}

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
except ImportError as e:
    sys.exit(
        f"\nMissing required package: {e}\n"
        "Install with: pip install scikit-learn\n"
    )


# ===========================================================================
# Helpers
# ===========================================================================

def load_and_validate(path: str) -> pd.DataFrame:
    """Load the processed dataset and apply pre-CV sanity checks."""
    print(f"  Loading: {path}")
    df = pd.read_csv(path)

    n_before = len(df)
    df = df.dropna(subset=["risk_class"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} row(s) with missing risk_class.")

    dups = df.duplicated(subset=["district", "state"]).sum()
    if dups:
        print(f"  WARNING: {dups} duplicate district+state pair(s) detected.")
    else:
        print(f"  district + state uniqueness: PASSED")

    bad = set(df["risk_class"].unique()) - set(CLASS_ORDER)
    if bad:
        sys.exit(f"  ERROR: Unexpected risk_class values: {bad}")

    print(f"  Rows after cleaning : {len(df)}")
    print(f"  Class distribution:")
    for cls in CLASS_ORDER:
        n = (df["risk_class"] == cls).sum()
        pct = 100.0 * n / len(df)
        print(f"    {cls:<12}: {n:>3}  ({pct:.1f}%)")

    return df


def build_pipeline() -> Pipeline:
    """
    Identical pipeline to train_models.py:
        SimpleImputer(strategy='median') -> RandomForestClassifier
    Imputer is fitted inside each CV fold -- no leakage.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def run_cv(df: pd.DataFrame, features: list, model_name: str) -> pd.DataFrame:
    """
    Run N_SPLITS-fold Stratified CV for one model.
    Returns a DataFrame with one row per fold plus a summary block.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(sep)
    print(f"  Features ({len(features)}): {features}")

    # Guard: dfsi and risk_class must never be features
    leaked = [f for f in features if f in ("dfsi", "risk_class")]
    if leaked:
        sys.exit(f"  ERROR: Target/reference column(s) in feature list: {leaked}")

    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        sys.exit(f"  ERROR: Feature(s) not found in dataset: {missing_cols}")

    X = df[features].copy()
    y = df["risk_class"].copy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    pipe = build_pipeline()

    print(f"\n  Running {N_SPLITS}-fold Stratified CV ...")
    cv_results = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=METRICS,
        return_train_score=False,
        n_jobs=1,   # parallelise at RF level (n_jobs=-1 inside RF)
    )

    # ---- Per-fold report ----
    metric_keys = list(METRICS.keys())
    scorer_keys = [f"test_{v.replace('_macro','_macro').replace('_weighted','_weighted')}"
                   for v in METRICS.values()]
    # cross_validate uses "test_<scoring_key>" where the key is the dict key
    # i.e. test_accuracy, test_macro_precision, etc.
    scorer_result_keys = [f"test_{k}" for k in metric_keys]

    print(f"\n  {'Fold':<6}", end="")
    for k in metric_keys:
        label = k.replace("_", " ").title()
        print(f"  {label:>18}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in metric_keys:
        print(f"  {'------------------':>18}", end="")
    print()

    fold_rows = []
    for fold_idx in range(N_SPLITS):
        print(f"  {fold_idx + 1:<6}", end="")
        row = {"model": model_name, "fold": fold_idx + 1}
        for k in metric_keys:
            val = float(cv_results[f"test_{k}"][fold_idx])
            row[k] = round(val, 4)
            print(f"  {val:>18.4f}", end="")
        print()
        fold_rows.append(row)

    # ---- Summary statistics ----
    print(f"\n  {'Metric':<25} {'Mean':>8} {'Std Dev':>10}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*65}")

    summary_rows = []
    for k in metric_keys:
        vals = cv_results[f"test_{k}"]
        mean = float(vals.mean())
        std  = float(vals.std())
        mn   = float(vals.min())
        mx   = float(vals.max())
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {mean:>8.4f}  {std:>8.4f}  {mn:>8.4f}  {mx:>8.4f}")
        summary_rows.append({
            "model":  model_name,
            "fold":   "summary_mean",
            k:        round(mean, 4),
        })
        summary_rows.append({
            "model":  model_name,
            "fold":   "summary_std",
            k:        round(std, 4),
        })

    return pd.DataFrame(fold_rows), summary_rows, cv_results


def print_comparison(
    m1_name: str, m1_cv: dict,
    m2_name: str, m2_cv: dict,
) -> None:
    """Print Macro F1 comparison and robustness interpretation."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  CROSS-VALIDATION COMPARISON")
    print(sep)

    # Header
    print(f"\n  {'Model':<48} {'Macro F1 Mean':>14} {'Macro F1 Std':>14}")
    print(f"  {'-'*78}")

    m1_f1 = m1_cv["test_macro_f1"]
    m2_f1 = m2_cv["test_macro_f1"]

    print(f"  {m1_name:<48} {m1_f1.mean():>14.4f} {m1_f1.std():>14.4f}")
    print(f"  {m2_name:<48} {m2_f1.mean():>14.4f} {m2_f1.std():>14.4f}")

    diff = m1_f1.mean() - m2_f1.mean()
    print(f"\n  Macro F1 difference (Model 1 - Model 2) : {diff:+.4f}")

    # Full metric summary table
    print(f"\n  Full metric summary (mean +/- std):")
    print(f"\n  {'Metric':<25} {'M1 Mean':>9} {'M1 Std':>8}   {'M2 Mean':>9} {'M2 Std':>8}")
    print(f"  {'-'*70}")
    for k in METRICS.keys():
        m1_v = m1_cv[f"test_{k}"]
        m2_v = m2_cv[f"test_{k}"]
        label = k.replace("_", " ").title()
        print(f"  {label:<25} {m1_v.mean():>9.4f} {m1_v.std():>8.4f}   "
              f"{m2_v.mean():>9.4f} {m2_v.std():>8.4f}")

    # Robustness interpretation
    print(f"\n{sep}")
    print("  ROBUSTNESS INTERPRETATION")
    print(sep)

    # 1. Consistent superiority?
    m1_wins = (m1_f1 > m2_f1).sum()
    print(f"\n  1. Is Model 1 consistently better than Model 2?")
    print(f"     Model 1 had higher Macro F1 in {m1_wins}/{N_SPLITS} folds.")
    if m1_wins == N_SPLITS:
        print(f"     Yes -- Model 1 outperforms Model 2 in every fold.")
    elif m1_wins >= 4:
        print(f"     Yes -- Model 1 outperforms Model 2 in nearly every fold.")
    else:
        print(f"     Inconsistent -- Model 1 does not reliably outperform Model 2.")

    # 2. Stability of Model 1?
    cv_coef = m1_f1.std() / m1_f1.mean() if m1_f1.mean() > 0 else float("inf")
    print(f"\n  2. Is Model 1's performance reasonably stable?")
    print(f"     Macro F1 = {m1_f1.mean():.4f} +/- {m1_f1.std():.4f}  "
          f"(CV = {cv_coef:.3f})")
    if m1_f1.std() <= 0.04:
        stability = "stable -- low variance across folds."
    elif m1_f1.std() <= 0.07:
        stability = "moderately stable -- some fold-to-fold variation."
    else:
        stability = "variable -- substantial fold-to-fold variation; interpret with caution."
    print(f"     Performance is {stability}")

    # 3. Support continuing with Model 1?
    print(f"\n  3. Does the result support continuing with Model 1?")
    if m1_f1.mean() > 0.55 and m1_f1.std() <= 0.07:
        print(f"     Yes. A mean Macro F1 of {m1_f1.mean():.4f} above a four-class")
        print(f"     random baseline of ~0.25 indicates genuine predictive signal.")
        print(f"     Model 1 is an appropriate starting point for the benchmark.")
    elif m1_f1.mean() > 0.40:
        print(f"     Cautiously yes. Mean Macro F1 of {m1_f1.mean():.4f} shows signal")
        print(f"     above chance, but results should be interpreted conservatively.")
    else:
        print(f"     With caution. Macro F1 of {m1_f1.mean():.4f} is only modestly above")
        print(f"     the ~0.25 random baseline. Additional features are recommended.")

    # 4. Was the 80/20 result unusually optimistic?
    # Compare CV mean to the single-split result reported from train_models.py
    TRAIN_MODELS_M1_F1 = 0.6536   # reported by train_models.py
    TRAIN_MODELS_M2_F1 = 0.3590
    delta1 = TRAIN_MODELS_M1_F1 - m1_f1.mean()
    delta2 = TRAIN_MODELS_M2_F1 - m2_f1.mean()
    print(f"\n  4. Was the original 80/20 result unusually optimistic?")
    print(f"     Model 1: 80/20 Macro F1 = {TRAIN_MODELS_M1_F1:.4f}, "
          f"CV mean = {m1_f1.mean():.4f}, delta = {delta1:+.4f}")
    print(f"     Model 2: 80/20 Macro F1 = {TRAIN_MODELS_M2_F1:.4f}, "
          f"CV mean = {m2_f1.mean():.4f}, delta = {delta2:+.4f}")
    if abs(delta1) <= 0.03:
        optimism = "The 80/20 result was consistent with CV -- no sign of optimism bias."
    elif delta1 > 0.03:
        optimism = ("The 80/20 result was slightly optimistic relative to CV mean. "
                    "CV estimate is more reliable for generalisation.")
    else:
        optimism = ("The 80/20 result was slightly conservative; "
                    "CV suggests the model generalises at least as well.")
    print(f"     {optimism}")

    print(f"\n  Scientific framing reminder:")
    print(f"     Both models perform district flood-risk CLASSIFICATION,")
    print(f"     not future flood prediction. Results represent how well")
    print(f"     the selected features reproduce the historical DFSI-based")
    print(f"     quartile ranking across India's districts.")
    print()


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\nDistrict Flood Risk -- Cross-Validation Experiment")
    print("=" * 50)
    print(f"Dataset    : {DATASET_PATH}")
    print(f"Target     : risk_class  (4-class DFSI-derived quartile label)")
    print(f"CV scheme  : {N_SPLITS}-fold Stratified K-Fold")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Estimators : {N_ESTIMATORS} trees per fold")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print("  DATA LOADING & VALIDATION")
    print("=" * 60)
    df = load_and_validate(DATASET_PATH)

    # ---- Run CV for both models ----
    fold_df1, _, cv1_raw = run_cv(df, MODEL1_FEATURES, MODEL1_NAME)
    fold_df2, _, cv2_raw = run_cv(df, MODEL2_FEATURES, MODEL2_NAME)

    # ---- Comparison + interpretation ----
    print_comparison(MODEL1_NAME, cv1_raw, MODEL2_NAME, cv2_raw)

    # ---- Save results CSV ----
    # Combine fold-level results for both models; add summary rows
    metric_keys = list(METRICS.keys())

    def make_summary_rows(model_name: str, cv_raw: dict) -> pd.DataFrame:
        rows = []
        for stat, fn in [("mean", np.mean), ("std", np.std),
                         ("min", np.min), ("max", np.max)]:
            row = {"model": model_name, "fold": f"summary_{stat}"}
            for k in metric_keys:
                row[k] = round(float(fn(cv_raw[f"test_{k}"])), 4)
            rows.append(row)
        return pd.DataFrame(rows)

    combined = pd.concat([
        fold_df1,
        make_summary_rows(MODEL1_NAME, cv1_raw),
        fold_df2,
        make_summary_rows(MODEL2_NAME, cv2_raw),
    ], ignore_index=True)

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"  Cross-validation results saved: {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("  Cross-validation completed successfully.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
