"""
explain_model.py
================
AI-Powered District Flood Risk Assessment & Decision Support System
-------------------------------------------------------------------
Explainability analysis for Model 1 -- Historical Flood Characteristics.

Produces:
  - Global permutation feature importance  (test-set)
  - Global SHAP feature importance         (test-set, if shap is installed)

Scientific framing
------------------
This script explains how Model 1 classifies districts into historical
DFSI-derived risk quartiles. It does NOT prove causal relationships and
does NOT constitute future flood prediction.

Usage
-----
    python explain_model.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants  -- must match train_models.py exactly
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("data", "processed", "district_flood_risk_dataset.csv")
OUTPUT_DIR   = "outputs"
RANDOM_SEED  = 42
TEST_SIZE    = 0.20
N_ESTIMATORS = 300
CLASS_ORDER  = ["Low", "Moderate", "High", "Very High"]

MODEL1_NAME  = "Model 1 -- Historical Flood Characteristics"

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

# Permutation importance: number of repeats (higher = more stable, slower)
PERM_N_REPEATS = 30

# ---------------------------------------------------------------------------
# Mandatory package imports
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
except ImportError as e:
    sys.exit(f"\nMissing required package: {e}\n"
             "Install with: pip install scikit-learn matplotlib\n")

# Optional seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Optional shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("\n  [INFO] shap package not installed.")
    print("  Permutation importance will still be produced.")
    print("  To enable SHAP: pip install shap\n")


# ===========================================================================
# Helpers
# ===========================================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.dropna(subset=["risk_class"])
    dropped = n_before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with missing risk_class.")
    dups = df.duplicated(subset=["district", "state"]).sum()
    if dups:
        print(f"  WARNING: {dups} duplicate district+state pair(s) detected.")
    else:
        print(f"  district + state uniqueness: PASSED")
    print(f"  Dataset shape: {df.shape}")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def ranked_table(title: str, feature_names: list, importances: np.ndarray,
                 std_devs: np.ndarray = None) -> pd.DataFrame:
    """Print and return a ranked importance DataFrame."""
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    })
    if std_devs is not None:
        df["std"] = std_devs
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    print(f"\n  {title}")
    if std_devs is not None:
        print(f"  {'Rank':<5}  {'Feature':<40}  {'Importance':>12}  {'Std Dev':>10}")
        print(f"  {'-'*72}")
        for _, row in df.iterrows():
            print(f"  {int(row['rank']):<5}  {row['feature']:<40}  "
                  f"{row['importance']:>12.4f}  {row['std']:>10.4f}")
    else:
        print(f"  {'Rank':<5}  {'Feature':<40}  {'Importance':>12}")
        print(f"  {'-'*60}")
        for _, row in df.iterrows():
            print(f"  {int(row['rank']):<5}  {row['feature']:<40}  "
                  f"{row['importance']:>12.4f}")
    return df


def save_importance_plot(fi_df: pd.DataFrame, title: str,
                         out_path: str, xlabel: str,
                         has_std: bool = False) -> None:
    """Horizontal bar chart of feature importances with optional error bars."""
    n = len(fi_df)
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * n + 1.8)))

    features_sorted = fi_df["feature"].tolist()
    values_sorted   = fi_df["importance"].tolist()
    errors_sorted   = fi_df["std"].tolist() if (has_std and "std" in fi_df.columns) else None

    colors = plt.cm.Blues(np.linspace(0.38, 0.88, n))

    bars = ax.barh(
        features_sorted,
        values_sorted,
        xerr=errors_sorted,
        color=colors,
        edgecolor="white",
        height=0.62,
        capsize=3,
        error_kw={"elinewidth": 1.1, "ecolor": "#555555"},
    )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # Value labels
    for bar, val in zip(bars, values_sorted):
        ax.text(
            bar.get_width() + (max(values_sorted) * 0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8.5, color="#333333",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ===========================================================================
# Permutation importance
# ===========================================================================

def compute_permutation_importance(pipe: Pipeline, X_test: pd.DataFrame,
                                   y_test: pd.Series) -> pd.DataFrame:
    """
    Compute permutation feature importance on the held-out test set.
    Uses macro F1 as the scoring metric (appropriate for 4-class balanced problem).
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  PERMUTATION FEATURE IMPORTANCE  (test set)")
    print(sep)
    print(f"  Metric : macro F1")
    print(f"  Repeats: {PERM_N_REPEATS}  (each feature permuted {PERM_N_REPEATS}x)")

    result = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )

    fi_df = ranked_table(
        "Permutation Importance (drop in Macro F1 when feature is shuffled):",
        FEATURES,
        result.importances_mean,
        result.importances_std,
    )

    csv_path  = os.path.join(OUTPUT_DIR, "permutation_feature_importance.csv")
    plot_path = os.path.join(OUTPUT_DIR, "permutation_feature_importance.png")
    fi_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved : {csv_path}")
    save_importance_plot(
        fi_df,
        title=f"Permutation Feature Importance\n{MODEL1_NAME}",
        out_path=plot_path,
        xlabel="Mean Drop in Macro F1 (higher = more important)",
        has_std=True,
    )
    return fi_df


# ===========================================================================
# SHAP analysis
# ===========================================================================

def compute_shap(pipe: Pipeline, X_train: pd.DataFrame,
                 X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values using TreeExplainer on the trained Random Forest.
    For multiclass output (shape: n_samples x n_classes x n_features),
    aggregate global importance as mean(|SHAP|) across all classes and samples.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SHAP FEATURE IMPORTANCE  (test set, TreeExplainer)")
    print(sep)

    # Extract the trained RF from the pipeline
    rf_model = pipe.named_steps["clf"]

    # Transform X_test through the pipeline's imputer only (no predict step)
    imputer   = pipe.named_steps["imputer"]
    X_test_t  = imputer.transform(X_test)
    X_train_t = imputer.transform(X_train)

    # Build explainer using a background sample for speed (max 150 rows)
    n_bg = min(150, len(X_train_t))
    rng  = np.random.default_rng(RANDOM_SEED)
    bg_idx     = rng.choice(len(X_train_t), size=n_bg, replace=False)
    X_bg       = X_train_t[bg_idx]

    print(f"  Building TreeExplainer (background sample: {n_bg} rows) ...")
    explainer = shap.TreeExplainer(
        rf_model,
        data=X_bg,
        feature_perturbation="interventional",
        model_output="raw",
    )

    print(f"  Computing SHAP values for {len(X_test_t)} test samples ...")
    shap_values = explainer.shap_values(X_test_t, check_additivity=False)

    # shap_values shape possibilities:
    #   - list of arrays [class_0, class_1, ...] each (n_samples, n_features)
    #   - single array  (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # List of (n_samples, n_features): stack -> (n_classes, n_samples, n_features)
        shap_arr = np.array(shap_values)              # (n_classes, n_samples, n_features)
    else:
        # (n_samples, n_features, n_classes) -> transpose
        shap_arr = np.transpose(shap_values, (2, 0, 1))  # (n_classes, n_samples, n_features)

    # Global importance: mean(|SHAP|) over samples and classes
    mean_abs_shap = np.mean(np.abs(shap_arr), axis=(0, 1))  # (n_features,)

    fi_df = ranked_table(
        "SHAP Global Importance (mean |SHAP value| across classes and test samples):",
        FEATURES,
        mean_abs_shap,
    )

    # Per-class SHAP table
    print(f"\n  Per-class mean |SHAP| breakdown:")
    header = f"  {'Feature':<40}" + "".join(f"  {c:>11}" for c in CLASS_ORDER)
    print(header)
    print(f"  {'-'*88}")
    for feat_idx, feat in enumerate(FEATURES):
        row_str = f"  {feat:<40}"
        for cls_idx in range(len(CLASS_ORDER)):
            val = float(np.mean(np.abs(shap_arr[cls_idx, :, feat_idx])))
            row_str += f"  {val:>11.4f}"
        print(row_str)

    csv_path  = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    plot_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
    fi_df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved : {csv_path}")
    save_importance_plot(
        fi_df,
        title=f"SHAP Global Feature Importance\n{MODEL1_NAME}",
        out_path=plot_path,
        xlabel="Mean |SHAP Value| (higher = more important across all classes)",
        has_std=False,
    )
    return fi_df


# ===========================================================================
# Interpretation
# ===========================================================================

def print_interpretation(perm_df: pd.DataFrame,
                          shap_df: pd.DataFrame = None) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  EXPLAINABILITY INTERPRETATION")
    print(sep)

    # Use SHAP ranking if available, else permutation
    if shap_df is not None:
        primary_df   = shap_df
        primary_name = "SHAP global importance"
    else:
        primary_df   = perm_df
        primary_name = "permutation importance (macro F1 drop)"

    top5 = primary_df.head(5)

    print(f"\n  Primary ranking method: {primary_name}")
    print(f"\n  Top 5 most important features:")
    for _, row in top5.iterrows():
        print(f"    {int(row['rank'])}. {row['feature']}  "
              f"(importance = {row['importance']:.4f})")

    print(f"""
  What importance values mean
  ---------------------------
  SHAP importance:
    A higher value means the feature shifts the model's predicted
    probability more (in absolute terms) across the four risk classes.
    A near-zero value means shuffling the feature barely changes
    the model's predictions.

  Permutation importance:
    A higher value means removing the feature's true signal
    (by random shuffling) causes a larger drop in macro F1.
    Negative values can occur when shuffling a weak feature
    accidentally improves predictions (noise effect).

  Consistency check
  -----------------""")

    # Check if top-2 agree between methods
    if shap_df is not None:
        perm_top2 = set(perm_df.head(2)["feature"].tolist())
        shap_top2 = set(shap_df.head(2)["feature"].tolist())
        overlap   = perm_top2 & shap_top2
        if overlap:
            print(f"    SHAP and permutation importance agree on top features:")
            for f in sorted(overlap):
                print(f"      - {f}")
        else:
            print(f"    SHAP and permutation importance top-2 differ.")
            print(f"    SHAP top-2 : {sorted(shap_top2)}")
            print(f"    Perm top-2 : {sorted(perm_top2)}")
            print(f"    This can occur when features are correlated; "
                  f"both methods remain valid.")

    print(f"""
  Scientific limitations
  ----------------------
  1. This analysis explains how Model 1 CLASSIFIES districts into
     historical DFSI-derived risk quartiles (Low / Moderate / High /
     Very High). It does NOT prove causal relationships between
     the features and actual flood severity.

  2. Higher feature importance means the feature is useful for
     reproducing the DFSI-based ranking -- it does not imply that
     the feature directly CAUSES higher flood risk.

  3. The model and its explanations are based entirely on historical
     data. Results must NOT be interpreted as predictions of future
     flood events or future risk.

  4. The honest framing of this project is:
     "AI-based district flood-risk classification and assessment
      benchmarked against the District Flood Severity Index (DFSI)."
""")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\nDistrict Flood Risk -- Model Explainability Analysis")
    print("=" * 52)
    print(f"Dataset    : {DATASET_PATH}")
    print(f"Model      : {MODEL1_NAME}")
    print(f"Target     : risk_class  (4-class DFSI-derived quartile label)")
    print(f"Features   : {len(FEATURES)}")
    print(f"SHAP       : {'available' if HAS_SHAP else 'not installed (permutation only)'}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load ----
    print("=" * 60)
    print("  DATA LOADING")
    print("=" * 60)
    df = load_data(DATASET_PATH)

    X = df[FEATURES].copy()
    y = df["risk_class"].copy()

    # ---- Split (identical to train_models.py) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n  Train : {len(X_train)} samples")
    print(f"  Test  : {len(X_test)} samples")

    # ---- Train ----
    print("\n  Fitting pipeline (SimpleImputer -> RandomForest) ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    mf1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Test Macro F1 : {mf1:.4f}  (matches train_models.py baseline)")

    # ---- Permutation importance ----
    perm_df = compute_permutation_importance(pipe, X_test, y_test)

    # ---- SHAP ----
    shap_df = None
    if HAS_SHAP:
        shap_df = compute_shap(pipe, X_train, X_test)
    else:
        # Write placeholder CSVs so outputs/ is complete
        placeholder = pd.DataFrame({
            "feature": FEATURES,
            "importance": [float("nan")] * len(FEATURES),
            "note": ["shap not installed"] * len(FEATURES),
        })
        placeholder.to_csv(
            os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
            index=False,
        )
        # Blank placeholder plot
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5,
                "shap package not installed\nInstall with: pip install shap",
                ha="center", va="center", fontsize=13, color="#888888",
                transform=ax.transAxes)
        ax.axis("off")
        fig.savefig(
            os.path.join(OUTPUT_DIR, "shap_feature_importance.png"),
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"\n  Placeholder SHAP files written (shap not installed).")

    # ---- Interpretation ----
    print_interpretation(perm_df, shap_df)

    # ---- Final summary ----
    print("=" * 60)
    print("  Output files written to: outputs/")
    if HAS_SHAP:
        print("    shap_feature_importance.csv")
        print("    shap_feature_importance.png")
    else:
        print("    shap_feature_importance.csv       (placeholder -- shap not installed)")
        print("    shap_feature_importance.png       (placeholder -- shap not installed)")
    print("    permutation_feature_importance.csv")
    print("    permutation_feature_importance.png")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
