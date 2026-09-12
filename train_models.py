"""
train_models.py
===============
AI-Powered District Flood Risk Assessment & Decision Support System
-------------------------------------------------------------------
Experiment: District flood-risk classification benchmarked against
the District Flood Severity Index (DFSI).

This script does NOT claim to predict future floods.
The target (risk_class) is a historically derived DFSI quartile label.

Two baseline Random Forest experiments are run:
  Model 1 -- Historical Flood Characteristics
  Model 2 -- Physical Exposure Only

Usage
-----
    python train_models.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH  = os.path.join("data", "processed", "district_flood_risk_dataset.csv")
OUTPUT_DIR    = "outputs"
RANDOM_SEED   = 42
TEST_SIZE     = 0.20
N_ESTIMATORS  = 300
CLASS_ORDER   = ["Low", "Moderate", "High", "Very High"]

MODEL1_NAME   = "Model 1 -- Historical Flood Characteristics"
MODEL2_NAME   = "Model 2 -- Physical Exposure Only"

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

# ---------------------------------------------------------------------------
# Imports that require installed packages (fail loudly if missing)
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (no display needed)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError as e:
    sys.exit(
        f"\nMissing required package: {e}\n"
        "Install with: pip install scikit-learn matplotlib seaborn\n"
    )

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("  [INFO] seaborn not found - using matplotlib for heatmaps.")


# ===========================================================================
# Helper functions
# ===========================================================================

def ensure_output_dir() -> None:
    """Create the outputs/ directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_validate(path: str) -> pd.DataFrame:
    """Load the processed dataset and run pre-training sanity checks."""
    print(f"  Loading: {path}")
    df = pd.read_csv(path)

    # Drop rows with missing target
    n_before = len(df)
    df = df.dropna(subset=["risk_class"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} row(s) with missing risk_class.")

    # Verify uniqueness of district + state
    dups = df.duplicated(subset=["district", "state"]).sum()
    if dups:
        print(f"  WARNING: {dups} duplicate district+state pair(s) detected.")
    else:
        print(f"  district + state uniqueness check: PASSED")

    # Verify risk_class contains only expected labels
    bad = set(df["risk_class"].unique()) - set(CLASS_ORDER)
    if bad:
        sys.exit(f"  ERROR: Unexpected risk_class values: {bad}")

    print(f"  Dataset shape after cleaning: {df.shape}")
    print(f"  Class distribution:")
    for cls in CLASS_ORDER:
        n = (df["risk_class"] == cls).sum()
        pct = 100.0 * n / len(df)
        print(f"    {cls:<12}: {n:>3}  ({pct:.1f}%)")
    return df


def build_pipeline(features: list) -> Pipeline:
    """
    Return an sklearn Pipeline:
        1. SimpleImputer (median strategy)
        2. RandomForestClassifier
    Imputation is fitted only on training data via the pipeline.
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


def train_and_evaluate(
    df: pd.DataFrame,
    features: list,
    model_name: str,
    out_prefix: str,
) -> dict:
    """
    Train a Random Forest pipeline and evaluate it on the held-out test set.
    Returns a dict of metrics.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(sep)
    print(f"  Features ({len(features)}): {features}")

    # ---- Verify features exist ----
    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        sys.exit(f"  ERROR: Feature(s) not found in dataset: {missing_cols}")

    # ---- Prepare X, y ----
    X = df[features].copy()
    y = df["risk_class"].copy()

    # ---- Train/test split (stratified) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"\n  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")

    print("\n  Class split (train / test):")
    for cls in CLASS_ORDER:
        n_tr = (y_train == cls).sum()
        n_te = (y_test  == cls).sum()
        print(f"    {cls:<12}: train={n_tr}, test={n_te}")

    # ---- Build pipeline and fit ----
    print("\n  Fitting pipeline (SimpleImputer -> RandomForest) ...")
    pipe = build_pipeline(features)
    pipe.fit(X_train, y_train)

    # ---- Predict ----
    y_pred = pipe.predict(X_test)

    # ---- Metrics ----
    acc   = accuracy_score(y_test, y_pred)
    mac_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    mac_r = recall_score(y_test, y_pred,    average="macro", zero_division=0)
    mac_f = f1_score(y_test, y_pred,        average="macro", zero_division=0)
    wgt_f = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

    print(f"\n  {'Metric':<25} {'Value':>8}")
    print(f"  {'-'*35}")
    print(f"  {'Accuracy':<25} {acc:>8.4f}")
    print(f"  {'Macro Precision':<25} {mac_p:>8.4f}")
    print(f"  {'Macro Recall':<25} {mac_r:>8.4f}")
    print(f"  {'Macro F1':<25} {mac_f:>8.4f}")
    print(f"  {'Weighted F1':<25} {wgt_f:>8.4f}")

    print(f"\n  Classification Report:")
    print(
        classification_report(
            y_test, y_pred,
            labels=CLASS_ORDER,
            zero_division=0,
        )
    )

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
    print(f"  Confusion Matrix (rows=Actual, cols=Predicted):")
    hdr = f"{'':>12}" + "".join(f"{c:>12}" for c in CLASS_ORDER)
    print(f"  {hdr}")
    for i, cls in enumerate(CLASS_ORDER):
        row = f"  {cls:<12}" + "".join(f"{cm[i,j]:>12}" for j in range(len(CLASS_ORDER)))
        print(row)

    _save_confusion_matrix(cm, model_name, out_prefix)

    # ---- Feature importance ----
    rf = pipe.named_steps["clf"]
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    features,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n  Feature Importances:")
    print(f"  {'Feature':<40} {'Importance':>10}")
    print(f"  {'-'*52}")
    for _, row in fi_df.iterrows():
        print(f"  {row['feature']:<40} {row['importance']:>10.4f}")

    fi_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"\n  Feature importance saved: {fi_path}")

    _save_feature_importance_plot(fi_df, model_name, out_prefix)

    return {
        "model":       model_name,
        "accuracy":    acc,
        "macro_p":     mac_p,
        "macro_r":     mac_r,
        "macro_f1":    mac_f,
        "weighted_f1": wgt_f,
        "fi_df":       fi_df,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    out_prefix: str,
) -> None:
    """Save a confusion-matrix heatmap to outputs/."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_ORDER,
            yticklabels=CLASS_ORDER,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_ORDER)
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("Actual Class",    fontsize=11)
    ax.set_title(
        f"Confusion Matrix\n{model_name}",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved: {out_path}")


def _save_feature_importance_plot(
    fi_df: pd.DataFrame,
    model_name: str,
    out_prefix: str,
) -> None:
    """Save a horizontal bar-chart of feature importances."""
    n = len(fi_df)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * n + 1.5)))

    colors = plt.cm.Blues(
        np.linspace(0.4, 0.9, n)[::-1]
    )
    bars = ax.barh(
        fi_df["feature"][::-1],
        fi_df["importance"][::-1],
        color=colors[::-1],
        edgecolor="white",
        height=0.65,
    )
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=10)
    ax.set_title(
        f"Feature Importances\n{model_name}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar, val in zip(bars, fi_df["importance"][::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8.5, color="#333333",
        )

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{out_prefix}_feature_importance.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Comparison + interpretation
# ---------------------------------------------------------------------------

def print_comparison(r1: dict, r2: dict) -> None:
    """Print a clean side-by-side comparison table and automated interpretation."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  MODEL COMPARISON")
    print(sep)

    cols = ["Model", "Accuracy", "Macro Precision", "Macro Recall",
            "Macro F1", "Weighted F1"]
    row1 = [r1["model"], r1["accuracy"], r1["macro_p"],
            r1["macro_r"], r1["macro_f1"], r1["weighted_f1"]]
    row2 = [r2["model"], r2["accuracy"], r2["macro_p"],
            r2["macro_r"], r2["macro_f1"], r2["weighted_f1"]]

    # Header
    print(f"\n  {'Model':<46} {'Accuracy':>9} {'Macro P':>9} "
          f"{'Macro R':>9} {'Macro F1':>9} {'Wgt F1':>9}")
    print(f"  {'-'*95}")
    for row in [row1, row2]:
        name = row[0]
        vals = [f"{v:.4f}" for v in row[1:]]
        print(f"  {name:<46} {vals[0]:>9} {vals[1]:>9} "
              f"{vals[2]:>9} {vals[3]:>9} {vals[4]:>9}")

    # --- Automated interpretation ---
    print(f"\n  INTERPRETATION")
    print(f"  {'-'*60}")

    f1_diff = r1["macro_f1"] - r2["macro_f1"]
    better  = r1["model"] if r1["macro_f1"] >= r2["macro_f1"] else r2["model"]
    print(f"\n  1. Overall performance:")
    print(f"     {better} achieves the higher Macro F1.")
    print(f"     Macro F1 difference: {abs(f1_diff):.4f} "
          f"({'Model 1 better' if f1_diff >= 0 else 'Model 2 better'})")

    print(f"\n  2. Top features in Model 1:")
    for _, row in r1["fi_df"].head(3).iterrows():
        print(f"     - {row['feature']} (importance={row['importance']:.4f})")

    print(f"\n  3. Physical-only model (Model 2):")
    if r2["macro_f1"] >= 0.45:
        signal = "meaningful predictive signal"
    elif r2["macro_f1"] >= 0.30:
        signal = "modest but non-trivial predictive signal"
    else:
        signal = "limited predictive signal on its own"
    print(f"     Physical exposure features alone contain {signal}.")
    print(f"     Macro F1 = {r2['macro_f1']:.4f}  (random-chance baseline ~= 0.25 for 4 classes)")

    print(f"\n  4. Scientific context:")
    print(f"     Both models perform district flood-risk CLASSIFICATION,")
    print(f"     not future flood prediction. The target (risk_class) is")
    print(f"     a historical DFSI-derived quartile label.")
    print(f"     Model quality should be interpreted as the degree to which")
    print(f"     the selected features reproduce the DFSI ranking.")

    print()


def save_comparison_csv(r1: dict, r2: dict) -> None:
    """Save the comparison metrics to outputs/model_comparison.csv."""
    rows = []
    for r in [r1, r2]:
        rows.append({
            "model":           r["model"],
            "accuracy":        round(r["accuracy"],    4),
            "macro_precision": round(r["macro_p"],     4),
            "macro_recall":    round(r["macro_r"],     4),
            "macro_f1":        round(r["macro_f1"],    4),
            "weighted_f1":     round(r["weighted_f1"], 4),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    out.to_csv(path, index=False)
    print(f"  Comparison table saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("\nDistrict Flood Risk ML Experiment")
    print("=" * 34)
    print(f"Dataset    : {DATASET_PATH}")
    print(f"Target     : risk_class  (4-class DFSI-derived quartile label)")
    print(f"Models     : 2 Random Forest experiments")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Test size  : {int(TEST_SIZE * 100)}%  (stratified split)")
    print(f"Estimators : {N_ESTIMATORS} trees")
    print()

    ensure_output_dir()

    # ---- Load data ----
    print("=" * 60)
    print("  DATA LOADING & VALIDATION")
    print("=" * 60)
    df = load_and_validate(DATASET_PATH)

    # ---- Confirm dfsi is NOT in any feature list ----
    for feat_list, name in [(MODEL1_FEATURES, "Model1"), (MODEL2_FEATURES, "Model2")]:
        leaked = [f for f in feat_list if f in ("dfsi", "risk_class")]
        if leaked:
            sys.exit(f"ERROR: Target/reference column(s) in {name} features: {leaked}")

    # ---- Run experiments ----
    r1 = train_and_evaluate(df, MODEL1_FEATURES, MODEL1_NAME, "model1")
    r2 = train_and_evaluate(df, MODEL2_FEATURES, MODEL2_NAME, "model2")

    # ---- Comparison ----
    print_comparison(r1, r2)
    save_comparison_csv(r1, r2)

    print("=" * 60)
    print("  Training and evaluation completed successfully.")
    print("=" * 60)
    print(f"\n  Output files written to: {OUTPUT_DIR}/")
    print(f"    model_comparison.csv")
    print(f"    model1_confusion_matrix.png")
    print(f"    model2_confusion_matrix.png")
    print(f"    model1_feature_importance.png")
    print(f"    model1_feature_importance.csv")
    print(f"    model2_feature_importance.png")
    print(f"    model2_feature_importance.csv")
    print()


if __name__ == "__main__":
    main()
