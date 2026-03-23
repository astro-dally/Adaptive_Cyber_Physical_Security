"""
Hybrid IDS Model — Random Forest + One-Class SVM

Logic:
  1. RF classifies traffic as benign or attack (known threats).
  2. OCSVM flags anomalies (potential zero-day threats).
  3. Combined: if EITHER model says "attack," we flag it.
     → RF catches known attacks with high precision.
     → OCSVM catches novel attacks RF has never seen.

Runs on a sample for speed.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ─── Paths ───
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "processed" / "cic18_full_processed.csv"
PLOT_DIR = ROOT_DIR / "outputs" / "plots" / "models"

# ─── Config ───
SEEN_ATTACKS = ["dos attacks-hulk"]
UNSEEN_ATTACKS = ["bot", "dos attacks-slowhttptest"]
SAMPLE_SIZE = 50_000  # Total sample size for speed
MAX_OCSVM_TRAIN = 20_000


def safe_recall(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def load_and_sample() -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df["AttackType"] = df["AttackType"].astype(str).str.lower().str.strip()
    df = df[df["AttackType"] != "label"].copy()
    df["BinaryLabel"] = df["BinaryLabel"].astype(int)

    # Sample for speed while keeping class balance
    print(f"Sampling {SAMPLE_SIZE} records...")
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    return df


def build_zero_day_split(df):
    print("Creating zero-day split...")
    benign = df[df["BinaryLabel"] == 0]
    attacks = df[df["BinaryLabel"] == 1]
    seen = attacks[attacks["AttackType"].isin(SEEN_ATTACKS)]
    unseen = attacks[attacks["AttackType"].isin(UNSEEN_ATTACKS)]

    benign_train, benign_test = train_test_split(benign, test_size=0.2, random_state=42)
    seen_train, seen_test = train_test_split(seen, test_size=0.2, random_state=42) if len(seen) > 0 else (seen, seen)

    train_df = pd.concat([benign_train, seen_train], ignore_index=True)
    test_df = pd.concat([benign_test, seen_test, unseen], ignore_index=True)
    test_df["ZeroDay"] = test_df["AttackType"].isin(UNSEEN_ATTACKS).astype(int)

    return train_df, test_df


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_and_sample()
    train_df, test_df = build_zero_day_split(df)

    drop_cols = ["Label", "BinaryLabel", "AttackType", "ZeroDay"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df["BinaryLabel"]
    X_test = test_df.drop(columns=drop_cols, errors="ignore")
    y_test = test_df["BinaryLabel"]

    # ─── Train Random Forest ───
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = pd.Series(rf.predict(X_test), index=y_test.index)

    # ─── Train One-Class SVM (benign-only) ───
    print("Training One-Class SVM...")
    benign_train_data = train_df[train_df["BinaryLabel"] == 0]
    X_benign = benign_train_data.drop(columns=drop_cols, errors="ignore")

    # Subsample for OCSVM speed
    sample_n = min(MAX_OCSVM_TRAIN, len(X_benign))
    X_benign_sample = X_benign.sample(n=sample_n, random_state=42)

    scaler = StandardScaler()
    X_benign_scaled = scaler.fit_transform(X_benign_sample)
    X_test_scaled = scaler.transform(X_test)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=4000)
    ocsvm.fit(X_benign_scaled)
    ocsvm_raw = ocsvm.predict(X_test_scaled)
    ocsvm_pred = pd.Series((ocsvm_raw == -1).astype(int), index=y_test.index)

    # ─── Hybrid: OR logic ───
    # If EITHER model says attack → flag as attack
    print("Combining predictions (OR logic)...")
    hybrid_pred = ((rf_pred == 1) | (ocsvm_pred == 1)).astype(int)

    # ─── Metrics ───
    zero_day_mask = test_df["ZeroDay"] == 1

    print("\n" + "=" * 50)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 50)

    print("\n--- Random Forest (alone) ---")
    print(f"  Accuracy:        {accuracy_score(y_test, rf_pred):.4f}")
    print(f"  Precision:       {precision_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"  Recall:          {recall_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"  F1-score:        {f1_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"  Zero-Day Recall: {safe_recall(y_test[zero_day_mask], rf_pred[zero_day_mask]):.4f}")

    print("\n--- One-Class SVM (alone) ---")
    print(f"  Precision:       {precision_score(y_test, ocsvm_pred, zero_division=0):.4f}")
    print(f"  Recall:          {recall_score(y_test, ocsvm_pred, zero_division=0):.4f}")
    print(f"  F1-score:        {f1_score(y_test, ocsvm_pred, zero_division=0):.4f}")
    print(f"  Zero-Day Recall: {safe_recall(y_test[zero_day_mask], ocsvm_pred[zero_day_mask]):.4f}")

    print("\n" + "=" * 50)
    print("HYBRID MODEL RESULTS (RF + OCSVM)")
    print("=" * 50)
    print(f"  Accuracy:        {accuracy_score(y_test, hybrid_pred):.4f}")
    print(f"  Precision:       {precision_score(y_test, hybrid_pred, zero_division=0):.4f}")
    print(f"  Recall:          {recall_score(y_test, hybrid_pred, zero_division=0):.4f}")
    print(f"  F1-score:        {f1_score(y_test, hybrid_pred, zero_division=0):.4f}")
    print(f"  Zero-Day Recall: {safe_recall(y_test[zero_day_mask], hybrid_pred[zero_day_mask]):.4f}")

    # ─── Save Confusion Matrix ───
    print("\nSaving Hybrid Confusion Matrix...")
    cm = confusion_matrix(y_test, hybrid_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hybrid Model Confusion Matrix (RF + OCSVM)")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "hybrid_confusion_matrix.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
