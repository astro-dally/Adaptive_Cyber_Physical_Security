from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "processed" / "cic18_full_processed.csv"
PLOT_DIR = ROOT_DIR / "outputs" / "plots" / "models"
SEEN_ATTACKS = ["dos attacks-hulk"]
UNSEEN_ATTACKS = ["bot", "dos attacks-slowhttptest"]


def safe_recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def load_dataset() -> pd.DataFrame:
    print("Loading dataset for Random Forest...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_FILE}. Run preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)
    df["AttackType"] = df["AttackType"].astype(str).str.lower().str.strip()
    df = df[df["AttackType"] != "label"].copy()
    df["BinaryLabel"] = df["BinaryLabel"].astype(int)
    return df


def build_zero_day_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Creating zero-day split...")
    benign_data = df[df["BinaryLabel"] == 0]
    attack_data = df[df["BinaryLabel"] == 1]
    seen_attack_data = attack_data[attack_data["AttackType"].isin(SEEN_ATTACKS)]
    unseen_attack_data = attack_data[attack_data["AttackType"].isin(UNSEEN_ATTACKS)]

    if benign_data.empty:
        raise ValueError("No benign samples found in the processed dataset.")
    if seen_attack_data.empty:
        raise ValueError(
            f"No seen-attack samples found for the configured classes: {SEEN_ATTACKS}"
        )

    benign_train, benign_test = train_test_split(
        benign_data, test_size=0.2, random_state=42
    )
    seen_train, seen_test = train_test_split(
        seen_attack_data, test_size=0.2, random_state=42
    )

    train_df = pd.concat([benign_train, seen_train], ignore_index=True)
    test_df = pd.concat(
        [benign_test, seen_test, unseen_attack_data], ignore_index=True
    )
    test_df["ZeroDay"] = test_df["AttackType"].isin(UNSEEN_ATTACKS).astype(int)
    return train_df, test_df


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_dataset()
    train_df, test_df = build_zero_day_split(df)

    feature_drop_columns = ["Label", "BinaryLabel", "AttackType", "ZeroDay"]
    X_train = train_df.drop(columns=feature_drop_columns, errors="ignore")
    y_train = train_df["BinaryLabel"]
    X_test = test_df.drop(columns=feature_drop_columns, errors="ignore")
    y_test = test_df["BinaryLabel"]

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print("Evaluating Random Forest...")
    y_pred = pd.Series(rf.predict(X_test), index=y_test.index)
    zero_day_mask = test_df["ZeroDay"] == 1
    zero_day_recall = safe_recall(y_test[zero_day_mask], y_pred[zero_day_mask])

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Zero-Day Recall: {zero_day_recall:.4f}")

    print("Saving Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "rf_confusion_matrix.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
