from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "processed" / "cic18_full_processed.csv"
PLOT_DIR = ROOT_DIR / "outputs" / "plots" / "models"
MAX_BENIGN_TRAIN_SAMPLES = 20_000


def safe_recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.empty:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def load_dataset() -> pd.DataFrame:
    print("Loading dataset for One-Class SVM...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_FILE}. Run preprocessing first."
        )

    df = pd.read_csv(DATA_FILE)
    df["AttackType"] = df["AttackType"].astype(str).str.lower().str.strip()
    df = df[df["AttackType"] != "label"].copy()
    df["BinaryLabel"] = df["BinaryLabel"].astype(int)
    return df


def main() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = load_dataset()

    print("Creating zero-day split...")
    benign_data = df[df["BinaryLabel"] == 0]
    attack_data = df[df["BinaryLabel"] == 1]

    if benign_data.empty:
        raise ValueError("No benign samples found in the processed dataset.")
    if attack_data.empty:
        raise ValueError("No attack samples found in the processed dataset.")

    X_benign = benign_data.drop(columns=["Label", "BinaryLabel", "AttackType"])
    y_benign = benign_data["BinaryLabel"]

    X_train_benign, X_test_benign, _, y_test_benign = train_test_split(
        X_benign, y_benign, test_size=0.2, random_state=42
    )

    print("Subsampling training data...")
    sample_size = min(MAX_BENIGN_TRAIN_SAMPLES, len(X_train_benign))
    X_train_benign = X_train_benign.sample(n=sample_size, random_state=42)

    X_attack = attack_data.drop(columns=["Label", "BinaryLabel", "AttackType"])
    y_attack = attack_data["BinaryLabel"]

    X_test = pd.concat([X_test_benign, X_attack], ignore_index=True)
    y_test = pd.concat([y_test_benign, y_attack], ignore_index=True)

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_benign)
    X_test_scaled = scaler.transform(X_test)

    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=4000)
    ocsvm.fit(X_train_scaled)

    print("Evaluating One-Class SVM...")
    y_pred_svm = ocsvm.predict(X_test_scaled)
    y_pred = pd.Series((y_pred_svm == -1).astype(int), index=y_test.index)
    zero_day_recall = safe_recall(y_test[y_test == 1], y_pred[y_test == 1])

    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Zero-Day Recall: {zero_day_recall:.4f}")

    print("Saving Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("One-Class SVM Confusion Matrix")

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ocsvm_confusion_matrix.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
