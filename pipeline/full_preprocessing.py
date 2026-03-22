from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "cic18_full_processed.csv"
IDENTIFIER_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]


def load_raw_datasets(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")

    print(f"Loading {len(csv_files)} raw dataset files...")
    frames = [pd.read_csv(csv_path, low_memory=False) for csv_path in csv_files]
    return pd.concat(frames, ignore_index=True)


def select_redundant_feature(feature_a: str, feature_b: str) -> str:
    feature_a_lower = feature_a.lower()
    feature_b_lower = feature_b.lower()

    if "subflow" in feature_a_lower and "total" not in feature_a_lower:
        return feature_a
    if "subflow" in feature_b_lower and "total" not in feature_b_lower:
        return feature_b
    if "mean" in feature_a_lower and any(
        token in feature_b_lower for token in ["max", "min", "std"]
    ):
        return feature_b
    if "mean" in feature_b_lower and any(
        token in feature_a_lower for token in ["max", "min", "std"]
    ):
        return feature_a
    return feature_b


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning columns...")
    df = df.copy()
    df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore", inplace=True)

    feature_columns = [column for column in df.columns if column != "Label"]
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print("\nNaN values per column (Top 10):")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))

    nan_ratio = df.isna().mean()
    sparse_columns = nan_ratio[nan_ratio > 0.5].index.tolist()
    df.drop(columns=sparse_columns, inplace=True)
    print(f"Dropped columns due to NaN threshold: {len(sparse_columns)}")

    df.fillna(0, inplace=True)
    print("Shape after NaN handling:", df.shape)

    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column in the raw dataset.")

    print("\nProcessing labels...")
    df["Label"] = df["Label"].astype(str).str.lower().str.strip()
    df = df[df["Label"] != "label"].copy()
    df["BinaryLabel"] = (df["Label"] != "benign").astype(int)
    df["AttackType"] = df["Label"]

    print("\nRemoving constant features...")
    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]
    df.drop(columns=constant_columns, inplace=True)
    print(f"Removed {len(constant_columns)} constant features.")

    print("\nRemoving heavily correlated features...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if "BinaryLabel" in numeric_columns:
        numeric_columns.remove("BinaryLabel")

    correlation_matrix = df[numeric_columns].corr().abs()
    features_to_drop: set[str] = set()

    for row_idx, column in enumerate(correlation_matrix.columns):
        for col_idx in range(row_idx):
            if correlation_matrix.iloc[row_idx, col_idx] > 0.98:
                other_column = correlation_matrix.columns[col_idx]
                if column not in features_to_drop and other_column not in features_to_drop:
                    features_to_drop.add(
                        select_redundant_feature(column, other_column)
                    )

    df.drop(columns=list(features_to_drop), inplace=True, errors="ignore")
    print(f"Dropped {len(features_to_drop)} highly redundant features.")

    return df


def main() -> None:
    print("Starting Full Data Preprocessing Pipeline...")
    df = load_raw_datasets(RAW_DATA_DIR)

    print("Initial shape:", df.shape)
    print("Number of columns:", len(df.columns))

    processed_df = preprocess_dataframe(df)

    print("\n=== Final Data Check ===")
    print("Final shape:", processed_df.shape)
    print("Final number of features:", len(processed_df.columns))
    print("\nBinaryLabel distribution:")
    print(processed_df["BinaryLabel"].value_counts())

    print("\nSaving processed dataset...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
