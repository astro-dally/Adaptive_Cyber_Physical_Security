# Data Preprocessing Plan

## 1. Overview
The CIC-IDS-2018 dataset contains millions of network traffic flows. However, raw flow data is often unrefined, containing missing values, infinities, and structural imbalances. This data preprocessing plan outlines the steps required to transform the raw benchmark into a robust foundation for machine learning modeling.

## 2. Initial Data Cleaning
Before applying any mathematical transformations, the dataset must be cleaned to ensure structural integrity.
- **Header Standardization:** Remove any stray, repeated header rows (e.g., rows where the target column contains the string "Label" instead of a true class).
- **Identifier Removal:** Drop non-generalizable identifiers such as `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, `Destination Port`, and `Timestamp`. These columns leak specific network context and encourage models to memorize topology rather than learn fundamental traffic behaviors.
- **Handling Corrupt Values:** Locate and treat infinite values (`inf` / `-inf`) generated during flow calculation by substituting them with `NaN`.

## 3. Handling Missing Values and Duplicates
- **Missing Values:** Impute critical missing values (e.g., using median imputation for skewed flow statistics) or drop rows if the missingness exceeds a predefined acceptable threshold (e.g., dropping rows if more than 5% of features are `NaN`). 
- **Duplicates:** Identify and remove exact duplicate rows to prevent data leakage and to avoid artificially inflating the model's apparent performance during evaluation.

## 4. Addressing Class Imbalance
The CIC-IDS-2018 dataset is heavily skewed, with Benign traffic dominating the distribution.
- **Undersampling:** Mildly undersample the majority class (Benign traffic) to reduce computational overhead during initial algorithm selection.
- **Class Weights:** Instead of aggressive SMOTE oversampling (which can introduce noise in high-dimensional spaces), rely on algorithm-level class weighting (e.g., `class_weight='balanced'`) during model training to penalize misclassifications of minority attack classes.

## 5. Encoding and Scaling
- **Categorical Encoding:** The primary categorical feature is the target `Label`. For multi-class classification, apply Label Encoding mapping string labels to integers. For anomaly detection, binarize the labels (Benign = 0, Attack = 1).
- **Feature Scaling:** Since different network statistics operate on varying scales (e.g., packet counts vs. microsecond durations), scaling is crucial.
  - Apply **Standard Normalization (Z-score scaling)** for distance-based and geometric models (e.g., One-Class SVM, Logistic Regression).
  - Note: Tree-based models (Random Forest, XGBoost) do not strictly require scaling but keeping a standardized dataset ensures pipeline consistency.

## 6. Train-Test Split Strategy
To rigorously evaluate the system's ability to handle adaptive cyber-physical threats, we will implement a dual splitting strategy:
- **Standard Supervised Split:** An 80/20 train-test split for evaluating performance on known attack signatures. Stratified sampling will be used to ensure the proportional representation of all classes in both sets.
- **Zero-Day (Unseen) Attack Split:** To evaluate anomaly detection under open-world conditions, specific attack families (e.g., `Bot` and `DoS attacks-SlowHTTPTest`) will be entirely withheld from the training set. The model will be trained exclusively on Benign traffic and a subset of known attacks, testing its ability to flag the withheld families as novel anomalies.
