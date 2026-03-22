# Theoretical Rigor of the Project

## Purpose

This project studies adaptive intrusion detection for cyber-physical security using the CIC-IDS-2018 network-flow benchmark. The main question is not only whether a model can classify attacks it has already seen, but whether it can still detect malicious behavior when the attack family was absent from training. That is the core zero-day problem addressed by the repository.

The codebase approaches that problem with two complementary model classes:

- a supervised `RandomForestClassifier` for known-pattern detection;
- a semi-supervised `OneClassSVM` for novelty detection from benign-only training.

This is theoretically sound because cyber defense is an open-world problem. A standard classifier can learn labeled attack categories that appear in training data, but it is not guaranteed to recognize new attack families that lie outside that training distribution.

## Why the Approach Is Rigorous

### 1. The project uses an explicit zero-day evaluation setup

The repository does not stop at standard train/test classification. The Random Forest experiment defines:

- one seen attack used in training: `dos attacks-hulk`
- two unseen attacks withheld from training: `bot`, `dos attacks-slowhttptest`

That design is important because it tests out-of-distribution behavior, not just ordinary in-distribution accuracy.

### 2. The preprocessing reduces leakage and instability

The full preprocessing pipeline in `pipeline/full_preprocessing.py` makes the following defensible choices:

- drops identifier columns: `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`
- converts non-label columns to numeric form
- replaces `inf` and `-inf` with missing values
- drops columns with more than 50% missing values
- fills remaining missing values with `0`
- removes constant features
- removes highly correlated features using a `0.98` threshold

These steps improve rigor because:

- identifier removal reduces the chance of learning environment-specific shortcuts;
- sparse and constant-feature removal reduces noise and degenerate predictors;
- correlation pruning reduces redundancy and helps stabilize downstream learning.

### 3. The model choices match the security problem

#### Random Forest

Random Forest is a sensible baseline for tabular intrusion-detection data because:

- it can model nonlinear decision boundaries;
- it works well with many numeric features;
- it is fairly robust to correlated predictors;
- it does not require feature scaling.

In simple terms, the model trains many decision trees on slightly different samples of the training data and lets them vote on the final prediction. The predicted class is the one that receives the most votes across the trees.

That matters here because network-flow data contains many interacting traffic statistics, and ensemble trees are a strong baseline for this type of structured data.

#### One-Class SVM

One-Class SVM is appropriate for zero-day detection because it learns the boundary of normal traffic instead of relying on a complete catalog of malicious labels.

In this repository:

- training uses benign traffic only;
- testing uses held-out benign traffic plus attack traffic;
- `StandardScaler` is applied before fitting, which is necessary for distance-sensitive kernel methods.

In plain language, the One-Class SVM learns a region that contains most benign samples. If a new sample falls far enough outside that learned benign region, it is flagged as anomalous. That is why the method is theoretically appropriate for novelty detection.

### 4. The evaluation uses security-relevant metrics

The project evaluates:

- accuracy
- precision
- recall
- F1-score
- zero-day recall

Zero-day recall is especially important. It measures how many truly unseen attacks are detected out of all unseen attacks in the test set.

Written plainly:

- true positives on unseen attacks / all unseen attacks

This is a necessary metric because a model can still look strong on aggregate metrics while failing completely on new attack types. The current repository already demonstrates that risk: the Random Forest is highly conservative and misses the configured unseen attacks, while the One-Class SVM recovers at least some of them.

## Evidence of Good Research Engineering

The codebase also shows several disciplined implementation choices:

- root-relative paths instead of fragile working-directory assumptions;
- explicit failure when the processed dataset is missing;
- explicit label normalization to lowercase stripped strings;
- explicit removal of stray rows where the label became `label`;
- bounded benign subsampling in the One-Class SVM pipeline for tractability;
- fixed `random_state=42` in the train/test splits.

These are not theoretical contributions by themselves, but they make the experimental results more reproducible and easier to trust.

## Current Dataset Reality

The checked-in processed dataset currently contains:

- `8,284,195` rows
- `52` columns
- `6,112,151` benign flows
- `2,172,044` attack flows

The processed file includes more attack classes than the smaller subset discussed in the report. That is not a contradiction in the codebase. The Random Forest script intentionally selects a smaller seen/unseen split from the larger processed dataset in order to test a specific zero-day scenario, while the One-Class SVM treats all attacks as anomalous during evaluation.

## Current Limits

The project is methodologically coherent, but several limits remain:

- no cross-validation or repeated experimental runs are built into the scripts;
- no systematic hyperparameter search is recorded;
- the Random Forest seen/unseen split is hand-configured rather than rotated across many attack families;
- missing values are filled with `0`, which is simple and reproducible but not always statistically neutral;
- feature reduction is correlation-based only, without ablation studies or feature-importance validation;
- concept drift and time-aware validation are not yet modeled.

These are reasonable limits for a Phase-1 research implementation, but they should be addressed in later iterations if the goal is a stronger experimental study.

## Bottom Line

The project is theoretically rigorous in the ways that matter most for an early adaptive-security study:

- it treats intrusion detection as an open-world problem;
- it removes obvious leakage and instability from the data pipeline;
- it uses two model families with complementary roles;
- it evaluates novelty detection explicitly instead of hiding behind aggregate accuracy.

In short, the repository does not just train models on a benchmark. It makes a defensible security-learning argument: known-attack recognition and zero-day detection are different tasks, and adaptive cyber-physical defense needs both.
