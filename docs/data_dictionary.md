# Data Dictionary

## Overview

This repository uses the CIC-IDS-2018 flow-based intrusion-detection dataset and produces two processed tabular artifacts:

- `data/processed/cic18_full_processed.csv`: main cleaned dataset used by the model scripts
- `data/processed/clean_features.csv`: notebook-derived feature-engineering artifact with a slightly different selected feature set

The primary modeling scripts in `experiments/models/` use `data/processed/cic18_full_processed.csv`.

## File-Level Dictionary

| File | Role | Notes |
|---|---|---|
| `data/raw/*.csv` | Raw daily CIC-IDS-2018 flow files | Original flow statistics plus `Label` |
| `data/processed/cic18_full_processed.csv` | Main cleaned dataset | 52 columns; produced by `pipeline/full_preprocessing.py` |
| `data/processed/clean_features.csv` | Notebook feature-selection output | Also 52 columns, but not identical to `cic18_full_processed.csv` |
| `data/processed/clean_features.csv.zip` | Archived copy of `clean_features.csv` | Convenience artifact |

## Current Processed Dataset Snapshot

For `data/processed/cic18_full_processed.csv`:

- rows: `8,284,195`
- columns: `52`
- numeric predictors: `49`
- label columns: `3`

### Binary Label Distribution

| BinaryLabel | Meaning | Count |
|---|---|---:|
| `0` | Benign traffic | 6,112,151 |
| `1` | Attack traffic | 2,172,044 |

### Attack Taxonomy in the Checked-In Processed File

| AttackType | Count |
|---|---:|
| `benign` | 6,112,151 |
| `ddos attack-hoic` | 686,012 |
| `dos attacks-hulk` | 461,912 |
| `bot` | 286,191 |
| `ftp-bruteforce` | 193,360 |
| `ssh-bruteforce` | 187,589 |
| `infilteration` | 161,934 |
| `dos attacks-slowhttptest` | 139,890 |
| `dos attacks-goldeneye` | 41,508 |
| `dos attacks-slowloris` | 10,990 |
| `ddos attack-loic-udp` | 1,730 |
| `brute force -web` | 611 |
| `brute force -xss` | 230 |
| `sql injection` | 87 |

## Raw Dataset Fields

The raw daily CIC-IDS-2018 CSVs contain flow-level network features such as packet counts, byte counts, inter-arrival times, TCP flag counters, segment statistics, and the final `Label` column.

Examples of raw-only fields that are later dropped or removed during preprocessing:

- `Timestamp`
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Bwd Pkts`
- `Fwd Pkt Len Mean`
- `Bwd Pkt Len Mean`
- `Flow IAT Mean`
- `Flow IAT Std`
- `Flow IAT Max`
- `Flow IAT Min`
- `Fwd PSH Flags`
- `Bwd PSH Flags`
- `Fwd URG Flags`
- `Bwd URG Flags`
- `RST Flag Cnt`
- `Fwd Byts/b Avg`
- `Fwd Pkts/b Avg`
- `Fwd Blk Rate Avg`
- `Bwd Byts/b Avg`
- `Bwd Pkts/b Avg`
- `Bwd Blk Rate Avg`
- `Subflow Fwd Pkts`
- `Subflow Fwd Byts`
- `Subflow Bwd Pkts`
- `Subflow Bwd Byts`
- `Idle Std`
- `Idle Max`

The preprocessing pipeline removes identifier columns and then prunes constant, sparse, and highly redundant features.

## Main Processed Dataset Dictionary

The following dictionary describes `data/processed/cic18_full_processed.csv`.

| Column | Type | Meaning |
|---|---|---|
| `Dst Port` | float | Destination transport-layer port |
| `Protocol` | float | IP protocol number |
| `TotLen Fwd Pkts` | float | Total bytes sent in the forward direction |
| `Fwd Pkt Len Max` | float | Maximum forward packet length |
| `Fwd Pkt Len Min` | float | Minimum forward packet length |
| `Fwd Pkt Len Std` | float | Standard deviation of forward packet length |
| `Bwd Pkt Len Max` | float | Maximum backward packet length |
| `Bwd Pkt Len Min` | float | Minimum backward packet length |
| `Bwd Pkt Len Std` | float | Standard deviation of backward packet length |
| `Flow Byts/s` | float | Flow byte rate |
| `Flow Pkts/s` | float | Flow packet rate |
| `Fwd IAT Tot` | float | Total forward inter-arrival time |
| `Fwd IAT Mean` | float | Mean forward inter-arrival time |
| `Fwd IAT Std` | float | Standard deviation of forward inter-arrival time |
| `Fwd IAT Max` | float | Maximum forward inter-arrival time |
| `Fwd IAT Min` | float | Minimum forward inter-arrival time |
| `Bwd IAT Tot` | float | Total backward inter-arrival time |
| `Bwd IAT Mean` | float | Mean backward inter-arrival time |
| `Bwd IAT Std` | float | Standard deviation of backward inter-arrival time |
| `Bwd IAT Max` | float | Maximum backward inter-arrival time |
| `Bwd IAT Min` | float | Minimum backward inter-arrival time |
| `Bwd Header Len` | float | Total backward header length |
| `Fwd Pkts/s` | float | Forward packet rate |
| `Bwd Pkts/s` | float | Backward packet rate |
| `Pkt Len Min` | float | Minimum packet length across the flow |
| `Pkt Len Max` | float | Maximum packet length across the flow |
| `Pkt Len Std` | float | Standard deviation of packet length |
| `Pkt Len Var` | float | Variance of packet length |
| `FIN Flag Cnt` | float | Count of FIN flags in the flow |
| `SYN Flag Cnt` | float | Count of SYN flags in the flow |
| `PSH Flag Cnt` | float | Count of PSH flags in the flow |
| `ACK Flag Cnt` | float | Count of ACK flags in the flow |
| `URG Flag Cnt` | float | Count of URG flags in the flow |
| `CWE Flag Count` | float | Count of congestion-window-reduced flags |
| `ECE Flag Cnt` | float | Count of ECN-echo flags |
| `Down/Up Ratio` | float | Ratio of downstream to upstream traffic volume |
| `Pkt Size Avg` | float | Average packet size |
| `Fwd Seg Size Avg` | float | Average forward segment size |
| `Bwd Seg Size Avg` | float | Average backward segment size |
| `Init Fwd Win Byts` | float | Initial forward TCP window size in bytes |
| `Init Bwd Win Byts` | float | Initial backward TCP window size in bytes |
| `Fwd Act Data Pkts` | float | Forward packets carrying active payload data |
| `Fwd Seg Size Min` | float | Minimum forward segment size |
| `Active Mean` | float | Mean duration of active periods in the flow |
| `Active Std` | float | Standard deviation of active-period duration |
| `Active Max` | float | Maximum active-period duration |
| `Active Min` | float | Minimum active-period duration |
| `Idle Mean` | float | Mean duration of idle periods in the flow |
| `Idle Min` | float | Minimum idle-period duration |
| `Label` | string | Original CIC-IDS-2018 label text |
| `BinaryLabel` | integer | `0` for benign, `1` for any attack |
| `AttackType` | string | Lowercased normalized attack label used by the scripts |

## Label Semantics

| Column | Semantics |
|---|---|
| `Label` | Original class label retained for traceability |
| `BinaryLabel` | Collapses all non-benign labels into a binary attack indicator |
| `AttackType` | Cleaned label string used for filtering seen and unseen attack classes |

## Notes on `clean_features.csv`

`data/processed/clean_features.csv` is a related processed dataset but not identical to the main modeling file. Compared with `cic18_full_processed.csv`, it includes fields such as:

- `Tot Fwd Pkts`
- `Flow IAT Std`
- `Fwd Header Len`
- `Idle Std`

and excludes some fields present in the main processed file, such as:

- `Fwd Pkt Len Max`
- `Idle Min`

This suggests the notebook feature-engineering path and the scripted preprocessing path are aligned in purpose but not fully identical in final column selection.

## Preprocessing Rules That Define the Dictionary

The processed dictionary is the result of the following rules in `pipeline/full_preprocessing.py`:

1. Drop identifier columns: `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`
2. Convert non-label features to numeric
3. Replace infinite values with missing values
4. Drop columns with more than 50% missingness
5. Fill remaining missing values with `0`
6. Normalize labels to lowercase stripped strings
7. Create `BinaryLabel`
8. Copy normalized labels into `AttackType`
9. Remove constant columns
10. Remove highly correlated numeric features above `0.98`

## Modeling-Specific Notes

The model scripts use this dictionary differently:

- `experiments/models/rf_model.py` trains on benign traffic plus one seen attack class and tests on benign, seen, and configured unseen attack classes
- `experiments/models/ocsvm_model.py` trains on benign traffic only and treats all attacks as out-of-distribution anomalies at evaluation time

That means the dataset dictionary serves two roles at once:

- a standard tabular classification dataset
- an open-world anomaly-detection benchmark
