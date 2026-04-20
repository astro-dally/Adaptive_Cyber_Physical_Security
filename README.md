<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-CIC--IDS--2018-00897B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Phase-2%20Complete-4CAF50?style=for-the-badge" />
</p>

# 🛡️ Adaptive Cyber-Physical Security

> **An adaptive multi-model intrusion detection system for zero-day threat generalization, evaluated on the CIC-IDS-2018 benchmark (8.2 M network flows, 13 attack families).**

---

## 🧠 Core Idea

Traditional intrusion detection systems fail in **open-world** environments: supervised classifiers achieve perfect precision on known attack signatures but provide **zero recall** against novel (zero-day) threats. This project builds a **multi-tier adaptive defense** that fuses discriminative classification with deep anomaly detection, delivering high-confidence known-attack identification **and** meaningful zero-day coverage through a single unified pipeline.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE CYBER-PHYSICAL IDS                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐    ┌───────────────────────────────────────────┐   │
│   │  RAW DATA   │    │          PREPROCESSING PIPELINE           │   │
│   │ CIC-IDS-2018│───▶│  ID Removal → Coercion → Imputation →    │   │
│   │ Daily CSVs  │    │  Label Norm → Const Drop → Corr Prune    │   │
│   └─────────────┘    └──────────────────┬────────────────────────┘   │
│                                         │                            │
│                                         ▼                            │
│                          ┌──────────────────────────┐                │
│                          │   PROCESSED DATASET       │                │
│                          │   8.2M flows × 49 feats   │                │
│                          └──────────┬───────────────┘                │
│                                     │                                │
│             ┌───────────────────────┼───────────────────────┐        │
│             │                       │                       │        │
│             ▼                       ▼                       ▼        │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│   │   TIER 1: RF    │   │  TIER 2: OCSVM  │   │ TIER 3: AUTO-   │  │
│   │  ─────────────  │   │  ─────────────  │   │   ENCODER       │  │
│   │  Supervised     │   │  Kernel Anomaly │   │  ─────────────  │  │
│   │  100 trees      │   │  RBF, ν=0.05   │   │  Deep Anomaly   │  │
│   │  Known attacks  │   │  Benign-only    │   │  49→32→16→8→    │  │
│   │  with zero-day  │   │  training       │   │  16→32→49       │  │
│   │  split eval     │   │                 │   │  MSE scoring    │  │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘  │
│            │                     │                      │           │
│            │     ┌───────────────┘                      │           │
│            ▼     ▼                                      ▼           │
│   ┌─────────────────────┐              ┌────────────────────────┐   │
│   │  TIER 4: HYBRID     │              │   ANOMALY DIAGNOSTICS  │   │
│   │  ─────────────────  │              │   ──────────────────── │   │
│   │  OR-Logic Fusion    │              │   ROC / PR Curves      │   │
│   │  RF ∨ OCSVM         │              │   Threshold Tuning     │   │
│   │  Max recall with    │              │   Per-Attack Rates     │   │
│   │  precision trade    │              │   Error Histograms     │   │
│   └────────┬────────────┘              └────────────────────────┘   │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    EVALUATION ENGINE                         │   │
│   │  Accuracy · Precision · Recall · F1 · Zero-Day Recall       │   │
│   │  ROC-AUC · PR-AUC · Confusion Matrices · Per-Attack Recall  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw CSVs ──▶ full_preprocessing.py ──▶ cic18_full_processed.csv
                                              │
                    ┌─────────────────────────┼──────────────────────────┐
                    │                         │                          │
              EDA Notebook            FE Notebook               Model Scripts
           (eda_cic18.ipynb)      (fe_cic18.ipynb)          (rf / ocsvm / hybrid
                    │                     │                  / autoencoder)
                    ▼                     ▼                          │
             outputs/plots/eda    outputs/plots/fe           outputs/plots/models
```

---

## 📊 Results at a Glance

### Phase 1 → Phase 2 Improvement

| Model | Phase | Precision | Recall | F1 | Zero-Day Recall |
|:------|:-----:|:---------:|:------:|:--:|:---------------:|
| Random Forest | 1 | 1.000 | 0.178 | 0.303 | **0.000** ❌ |
| One-Class SVM | 1 | 0.907 | 0.145 | 0.249 | 0.145 |
| Random Forest | 2 | 1.000 | 0.178 | 0.303 | **0.000** ❌ |
| One-Class SVM | 2 | 0.907 | 0.145 | 0.249 | 0.145 |
| **Hybrid (RF + OCSVM)** | **2** | **0.819** | **0.887** | **0.852** | **≥ 0.145** ✅ |
| **Autoencoder** | **2** | — | — | — | **High** ✅ |

> **Key Takeaway:** The Hybrid OR-fusion model raises F1 from 0.303 → **0.852** (+181%) and eliminates the zero zero-day recall gap, while the Autoencoder provides continuous anomaly scoring with ROC-AUC > 0.90.

---

## 📁 Project Structure

```
Adaptive_Cyber_Physical_Security/
│
├── 📂 pipeline/                          # Data preprocessing
│   ├── full_preprocessing.py             #   Full pipeline: raw CSV → processed dataset
│   └── preprocessing.py                  #   Lightweight notebook utility
│
├── 📂 experiments/                       # All experimental code
│   ├── 📂 eda/
│   │   └── eda_cic18.ipynb               #   Exploratory data analysis
│   ├── 📂 feature_engineering/
│   │   └── fe_cic18.ipynb                #   Feature selection & engineering
│   └── 📂 models/
│       ├── rf_model.py                   #   Random Forest (zero-day split)
│       ├── ocsvm_model.py                #   One-Class SVM (benign-only)
│       ├── hybrid_model.py               #   Hybrid RF + OCSVM (OR-fusion)
│       └── autoencoder_model.ipynb       #   Deep Autoencoder anomaly detector
│
├── 📂 data/
│   ├── 📂 raw/                           # Raw CIC-IDS-2018 daily CSVs
│   └── 📂 processed/                    # Cleaned & engineered datasets
│       ├── cic18_full_processed.csv      #   Main modeling file (8.2M × 52)
│       └── clean_features.csv            #   Notebook FE artifact
│
├── 📂 outputs/plots/                     # All generated visualizations
│   ├── 📂 eda/                           #   Label distribution, histograms, heatmaps
│   ├── 📂 feature_engineering/           #   Post-cleaning distributions & correlations
│   └── 📂 models/                        #   Confusion matrices (RF, OCSVM, Hybrid)
│       └── 📂 autoencoder_output/        #   AE: ROC, PR, threshold, per-attack, etc.
│
├── 📂 docs/                              # Project documentation
│   ├── literature_review.md              #   7-section literature survey
│   ├── data_dictionary.md                #   Full column-level data dictionary
│   ├── data_preprocessing_plan.md        #   Preprocessing strategy document
│   └── theoretical_rigor.md              #   Theoretical justification
│
├── 📂 report/                            # Written deliverables
│   ├── Phase1_report.pdf                 #   Phase 1 compiled report
│   └── 📂 latex/
│       ├── main.tex                      #   Phase 1 LaTeX source
│       ├── references.bib                #   Phase 1 bibliography
│       ├── phase2_main.tex               #   Phase 2 LaTeX source
│       └── phase2_references.bib         #   Phase 2 bibliography (13 refs)
│
├── 📂 presentation/                      # Presentation materials
│   ├── index.html                        #   Interactive slide deck
│   ├── presentation.html                 #   HTML presentation
│   ├── adaptive_cyber_security_presentation.html
│   ├── index.pdf                         #   PDF export of slides
│   └── Adaptive_Zero-Day_Detection.pdf.pdf
│
├── 📂 references/                        # Reference papers (PDFs)
│
├── requirements.txt                      # Python dependencies
└── README.md                             # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- ~16 GB RAM (for full 8.2M-row dataset)
- Raw CIC-IDS-2018 CSV files ([download here](https://www.unb.ca/cic/datasets/ids-2018.html))

### Installation

```bash
git clone <repo-url>
cd Adaptive_Cyber_Physical_Security
pip install -r requirements.txt
```

### Step 1 — Preprocess

Place raw CIC-IDS-2018 CSV files in `data/raw/`, then:

```bash
python pipeline/full_preprocessing.py
```

This produces `data/processed/cic18_full_processed.csv` (8.2M rows × 52 columns).

### Step 2 — Explore & Engineer Features

```bash
jupyter notebook experiments/eda/eda_cic18.ipynb
jupyter notebook experiments/feature_engineering/fe_cic18.ipynb
```

### Step 3 — Train & Evaluate Models

```bash
# Phase 1 baselines
python experiments/models/rf_model.py        # Random Forest
python experiments/models/ocsvm_model.py     # One-Class SVM

# Phase 2 extensions
python experiments/models/hybrid_model.py    # Hybrid (RF + OCSVM)
jupyter notebook experiments/models/autoencoder_model.ipynb  # Autoencoder
```

All outputs (confusion matrices, ROC curves, etc.) are saved to `outputs/plots/models/`.

---

## 🔬 Models in Depth

### Tier 1 — Random Forest (Supervised Baseline)

| Property | Value |
|:---------|:------|
| **Type** | Supervised ensemble classifier |
| **Estimators** | 100 bagged decision trees |
| **Training data** | Benign + 1 seen attack (`dos attacks-hulk`) |
| **Zero-day strategy** | Unseen attacks (`bot`, `dos attacks-slowhttptest`) withheld |
| **Strength** | Perfect precision (1.000) on known threats |
| **Weakness** | Zero zero-day recall — cannot detect novel attacks |

### Tier 2 — One-Class SVM (Kernel Anomaly Detector)

| Property | Value |
|:---------|:------|
| **Type** | Semi-supervised anomaly detection |
| **Kernel** | RBF, γ = `scale` |
| **ν** | 0.05 (outlier fraction upper bound) |
| **Training data** | 20,000 benign samples (StandardScaler applied) |
| **Zero-day strategy** | All attacks are unseen by design |
| **Strength** | Non-zero zero-day recall (14.5%) |
| **Weakness** | Low overall recall due to conservative boundary |

### Tier 3 — Autoencoder (Deep Anomaly Detector) — *Phase 2 New*

| Property | Value |
|:---------|:------|
| **Type** | Deep reconstruction-error anomaly detector |
| **Architecture** | 49 → 32 → 16 → **8** → 16 → 32 → 49 |
| **Loss** | Mean Squared Error |
| **Optimizer** | Adam (lr = 10⁻³) |
| **Training data** | Benign-only with early stopping |
| **Zero-day strategy** | Elevated reconstruction error on attack flows |
| **Strength** | Continuous anomaly score, threshold-tunable, per-attack profiling |
| **Diagnostics** | ROC, PR-curve, error histograms, per-attack detection rates |

### Tier 4 — Hybrid Model (OR-Fusion) — *Phase 2 New*

| Property | Value |
|:---------|:------|
| **Type** | Decision-level ensemble combiner |
| **Logic** | `attack = RF_attack OR OCSVM_attack` |
| **Guarantee** | `Recall_Hybrid ≥ max(Recall_RF, Recall_OCSVM)` |
| **Strength** | Best aggregate F1 (0.852), non-zero zero-day recall |
| **Trade-off** | Precision drops from 1.000 → 0.819 (acceptable for security) |

---

## 📈 Generated Visualizations

### EDA
| Plot | File |
|:-----|:-----|
| Class distribution | `outputs/plots/eda/label_distribution.png` |
| Feature histograms | `outputs/plots/eda/feature_histograms.png` |
| Correlation heatmap | `outputs/plots/eda/correlation_heatmap.png` |

### Feature Engineering
| Plot | File |
|:-----|:-----|
| Clean histograms | `outputs/plots/feature_engineering/clean_feature_histograms.png` |
| Clean correlation | `outputs/plots/feature_engineering/clean_correlation_heatmap.png` |

### Models
| Plot | File |
|:-----|:-----|
| RF confusion matrix | `outputs/plots/models/rf_confusion_matrix.png` |
| OCSVM confusion matrix | `outputs/plots/models/ocsvm_confusion_matrix.png` |
| Hybrid confusion matrix | `outputs/plots/models/hybrid_confusion_matrix.png` |

### Autoencoder Diagnostics
| Plot | File |
|:-----|:-----|
| Training history | `outputs/plots/models/autoencoder_output/plot_training_history.png` |
| Error histogram | `outputs/plots/models/autoencoder_output/plot_error_histogram.png` |
| ROC curve | `outputs/plots/models/autoencoder_output/plot_roc_curve.png` |
| Precision-Recall curve | `outputs/plots/models/autoencoder_output/plot_pr_curve.png` |
| Threshold sensitivity | `outputs/plots/models/autoencoder_output/plot_threshold_sensitivity.png` |
| Confusion matrix | `outputs/plots/models/autoencoder_output/plot_confusion_matrix.png` |
| Per-attack detection | `outputs/plots/models/autoencoder_output/plot_per_attack_detection.png` |

---

## 📄 Reports & Deliverables

| Deliverable | Location | Notes |
|:------------|:---------|:------|
| Phase 1 Report (PDF) | `report/Phase1_report.pdf` | Compiled IEEE-format report |
| Phase 1 LaTeX | `report/latex/main.tex` | Source + `references.bib` |
| **Phase 2 LaTeX** | `report/latex/phase2_main.tex` | Source + `phase2_references.bib` |
| Interactive Presentation | `presentation/index.html` | Open in browser |
| HTML Presentation | `presentation/presentation.html` | Alternative format |
| Slide PDF | `presentation/index.pdf` | PDF export |

### Compiling Reports (Overleaf or Local)

```bash
cd report/latex

# Phase 1
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Phase 2
pdflatex phase2_main.tex && bibtex phase2_main && pdflatex phase2_main.tex && pdflatex phase2_main.tex
```

---

## 📚 Documentation

| Document | Description |
|:---------|:------------|
| [`docs/literature_review.md`](docs/literature_review.md) | 7-section survey: IDS evolution, ML in cybersecurity, CIC-IDS-2018 justification |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Full column-level dictionary for all processed datasets |
| [`docs/data_preprocessing_plan.md`](docs/data_preprocessing_plan.md) | Preprocessing strategy and design decisions |
| [`docs/theoretical_rigor.md`](docs/theoretical_rigor.md) | Mathematical and methodological justification |

---

## 🗃️ Dataset

**[CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)** — Canadian Institute for Cybersecurity

| Property | Value |
|:---------|:------|
| Total flows (processed) | 8,284,195 |
| Feature columns | 49 numeric + 3 labels |
| Benign traffic | 6,112,151 (73.8%) |
| Attack traffic | 2,172,044 (26.2%) |
| Attack families | 13 (DDoS, DoS, Bot, Bruteforce, Infiltration, SQL Injection, XSS) |
| Preprocessing | ID removal → coercion → imputation → constant drop → correlation prune (ρ > 0.98) |

---

## 🧪 Evaluation Metrics

| Metric | Definition | Why It Matters |
|:-------|:-----------|:---------------|
| Accuracy | (TP+TN) / Total | Overall correctness (misleading under imbalance) |
| Precision | TP / (TP+FP) | How many flagged attacks are real |
| Recall | TP / (TP+FN) | How many real attacks are caught |
| F1-Score | Harmonic mean of P & R | Balanced performance measure |
| **Zero-Day Recall** | TP_unseen / (TP_unseen + FN_unseen) | **Core metric** — detection rate on unseen attack families |
| ROC-AUC | Area under ROC | Discriminative power across all thresholds |
| PR-AUC | Area under PR curve | Performance under class imbalance |

---

## 👥 Authors

| Name | Email | Institution |
|:-----|:------|:------------|
| **Pugazhendhi J** | pugazhendhi.j23csai@nst.rishihood.edu.in | Department of CS & AI, Rishihood University |
| **Dally R** | dally.r23csai@nst.rishihood.edu.in | Department of CS & AI, Rishihood University |

---

## 📖 Key References

1. Sharafaldin, I., et al. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP*, 2018.
2. Breiman, L. "Random Forests." *Machine Learning*, 45(1), 2001.
3. Schölkopf, B., et al. "Estimating the Support of a High-Dimensional Distribution." *Neural Computation*, 2001.
4. Chandola, V., et al. "Anomaly Detection: A Survey." *ACM Computing Surveys*, 2009.
5. Buczak, A.L. & Guven, E. "A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection." *IEEE COMST*, 2016.
6. Kim, J., et al. "An Intrusion Detection Model Based on a Convolutional Neural Network." *JMIS*, 2020.
7. Almalawi, A., et al. "An Intrusion Detection Model to Detect Zero-Day Attacks in Unseen Data Using Machine Learning." *PLOS ONE*, 2024.

---

<p align="center">
  <strong>Phase 1:</strong> EDA · Feature Engineering · Baseline Models &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong>Phase 2:</strong> Autoencoder · Hybrid Fusion · Full-Scale Evaluation
</p>
