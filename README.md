# Adaptive Cyber-Physical Security

Adaptive network intrusion detection system using the **CIC-IDS-2018** dataset, with a focus on **zero-day attack detection**.

## Project Structure

```
├── pipeline/                  # Data preprocessing scripts
│   ├── full_preprocessing.py  # Full pipeline: raw CSVs → processed dataset
│   └── preprocessing.py       # Lightweight cleaning utility for notebooks
├── experiments/
│   ├── eda/                   # Exploratory Data Analysis notebook
│   ├── feature_engineering/   # Feature selection & engineering notebook
│   └── models/                # Model training & evaluation scripts
│       ├── rf_model.py        # Random Forest (supervised zero-day split)
│       └── ocsvm_model.py     # One-Class SVM (anomaly-based zero-day)
├── data/
│   ├── raw/                   # Raw CIC-IDS-2018 daily CSV files
│   ├── interim/               # Intermediate data (if needed)
│   └── processed/             # Cleaned & feature-engineered datasets
├── outputs/plots/             # Generated visualizations
│   ├── eda/                   # EDA plots
│   ├── feature_engineering/   # Feature engineering plots
│   └── models/                # Confusion matrices
├── configs/                   # Configuration files
└── report/                    # LaTeX report and figures
```

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Adaptive_Cyber_Physical_Security

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing

Place raw CIC-IDS-2018 CSV files in `data/raw/`, then run:

```bash
python pipeline/full_preprocessing.py
```

This produces `data/processed/cic18_full_processed.csv`.

### 2. EDA & Feature Engineering

Run the Jupyter notebooks in order:

```bash
jupyter notebook experiments/eda/eda_cic18.ipynb
jupyter notebook experiments/feature_engineering/fe_cic18.ipynb
```

Feature engineering produces `data/processed/clean_features.csv`.

### 3. Model Training & Evaluation

```bash
python experiments/models/rf_model.py       # Random Forest
python experiments/models/ocsvm_model.py    # One-Class SVM
```

Confusion matrices are saved to `outputs/plots/models/`.

## Deliverables

The repository now includes the Phase-1 written and presentation artifacts:

- LaTeX report: `report/latex/main.tex`
- Bibliography: `report/latex/references.bib`
- HTML presentation: `presentation/index.html`

To compile the report locally:

```bash
cd report/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

To open the presentation, load `presentation/index.html` in a browser.

## Models

| Model | Approach | Training Data | Zero-Day Strategy |
|-------|----------|---------------|-------------------|
| **Random Forest** | Supervised | Benign + 1 seen attack | Unseen attack types withheld from training |
| **One-Class SVM** | Anomaly detection | Benign only (20k samples) | All attacks unseen by design |

## Dataset

[CIC-IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html) — a realistic network intrusion detection dataset containing benign traffic and multiple attack types.

**Phase-1:** EDA, Feature Engineering, and Model Application
