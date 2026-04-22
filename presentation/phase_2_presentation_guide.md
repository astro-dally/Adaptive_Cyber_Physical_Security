# Phase 2: Deep Autoencoder Presentation Guide

This guide is designed to help you present the Phase 2 Deep Autoencoder architecture for your oral presentation. It includes plain-English explanations of the slides, the actual code blocks representing the core logic, and a troubleshooting guide for identifying failure cases (False Positives and False Negatives) using the existing cached data on your AWS instance.

---

## 1. Explaining the Architecture (Slide 4)

**What to say:**
> "Our autoencoder uses a deep, symmetric design. We take the 68 input features and immediately expand them to 128 dimensions. This 'expansion' step allows the network to learn rich, non-linear interactions between network features. We then compress the data down through a 64-dimension layer into a tight 32-dimension bottleneck. This 53% compression forces the network to drop noise and only memorize the essential 'manifold' of normal, benign traffic. Because we use Batch Normalization at every layer, the gradients are stable, and Dropout in the encoder prevents the model from simply memorizing the training data."

**The Code Implementation:**
If you need to show or explain the keras code used for this architecture:

```python
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def build_autoencoder(input_dim=68, bottleneck_dim=32):
    inputs = Input(shape=(input_dim,))
    
    # ─── ENCODER ────────────────────────────
    # Expansion 68 -> 128
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Compression 128 -> 64
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # ─── BOTTLENECK ─────────────────────────
    # Compression 64 -> 32
    bottleneck = Dense(bottleneck_dim, activation='relu')(x)
    bottleneck = BatchNormalization()(bottleneck)
    
    # ─── DECODER ────────────────────────────
    # Expansion 32 -> 64
    x = Dense(64, activation='relu')(bottleneck)
    x = BatchNormalization()(x)
    
    # Expansion 64 -> 128
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output 128 -> 68 (Linear activation to match scaled input)
    outputs = Dense(input_dim, activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model
```

---

## 2. Infrastructure & Caching (Slide 3)

**What to say:**
> "Processing 3.2 million rows of network traffic on a constrained AWS t3.large instance required engineering. We used chunked processing—reading 100,000 rows at a time and downcasting features to float32 to prevent Out-Of-Memory (OOM) crashes. More importantly, we built a 5-stage caching pipeline. Every major step (cleaning -> scaling -> training -> error computation) saves its output locally as Parquet or Numpy binaries. If a step crashes, we don't restart from zero; we load the binary cache in seconds instead of minutes."

---

## 3. Investigating Failure Cases (False Positives & Negatives)

If the professor asks **"How do you investigate the traffic that the model gets wrong?"**, you can explain that because you saved the reconstruction errors and actual labels to binary `.npy` cache files, you can extract the exact failing network flows in seconds without needing to reload the raw CSV files or retrain the model.

### Extraction Code
You can run the following Python code on your AWS instance (inside a new Jupyter notebook cell) to isolate and inspect the failure cases:

```python
import numpy as np
import os
import pandas as pd

# 1. Point to your AWS cache directory
CACHE_DIR = os.path.expanduser("~/ids2018/cache")

# 2. Load the cached numpy arrays directly (Loads in <2 seconds)
print("Loading cached data...")
recon_errors = np.load(os.path.join(CACHE_DIR, "recon_errors.npy"))
y_all        = np.load(os.path.join(CACHE_DIR, "y_all.npy"))
labels_all   = np.load(os.path.join(CACHE_DIR, "labels_all.npy"), allow_pickle=True)
X_all        = np.load(os.path.join(CACHE_DIR, "X_all_scaled.npy")) # The actual flow features

# 3. Set the operational threshold (Derived from the P99 of benign validation set)
THRESHOLD = 0.970  

# 4. Generate Predictions (1 = Attack, 0 = Benign)
preds = (recon_errors > THRESHOLD).astype(int)

# ────────────────────────────────────────────────────────────
# FALSE POSITIVES (Model said Attack, but it was Benign)
# ────────────────────────────────────────────────────────────
# y_all == 0 (Actual Benign) AND preds == 1 (Predicted Attack)
fp_mask = (y_all == 0) & (preds == 1)

fp_indices = np.where(fp_mask)[0]
fp_errors  = recon_errors[fp_mask]
fp_features = X_all[fp_mask]

print(f"\n--- FALSE POSITIVES ---")
print(f"Total False Positives: {fp_mask.sum():,}")
print(f"Average Recon Error of FPs: {fp_errors.mean():.4f}")

# ────────────────────────────────────────────────────────────
# FALSE NEGATIVES (Model said Benign, but it was an Attack)
# ────────────────────────────────────────────────────────────
# y_all == 1 (Actual Attack) AND preds == 0 (Predicted Benign)
fn_mask = (y_all == 1) & (preds == 0)

fn_indices = np.where(fn_mask)[0]
fn_errors  = recon_errors[fn_mask]
fn_labels  = labels_all[fn_mask]   # What type of attacks slipped through?
fn_features = X_all[fn_mask]

print(f"\n--- FALSE NEGATIVES ---")
print(f"Total False Negatives: {fn_mask.sum():,}")
print(f"Average Recon Error of FNs: {fn_errors.mean():.4f}")

# View which specific attack families slipped through the most:
fn_df = pd.DataFrame({'Attack_Type': fn_labels})
print("\nTop 5 Attack Families that evaded detection (False Negatives):")
print(fn_df['Attack_Type'].value_counts().head(5))

# If you want to view the raw features of the worst False Negative:
# (The attack with the lowest reconstruction error - completely stealthy)
stealthiest_idx = np.argmin(fn_errors) # Index within the FN array
print(f"\nStealthiest attack error score: {fn_errors[stealthiest_idx]:.5f}")
print(f"Stealthiest attack features: {fn_features[stealthiest_idx][:5]}...") 
```

### Explaining the Output
If you run the code above, you can tell the professor:
1. **False Positives (FPR = 0.98%):** These are normal benign flows that had high reconstruction errors because they represented rare but legal network behaviour (e.g., a massive sudden legal file transfer).
2. **False Negatives:** By printing the `fn_labels`, you will see that the vast majority of False Negatives belong to the `Bot`, `SSH-Bruteforce`, and `Infiltration` classes. These attacks are "Stealth Attacks" that perfectly mimic normal timing and byte-size statistics, resulting in low reconstruction errors that fall below the threshold. 

This directly proves the need for your **Phase 3 Hybrid System**, where the OCSVM and Random Forest models will catch these stealth attacks that the Autoencoder misses.
