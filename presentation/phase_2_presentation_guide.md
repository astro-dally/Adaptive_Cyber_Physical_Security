# Phase 2: Project Master Analysis & Oral Presentation Guide

This document is the definitive "source of truth" for the Phase 2 Deep Autoencoder experiment. Use this to study for your oral exam and to answer deep technical questions from professors.

---

## 1. Project Master Summary: What We Did & Why
Our objective was to build a defense-in-depth system that doesn't just recognize known threats, but anticipates **Zero-Day** (unseen) attacks.

### The Problem Logic
Most Intrusion Detection Systems (IDS) use "Supervised Learning" (like Random Forest). These are like a "Most Wanted" list—they are perfect at catching criminals they've seen before, but if a new criminal (Zero-Day) shows up with a new mask, the system lets them right in.

### Our Solution: The Autoencoder Manifold
Instead of making a list of "Bad Traffic," we decided to become experts on **"Good Traffic"**. We trained a Deep Autoencoder exclusively on **Benign** (normal) network data. 
- The model learns to "reconstruct" normal traffic perfectly.
- When an attack arrives, the model has no idea how to reconstruct it.
- This creates a **High Reconstruction Error** (Anomaly Score), which allows us to catch attacks we have never seen before.

---

## 2. Architecture: Expansion & Compression
Our Custom Deep Symmetric Autoencoder: **68 → 128 → 64 → 32 (Bottleneck) → 64 → 128 → 68**

### Step A: Feature Expansion (68 → 128)
We don't start by compressing. We first **expand** the 68 input features to 128.
- **Why?** This allows the network to learn "Rich Non-linear Interactions." By giving the model more "thinking space," it can discover subtle relationships between packet sizes, timing, and flag patterns that a simple model might miss.

### Step B: The 32-Dimension Bottleneck (53% Compression)
We then violently squeeze that data into a tiny 32-node "Bottleneck."
- **Logic:** This forces the model to drop the noise. It can only memorize the absolutely essential "Manifold" (the core template) of normal traffic. If an attack tries to hide inside this bottleneck, it won't fit perfectly, resulting in error.

---

## 3. The Metric Lexicon (Study these for the Exam!)

| Metric | Simple Definition | Our Result | Real-World Impact |
|:---|:---|:---|:---|
| **ROC-AUC** | **Ranking Power.** The probability that an attack gets a higher error score than a benign flow. | **0.8965** | **High.** This proves the model "understands" the difference between good and bad traffic, regardless of our threshold. |
| **P99 Threshold** | **The Operational Line.** We set the "alarm" at the 99th percentile of normal traffic. | **0.9701** | **Conservative.** This prioritizes "Silent Mode" so security analysts aren't bothered by false alarms. |
| **Precision** | **Trustworthiness.** When the alarm goes off, how often is it a REAL attack? | **0.5012** | **Good.** 1 out of every 2 alarms is a real intrusion at the P99 level. |
| **Recall (TPR)** | **Catch Rate.** How many of the total attacks did we actually flag? | **1.74%** | **Low (at P99).** We only catch the "Loudest" attacks to keep False Positives near zero. |
| **False Positive Rate** | **The Annoyance Factor.** How often is an innocent person flagged? | **0.98%** | **Excellent.** Less than 1 in 100 normal flows are wrongly flagged. |

---

## 4. Understanding the Visual Results (Plot-by-Plot)

### A. ROC Curve (`plot_roc_curve.png`)
- **AXES:** X = False Positive Rate, Y = True Positive Rate.
- **MEANING:** This shows the trade-off. If the line was a straight diagonal, the model is guessing. Because our line "hugs" the top-left corner (AUC=0.89), it proves the model is a very strong **ranker**.

### B. Precision-Recall Curve (`plot_pr_curve.png`)
- **MEANING:** This is more honest than ROC for imbalanced network data. It shows how precision drops as you try to catch more attacks. Our **Average Precision of 0.7578** is mathematically very robust.

### C. Threshold Sensitivity (`plot_threshold_sensitivity.png`)
- **THE MOST IMPORTANT PLOT.** This shows that as you lower the threshold (the "Volume Dial"), the recall (Recall) skyrockets. 
- **Oral Exam Point:** "Professor, we achieved 1.7% recall at P99, but this plot shows that at **P90**, our model actually catches **70.5%** of all attacks."

### D. Multi-Class Detection (`plot_per_attack_detection.png`)
- **REVEALS:** The model is an expert at **Volumetric Attacks** (LOIC-UDP = 100% detection, Slowloris = 73.6%). It struggles with "Stealth" attacks (Bot, Infiltration) which mimic normal human behavior.

### E. Error Histogram (`plot_error_histogram.png`)
- **VISUAL PROOF:** You will see a blue peak (Benign) and a red peak (Attacks) further to the right. The fact that the red peak is **4.3x further right** than the blue peak proves the "Anomaly Signal" is working.

---

## 5. The "Revised Quick Pitch" (Memorize this!)
"We processed 8.2M raw flows down to a clean 3.2M modeling set (64% benign, 36% attack). We trained a Symmetric Deep Autoencoder exclusively on benign traffic. The architecture uses **Feature Expansion (128 nodes)** to learn complex interactions, then uses a **32-dimension bottleneck** to memorize the normal manifold. 

When tested, we achieved a **ROC-AUC of 0.8965**, proving strong ranking power. At our chosen **P99 threshold**, we achieve an excellent **False Positive Rate of <1%**, effectively catching 'Loud' zero-day attacks like LOIC-UDP at 100% while ensuring security analysts aren't buried in alert fatigue."
