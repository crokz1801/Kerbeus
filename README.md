

# Kerbeus: Detecting and Preventing Modality Collapse in Multimodal Skin Lesion Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Derm7pt](https://img.shields.io/badge/Dataset-Derm7pt-blue)](https://derm7pt.github.io/)

**Kerbeus** is a robust multimodal deep learning architecture designed to detect, measure, and eliminate **modality collapse** in clinical medical image analysis. Developed for skin lesion diagnosis on the **Derm7pt** benchmark, Kerbeus regulates gradient flow dynamics and aligns high-dimensional representations across clinical images, dermoscopy photographs, and structured tabular metadata.

---

## 📌 Problem Overview

In standard multimodal pipelines, joint models frequently suffer from **modality collapse**—a failure mode where high-capacity vision backbones dominate backpropagation and cause the optimizer to ignore tabular clinical metadata. 

* **Baseline Collapse:** A standard concatenation baseline exhibits an **8.82× gradient ratio** favoring image inputs. This results in images contributing **82.98%** to final predictions, while tabular metadata contributes only **17.03%**.
* **Clinical Risk:** The system degenerates into a pseudo-unimodal image classifier, losing the ability to resolve visual ambiguities using patient history and metadata on critical diagnostic edge cases.

---

## 🌟 Key Features & Architectural Interventions

Kerbeus eliminates modality collapse through five layered interventions:

1. **Dual Vision & FT-Transformer Encoders:** Features dual InceptionV3 backbones for dermoscopic and clinical images, alongside an FT-Transformer encoding 11 categorical attributes and 1 numeric feature (7-point score).
2. **Cross-Modal Attention Fusion:** Projects image ($f_{img} \in \mathbb{R}^{512}$) and tabular ($f_{tab}$) feature maps into a 2-token sequence processed via 8-head self-attention.
3. **Triple CLIP Alignment:** Constrains $f_{img}$, $f_{tab}$, and $f_{fused}$ within a shared 512-D embedding space via symmetric NCE losses to prevent feature space drift.
4. **Two-Sided Gradient Balancing:** Dynamically scales gradient norm ratios during backpropagation, enforcing a safe bound within $[1/3, 3]$[cite: 1].
5. **Perturbation-Aware Reliability Gate:** Employs an MLP to dynamically estimate per-sample input corruption and adjust modality weights ($w_{img}, w_{tab}$) during inference[cite: 1].

---

## 📊 Performance Comparison

Experimental evaluation on the 5-class diagnosis task of the **Derm7pt** test split[cite: 1]:

| Metric | Baseline (Concat) | Kerbeus (Ours) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 67.85%[cite: 1] | **83.04%**[cite: 1] | **+15.19%**[cite: 1] |
| **Macro-F1** | 0.4757[cite: 1] | **0.7311**[cite: 1] | **+0.2554**[cite: 1] |
| **Weighted-F1** | 0.6653[cite: 1] | **0.8253**[cite: 1] | **+0.1600**[cite: 1] |
| **Gradient Ratio ($\rho$)** | 8.82 (Severe Imbalance)[cite: 1] | **1.60 (Balanced)**[cite: 1] | **Equilibrated**[cite: 1] |
| **Tabular Contribution** | 17.03%[cite: 1] | **38.59%**[cite: 1] | **+21.56%**[cite: 1] |

### Per-Class F1 Score Improvements
* **BCC (Basal Cell Carcinoma):** $0.33 \rightarrow \mathbf{0.70}$[cite: 1]
* **MEL (Melanoma):** $0.56 \rightarrow \mathbf{0.80}$[cite: 1]
* **MISC (Miscellaneous):** $0.52 \rightarrow \mathbf{0.72}$[cite: 1]
* **NEV (Nevus):** $0.80 \rightarrow \mathbf{0.88}$[cite: 1]
* **SK (Seborrheic Keratosis):** $0.15 \rightarrow \mathbf{0.54}$[cite: 1]

---

## 📁 Repository Structure

```text
├── data/
│   └── derm7pt/             # Preprocessed Derm7pt dataset splits
├── models/
│   ├── kerbeus.py           # Core Kerbeus model class
│   ├── backbones.py         # Dual InceptionV3 and FT-Transformer modules
│   ├── fusion.py            # Cross-modal self-attention & Reliability Gate
│   └── clip_heads.py        # Projection heads for Triple CLIP contrastive loss
├── utils/
│   ├── gradient_balance.py  # Two-sided dynamic gradient balancing
│   ├── fragility_sampler.py # Uncertainty-aware fragility sampler
│   └── metrics.py           # Functions for gradient ratios and logit contributions
├── train.py                 # 3-Phase training schedule script
├── evaluate.py              # Model inference and out-of-distribution evaluation
├── requirements.txt         # Dependencies
└── README.md
