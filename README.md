# Patient Knowledge Distillation (PKD) for SST-2

This repository implements **Patient Knowledge Distillation (PKD-Skip)** to compress a 109.5M parameter BERT-base classifier into a lightweight 66.9M parameter student model (~38.8% parameter reduction) on the Stanford Sentiment Treebank (SST-2) dataset while maintaining high classification accuracy.

---

## Core Architecture & Innovation

Standard Knowledge Distillation (e.g., Hinton et al.) transfers knowledge solely from the teacher's final output logits, discarding rich intermediate contextual representations learned across hidden layers.

PKD-Skip resolves this limitation by encouraging the student to patiently imitate the teacher's intermediate hidden state representations layer-by-layer at the `[CLS]` token position:

1. **Soft Logit & Cross-Entropy Transfer:** Combines hard ground-truth labels with softened teacher output probabilities via KL-Divergence at temperature $T = 2.0$:

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}\left( \text{Softmax}\left(\frac{z_s}{T}\right) \parallel \text{Softmax}\left(\frac{z_t}{T}\right) \right) \times T^2$$

2. **Patient Hidden-State Transfer (PKD-Skip):** Forces the student layers $\{1, 2, 3, 4, 5, 6\}$ to learn normalized `[CLS]` representations from the corresponding teacher layers $\{2, 4, 6, 8, 10, 12\}$ via Mean Squared Error (MSE):

$$\mathcal{L}_{\text{PT}} = \frac{1}{\vert{}L\vert{}} \sum_{(s, t) \in L} \left\Vert{} \frac{h_s}{\Vert{}h_s\Vert{}_2} - \frac{h_t}{\Vert{}h_t\Vert{}_2} \right\Vert{}_2^2$$

3. **Total Unified Loss Function:**

$$\mathcal{L}_{\text{PKD}} = (1 - \alpha)\mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{PT}}$$

By skipping every second layer of the teacher, the 6-layer student retains deep contextual representations without requiring additional projection matrices, enabling significantly faster inference with lower memory overhead.

---

## Benchmark Results (SST-2 Sentiment Analysis)

| Metric | Teacher Baseline (BERT-base) | PKD Student (6-layer) |
| :--- | :--- | :--- |
| **Encoder Layers** | 12 | 6 |
| **Hidden Dimension** | 768 | 768 |
| **Parameters** | 109.5M | **66.9M (-38.8%)** |
| **SST-2 Accuracy** | 92.5% | **90.1%** |
| **Model Size (.pt)** | ~438 MB | **~268 MB** |

**Business Application:** This pipeline provides an optimal trade-off for real-time NLP microservices and edge deployments, sacrificing less than ~2.5% in classification accuracy while reducing model footprint by ~39% and significantly cutting GPU memory requirements.

---

## Repository Structure

```text
├── checkpoints/          # Saved student model checkpoints (Ignored in git)
├── notebooks/            # Jupyter Notebooks for Kaggle / Colab training
│   └── pkd_bert_sst2.ipynb
├── src/                  # Core source code directory
│   ├── config.py         # Hyperparameters and model configuration
│   ├── dataset.py        # SST-2 dataset loader and preprocessing
│   ├── model_student.py  # 6-layer student BERT architecture
│   ├── model_teacher.py  # Pretrained 12-layer teacher BERT wrapper
│   ├── train_distill.py  # Main distillation training loop & evaluation
│   └── utils.py          # Custom PKD loss computation functions
├── .gitignore            # Git ignore rules for checkpoints and cache
├── README.md             # Project documentation
└── requirements.txt      # Python package dependencies
