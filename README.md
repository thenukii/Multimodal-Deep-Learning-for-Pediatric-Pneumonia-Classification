# Multimodal Deep Learning for Pediatric Pneumonia Classification
### Using Chest X-Rays and Cough Sounds
This repository contains the implementation of a multimodal deep learning framework for pediatric pneumonia classification using chest X-ray images and cough sounds. The project combines visual and acoustic information to improve diagnostic performance compared to unimodal approaches.

---

## Pipeline

The implementation is structured in three main stages:

1. **X-ray Classification** — Transfer learning with ResNet50, MobileNet, DenseNet121, and VGG19 (ResNet50 selected as best)
2. **Cough Sound Classification** — MFCC feature extraction + SVM, LSTM, BiLSTM, and CNN-BiLSTM models (CNN-BiLSTM selected as best)
3. **Multimodal Fusion** — Late Fusion (weighted averaging) and Gated Fusion combining both modality outputs

---
## Datasets

> ⚠️ Datasets and trained models are **not included** in this repository due to storage limits. Download them separately and place them in the appropriate directories before running the notebooks.

| Modality | Dataset | Source |
|---|---|---|
| Chest X-ray | Kermany et al. (2018) — 5,856 labeled images | [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) |
| Cough Audio | Liao et al. (2022) — 173 pediatric recordings | [(https://doi.org/10.6084/m9.figshare.21176197.v1)] |

---

## Key Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| ResNet50 (X-ray only) | 0.8478 | 0.8868 | 0.9524 |
| CNN-BiLSTM (Cough only) | 0.7403 | 0.7297 | 0.7459 |
| **Late Fusion** | **0.8773** | **0.9073** | 0.9389 |
| Gated Fusion | 0.8707 | 0.9016 | 0.9284 |

Late fusion with weighted averaging achieved the best overall performance, outperforming both unimodal baselines.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install tensorflow keras librosa audiomentations scikit-learn numpy matplotlib
```

Then download the datasets, place them in the correct paths as referenced in each notebook, and run the notebooks in order.

---

## Notes

- Optimal classification thresholds are selected using **Youden's J Statistic** on validation data
- Fusion evaluation uses a **20-seed multi-run strategy** to account for pseudo-pairing randomness
- All ethical considerations followed; datasets are fully anonymised and publicly available
