# Multimodal Deep Learning for Pediatric Pneumonia Classification
### Using Chest X-Rays and Cough Sounds
This repository contains the implementation of a multimodal deep learning framework for pediatric pneumonia classification using chest X-ray images and cough sounds. The project combines visual and acoustic information to improve diagnostic performance compared to unimodal approaches.

---

## Pipeline

The implementation is structured in three main stages:

1. **X-ray Classification** — Transfer learning with ResNet50, MobileNet, DenseNet121, and VGG19  
   *(ResNet50 selected as the best model)*  

2. **Cough Sound Classification** — MFCC feature extraction with SVM, LSTM, BiLSTM, and CNN-BiLSTM  
   *(CNN-BiLSTM selected as the best model)*  

3. **Multimodal Fusion** — Late Fusion (weighted averaging) and Gated Fusion combining both modality outputs  

---

## Datasets

> ⚠️ Datasets and trained models are **not included** in this repository due to storage limitations. Please download them separately and update the file paths in the notebooks before running.

| Modality | Dataset | Source |
|---|---|---|
| Chest X-ray | Kermany et al. (2018) — 5,856 labeled images | [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) |
| Cough Audio | Liao et al. (2022) — 173 pediatric recordings | [Paper reference](https://doi.org/10.6084/m9.figshare.21176197.v1) |

---

## Key Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| ResNet50 (X-ray only) | 0.8478 | 0.8868 | 0.9524 |
| CNN-BiLSTM (Cough only) | 0.7274 | 0.7522 | 0.7472 |
| **Late Fusion** | **0.8773** | **0.9073** | 0.9389 |
| Gated Fusion | 0.8707 | 0.9016 | 0.9284 |

Late fusion with weighted averaging achieved the best overall performance, outperforming both unimodal baselines.

---
## Setup

```bash
# Clone the repository
https://github.com/thenukii/Multimodal-Deep-Learning-for-Pediatric-Pneumonia-Classification.git
cd Multimodal-Deep-Learning-for-Pediatric-Pneumonia-Classification
```

### ⚠️ Path Configuration

All notebooks were developed using Google Colab with Google Drive integration. After cloning the repository, update the dataset and model paths inside each notebook before running.

---

## Notes

- Optimal classification thresholds are selected using **Youden's J Statistic** on validation data
- Fusion evaluation uses a **20-seed multi-run strategy** to account for pseudo-pairing randomness
- All ethical considerations followed; datasets are fully anonymised and publicly available
