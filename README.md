# Multimodal-Deep-Learning-for-Pediatric-Pneumonia-Classification
This repository contains the implementation of my Final Year Individual Research Project, which focuses on developing a multimodal deep learning system for pediatric pneumonia classification using chest X-ray images and cough sound recordings.

---

## 📌 Project Overview

Pneumonia is one of the leading causes of mortality among children worldwide. Early and accurate diagnosis is critical for effective treatment. Traditional diagnosis methods rely heavily on radiological analysis and clinical expertise, which can be time-consuming and subjective.

This project proposes a multimodal deep learning framework that combines:

- Chest X-ray images (structural information)
- Cough sound recordings (acoustic and functional information)

The goal is to investigate whether combining these two modalities improves classification performance compared to unimodal approaches.

---
## 🎯 Objectives

- Develop a deep learning model for chest X-ray classification
- Develop a deep learning model for cough sound classification
- Design and implement multimodal fusion strategies
- Compare unimodal and multimodal performance

---

## 🧠 Methodology

The project is structured into three main stages:

### 1. X-ray Classification
- Models: ResNet50, DenseNet121, MobileNet, VGG19
- Transfer learning with ImageNet pretrained weights
- Image preprocessing and augmentation
- Best model selected: **ResNet50**

---

### 2. Cough Sound Classification
- Feature extraction: MFCC + Delta + Delta-Delta
- Models:
  - SVM (baseline)
  - LSTM
  - BiLSTM
  - CNN-BiLSTM
- Best model selected: **CNN-BiLSTM**

---

### 3. Multimodal Fusion
Two decision-level fusion strategies were implemented:

#### 🔹 Late Fusion (Weighted Averaging)
- Combines probabilities from both models
- Optimal weights selected using validation data

#### 🔹 Gated Fusion (Adaptive Fusion)
- Learnable gating mechanism
- Dynamically adjusts modality importance per sample

---

## 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- ROC-AUC

Threshold selection was performed using **Youden’s J statistic**.

---

## 📈 Key Results

- X-ray model achieved strong baseline performance
- Cough model showed moderate performance due to dataset limitations
- Multimodal fusion improved results over cough-only models
- Late fusion achieved the best overall performance
- Gated fusion achieved higher recall and AUC but lower accuracy

---

## ⚠️ Dataset Notes

- X-ray and cough datasets are from different sources
- No patient-level alignment between modalities
- Fusion was performed using class-based random pairing
- Multi-seed evaluation was used to ensure robustness

---
