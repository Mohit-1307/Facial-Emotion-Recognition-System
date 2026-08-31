<div align="center">

# DeepFER

**Facial Emotion Recognition on FER-2013 with a From-Scratch CNN and Transfer Learning**

An end-to-end deep learning project that classifies facial expressions into seven emotions using a from-scratch CNN and a fine-tuned MobileNetV2 transfer-learning model — with real-time inference, a web application, and post-training quantization for deployment.

**[Live App →](https://deepfer-q8kd.onrender.com)**

</div>

---

## Overview

This project trains on the FER-2013 dataset (35,887 grayscale facial images) to:

1. **Classify facial expressions** into one of seven emotions — angry, disgust, fear, happy, neutral, sad, surprise — using two independently trained models built and evaluated under identical conditions.
2. **Deploy the model for real-time use** via desktop webcam inference (OpenCV) and a browser-based Flask application, with a TFLite export pipeline benchmarking float32 and int8 quantization.

Both models are trained on FER-2013, evaluated on its held-out 7,178-image test partition, and shipped with real-time inference and a Flask web app.

---

## Data Pipeline

The raw FER-2013 dataset (35,887 grayscale 48×48 images, 7 classes) was processed as follows:

| Step | Action |
|---|---|
| 1 | Built `data/processed/{train, val, test}` from the raw FER-2013 archive via `scripts/prepare_dataset.py` |
| 2 | Carved out a stratified 10% validation split from the original training partition (test partition never touched during training/selection) |
| 3 | Computed balanced class weights from on-disk file counts (`w_i = N / (n_classes × n_i)`) to correct FER-2013's severe class imbalance |
| 4 | Applied data augmentation (horizontal flip, ±~36° rotation, ±15% zoom, ±10% translation) to the training split only |
| 5 | Built two parallel input pipelines — 48×48 grayscale for the scratch CNN, 160×160 RGB for the transfer model |

**Result: 25,837 train / 2,872 validation / 7,178 test images across 7 emotion classes.**

| Split | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise | Total |
|---|---|---|---|---|---|---|---|---|
| Train | 3,595 | 392 | 3,687 | 6,493 | 4,469 | 4,347 | 2,854 | 25,837 |
| Validation | 400 | 44 | 410 | 722 | 496 | 483 | 317 | 2,872 |
| Test | 958 | 111 | 1,024 | 1,774 | 1,233 | 1,247 | 831 | 7,178 |

FER-2013 is heavily imbalanced — `happy` has roughly 16.6× as many training images as `disgust` — corrected via class weighting during training of both models.

---

## Facial Emotion Classification

### From-Scratch CNN vs. MobileNetV2 Transfer Learning

Two models were trained and evaluated on the same test partition: a **from-scratch CNN** (4 convolutional blocks, 32→64→128→256 filters, L2 regularization + dropout, 617,511 parameters, native 48×48 grayscale input) and a **MobileNetV2 backbone pretrained on ImageNet**, fine-tuned in two phases at 160×160 RGB — a resolution chosen empirically after comparing 96×96, 160×160, and MobileNetV2's native 224×224.

### Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| **Transfer (MobileNetV2, fine-tuned, 160×160)** | **59.46%** | **57.48%** | **58.10%** | **57.07%** | **58.46%** |
| Scratch CNN | 52.27% | 44.73% | 51.11% | 45.73% | 50.89% |

**The fine-tuned transfer model was selected as the stronger classifier** — it outperformed the from-scratch CNN on every metric and on every individual class, most notably on `disgust` (F1 0.298 → 0.609) despite that class having only 392 training images. Both models are strongest on `happy` and `surprise`, and weakest on `fear` — a reflection of genuine visual overlap between fear, sad, and angry in FER-2013 (estimated human-rater agreement on this dataset is only 65–68%), not a data-scarcity artifact.

### Training Configuration

| Phase | Trainable Params | Learning Rate | Epochs Run |
|---|---|---|---|
| Transfer — head (backbone frozen) | 329,735 | 1e-3 | 15 |
| Transfer — fine-tune (top 30 backbone layers unfrozen) | 1,840,455 | 1e-5 | 60 |
| Scratch CNN | 617,511 (full) | 1e-3 | 40 |

Resolution and label smoothing were both validated empirically: 96×96 reached 49.6% test accuracy versus 160×160's 59.46%, and MobileNetV2's native 224×224 underperformed 160×160 under a comparable training budget. Label smoothing (ε=0.1) underperformed standard cross-entropy by ~2.6 validation points, so standard cross-entropy was kept.

---

## Deployment Optimization

Each trained model was converted to TFLite and benchmarked across float32, dynamic-range int8, and full int8 quantization (measured latency, 100-run average; measured accuracy, 2,000-image seeded subset of the test set):

**Transfer model (fine-tuned, 160×160)**

| Variant | Accuracy | Latency | Speedup | Size |
|---|---|---|---|---|
| Keras baseline (float32) | 58.35% | 79.16 ms | 1.0× | 10.35 MB |
| TFLite float32 | 58.35% | 4.15 ms | 19.1× | 10.21 MB |
| TFLite dynamic-range int8 | 58.50% | 17.43 ms | 4.5× | 2.87 MB (−72%) |
| **TFLite full int8** | 56.35% | **1.61 ms** | **49.3×** | **3.08 MB (−70%)** |

**Scratch CNN**

| Variant | Accuracy | Latency | Speedup | Size |
|---|---|---|---|---|
| Keras baseline (float32) | 50.35% | 55.43 ms | 1.0× | 2.47 MB |
| TFLite full int8 | 50.35% | 0.51 ms | 108.3× | 0.65 MB (−74%) |

Full int8 gives the largest speedup and smallest footprint for both models, at a small accuracy cost for the transfer model — dynamic-range int8 is the stronger choice there when accuracy preservation matters most.

---

## Repository Structure

```
Facial-Emotion-Recognition-System/
├── data/    
├── deepfer/   
├── outputs/
├── saved_models/     
├── scripts/             
├── tests/
├── webapp/              
├── .dockerignore
├── .gitignore
├── Dockerfile            
├── README.md
├── evaluate.py                  
├── optimize_export.py
├── realtime_webcam.py
├── requirements.txt
├── train.py 
└── train_transfer.py          
```

---

## Running Locally

```bash
git clone https://github.com/Mohit-1307/Facial-Emotion-Recognition-System.git
cd Facial-Emotion-Recognition-System
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_dataset.py --raw-dir /path/to/fer2013 --out-dir data/processed --val-fraction 0.1 --seed 42
python webapp/app.py
```

The app expects a trained checkpoint (default `saved_models/transfer_finetune_best.keras`) produced by running `train.py` / `train_transfer.py` end-to-end, or can be used as already provided in this repo. The same app can also be built and run via Docker: `docker build -t deepfer . && docker run -p 7860:7860 deepfer`.

---

## Tech Stack

- **Data / ML:** TensorFlow / Keras, MobileNetV2 (ImageNet pretrained), NumPy
- **Computer Vision:** OpenCV (Haar-cascade face detection)
- **Deployment:** Flask, Docker, TFLite (post-training quantization), Render
- **Experiment artifacts:** confusion matrices, training curves, metrics JSON (checked into `outputs/`)

---

# Author

**MOHIT SINGH RAJPUT — AI/ML Engineer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/mohitsingh1307)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/Mohit-1307)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohitsinghrajput1307)
[![LeetCode](https://img.shields.io/badge/LeetCode-181717?style=flat-square&logo=leetcode&logoColor=FFA116)](https://leetcode.com/u/MOHIT_SINGH_RAJPUT/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:mohitsinghrajput1307@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>