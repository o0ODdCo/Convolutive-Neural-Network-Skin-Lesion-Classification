# HAM10000 Skin Lesion Classification — CNN from Scratch

## Overview

Implementation of a convolutional neural network from scratch using
NumPy, with optional CuPy/CUDA acceleration, for multiclass skin lesion
classification on the HAM10000 dataset.

No deep-learning framework such as PyTorch or TensorFlow is used.

## Classes

- akiec — Actinic keratoses
- bcc — Basal cell carcinoma
- bkl — Benign keratosis
- df — Dermatofibroma
- mel — Melanoma
- nv — Melanocytic nevi
- vasc — Vascular lesions

## Architecture

Input RGB image
→ 3 convolutional blocks
→ Global Average Pooling
→ Dense layer
→ Dropout
→ 7-class classifier

Each convolutional block contains:
Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool

## Features

- CNN implemented from scratch
- Manual forward and backward propagation
- Adam optimizer
- Batch normalization
- Global Average Pooling
- Dropout
- Weighted cross-entropy
- Class balancing
- Data augmentation
- Grouped stratified train/validation split
- NumPy CPU backend
- Optional CuPy/CUDA GPU backend
- Checkpoint saving/resume
- Batch-level metrics logging
- Single-image inference

## Installation

```bash
git clone ...
cd ham10000-cnn

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
