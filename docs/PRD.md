# L39 -- Cats vs Dogs Classification using PyTorch

## Deep Learning Architecture Study (Professional AI Development Project)

------------------------------------------------------------------------

## 1. Project Overview

### Project Name

L39 -- Cats vs Dogs Classification using PyTorch

### Description

A professional deep learning project that classifies images of cats and
dogs using PyTorch, while comparing 10 different CNN architectures in an
educational and research-driven way.

### Problem Statement

The goal is to train a neural network to classify images into: - Cat -
Dog

This project focuses on architecture comparison and deep learning
experimentation.

------------------------------------------------------------------------

## 2. Dataset

-   Source: KERAS Dogs vs Cats dataset (cats_and_dogs_filtered)
-   Binary classification
-   Images resized to: 150x150
-   Output classes: 2 (Cat, Dog)

### Dataset Sizes by Environment

| Environment | Train Images | Validation Images | Total |
|-------------|-------------|-------------------|-------|
| **Local (CPU)** | ~2,000 (1,000 cats + 1,000 dogs) | ~1,000 (500 cats + 500 dogs) | ~3,000 |
| **Colab (GPU)** | ~20,000 (10,000 cats + 10,000 dogs) | ~5,000 (2,500 cats + 2,500 dogs) | ~25,000 |

> **Note:** Local environment uses the small filtered Keras dataset (~3,000 images)
> to keep training feasible on CPU. Colab uses the full Kaggle Dogs vs Cats
> dataset (~25,000 labeled images) for GPU-accelerated training.

### Preprocessing Modes

1.  Grayscale (1x150x150)
2.  RGB (3x150x150)

Normalization: pixel_value / 255.0

Optional augmentations: - Random horizontal flip - Rotation - Random
crop

------------------------------------------------------------------------

## 3. Functional Requirements

-   Load dataset from KERAS
-   Convert to PyTorch Dataset
-   Implement 10 CNN architectures
-   Train and evaluate all models
-   Generate comparison graphs
-   Support Local and Google Colab environments

------------------------------------------------------------------------

## 4. Model Architectures (10 Total)

1.  Baseline (Converted from Keras example)
2.  Shallow CNN
3.  Deep CNN (6 Conv layers)
4.  Wider CNN
5.  CNN + Dropout
6.  CNN + BatchNorm
7.  Small Fully Connected variant
8.  LeNet-style CNN
9.  Transfer Learning (ResNet18)
10. Lightweight Efficient model

------------------------------------------------------------------------

## 5. Technical Requirements

Python 3.10+

Libraries: - torch - torchvision - numpy - matplotlib - scikit-learn

Two execution environments:
1. Local (UV virtual environment) — **CPU only**, ~3,000 images, 10 epochs default
2. Google Colab (GPU enabled) — ~25,000 images, 15 epochs default

------------------------------------------------------------------------

## 6. Success Criteria

-   All 10 architectures train successfully
-   Accuracy target: \>80%
-   Works locally and in Colab
-   Graphs generated:
    -   Accuracy vs Epoch
    -   Loss vs Epoch
    -   Confusion Matrix
    -   Model Comparison Chart

------------------------------------------------------------------------

## 7. Learning Objectives

-   Understanding CNN architecture design
-   Comparing depth vs width
-   Regularization techniques
-   Transfer learning basics
-   PyTorch vs Keras mindset differences

------------------------------------------------------------------------

End of PRD
