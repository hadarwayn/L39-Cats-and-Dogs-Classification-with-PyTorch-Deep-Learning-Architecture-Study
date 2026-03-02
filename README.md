# L39 — Cats vs Dogs Classification using PyTorch

## A Comparative Study of 10 CNN Architectures

A professional deep learning project that classifies images of cats and dogs using **PyTorch**, while comparing **10 different CNN architectures** in an educational and research-driven way.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hadarwayn/L39-Cats-and-Dogs-Classification-with-PyTorch-Deep-Learning-Architecture-Study/blob/main/notebooks/L39_Cats_vs_Dogs_PyTorch.ipynb)

---

## Table of Contents

1. [What Does This Project Do?](#what-does-this-project-do)
2. [What is a CNN?](#what-is-a-cnn)
3. [The 10 Architectures](#the-10-architectures)
4. [Dataset](#dataset)
5. [Installation & Setup](#installation--setup)
6. [How to Run](#how-to-run)
7. [Local Results (CPU)](#local-results-cpu)
8. [Colab Results (GPU)](#colab-results-gpu)
9. [Local vs Colab Comparison](#local-vs-colab-comparison)
10. [Key Findings](#key-findings)
11. [What I Learned](#what-i-learned)
12. [Project Structure](#project-structure)
13. [Troubleshooting](#troubleshooting)

---

## What Does This Project Do?

This project trains 10 different neural networks to answer one question: **Is this a picture of a cat or a dog?**

But the real goal is not just getting the right answer — it is about **understanding how different network designs affect performance**. We compare tiny networks (30K parameters) to giant ones (38M parameters), simple architectures to complex ones, and networks that learn from scratch to ones that already know how to see.

### Real-World Applications

- Medical imaging (tumour detection)
- Self-driving cars (object recognition)
- Social media (photo tagging)
- Security systems (face detection)

---

## What is a CNN?

A **Convolutional Neural Network** (CNN) is a type of artificial brain that is really good at understanding images.

Imagine you are looking at a picture of a cat:
1. First, your eyes notice **small details** — edges, colours, textures (like fur)
2. Then your brain puts these together into **bigger patterns** — ears, whiskers, paws
3. Finally, you recognise the whole thing — "That is a cat!"

A CNN does the exact same thing, layer by layer:

| Layer Type | What It Does | Analogy |
|-----------|-------------|---------|
| **Conv2d** | Looks for patterns (edges, shapes) | Like a magnifying glass scanning the image |
| **ReLU** | Keeps only useful information | Like highlighting important notes |
| **MaxPool** | Shrinks the image, keeping the best parts | Like making a summary of a chapter |
| **Flatten** | Turns the 2D image into a 1D list | Like reading a grid left-to-right |
| **Linear** | Makes the final decision (cat or dog) | Like the brain making a choice |

### Special Techniques We Test

- **Dropout**: Randomly turns off neurons during training — forces the model to not rely on any single neuron (like studying with a blindfold sometimes)
- **BatchNorm**: Resets numbers between layers to a nice range — helps the model train faster and more stably (like a reset button)
- **Transfer Learning**: Uses a model that already learned from millions of images — we only teach it the final step (like hiring an expert and giving them one simple task)
- **Global Average Pooling**: Instead of a huge list of numbers, averages each channel to one number — dramatically reduces model size

---

## The 10 Architectures

We organised the 10 models into 4 research groups:

### Group A — Depth Study
| # | Model | Layers | Parameters | Key Question |
|---|-------|--------|-----------|-------------|
| 1 | Baseline CNN | 4 conv | ~3.45M | Starting point (converted from Keras) |
| 2 | Shallow CNN | 2 conv | ~11.2M | How little depth can we get away with? |
| 3 | Deep CNN | 6 conv | ~30.6M | Does deeper always mean better? |
| 4 | Wide CNN | 3 conv (wide) | ~38.3M | Is width better than depth? |

### Group B — Regularisation
| # | Model | Technique | Parameters | Key Question |
|---|-------|-----------|-----------|-------------|
| 6 | Dropout CNN | Dropout (50%+30%) | ~3.45M | Does dropout prevent overfitting? |
| 7 | BatchNorm CNN | BatchNorm after conv | ~3.45M | Does BatchNorm speed up training? |

### Group C — Architecture Variants
| # | Model | Design | Parameters | Key Question |
|---|-------|--------|-----------|-------------|
| 5 | Small FC CNN | Tiny classifier (128) | ~4.8M | Do we need a big classifier head? |
| 8 | LeNet-Style | 5x5 kernels, AvgPool | ~2.2M | How does a classic design perform? |

### Group D — Advanced
| # | Model | Technique | Parameters | Key Question |
|---|-------|-----------|-----------|-------------|
| 9 | Transfer ResNet18 | Pretrained, frozen | ~11.3M (130K trainable) | Can transfer learning dominate? |
| 10 | Lightweight CNN | Global AvgPool | ~24K | Can a tiny model still work? |

---

## Dataset

We use the **Kaggle Cats and Dogs** dataset (hosted by Microsoft).

| Environment | Train Images | Val Images | Total |
|-------------|-------------|-----------|-------|
| **Local (CPU)** | 2,000 | 1,000 | 3,000 |
| **Colab (GPU)** | 20,000 | 5,000 | 25,000 |

- Images resized to **150x150** pixels
- Normalised to [-1, 1] range
- Training images get augmentation (random flips, rotations, colour jitter)
- Validation images get no augmentation (fair evaluation)

![Sample Images](results/graphs/sample_images.png)
![Class Distribution](results/graphs/class_distribution.png)

---

## Installation & Setup

### Option 1: Local (UV Virtual Environment)

```bash
# 1. Clone the repository
git clone https://github.com/hadarwayn/L39-Cats-and-Dogs-Classification-with-PyTorch-Deep-Learning-Architecture-Study.git
cd L39-Cats-and-Dogs-Classification-with-PyTorch-Deep-Learning-Architecture-Study

# 2. Create and activate virtual environment
uv venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Run training
python main.py
```

### Option 2: Google Colab (Recommended for GPU)

1. Click the badge to open directly: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hadarwayn/L39-Cats-and-Dogs-Classification-with-PyTorch-Deep-Learning-Architecture-Study/blob/main/notebooks/L39_Cats_vs_Dogs_PyTorch.ipynb)
2. Set runtime to **GPU** (Runtime > Change runtime type > GPU)
3. Click **Runtime > Run all**
4. The notebook is completely self-contained — no setup needed

---

## How to Run

```bash
# Train all 10 models (default: RGB, 10 epochs)
python main.py

# Train a specific model
python main.py --model 1

# Train with grayscale images
python main.py --mode grayscale

# Custom epoch count
python main.py --epochs 5
```

---

## Local Results (CPU)

**Environment:** CPU only | **Dataset:** 3,000 images | **Epochs:** 10

### Results Summary

| Rank | Model | Accuracy | Parameters | Training Time |
|------|-------|----------|-----------|---------------|
| 1 | Transfer ResNet18 | **94.6%** | 11.3M (130K trainable) | 8m 8s |
| 2 | LeNet-Style CNN | 81.0% | 2.2M | 3m 27s |
| 3 | Small FC CNN | 79.4% | 4.8M | 7m 59s |
| 4 | BatchNorm CNN | 75.2% | 3.5M | 7m 33s |
| 5 | Dropout CNN | 75.0% | 3.5M | 7m 9s |
| 6 | Shallow CNN | 74.9% | 11.2M | 6m 19s |
| 7 | Lightweight CNN | 70.6% | 24K | 4m 43s |
| 8 | Baseline CNN | 69.2% | 3.5M | 7m 15s |
| 9 | Deep CNN | 68.0% | 30.6M | 31m 21s |
| 10 | Wide CNN | 50.0% | 38.3M | 16m 8s |

### Comparison Charts

![Accuracy Comparison](results/graphs/comparison_accuracy.png)
![Parameter Comparison](results/graphs/comparison_params.png)
![Training Time Comparison](results/graphs/comparison_time.png)
![Accuracy vs Parameters](results/graphs/comparison_acc_vs_params.png)
![Group Comparison](results/graphs/comparison_groups.png)

### Per-Model Training Curves

<details>
<summary>Click to expand training curves for all models</summary>

#### Model 1 — Baseline CNN (69.2%)
![Baseline Training](results/graphs/training_baseline_cnn.png)
![Baseline Confusion](results/graphs/confusion_baseline_cnn.png)

#### Model 2 — Shallow CNN (74.9%)
![Shallow Training](results/graphs/training_shallow_cnn.png)
![Shallow Confusion](results/graphs/confusion_shallow_cnn.png)

#### Model 3 — Deep CNN (68.0%)
![Deep Training](results/graphs/training_deep_cnn.png)
![Deep Confusion](results/graphs/confusion_deep_cnn.png)

#### Model 4 — Wide CNN (50.0%)
![Wide Training](results/graphs/training_wide_cnn.png)
![Wide Confusion](results/graphs/confusion_wide_cnn.png)

#### Model 5 — Small FC CNN (79.4%)
![SmallFC Training](results/graphs/training_small_fc_cnn.png)
![SmallFC Confusion](results/graphs/confusion_small_fc_cnn.png)

#### Model 6 — Dropout CNN (75.0%)
![Dropout Training](results/graphs/training_dropout_cnn.png)
![Dropout Confusion](results/graphs/confusion_dropout_cnn.png)

#### Model 7 — BatchNorm CNN (75.2%)
![BatchNorm Training](results/graphs/training_batchnorm_cnn.png)
![BatchNorm Confusion](results/graphs/confusion_batchnorm_cnn.png)

#### Model 8 — LeNet-Style CNN (81.0%)
![LeNet Training](results/graphs/training_lenet-style_cnn.png)
![LeNet Confusion](results/graphs/confusion_lenet-style_cnn.png)

#### Model 9 — Transfer ResNet18 (94.6%)
![Transfer Training](results/graphs/training_transfer_resnet18.png)
![Transfer Confusion](results/graphs/confusion_transfer_resnet18.png)

#### Model 10 — Lightweight CNN (70.6%)
![Lightweight Training](results/graphs/training_lightweight_cnn.png)
![Lightweight Confusion](results/graphs/confusion_lightweight_cnn.png)

</details>

---

## Colab Results (GPU)

**Environment:** L4 GPU (CUDA) | **Dataset:** 25,000 images | **Epochs:** 15

### Results Summary

| Rank | Model | Accuracy | Parameters | Training Time | Epochs |
|------|-------|----------|-----------|---------------|--------|
| 1 | BatchNorm CNN | **96.4%** | 3.45M | 11m 42s | 15/15 |
| 2 | Transfer ResNet18 | 96.1% | 11.3M (130K trainable) | 11m 36s | 15/15 |
| 3 | Small FC CNN | 95.0% | 4.83M | 11m 50s | 15/15 |
| 4 | Wide CNN | 93.7% | 38.25M | 11m 48s | 15/15 |
| 5 | Baseline CNN | 92.4% | 3.45M | 11m 49s | 15/15 |
| 6 | Dropout CNN | 89.4% | 3.45M | 11m 48s | 15/15 |
| 7 | LeNet-Style CNN | 88.0% | 2.23M | 11m 37s | 15/15 |
| 8 | Shallow CNN | 87.3% | 11.23M | 11m 49s | 15/15 |
| 9 | Lightweight CNN | 77.2% | 23.9K | 11m 36s | 15/15 |
| 10 | Deep CNN | 50.0% | 30.62M | 5m 37s | 7/15 |

### Detailed Per-Class Metrics

| Model | Precision (Cat) | Recall (Cat) | F1 (Cat) | Precision (Dog) | Recall (Dog) | F1 (Dog) |
|-------|----------------|-------------|---------|----------------|-------------|---------|
| BatchNorm CNN | 0.974 | 0.954 | 0.964 | 0.955 | 0.975 | 0.965 |
| Transfer ResNet18 | 0.962 | 0.959 | 0.961 | 0.959 | 0.962 | 0.961 |
| Small FC CNN | 0.953 | 0.946 | 0.949 | 0.946 | 0.954 | 0.950 |
| Wide CNN | 0.920 | 0.957 | 0.938 | 0.955 | 0.917 | 0.936 |
| Baseline CNN | 0.939 | 0.906 | 0.922 | 0.909 | 0.942 | 0.925 |
| Dropout CNN | 0.893 | 0.896 | 0.894 | 0.895 | 0.893 | 0.894 |
| LeNet-Style CNN | 0.907 | 0.846 | 0.875 | 0.855 | 0.914 | 0.884 |
| Shallow CNN | 0.912 | 0.824 | 0.866 | 0.840 | 0.921 | 0.878 |
| Lightweight CNN | 0.708 | 0.923 | 0.802 | 0.890 | 0.620 | 0.731 |
| Deep CNN | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 0.667 |

### Group Analysis

| Group | Description | Avg Accuracy | Models |
|-------|-------------|-------------|--------|
| B | Regularisation | **92.9%** | Dropout CNN, BatchNorm CNN |
| C | Architecture Variants | 91.5% | Small FC CNN, LeNet-Style CNN |
| D | Advanced | 86.6% | Transfer ResNet18, Lightweight CNN |
| A | Depth Study | 80.8% | Baseline, Shallow, Deep, Wide |

---

## Local vs Colab Comparison

### Side-by-Side Results

| Model | Local (CPU) | Colab (GPU) | Change | Rank Change |
|-------|------------|------------|--------|-------------|
| BatchNorm CNN | 75.2% | **96.4%** | +21.2% | #4 → **#1** |
| Transfer ResNet18 | **94.6%** | 96.1% | +1.5% | #1 → #2 |
| Small FC CNN | 79.4% | 95.0% | +15.6% | #3 → #3 |
| Wide CNN | 50.0% | 93.7% | **+43.7%** | #10 → #4 |
| Baseline CNN | 69.2% | 92.4% | +23.2% | #8 → #5 |
| Dropout CNN | 75.0% | 89.4% | +14.4% | #5 → #6 |
| LeNet-Style CNN | 81.0% | 88.0% | +7.0% | #2 → #7 |
| Shallow CNN | 74.9% | 87.3% | +12.4% | #6 → #8 |
| Lightweight CNN | 70.6% | 77.2% | +6.6% | #7 → #9 |
| Deep CNN | 68.0% | 50.0% | **-18.0%** | #9 → #10 |

### Environment Comparison

| Factor | Local | Colab |
|--------|-------|-------|
| **Device** | CPU | L4 GPU (CUDA) |
| **Training images** | 2,000 | 20,000 (10x more) |
| **Validation images** | 1,000 | 5,000 (5x more) |
| **Epochs** | 10 | 15 |
| **Total training time** | ~110 minutes | ~108 minutes |
| **Average accuracy** | 73.7% | 86.9% |
| **Best model** | Transfer ResNet18 (94.6%) | BatchNorm CNN (96.4%) |
| **Worst model** | Wide CNN (50.0%) | Deep CNN (50.0%) |

### What Changed and Why

**Biggest winners with more data:**

1. **Wide CNN (+43.7%)** — The most dramatic turnaround. With only 3,000 images locally, its 38.3M parameters had nothing to learn from — pure overfitting leading to random guessing. With 20,000 training images, those same parameters finally had enough data to learn meaningful features, jumping to 93.7%.

2. **Baseline CNN (+23.2%)** — Went from a mediocre 69.2% to a strong 92.4%. The basic 4-layer architecture was actually a solid design all along — it just needed more data to show it.

3. **BatchNorm CNN (+21.2%)** — Rose from mid-pack (#4) to **first place**. BatchNorm's ability to stabilise training and allow higher effective learning rates shines when there is enough data to train on. With 20,000 images, it outperformed even Transfer Learning.

**Models that barely changed:**

4. **Transfer ResNet18 (+1.5%)** — Was already at 94.6% with just 3,000 images and only improved to 96.1%. This makes sense: the pretrained backbone already knows how to "see" from ImageNet training. More cat/dog images add little new knowledge.

5. **Lightweight CNN (+6.6%)** — With only 24K parameters, there is a hard ceiling on what this model can learn. More data helps, but the model simply does not have enough capacity to capture complex patterns.

**The one model that got worse:**

6. **Deep CNN (-18.0%)** — Dropped from 68.0% to 50.0% (random guessing) and triggered early stopping at epoch 7. With 6 convolutional layers and 30.6M parameters, this model suffers from the **vanishing gradient problem** — gradients become too small to update earlier layers. More data made this worse because the model had more opportunities to get stuck. This architecture needs skip connections (like ResNet) to train successfully at this depth.

---

## Key Findings

### 1. Data Quantity is the Great Equaliser
The average accuracy jumped from **73.7% to 86.9%** when going from 3,000 to 25,000 images. Models that appeared broken locally (Wide CNN at 50.0%) became competitive (93.7%) with enough data. This is the single most important takeaway: **before blaming the architecture, check if you have enough data**.

### 2. BatchNorm is the Overall Champion
BatchNorm CNN achieved the highest accuracy on GPU (**96.4%**) — beating even Transfer Learning. With only 3.45M parameters, it proves that normalising layer outputs is one of the most effective techniques in deep learning. It stabilises training, allows faster convergence, and acts as a mild regulariser.

### 3. Transfer Learning is the Safe Bet
Transfer ResNet18 was the **most consistent** model across both environments: 94.6% locally, 96.1% on Colab. When you have limited data, transfer learning gives the best results. When you have plenty of data, it is still near the top. It is the safest choice for any real-world project.

### 4. Bigger is NOT Always Better
The Deep CNN (30.6M parameters) scored **50.0%** on both local and Colab — the only model to fail completely in both environments. Meanwhile, the Baseline CNN (3.45M parameters) reached 92.4% on Colab. More depth without proper architecture design (e.g., skip connections) leads to untrainable models.

### 5. Regularisation Scales with Data
Both Dropout and BatchNorm improved significantly with more data (Dropout: +14.4%, BatchNorm: +21.2%). Regularisation does not just prevent overfitting — it helps the model **generalise better** when given enough training examples to learn from.

### 6. Small Models Have a Ceiling
The Lightweight CNN (24K parameters) improved only modestly (+6.6%) despite having 10x more data. With such limited capacity, there is a hard limit on what the model can represent. Global Average Pooling is powerful for reducing parameters, but you still need enough filters to capture the relevant features.

### 7. GPU Makes Training Practical
Local CPU training took ~110 minutes total for 10 models across 10 epochs. Colab GPU training took ~108 minutes for 10 models across 15 epochs with 10x more data. The GPU processed roughly **15x more work** in the same wall-clock time, making larger experiments feasible.

---

## What I Learned

1. **Data matters more than architecture** — Most "bad" models locally became good with 10x more data. The Wide CNN went from 50% to 94%.
2. **BatchNorm is underrated** — It beat transfer learning on GPU, proving that simple techniques applied correctly can rival complex ones.
3. **Transfer learning is the safest starting point** — Consistent top-2 results in both environments, even with a frozen backbone.
4. **Depth without skip connections is dangerous** — The Deep CNN failed in both environments. Modern deep networks (ResNet, etc.) solved this with residual connections.
5. **Regularisation and data are partners** — Dropout and BatchNorm showed their full potential only when given enough data.
6. **GPU vs CPU is not just about speed** — GPU lets you train with more data and more epochs, which fundamentally changes which architectures succeed.
7. **PyTorch is flexible** — Building 10 custom architectures and a full training pipeline was straightforward with `nn.Module`.
8. **Always test on multiple scales** — Running locally (small data) and on Colab (large data) revealed completely different rankings, giving much deeper insight.

---

## Project Structure

```
L39/
├── main.py                          # Entry point for local training
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files to exclude from git
├── README.md                        # This file
├── config/
│   └── settings.yaml                # Configuration settings
├── docs/
│   ├── PRD.md                       # Product Requirements Document
│   ├── PROJECT_GUIDELINES.md        # Coding standards & conventions
│   └── tasks.json                   # Task breakdown
├── logs/
│   └── config/log_config.json       # Logging configuration
├── notebooks/
│   └── L39_Cats_vs_Dogs_PyTorch.ipynb  # Google Colab notebook
├── results/
│   ├── graphs/                      # All generated plots (38 files)
│   ├── tables/                      # Results CSV/JSON
│   ├── models/                      # Saved model weights
│   └── examples/                    # Example predictions
├── src/
│   ├── __init__.py
│   ├── config.py                    # Central configuration
│   ├── data/
│   │   ├── dataset.py               # Dataset download & loading
│   │   ├── transforms.py            # Image transforms & augmentation
│   │   └── loader.py                # DataLoader creation
│   ├── models/
│   │   ├── __init__.py              # Model registry
│   │   ├── model_01_baseline.py     # Baseline CNN
│   │   ├── model_02_shallow.py      # Shallow CNN
│   │   ├── model_03_deep.py         # Deep CNN
│   │   ├── model_04_wide.py         # Wide CNN
│   │   ├── model_05_small_fc.py     # Small FC CNN
│   │   ├── model_06_dropout.py      # Dropout CNN
│   │   ├── model_07_batchnorm.py    # BatchNorm CNN
│   │   ├── model_08_lenet.py        # LeNet-Style CNN
│   │   ├── model_09_transfer.py     # Transfer ResNet18
│   │   └── model_10_lightweight.py  # Lightweight CNN
│   ├── training/
│   │   ├── trainer.py               # Training loop
│   │   ├── evaluator.py             # Evaluation & metrics
│   │   └── results.py               # Results saving & summary
│   ├── visualization/
│   │   ├── sample_plots.py          # Sample image plots
│   │   ├── training_plots.py        # Training curves
│   │   ├── confusion_plots.py       # Confusion matrices
│   │   └── comparison_plots.py      # Cross-model comparisons
│   └── utils/
│       ├── paths.py                 # Path utilities
│       ├── helpers.py               # General helpers
│       └── logger.py                # Ring buffer logging
└── venv/                            # Virtual environment (not in git)
```

### Code Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| main.py | ~150 | Entry point, orchestrates training |
| src/config.py | ~115 | Central configuration |
| src/data/dataset.py | ~140 | Dataset download & PyTorch Dataset |
| src/data/transforms.py | ~75 | Image transforms |
| src/data/loader.py | ~65 | DataLoader creation |
| src/models/*.py | ~50-60 each | 10 CNN architectures |
| src/training/trainer.py | ~150 | Training loop with time limit |
| src/training/evaluator.py | ~115 | Evaluation & metrics |
| src/visualization/*.py | ~80-150 each | All visualization code |
| src/utils/*.py | ~60-120 each | Logging, paths, helpers |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'src'` | Make sure you run from the project root: `cd L39` then `python main.py` |
| `torch not found` | Activate your virtual environment: `venv\Scripts\activate` (Windows) |
| GPU not detected on Colab | Go to Runtime > Change runtime type > GPU |
| Download fails (403 error) | The dataset URL may have changed. Check `src/data/dataset.py` |
| Training is very slow | Expected on CPU. Use Colab with GPU for faster training |
| Images look wrong | Check that transforms are applied correctly (RGB vs Grayscale) |
| `AttributeError: total_mem` on Colab | PyTorch renamed this attribute. Change `total_mem` to `total_memory` |

---

## Author

**Student Project** — Professional AI Development Course

## License

This project is for educational purposes.
