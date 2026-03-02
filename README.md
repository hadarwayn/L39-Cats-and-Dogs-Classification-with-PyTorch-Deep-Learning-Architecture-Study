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
| **Colab (GPU)** | ~20,000 | ~5,000 | ~25,000 |

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

> **Pending:** Run the Colab notebook and paste results here.

<!-- COLAB_RESULTS_PLACEHOLDER -->

---

## Local vs Colab Comparison

> **Pending:** Will be completed after Colab notebook run.

<!-- LOCAL_VS_COLAB_PLACEHOLDER -->

---

## Key Findings

### 1. Transfer Learning Dominates
The Transfer ResNet18 model achieved **94.6%** accuracy — far ahead of all other models. Using a pretrained backbone (trained on millions of ImageNet images) gives a massive head start, even when we freeze all the backbone layers and only train a tiny classifier head (~130K parameters).

### 2. Bigger is NOT Always Better
The Wide CNN (38.3M parameters) scored only **50.0%** — essentially random guessing. Meanwhile, the LeNet-Style CNN (2.2M parameters) scored **81.0%**. More parameters can actually hurt when the model is too complex for the dataset size.

### 3. Classic Designs Still Work
The LeNet-Style CNN, inspired by a 1998 architecture, was the **second-best** custom model at 81.0%. Its 5x5 kernels and average pooling provide good feature extraction even by modern standards.

### 4. Regularisation Helps (Slightly)
Both Dropout (75.0%) and BatchNorm (75.2%) outperformed the vanilla Baseline (69.2%). BatchNorm had a slight edge, likely due to training stability benefits.

### 5. Small Models Can Compete
The Lightweight CNN with only **24K parameters** (100x fewer than baseline) still achieved 70.6% — surprisingly close to models with millions more parameters. Global Average Pooling is a powerful technique.

### 6. Depth vs Width Tradeoff
Deep (68.0%) and Wide (50.0%) both underperformed the Baseline (69.2%). On a small dataset, adding more layers or filters leads to overfitting. The sweet spot is moderate complexity.

---

## What I Learned

1. **Architecture matters more than size** — A well-designed small model beats a poorly designed big one
2. **Transfer learning is incredibly powerful** — Even a frozen backbone gives near-perfect results
3. **More data would help all models** — With only 3,000 images locally, many models overfit quickly
4. **Regularisation is essential** — Dropout and BatchNorm both help prevent overfitting
5. **PyTorch is flexible** — Building custom architectures is straightforward with `nn.Module`
6. **GPU vs CPU makes a huge difference** — Training that takes hours on CPU takes minutes on GPU

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
│   │   └── evaluator.py             # Evaluation & metrics
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

---

## Author

**Student Project** — Professional AI Development Course

## License

This project is for educational purposes.
