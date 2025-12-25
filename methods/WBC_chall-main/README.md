# WBCBench 2026: Robust White Blood Cell Classification

Complete implementation for the WBCBench 2026 challenge using pseudo-labeling with ResNet-50 architecture.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Pipeline](#data-pipeline)
- [Training Pipeline](#training-pipeline)
- [Pseudo-Labeling Method](#pseudo-labeling-method)
- [Model Architecture](#model-architecture)
- [Running Experiments](#running-experiments)
- [Submission Generation](#submission-generation)
- [Reproducibility](#reproducibility)

## Overview

This project implements a deep learning solution for white blood cell classification using:
- **Model**: ResNet-50 (with support for 13+ architectures)
- **Method**: Pseudo-labeling with iterative training
- **Split Strategy**: Old split method (phase1+phase2_train / phase2_eval)
- **Best Score**: 0.64887 (8th place)

## System Architecture

### Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPROCESSING                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼──────────┐        ┌──────────▼──────────┐
        │  Training Data       │        │  Test Data          │
        │  - phase1 (8,288)    │        │  - phase2/test     │
        │  - phase2_train      │        │    (16,477 images)  │
        │    (24,897)          │        │                     │
        │  Total: ~33,185      │        │                     │
        └───────────┬──────────┘        └──────────┬──────────┘
                    │                               │
                    │                               │
┌───────────────────▼───────────────────────────────▼───────────────────┐
│                      DATA AUGMENTATION                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Training Augmentations:                                         │    │
│  │  • Geometric: Horizontal/Vertical flips, Rotation (±15°)        │    │
│  │  • Elastic Transform, Grid Distortion, Optical Distortion     │    │
│  │  • Color: ColorJitter, Brightness, Contrast, Saturation        │    │
│  │  • Medical: CLAHE, Sharpening                                  │    │
│  │  • Noise: Gaussian Noise, Gaussian Blur, Motion Blur           │    │
│  │  • Rare Class Boost: 20x for PLY, 8x for PC, 10x for PMY       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Validation/Test Augmentations:                                  │    │
│  │  • Resize to 384x384 (or 224x224 for transformers)              │    │
│  │  • Normalization                                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                      DATA LOADING & SAMPLING                              │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ • WBCDataset: Handles image loading and augmentation           │     │
│  │ • WeightedRandomSampler: Balances classes during training      │     │
│  │ • Class Weights: Inverse frequency weighting                   │     │
│  │ • Min Samples: Guarantees 100+ samples per class per epoch     │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                         MODEL ARCHITECTURE                                │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ ResNet-50 (Best Model)                                          │     │
│  │  ┌──────────────────────────────────────────────────────────┐  │     │
│  │  │ Input: 384×384×3 RGB images                               │  │     │
│  │  │                                                           │  │     │
│  │  │ Conv1: 7×7 conv, stride 2 → 192×192×64                   │  │     │
│  │  │ MaxPool: 3×3, stride 2 → 96×96×64                        │  │     │
│  │  │                                                           │  │     │
│  │  │ ResBlock1: 3× conv blocks → 96×96×256                    │  │     │
│  │  │ ResBlock2: 4× conv blocks → 48×48×512                    │  │     │
│  │  │ ResBlock3: 6× conv blocks → 24×24×1024                   │  │     │
│  │  │ ResBlock4: 3× conv blocks → 12×12×2048                   │  │     │
│  │  │                                                           │  │     │
│  │  │ Global Average Pooling → 2048                             │  │     │
│  │  │ Dropout (0.2)                                             │  │     │
│  │  │ Fully Connected → 13 classes                             │  │     │
│  │  │ Softmax → Class Probabilities                             │  │     │
│  │  └──────────────────────────────────────────────────────────┘  │     │
│  │                                                                 │     │
│  │ Parameters: ~25M                                               │     │
│  │ Pretrained: ImageNet weights                                    │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                         TRAINING PROCESS                                  │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ Loss Function: Weighted CrossEntropyLoss                        │     │
│  │   • Class weights: Inverse frequency                           │     │
│  │   • PLY boost: 20x, PC: 8x, PMY: 10x                           │     │
│  │                                                                 │     │
│  │ Optimizer: AdamW                                                │     │
│  │   • Learning rate: 1e-4                                         │     │
│  │   • Weight decay: 1e-4                                         │     │
│  │                                                                 │     │
│  │ Scheduler: CosineAnnealingLR                                    │     │
│  │   • T_max: num_epochs                                           │     │
│  │                                                                 │     │
│  │ Training Features:                                              │     │
│  │   • Mixed Precision (FP16)                                     │     │
│  │   • Gradient Accumulation (effective batch size: 24)            │     │
│  │   • Early Stopping (patience: 5 epochs)                        │     │
│  │   • Best Model Checkpointing                                    │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                      PSEUDO-LABELING ITERATIONS                            │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ Iteration 0: Initial Training                                    │     │
│  │   • Train on: phase1 + phase2_train (~33,185 samples)            │     │
│  │   • Validate on: phase2_eval (5,350 samples)                    │     │
│  │   • Save: checkpoint_iter0/best.pth                             │     │
│  │                                                                 │     │
│  │ Iteration 1+: Pseudo-Labeling Loop                              │     │
│  │   1. Load model from previous iteration                          │     │
│  │   2. Generate pseudo-labels from test set                        │     │
│  │      • Confidence threshold: 0.9 (increases per iteration)       │     │
│  │      • TTA: Horizontal/Vertical flips                           │     │
│  │   3. Merge with previous training data                          │     │
│  │   4. Retrain on expanded dataset                                │     │
│  │   5. Save checkpoint and submission                              │     │
│  │                                                                 │     │
│  │ Output Files:                                                   │     │
│  │   • pseudo_labels_{model}_iter{N}.csv                           │     │
│  │   • expanded_training_{model}_iter{N}.csv                        │     │
│  │   • submission_pseudo_labeling_{model}_iter{N}.csv               │     │
│  │   • checkpoints/pseudo_labeling_{model}_iter{N}/best.pth        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────────┐
│                      SUBMISSION GENERATION                                 │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ 1. Load best checkpoint                                         │     │
│  │ 2. Generate predictions on test set                             │     │
│  │ 3. Apply PLY post-processing (ensures 50 PLY predictions)       │     │
│  │ 4. Format as CSV: ID, labels                                    │     │
│  │ 5. Validate submission format                                   │     │
│  │ 6. Save: submission_pseudo_labeling_{model}_iter{N}.csv        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                            ┌───────▼───────┐
                            │  Submit to    │
                            │    Kaggle     │
                            └───────────────┘
```

## Project Structure

```
WBC_Challenge/
├── src/                          # Core source code
│   ├── config.py                # Configuration management (auto-detects local/Kaggle)
│   ├── dataset.py               # Dataset classes, data loaders, augmentation
│   ├── models.py                # Model definitions (ResNet, EfficientNet, ViT, etc.)
│   ├── training.py              # Trainer class with training loop
│   ├── evaluation.py            # Metrics calculation (macro-F1, per-class F1)
│   ├── submission.py             # Submission generation with PLY post-processing
│   ├── pseudo_labeling.py       # Pseudo-labeling utilities
│   ├── ensemble.py              # Ensemble prediction utilities
│   └── postprocessing.py        # Post-processing utilities
│
├── scripts/                      # Executable scripts
│   ├── train_model.py           # Train individual models (no pseudo-labeling)
│   ├── pseudo_labeling_iterations.py  # Main pseudo-labeling script (ResNet-50 standard)
│   ├── generate_submission_advanced_tta.py  # Advanced TTA submission
│   ├── regenerate_submission_with_ply_fix.py  # Regenerate with PLY fixes
│   └── run_all_models_iterations.py  # Run iterations for multiple models
│
├── configs/                      # Model configurations
│   ├── config_resnet50.yaml     # ResNet-50 config (standard/best model)
│   ├── config_swin_tiny.yaml
│   ├── config_efficientnet_b3.yaml
│   ├── config_efficientnet_b4.yaml
│   ├── config_convnext_tiny.yaml
│   └── ... (other model configs)
│
├── wbc-bench-2026/              # Dataset directory (not in git)
│   ├── phase1/                  # Phase 1 training images
│   ├── phase1_label.csv
│   ├── phase2/
│   │   ├── train/               # Phase 2 training images
│   │   ├── eval/                # Phase 2 evaluation images
│   │   └── test/                # Phase 2 test images
│   ├── phase2_train.csv
│   ├── phase2_eval.csv
│   └── phase2_test.csv
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Setup and Installation

### 1. Clone Repository

```bash
git clone https://github.com/nghianguyen7171/WBC_chall.git
cd WBC_chall
```

### 2. Create Conda Environment

```bash
# Create environment
conda create -n WBC python=3.9 -y
conda activate WBC

# Install PyTorch with CUDA (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt
```

### 3. Dataset Setup

Download the WBCBench 2026 dataset and extract to `wbc-bench-2026/` directory:

```
wbc-bench-2026/
├── phase1/                    # 8,288 images
├── phase1_label.csv
├── phase2/
│   ├── train/                 # 24,897 images
│   ├── eval/                  # 5,350 images
│   └── test/                  # 16,477 images
├── phase2_train.csv
├── phase2_eval.csv
└── phase2_test.csv
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Pipeline

### Data Split Strategy (Old Method - Default)

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING DATA                            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   phase1         │   +     │  phase2_train   │         │
│  │   8,288 images   │         │  24,897 images  │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                    │
│           └────────────┬───────────────┘                    │
│                        │                                    │
│              Total: ~33,185 samples                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  VALIDATION DATA                            │
│  ┌──────────────────┐                                       │
│  │   phase2_eval    │                                       │
│  │   5,350 images   │  (Fixed validation set)               │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    TEST DATA                                │
│  ┌──────────────────┐                                       │
│  │   phase2_test    │                                       │
│  │   16,477 images  │  (For pseudo-labeling & submission)  │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

### Class Distribution

13 WBC classes:
- **BA** (Basophil), **BL** (Blast), **BNE** (Band Neutrophil), **EO** (Eosinophil)
- **LY** (Lymphocyte), **MMY** (Metamyelocyte), **MO** (Monocyte), **MY** (Myelocyte)
- **PC** (Platelet Clumps), **PLY** (Prolymphocyte - rarest, 14 samples), **PMY** (Promyelocyte)
- **SNE** (Segmented Neutrophil), **VLY** (Variant Lymphocyte)

### Data Augmentation Pipeline

```
Input Image (368×368)
        │
        ├─→ Resize to 384×384
        │
        ├─→ Geometric Augmentations (Training only)
        │   ├─→ Horizontal Flip (50%)
        │   ├─→ Vertical Flip (50%)
        │   ├─→ Rotation (±15°)
        │   ├─→ Elastic Transform (30%)
        │   ├─→ Grid Distortion (30%)
        │   └─→ Optical Distortion (20%)
        │
        ├─→ Color Augmentations (Training only)
        │   ├─→ ColorJitter (20%)
        │   ├─→ Brightness/Contrast (20%)
        │   ├─→ Saturation (20%)
        │   └─→ Hue (10%)
        │
        ├─→ Medical-Specific (Training only)
        │   ├─→ CLAHE (30%)
        │   └─→ Sharpening (30%)
        │
        ├─→ Noise Augmentations (Training only)
        │   ├─→ Gaussian Noise (10%)
        │   ├─→ Gaussian Blur (10%)
        │   └─→ Motion Blur (20%)
        │
        ├─→ Rare Class Boost (if PLY/PC/PMY)
        │   └─→ 1.5x augmentation probability
        │
        └─→ Normalize → Tensor
```

## Training Pipeline

### Standard Training (No Pseudo-Labeling)

```bash
# Train ResNet-50
python scripts/train_model.py --model resnet50

# Train other models
python scripts/train_model.py --model swin_tiny
python scripts/train_model.py --model efficientnet_b3
```

### Pseudo-Labeling Training (Recommended)

The standard approach uses ResNet-50 with old split method:

```bash
# Iteration 0: Initial training
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 0

# Iteration 1: First pseudo-labeling
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 1

# Iterations 2-5: Continue pseudo-labeling
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 4 \
    --start_iter 2 \
    --confidence_threshold 0.9 \
    --confidence_increment 0.01
```

### Training Process Details

```
┌─────────────────────────────────────────────────────────────┐
│                    EPOCH TRAINING                           │
│                                                             │
│  For each epoch:                                            │
│    1. WeightedRandomSampler selects balanced batches       │
│    2. Forward pass with mixed precision (FP16)             │
│    3. Calculate weighted loss                              │
│    4. Backward pass with gradient accumulation             │
│    5. Update weights (AdamW optimizer)                      │
│    6. Update learning rate (CosineAnnealingLR)            │
│                                                             │
│  After each epoch:                                          │
│    1. Validate on phase2_eval                              │
│    2. Calculate macro-F1 score                             │
│    3. Save best model if improved                           │
│    4. Early stopping check (patience: 5)                    │
└─────────────────────────────────────────────────────────────┘
```

## Pseudo-Labeling Method

### Algorithm Flow

```
┌─────────────────────────────────────────────────────────────┐
│              PSEUDO-LABELING ITERATION N                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                         │
┌───────▼────────┐                    ┌──────────▼────────┐
│ Load Model     │                    │ Load Training    │
│ from Iter N-1  │                    │ Data (Iter N-1)  │
└───────┬────────┘                    └──────────┬────────┘
        │                                         │
        └───────────────────┬─────────────────────┘
                            │
                ┌───────────▼───────────┐
                │ Generate Pseudo-Labels│
                │ from Test Set         │
                │ • Confidence ≥ 0.9    │
                │ • TTA: 4 augmentations│
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │ Merge Pseudo-Labels   │
                │ with Training Data    │
                │ • Preserve all data   │
                │ • No class balancing  │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │ Retrain Model on      │
                │ Expanded Dataset      │
                │ • 30 epochs           │
                │ • Same hyperparams    │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │ Save Checkpoint       │
                │ & Submission          │
                └─────────────────────┘
```

### Key Parameters

- **Confidence Threshold**: Starts at 0.9, increases by 0.01 per iteration
- **TTA**: Horizontal/Vertical flips for pseudo-label generation
- **Merging Strategy**: Preserve all training data, add high-confidence pseudo-labels
- **Validation**: Always uses phase2_eval (fixed, 5,350 samples)

## Model Architecture

## Running Experiments

### Quick Start: ResNet-50 Pseudo-Labeling

```bash
# Complete workflow: iterations 0-5
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 6 \
    --start_iter 0 \
    --confidence_threshold 0.9 \
    --confidence_increment 0.01
```

### Step-by-Step Execution

**Step 1: Initial Training (Iteration 0)**
```bash
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 0
```
- Trains ResNet-50 on phase1 + phase2_train
- Validates on phase2_eval
- Saves: `outputs/checkpoints/pseudo_labeling_resnet50_iter0/best.pth`

**Step 2: First Pseudo-Labeling (Iteration 1)**
```bash
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 1
```
- Loads model from iteration 0
- Generates pseudo-labels (confidence ≥ 0.9)
- Merges with training data
- Retrains on expanded dataset

**Step 3: Continue Iterations**
```bash
# Iterations 2-5 with increasing confidence
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 4 \
    --start_iter 2 \
    --confidence_threshold 0.91 \
    --confidence_increment 0.01
```

### Training Other Models

```bash
# Swin-Tiny
python scripts/pseudo_labeling_iterations.py \
    --model swin_tiny \
    --iterations 3 \
    --start_iter 0

# EfficientNet-B3
python scripts/pseudo_labeling_iterations.py \
    --model efficientnet_b3 \
    --iterations 3 \
    --start_iter 0
```

### Advanced Options

```bash
# Use new split method (merged split)
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --use_new_split \
    --iterations 3 \
    --start_iter 0

# Custom confidence threshold
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 5 \
    --confidence_threshold 0.93

# Disable TTA
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 1 \
    --no_tta
```

## Submission Generation

### Automatic (During Pseudo-Labeling)

Submissions are automatically generated after each iteration:
- Location: `outputs/submission_pseudo_labeling_resnet50_iter{N}.csv`
- Includes PLY post-processing (ensures 50 PLY predictions)

### Manual Generation

```bash
# Regenerate submission from checkpoint
python scripts/regenerate_submission_with_ply_fix.py \
    --model resnet50 \
    --checkpoint outputs/checkpoints/pseudo_labeling_resnet50_iter6/best.pth
```

### Advanced TTA Submission

```bash
# Generate with advanced TTA (multi-scale, rotation, color)
python scripts/generate_submission_advanced_tta.py \
    --model resnet50 \
    --checkpoint outputs/checkpoints/pseudo_labeling_resnet50_iter6/best.pth
```

### Submission Format

```csv
ID,labels
phase2_test_00001.jpg,LY
phase2_test_00002.jpg,SNE
...
phase2_test_16477.jpg,MO
```

- **Total rows**: 16,477 (one per test image)
- **Columns**: ID, labels
- **Labels**: One of 13 class codes (BA, BL, BNE, EO, LY, MMY, MO, MY, PC, PLY, PMY, SNE, VLY)

## Reproducibility

### Environment Setup

To ensure reproducibility when cloning on another machine:

1. **Python Version**: Python 3.9
2. **Conda Environment**: Create `WBC` environment
3. **Dependencies**: Install from `requirements.txt`
4. **Random Seeds**: All scripts use seed=42 by default

### Reproducible Training

```bash
# Set seed explicitly (default: 42)
python scripts/pseudo_labeling_iterations.py \
    --model resnet50 \
    --iterations 1 \
    --start_iter 0 \
    --seed 42
```

### Key Reproducibility Factors

1. **Random Seed**: 42 (set in all scripts)
2. **Data Split**: Old method uses fixed phase2_eval (deterministic)
3. **Model Initialization**: Pretrained weights (deterministic)
4. **Training**: `torch.backends.cudnn.deterministic = True`

### Expected Results

With seed=42 and old split method:
- **Iteration 0**: Validation F1 ~0.64-0.65
- **Iteration 1**: Validation F1 ~0.65-0.66
- **Iteration 2-5**: Gradual improvement to ~0.66-0.67
- **Test Score**: ~0.64-0.65 (may vary slightly)

## Configuration

### Model Configuration (`configs/config_resnet50.yaml`)

Key settings:
- **Image size**: 384×384
- **Batch size**: 12 (effective: 24 with gradient accumulation)
- **Learning rate**: 1e-4
- **Epochs**: 30
- **Early stopping**: Patience 5
- **Class weights**: Enabled
- **PLY boost**: 20x, PC: 8x, PMY: 10x

### Environment Detection

The code automatically detects:
- **Local**: Uses `configs/config_resnet50.yaml`
- **Kaggle**: Auto-detects Kaggle environment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable mixed precision (already enabled)
   - Reduce image size (384 → 320)

2. **Import Errors**
   - Ensure you're in project root
   - Activate conda environment: `conda activate WBC`
   - Check Python path includes project root

3. **Checkpoint Not Found**
   - Ensure previous iteration completed
   - Check checkpoint path in outputs/checkpoints/

4. **Low PLY Predictions**
   - PLY post-processing is automatic
   - Check submission file for PLY count
   - Should have ~50 PLY predictions

## Performance

### Best Results (ResNet-50)

- **Iteration 6**: 0.64887 (8th place)
- **Validation F1**: ~0.69
- **Training Time**: ~2-3 hours per iteration (on GPU)

### Model Comparison

| Model | Best Score | Notes |
|-------|-----------|-------|
| ResNet-50 | 0.64887 | Best performing, used for submission |
| Swin-Tiny | 0.64256 | Strong transformer baseline |
| EfficientNet-B3 | 0.63510 | Efficient architecture |
| ConvNeXt-Tiny | 0.63+ | Modern CNN |

## Citation

If you use this code, please cite:

```bibtex
@misc{wbcbench2026,
  title={WBCBench 2026: Robust White Blood Cell Classification},
  author={Your Name},
  year={2026},
  url={https://github.com/nghianguyen7171/WBC_chall}
}
```

## License

This project is for the WBCBench 2026 challenge. See competition rules for usage terms.

## Contact

- **Repository**: https://github.com/nghianguyen7171/WBC_chall
- **Challenge**: https://xudong-ma.github.io/WBCBench2026-Robust-White-Blood-Cell-Classification/
- **Kaggle**: https://www.kaggle.com/competitions/wbc-bench-2026
