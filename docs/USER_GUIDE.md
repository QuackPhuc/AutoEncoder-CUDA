# AutoEncoder CUDA - User Guide

A CUDA-accelerated convolutional autoencoder for CIFAR-10 feature learning and image classification.

## Overview

This project implements a two-stage machine learning pipeline:

1. **Unsupervised Feature Learning**: Train a convolutional autoencoder to compress 32x32x3 CIFAR-10 images into 8,192-dimensional feature representations.
2. **Supervised Classification**: Train an SVM classifier on the extracted features to classify images into 10 categories.

The primary focus is GPU acceleration using CUDA, achieving up to **226x speedup** over CPU baseline.

---

## Prerequisites

### System Requirements

- **Operating System**: Linux, Windows, or macOS
- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- **VRAM**: Minimum 2 GB, recommended 4 GB or more
- **RAM**: Minimum 8 GB, **~17 GB peak for full pipeline** (SVM training phase)

### Software Requirements

| Component         | Version      |
|-------------------|--------------|
| CMake             | 3.18+        |
| C++ Compiler      | C++17 (GCC 7+, Clang 5+, MSVC 2017+) |
| CUDA Toolkit      | 11.0+        |
| ThunderSVM        | Latest (via submodule) |

### Verify Installation

```bash
cmake --version     # Should be 3.10+
nvcc --version      # Should be 11.0+
nvidia-smi          # Verify GPU detection
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive https://github.com/QuackPhuc/AutoEncoder-CUDA.git
cd AutoEncoder-CUDA
```

### 2. Download CIFAR-10 Dataset

```bash
./scripts/download_cifar10.sh
```

This downloads and extracts the CIFAR-10 binary files to `./data/`.

### 3. Download Pretrained Weights (Optional)

Skip training and use pretrained weights for quick evaluation:

```bash
./scripts/download_weights.sh
```

This downloads encoder weights and SVM model from Google Drive to `./checkpoints/`.

### 4. Setup ThunderSVM

```bash
git submodule update --init --recursive
```

### 5. Build the Project

```bash
./build.sh --clean
```

This creates three executables in `./build/bin/`:

| Executable              | Description                          |
|-------------------------|--------------------------------------|
| `autoencoder_cpu`       | CPU baseline training                |
| `autoencoder_gpu`       | GPU training (all optimization levels) |
| `autoencoder_inference` | Feature extraction + SVM pipeline    |

### 6. Run the Pipeline

```bash
# Full pipeline (default): train autoencoder -> train SVM -> evaluate
# Requires ~17GB peak RAM - use Google Colab Pro/Pro+ with High RAM mode
./run.sh

# Train autoencoder only
./run.sh train-autoencoder --epochs 20

# Train SVM using pre-trained encoder weights
./run.sh train-svm

# Evaluate with pre-trained weights (no training)
./run.sh evaluate
```

---

## Usage

### Training Options

```bash
./run.sh [COMMAND] [OPTIONS]

COMMANDS:
    train-autoencoder   Train autoencoder only, save encoder weights
    train-svm           Train SVM using pre-trained encoder weights
    evaluate            Evaluate using pre-trained encoder + SVM weights
    pipeline            Full: train-autoencoder -> train-svm -> evaluate (default)

OPTIONS:
    --device cpu|gpu        Device to use (default: gpu)
    --version v             GPU version: naive | v1 | v2 (default: v2)
    --epochs N              Training epochs (default: 20)
    --samples N             Limit samples, 0=all (default: 0)
    --batch-size N          Batch size, 0=auto (default: 0)
    --encoder-weights PATH  Input encoder weights (default: ./checkpoints/encoder.weights)
                            Used by: train-svm, evaluate
                            Ignored by: pipeline (uses own trained weights)
    --svm-model PATH        Input SVM model (default: ./checkpoints/svm.bin)
                            Used by: evaluate only
                            Ignored by: pipeline (uses own trained weights)
```

### GPU Version Selection

| Version  | Description                          | Expected Speedup |
|----------|--------------------------------------|------------------|
| `naive`  | Basic GPU parallelization            | 166.7x           |
| `v1`     | NCHW + 2D Grid + Warp Shuffle        | 237.0x           |
| `v2`     | im2col + cuBLAS GEMM                 | 266.0x           |

### Example Commands

```bash
# Full pipeline with best GPU version (default)
./run.sh

# Train autoencoder only
./run.sh train-autoencoder --device gpu --epochs 20

# Quick test (1000 samples, 5 epochs)
./run.sh train-autoencoder --epochs 5 --samples 1000

# Evaluate with custom weights
./run.sh evaluate --encoder-weights ./checkpoints/gpu_opt_v2.weights

# Benchmark all versions
./scripts/benchmark.sh
```

---

## CUDA Optimizations

### GPU Opt v1: NCHW Layout + 2D Grid

- NCHW memory layout (channels-first)
- 2D grid indexing for spatial locality
- 3x3 kernel compile-time unrolling
- Warp shuffle reduction for bias gradients

### GPU Opt v2: im2col + cuBLAS GEMM

- im2col transformation (patches to matrix)
- cuBLAS SGEMM for optimized matrix multiplication
- Fused bias addition + ReLU
- col2im for backward pass

---

## Executables

| Executable              | Description                          |
|-------------------------|--------------------------------------|
| `autoencoder_cpu`       | CPU baseline training                |
| `autoencoder_gpu`       | GPU training (all versions)          |
| `autoencoder_inference` | Phase 4: Feature extraction + SVM    |

---

## Output Files

### Checkpoints

Model weights are saved to `./checkpoints/`:

| File                  | Description                |
|-----------------------|----------------------------|
| `cpu_baseline`        | CPU-trained model          |
| `gpu_basic`           | GPU Basic (naive) model    |
| `gpu_opt_v1`          | GPU Opt v1 model           |
| `gpu_opt_v2`          | GPU Opt v2 model           |
| `svm_model.bin`       | Trained SVM classifier     |

### Results

Performance metrics and evaluation results are saved to `./results/`:

| File                        | Description                  |
|-----------------------------|------------------------------|
| `benchmark.csv`             | Benchmark timing data        |
| `benchmark.png`             | Performance comparison chart |
| `classification_report.txt` | Per-class accuracy report    |
| `confusion_matrix.csv`      | 10x10 confusion matrix       |

---

## Expected Performance

### Training Time (20 epochs, 50,000 images, Tesla T4)

| Version      | Training Time | Speedup |
|--------------|---------------|---------|
| CPU Baseline | ~400 hrs*     | 1.0x    |
| GPU Basic    | ~166 min      | 166.7x  |
| GPU Opt v1   | ~111 min      | 237.0x  |
| GPU Opt v2   | ~99 min       | 266.0x  |

*CPU time estimated from 100-sample benchmark.*

### Classification Accuracy

| Metric               | Achieved  |
|----------------------|-----------|
| Overall Accuracy     | 65%       |
| Feature Extraction   | ~7.6 sec  |
| SVM Training         | ~296.7 sec|

---

## Build Options

```bash
./build.sh [OPTIONS]

OPTIONS:
  --clean      Clean build directory before building
  --debug      Build in Debug mode (default: Release)
```

---

## Troubleshooting

### Build Errors

**CUDA not found**

```bash
# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**ThunderSVM not found**

```bash
# Download ThunderSVM as a submodule
git submodule update --init --recursive
./build.sh --clean
```

### Runtime Errors

**CUDA out of memory**

```bash
# Reduce batch size
./run.sh --gpu opt_v2 --batch-size 32
```

**Dataset not found**

```bash
./scripts/download_cifar10.sh
```

**Model weights not found (inference)**

```bash
# Train the model first
./run.sh train --device gpu --epochs 20
```

### Common Issues

| Issue                          | Solution                              |
|--------------------------------|---------------------------------------|
| CMake version too old          | Upgrade CMake to 3.18+                |
| GPU not detected               | Run `nvidia-smi` to verify driver     |
| Compilation errors on Windows  | Use Visual Studio 2017+ with CUDA     |
| Slow GPU performance           | Use `--version v2` for best speed     |

---

## Utility Scripts

The `scripts/` directory contains helper scripts to streamline common tasks.

### Benchmark Script (`scripts/benchmark.sh`)

Evaluates training performance across all available implementations (CPU, GPU Basic, Opt v1, Opt v2).

**Usage:**

```bash
./scripts/benchmark.sh [OPTIONS]
```

**Options:**

- `--epochs N`: Number of training epochs (default: 1)
- `--samples N`: Number of samples to use (default: 1000)
- `--gpu-only`: Skip CPU benchmark (useful for quick GPU comparisons)

**Output:**

- Generates `results/benchmark.csv` with timing and loss data.
- Automatically plots results if Python and Matplotlib are installed.

### Plotting Results (`scripts/plot_results.py`)

Generates visualization charts from benchmark CSV files. This is automatically called by `benchmark.sh` but can be run manually.

**Usage:**

```bash
python3 scripts/plot_results.py <path_to_benchmark.csv>
```

**Requirements:**

- Python 3
- Matplotlib (`pip install matplotlib`)

**Output:**

- Prints specific speedup metrics to the console.
- Saves `benchmark.png` chart in the same directory as the input CSV.

---

## Directory Structure

```
AutoEncoder-CUDA/
├── build/              # Build output
├── checkpoints/        # Saved model weights
├── data/               # CIFAR-10 dataset
├── docs/               # Documentation
├── external/thundersvm/# ThunderSVM library
├── results/            # Performance metrics
├── scripts/            # Utility scripts
├── src/                # Source code
├── build.sh            # Build script
├── run.sh              # Training script
└── CMakeLists.txt      # CMake configuration
```
