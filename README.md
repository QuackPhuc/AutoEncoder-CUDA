# AutoEncoder CUDA

High-Performance CUDA Convolutional Autoencoder for CIFAR-10 Feature Learning and Classification

[![Phase](https://img.shields.io/badge/Phase-4%20Complete-success)](.) [![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green)](https://developer.nvidia.com/cuda-toolkit) [![CMake](https://img.shields.io/badge/CMake-3.18%2B-blue)](https://cmake.org/)

## Overview

A complete two-stage machine learning pipeline:

1. **Unsupervised Feature Learning**: Train a convolutional autoencoder on CIFAR-10 images to learn 8,192-dimensional feature representations
2. **Supervised Classification**: Extract features and train an SVM classifier for image classification

The project demonstrates progressive CUDA optimization achieving significant GPU speedup over CPU baseline.

### Performance Summary

| Phase | Time/Epoch | Speedup vs CPU | Key Technique |
|-------|------------|----------------|---------------|
| CPU Baseline | 169.18 sec* | 1.0x | Sequential |
| GPU Naive (v1) | 500.57 sec | 169x | Basic Parallelization |
| GPU Opt v1 | 247.15 sec | 342x | NCHW + 2D Grid + Warp Shuffle |
| GPU Opt v2 | 50.52 sec | 1690x | im2col + cuBLAS GEMM |

***Notes:***

- *CPU baseline measured on 100 samples only. Estimated full training (50,000 samples): ~23.5 hours/epoch.*
- *GPU values: average per epoch from 50,000 samples, 3 epochs. (GPU: T4)*

## Quick Start

### Prerequisites

- CMake 3.18+, C++17 Compiler, CUDA Toolkit 11.0+, NVIDIA GPU (Pascal+).

### Build and Run

```bash
# 1. Clone repository (recursive for submodules)
git clone --recursive https://github.com/QuackPhuc/AutoEncoder-CUDA.git
cd AutoEncoder-CUDA

# 2. Download Dataset
./scripts/download_cifar10.sh

# 3. Setup ThunderSVM (if not cloned recursively)
git submodule update --init --recursive

# 4. Build Project
./build.sh

# 5. Run Full Pipeline (Train Autoencoder + Train SVM + Evaluate)
# Pipeline requires ~17GB peak RAM - use Google Colab Pro/Pro+ with High RAM mode
./run.sh

# Or train autoencoder only (outputs timestamped weights)
./run.sh train-autoencoder --epochs 20

# Train SVM using default encoder weights (./checkpoints/encoder.weights)
./run.sh train-svm

# Evaluate with default weights
./run.sh evaluate
```

> **Note**: For detailed instructions on building, training options, inference, and troubleshooting, please refer to the **[User Guide](docs/USER_GUIDE.md)**.

## GPU Versions

| Version | Description | Key Features |
|---------|-------------|--------------|
| `naive` | Basic GPU parallelization | Per-pixel thread mapping |
| `v1` | NCHW layout | 2D grid indexing, warp shuffle reduction |
| `v2` | GEMM-based convolution | im2col + cuBLAS SGEMM |

Use `--version` flag to select: `./run.sh train-autoencoder --version v2`

## Architecture

```text
Input: 32x32x3 CIFAR-10 Image

ENCODER:
  Conv1: 3->256 (3x3, pad=1) + ReLU + MaxPool(2x2) -> 16x16x256
  Conv2: 256->128 (3x3, pad=1) + ReLU + MaxPool(2x2) -> 8x8x128
  
LATENT: 8x8x128 = 8,192 features

DECODER:
  Conv3: 128->128 (3x3, pad=1) + ReLU + Upsample(2x) -> 16x16x128
  Conv4: 128->256 (3x3, pad=1) + ReLU + Upsample(2x) -> 32x32x256
  Conv5: 256->3 (3x3, pad=1) -> 32x32x3
  
Output: Reconstructed 32x32x3 Image

Total Parameters: 751,875
```

## Project Pipeline

| Step | Description | Component |
|------|-------------|-----------|
| 1 | Load CIFAR-10 | `cifar_loader.cpp` |
| 2 | Train Autoencoder | `gpu_autoencoder.cu` |
| 3 | Extract Features | `feature_extractor.cu` |
| 4 | Train SVM | `svm.cpp` (ThunderSVM) |
| 5 | Evaluate | `metrics.cpp` |

## Project Structure

```text
AutoEncoder-CUDA/
├───checkpoints/          # Saved model weights
├───data/                 # CIFAR-10 binary files
├───docs/                 # Documentation files
├───external/             # Third-party libraries (ThunderSVM)
├───notebooks/            # Python analysis notebooks
├───scripts/              # Build and run scripts
└───src/                  # Source code
    ├───benchmarking/     # Performance timing utilities
    ├───config/           # Configuration parsing
    ├───cpu/              # CPU baseline implementation
    │   ├───data/         # Data loading logic
    │   ├───evaluation/   # Metrics calculation
    │   ├───layers/       # Layer implementations
    │   ├───model/        # Autoencoder class
    │   └───training/     # Training loop
    ├───gpu/              # CUDA optimized implementation
    │   ├───core/         # Memory management
    │   ├───inference/    # Feature extraction
    │   ├───kernels/      # CUDA kernels (.cu)
    │   │   ├───backward/ # Backpropagation kernels
    │   │   ├───forward/  # Forward pass kernels
    │   │   └───gemm/     # im2col + GEMM kernels
    │   ├───model/        # GPU Autoencoder class
    │   └───svm/          # SVM wrapper
    └───utils/            # Helper functions
```

## Documentation

- [**User Guide**](docs/USER_GUIDE.md): Detailed usage, build options, and troubleshooting.
- [**Developer Guide**](docs/DEVELOPER.md): Code structure and contribution guide.
- [**Optimization Guide**](docs/OPTIMIZATION_GUIDE.md): CUDA optimization techniques explained.
- [**API Reference**](docs/API_REFERENCE.md): Class and function documentation.
