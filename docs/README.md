# AutoEncoder CUDA Documentation

Complete documentation for the CUDA-accelerated convolutional autoencoder project.

---

## Documentation Index

### For Users

- [User Guide](USER_GUIDE.md) - Setup, installation, and usage instructions

### For Developers

- [Developer Documentation](DEVELOPER.md) - Source code structure and implementation details
- [API Reference](API_REFERENCE.md) - Complete class and function reference
- [Network Architecture](ARCHITECTURE.md) - Autoencoder architecture specification
- [CUDA Optimization Guide](OPTIMIZATION_GUIDE.md) - GPU optimization techniques

---

## Quick Links

### Getting Started

1. [Prerequisites](USER_GUIDE.md#prerequisites)
2. [Quick Start](USER_GUIDE.md#quick-start)
3. [Training Options](USER_GUIDE.md#usage)

### Development

1. [Source Code Structure](DEVELOPER.md#source-code-structure)
2. [GPU Implementation](DEVELOPER.md#gpu-implementation)
3. [Extending the Project](DEVELOPER.md#extending-the-project)

### Reference

1. [CPU Layers API](API_REFERENCE.md#cpu-layers)
2. [GPU Autoencoder API](API_REFERENCE.md#gpu-autoencoder)
3. [CUDA Kernels](API_REFERENCE.md#cuda-kernel-functions)
4. [SVM Integration](API_REFERENCE.md#svm-components)

---

## Project Overview

This project implements a two-stage machine learning pipeline for CIFAR-10 image classification:

**Stage 1: Unsupervised Feature Learning**

- Convolutional autoencoder learns to compress images
- 8,192-dimensional latent representation
- Trained without labels (reconstruction loss)

**Stage 2: Supervised Classification**

- Extract features using trained encoder
- Train SVM classifier on features
- Evaluate on test set (target: 60-65% accuracy)

### Performance

| Phase | Time/Epoch | Speedup vs CPU | Key Technique |
|-------|------------|----------------|---------------|
| CPU Baseline | 169.18 sec* | 1.0x | Sequential |
| GPU Naive (v1) | 500.57 sec | 169x | Basic Parallelization |
| GPU Opt v1 | 247.15 sec | 342x | NCHW + 2D Grid + Warp Shuffle |
| GPU Opt v2 | 50.52 sec | 1690x | im2col + cuBLAS GEMM |

***Notes:***

- *CPU baseline measured on 100 samples only. Estimated full training (50,000 samples): ~23.5 hours/epoch.*
- *GPU values: average per epoch from 50,000 samples, 3 epochs. (GPU: T4)*

---

## Directory Structure

```
docs/
├── README.md           # This file (index)
├── USER_GUIDE.md       # User documentation
├── DEVELOPER.md        # Developer documentation
├── API_REFERENCE.md    # API reference
├── ARCHITECTURE.md     # Network architecture
└── OPTIMIZATION_GUIDE.md # CUDA optimization guide
```

---

## Related Files

- [README.md](../README.md) - Repository overview
