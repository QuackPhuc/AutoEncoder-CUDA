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

| Phase | Training Time | Speedup | Key Technique |
|-------|---------------|---------|---------------|
| CPU Baseline | 439.97 sec | 1.0x | Sequential |
| GPU Naive | 2.64 sec | 166.7x | Basic Parallelization |
| GPU Opt v1 | 1.85 sec | 237.0x | NCHW + 2D Grid + Warp Shuffle |
| GPU Opt v2 | 1.65 sec | 266.0x | im2col + cuBLAS GEMM |

***Notes:***

- *Benchmarked on Tesla T4, 100 CIFAR-10 images, 3 epochs.*
- *GPU Opt v2 full training on 50000 CIFAR-10 images, 20 epochs takes ~126 min.*

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

- [QUICKSTART.md](../QUICKSTART.md) - Repository quick start guide
- [FILE_MANIFEST.md](../FILE_MANIFEST.md) - Complete file listing
- [README.md](../README.md) - Repository overview
