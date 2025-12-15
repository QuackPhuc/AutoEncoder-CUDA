# AutoEncoder CUDA - Developer Documentation

Technical documentation for the CUDA-accelerated convolutional autoencoder implementation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Source Code Structure](#source-code-structure)
3. [Neural Network Architecture](#neural-network-architecture)
4. [CPU Implementation](#cpu-implementation)
5. [GPU Implementation](#gpu-implementation)
6. [CUDA Kernels](#cuda-kernels)
7. [Optimization Techniques](#optimization-techniques)
8. [Feature Extraction and SVM Integration](#feature-extraction-and-svm-integration)
9. [Build System](#build-system)
10. [Extending the Project](#extending-the-project)

---

## Architecture Overview

The project implements a convolutional autoencoder for unsupervised feature learning on CIFAR-10 images. The pipeline consists of four phases:

| Phase | Description                                | Components                    |
|-------|--------------------------------------------|-------------------------------|
| 1     | CPU baseline implementation                | `src/cpu/`                    |
| 2     | Naive GPU parallelization                  | `src/gpu/` (basic kernels)    |
| 3     | GPU optimization v1 (NCHW layout)          | `src/gpu/kernels/forward/`    |
| 4     | GPU optimization v2 (im2col + GEMM)        | `src/gpu/kernels/gemm/`       |
| 5     | SVM integration and evaluation             | `src/cpu/svm/`, `src/gpu/inference/` |

### Data Flow

```
CIFAR-10 Images (32x32x3)
       |
       v
  [Autoencoder Training]
       |
       v
  Encoder Weights Saved
       |
       v
  [Feature Extraction] --> 8,192-dim features
       |
       v
  [SVM Training] --> Classification Model
       |
       v
  [Evaluation] --> Accuracy Metrics
```

---

## Source Code Structure

```
src/
├── cpu/                          # CPU implementation
│   ├── data/                     # Data loading
│   │   ├── cifar_loader.cpp      # CIFAR-10 binary parser
│   │   └── cifar_loader.h
│   ├── evaluation/               # Metrics and visualization
│   │   ├── metrics.cpp           # Accuracy calculation
│   │   ├── metrics.h
│   │   ├── visualizer.cpp        # Confusion matrix output
│   │   └── visualizer.h
│   ├── layers/                   # Neural network layers
│   │   ├── conv2d.cpp            # 2D convolution forward/backward
│   │   ├── conv2d.h
│   │   ├── maxpool.cpp           # Max pooling layer
│   │   ├── maxpool.h
│   │   ├── mse_loss.cpp          # Mean squared error loss
│   │   ├── mse_loss.h
│   │   ├── relu.cpp              # ReLU activation
│   │   ├── relu.h
│   │   ├── upsample.cpp          # Nearest neighbor upsampling
│   │   └── upsample.h
│   ├── model/                    # Autoencoder model
│   │   ├── autoencoder.cpp       # CPU forward/backward pass
│   │   └── autoencoder.h
│   └── training/                 # Training orchestration
│       ├── trainer.cpp           # Training loop
│       └── trainer.h
├── gpu/                          # GPU implementation
│   ├── core/                     # CUDA utilities
│   │   ├── cuda_utils.h          # Error checking macros, printGPUInfo
│   │   ├── device_reset.cu/h     # GPU device reset
│   │   ├── kernel_config.h       # Block/grid size macros
│   │   └── layout_convert.cu/h   # NHWC<->NCHW conversion
│   ├── inference/                # Feature extraction
│   │   ├── feature_extractor.cu  # Batched feature extraction
│   │   └── feature_extractor.h
│   ├── kernels/                  # CUDA kernels
│   │   ├── backward/             # Backward pass kernels
│   │   │   ├── activation.cu     # ReLU gradient
│   │   │   ├── conv2d.cu         # Conv2D gradients
│   │   │   ├── optimizer.cu      # SGD update
│   │   │   └── pooling.cu        # Pooling/upsample gradients
│   │   └── forward/              # Forward pass kernels
│   │       ├── activation.cu     # ReLU activation
│   │       ├── conv2d.cu         # Conv2D forward
│   │       ├── loss.cu           # MSE loss computation
│   │       └── pooling.cu        # MaxPool/Upsample
│   │   └── gemm/                 # im2col + cuBLAS (V4)
│   │       ├── im2col.cu         # im2col/col2im kernels
│   │       ├── conv_gemm.cu      # cuBLAS SGEMM wrappers
│   │       └── conv_gemm.h       # cuBLAS handle management
│   └── model/                    # GPU autoencoder model
│   │   ├── autoencoder.cu        # Core class implementation
│   │   ├── autoencoder.h         # Class declaration
│   │   ├── backward_pass.cu      # Backward pass orchestration
│   │   └── forward_pass.cu       # Forward pass orchestration
│   └── svm/                      # ThunderSVM GPU wrapper
│       ├── svm.cpp               # GPU SVM trainer/predictor
│       └── svm.h
├── config/                       # Configuration
│   ├── gpu_config.cpp            # GPU version selection
│   └── gpu_config.h
├── benchmarking/                 # Performance profiling
│   ├── profiler.cpp              # Training metrics
│   └── profiler.h
├── utils/                        # Utilities
│   ├── logger.cpp                # Logging utility
│   ├── logger.h
│   ├── timer.cpp                 # Wall-clock timer
│   └── timer.h
├── main.cpp                      # CPU training entry point
├── main_gpu.cpp                  # GPU training entry point
└── main_inference.cpp            # Phase 4 pipeline entry point
```

---

## Neural Network Architecture

### Autoencoder Architecture

The autoencoder compresses 32x32x3 images into an 8,192-dimensional latent representation (8x8x128).

**Encoder Path**

| Layer    | Operation                       | Input Shape    | Output Shape   | Parameters  |
|----------|---------------------------------|----------------|----------------|-------------|
| Conv1    | Conv2D(3x3, 3 -> 256, pad=1)    | 32x32x3        | 32x32x256      | 7,168       |
| ReLU1    | ReLU                            | 32x32x256      | 32x32x256      | 0           |
| Pool1    | MaxPool2D(2x2, stride=2)        | 32x32x256      | 16x16x256      | 0           |
| Conv2    | Conv2D(3x3, 256 -> 128, pad=1)  | 16x16x256      | 16x16x128      | 295,040     |
| ReLU2    | ReLU                            | 16x16x128      | 16x16x128      | 0           |
| Pool2    | MaxPool2D(2x2, stride=2)        | 16x16x128      | 8x8x128        | 0           |

**Latent Representation**: 8 x 8 x 128 = 8,192 dimensions

**Decoder Path**

| Layer    | Operation                       | Input Shape    | Output Shape   | Parameters  |
|----------|---------------------------------|----------------|----------------|-------------|
| Conv3    | Conv2D(3x3, 128 -> 128, pad=1)  | 8x8x128        | 8x8x128        | 147,584     |
| ReLU3    | ReLU                            | 8x8x128        | 8x8x128        | 0           |
| Up1      | Upsample2D(2x)                  | 8x8x128        | 16x16x128      | 0           |
| Conv4    | Conv2D(3x3, 128 -> 256, pad=1)  | 16x16x128      | 16x16x256      | 295,168     |
| ReLU4    | ReLU                            | 16x16x256      | 16x16x256      | 0           |
| Up2      | Upsample2D(2x)                  | 16x16x256      | 32x32x256      | 0           |
| Conv5    | Conv2D(3x3, 256 -> 3, pad=1)    | 32x32x256      | 32x32x3        | 6,915       |

**Total Parameters**: 751,875

### Memory Layout

- **V1**: NCHW (batch, channels, height, width) - optimized for spatial operations
- **V2**: NCHW (batch, channels, height, width) - optimized for cuBLAS GEMM (via im2col buffer)

---

## CPU Implementation

### Key Classes

**`CIFAR10Dataset`** (`src/cpu/data/cifar_loader.h`)

Loads and manages CIFAR-10 binary data files.

```cpp
class CIFAR10Dataset {
public:
    CIFAR10Dataset(const std::string& dataDir);
    void loadTrainData(int maxSamples = 0);
    void loadTestData();
    std::vector<float> getBatch(int batchIdx, int batchSize);
    void shuffleTrainData();
    // Getters for images and labels...
};
```

**`Autoencoder`** (`src/cpu/model/autoencoder.h`)

CPU autoencoder model with forward/backward pass and weight management.

```cpp
class Autoencoder {
public:
    Autoencoder();
    std::vector<float> forward(const std::vector<float>& input);
    std::vector<float> encode(const std::vector<float>& input);
    void backward(const std::vector<float>& gradOutput);
    void updateWeights(float learningRate);
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);
};
```

**`Trainer`** (`src/cpu/training/trainer.h`)

Orchestrates the training loop.

```cpp
class Trainer {
public:
    Trainer(Autoencoder& model, CIFAR10Dataset& dataset, float learningRate);
    void train(int epochs, int batchSize);
    float trainEpoch(int batchSize);
};
```

---

## GPU Implementation

### Key Classes

**`GPUAutoencoder`** (`src/gpu/model/autoencoder.h`)

GPU autoencoder with multiple optimization levels.

```cpp
class GPUAutoencoder {
public:
    GPUAutoencoder(int batchSize = 64, float learningRate = 0.001f);
    ~GPUAutoencoder();
    
    void trainStep(const std::vector<float>& h_batch);
    std::vector<float> getFeatures(const std::vector<float>& h_input);
    std::vector<float> extractBatchFeatures(const std::vector<float>& images, int numImages);
    float getLoss() const;
    
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);

private:
    // Forward pass dispatchers
    void forward();
    void forwardBasic();
    void forwardOptV1();
    void forwardOptV2();
    
    // Backward pass dispatchers
    void backward();
    void backwardBasic();
    void backwardOptV1();
    void backwardOptV2();
    
    // GPU memory buffers...
};
```

**`GPUConfig`** (`src/config/gpu_config.h`)

Singleton for GPU version selection.

```cpp
    GPUVersion {
    CPU_BASELINE = 0,
    GPU_BASIC = 1,      // Naive GPU parallelization
    GPU_OPT_V1 = 2,     // NCHW + 2D grid + warp shuffle
    GPU_OPT_V2 = 3      // im2col + cuBLAS GEMM
};

class GPUConfig {
public:
    static GPUConfig& getInstance();
    void parseCommandLine(int argc, char** argv);
    GPUVersion getVersion() const;
    int getBatchSize() const;
};
```

**`FeatureExtractor`** (`src/gpu/inference/feature_extractor.h`)

Batched feature extraction for SVM training.

```cpp
class FeatureExtractor {
public:
    explicit FeatureExtractor(const std::string& encoderWeightsPath, int batchSize = 128);
    ~FeatureExtractor();
    
    std::vector<float> extract_all(
        const std::vector<float>& images,
        int numImages,
        int batchSize = -1);
    
    int get_feature_dim() const { return 8 * 8 * 128; }
};
```

### GPU Memory Management

Memory buffers are allocated in `GPUAutoencoder::allocateMemory()` during construction and reused across batches. Memory is managed as member variables of the `GPUAutoencoder` class:

```cpp
class GPUAutoencoder {
private:
    // Device memory pointers (member variables)
    float* d_input;           // (batch, 32, 32, 3)
    float* d_output;          // (batch, 32, 32, 3)
    float* d_enc_conv1_w;     // (256, 3, 3, 3)
    float* d_enc_conv1_b;     // (256)
    // ... additional layer buffers
};
```

---

## CUDA Kernels

### Forward Pass Kernels

**`conv2dForwardNaiveKernel`** (`src/gpu/kernels/forward/conv2d.cu`)

Basic convolution where each thread computes one output pixel.

```cpp
__global__ void conv2dForwardNaiveKernel(
    const float* input,   // (batch, inH, inW, inC)
    const float* weights, // (outC, kernelSize, kernelSize, inC)
    const float* bias,    // (outC)
    float* output,        // (batch, outH, outW, outC)
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride);
```

**`conv2dForwardSharedKernel`** (GPU Opt v1)

Shared memory tiling with 18x18 tiles for 3x3 convolutions.

```cpp
__global__ void conv2dForwardSharedKernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride);
```

**`conv2dForwardSharedReluKernel`** (GPU Opt v2)

Fused Conv2D + ReLU kernel eliminating intermediate memory writes.

```cpp
// Adds bias and applies ReLU inline
sum += d_constBias[oc];
sum = fmaxf(0.0f, sum);
output[outputIdx] = sum;
```

### Backward Pass Kernels

**`conv2dBackwardInputKernel`** (`src/gpu/kernels/backward/conv2d.cu`)

Computes gradient with respect to input.

**`conv2dBackwardWeightsKernel`**

Computes gradient with respect to weights.

**`conv2dBackwardBiasKernel`**

Sums gradients for bias update.

### Host Wrappers

Each kernel has a corresponding host wrapper function:

```cpp
void launchConv2dNaive(...);
void launchConv2dShared(...);
void launchConv2dSharedRelu(...);
void launchConv2dBackwardInput(...);
void launchConv2dBackwardWeights(...);
void launchConv2dBackwardBias(...);
```

---

## Optimization Techniques

### GPU Opt v1: NCHW Layout + 2D Grid

| Technique              | Description                                    |
|------------------------|------------------------------------------------|
| NCHW layout            | Channels-first memory layout                   |
| 2D grid indexing       | dim3 block(16,16) for spatial locality         |
| 3x3 unrolling          | Compile-time loop unrolling                    |
| Fused Conv+ReLU        | Eliminates intermediate memory writes          |
| `__restrict__` hints   | Compiler optimization for non-overlapping ptrs |
| Cache-optimized access | Relies on L1/L2 cache (no explicit shared mem) |
| Warp shuffle reduction | `__shfl_down_sync` for bias gradients          |

### GPU Opt v2: im2col + cuBLAS GEMM

| Technique              | Description                                    |
|------------------------|------------------------------------------------|
| im2col transformation  | Convert conv to matrix multiply                |
| cuBLAS SGEMM           | Optimized BLAS matrix multiplication           |
| Workspace buffer       | `d_im2col_workspace` for col matrix            |
| col2im backward        | Gradient accumulation for backward pass        |

### Block and Grid Configuration

```cpp
// 2D grid for NCHW convolution (cache-optimized, no explicit shared memory)
dim3 block(16, 16);
dim3 grid(
    (outW + block.x - 1) / block.x,
    (outH + block.y - 1) / block.y,
    batch * outC
);
```

---

## Feature Extraction and SVM Integration

### Feature Extraction

The `FeatureExtractor` class uses the trained encoder to extract 8,192-dimensional features:

```cpp
std::vector<float> FeatureExtractor::extract_all(
    const std::vector<float>& images,
    int numImages,
    int batchSize)
{
    std::vector<float> allFeatures(numImages * 8192);
    
    for (int start = 0; start < numImages; start += batchSize) {
        int currentBatch = std::min(batchSize, numImages - start);
        auto batchFeatures = m_encoder->extractBatchFeatures(
            images.data() + start * 3072, currentBatch);
        // Copy to output...
    }
    return allFeatures;
}
```

### SVM Training

**`ThunderSVMTrainer`** (`src/gpu/svm/svm.h`)

Wraps ThunderSVM for GPU-accelerated multi-class classification using RBF kernel.

```cpp
namespace gpu_svm {

class ThunderSVMTrainer {
public:
    explicit ThunderSVMTrainer(double C = 10.0, double gamma = -1.0);
    ~ThunderSVMTrainer();
    
    void train(
        const std::vector<float>& train_features,
        const std::vector<uint8_t>& train_labels,
        int num_samples,
        int feature_dim);
    
    std::vector<int> predict_batch(
        const std::vector<float>& features,
        int num_samples,
        int feature_dim) const;
        
    void save_model(const std::string& filepath) const;
    void load_model(const std::string& filepath);
};

} // namespace gpu_svm
```

### Metrics Calculation

**`MetricsCalculator`** (`src/cpu/evaluation/metrics.h`)

Computes classification accuracy and confusion matrix.

```cpp
struct ClassificationMetrics {
    float overall_accuracy;
    std::vector<float> per_class_accuracy;
    std::vector<std::vector<int>> confusion_matrix;
};

ClassificationMetrics MetricsCalculator::calculate(
    const std::vector<int>& predictions,
    const std::vector<uint8_t>& ground_truth,
    int num_classes);
```

---

## Build System

### CMake Configuration

The project uses CMake with CUDA language support.

**Executables**

| Target                  | Description                          | Sources                          |
|-------------------------|--------------------------------------|----------------------------------|
| `autoencoder_cpu`       | CPU baseline training                | `main.cpp`, CPU modules          |
| `autoencoder_gpu`       | GPU training                         | `main_gpu.cpp`, GPU modules      |
| `autoencoder_inference` | Phase 4 pipeline                     | `main_inference.cpp`, all modules |

**Compiler Flags**

```cmake
# CUDA
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)
target_compile_options(... PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

# C++
set(CMAKE_CXX_STANDARD 17)
target_compile_options(... PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
```

### Scripts

| Script                       | Purpose                              |
|------------------------------|--------------------------------------|
| `build.sh`                   | Build project with CMake             |
| `run.sh`                     | Run training/pipeline (see below)    |
| `scripts/download_cifar10.sh`| Download CIFAR-10 dataset            |
| `scripts/benchmark.sh`       | Benchmark all CPU/GPU versions       |
| `scripts/plot_results.py`    | Generate charts from benchmark CSV   |

#### `run.sh` Weight Path Behavior

| Mode | Input Weights | Output Weights |
|------|---------------|----------------|
| `train-autoencoder` | N/A | `encoder_<timestamp>.weights` |
| `train-svm` | `--encoder-weights` (default: `encoder.weights`) | `svm_<timestamp>.bin` |
| `evaluate` | `--encoder-weights`, `--svm-model` (defaults exist) | N/A |
| `pipeline` | **Ignored** (uses own trained weights) | Both timestamped |

---

## Extending the Project

### Adding a New GPU Optimization Version

1. Add new enum value in `src/config/gpu_config.h`:

```cpp
enum class GPUVersion {
    // ...
    GPU_OPT_V3 = 4  // New version
};
```

2. Update `GPUConfig::parseCommandLine()` and related methods.

3. Add new forward/backward implementations in `GPUAutoencoder`:

```cpp
void GPUAutoencoder::forwardOptV3() {
    // New optimized implementation
}
```

4. Update dispatcher in `forward()` and `backward()` methods.

### Adding New Kernels

1. Create kernel file in `src/gpu/kernels/forward/` or `backward/`.

2. Implement kernel with host wrapper:

```cpp
__global__ void myNewKernel(...) { ... }

void launchMyNewKernel(...) {
    dim3 block(...);
    dim3 grid(...);
    myNewKernel<<<grid, block>>>(...);
    CHECK_CUDA(cudaGetLastError());
}
```

3. Declare wrapper in appropriate header file.

4. Call from `GPUAutoencoder` implementation.

### Modifying Network Architecture

To change the autoencoder architecture:

1. Update layer dimensions directly in `src/gpu/model/autoencoder.cu`:

2. Update memory allocation in `GPUAutoencoder::allocateMemory()`.

3. Update forward/backward pass implementations.

4. Ensure latent dimension is updated in `FeatureExtractor`.

---

## Error Handling

### CUDA Error Checking

All CUDA calls use the `CHECK_CUDA` macro:

```cpp
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

### Common Pitfalls

| Issue                        | Solution                                      |
|------------------------------|-----------------------------------------------|
| Synchronization errors       | Use `cudaDeviceSynchronize()` for debugging   |
| Shared memory overflow       | Check tile size fits in 48KB shared memory    |
| Memory coalescing issues     | Ensure NHWC layout and aligned access         |
| Race conditions              | Use `__syncthreads()` between shared memory access |

---

## Performance Profiling

### Using GPUProfiler

```cpp
GPUProfiler profiler;
GPUProfiler::Metrics metrics = profiler.profileTraining(model, dataset, epochs);
profiler.printMetrics(metrics, "GPU_OPT_V2");
```

### Metrics Collected

| Metric                | Description                          |
|-----------------------|--------------------------------------|
| `trainingTimeSec`     | Total training time                  |
| `timePerEpochSec`     | Average time per epoch               |
| `finalLoss`           | MSE loss after training              |
| `gpuMemoryUsedBytes`  | GPU memory consumption               |

### NVIDIA Profiler Integration

For detailed kernel-level profiling:

```bash
# Using nvprof
nvprof ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1

# Using Nsight Compute
ncu --set full ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1
```
