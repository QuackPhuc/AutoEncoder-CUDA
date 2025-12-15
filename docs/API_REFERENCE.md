# API Reference

Complete API documentation for AutoEncoder CUDA classes and functions.

---

## Table of Contents

1. [Data Loading](#data-loading)
2. [CPU Layers](#cpu-layers)
3. [CPU Autoencoder](#cpu-autoencoder)
4. [GPU Autoencoder](#gpu-autoencoder)
5. [Feature Extraction](#feature-extraction)
6. [SVM Components](#svm-components)
7. [Evaluation](#evaluation)
8. [Configuration](#configuration)
9. [Utilities](#utilities)
10. [CUDA Kernel Functions](#cuda-kernel-functions)

---

## Data Loading

### CIFAR10Dataset

**Header**: `src/cpu/data/cifar_loader.h`

Loads and manages CIFAR-10 dataset in binary format.

#### Constructor

```cpp
CIFAR10Dataset(const std::string& dataDir);
```

| Parameter | Type          | Description                              |
|-----------|---------------|------------------------------------------|
| `dataDir` | `std::string` | Path to directory containing CIFAR-10 binary files |

#### Methods

```cpp
void loadTrainData(int maxSamples = 0);
```

Loads training data from 5 batch files (data_batch_1.bin to data_batch_5.bin).

| Parameter    | Type  | Description                              |
|--------------|-------|------------------------------------------|
| `maxSamples` | `int` | Maximum samples to load (0 = load all 50,000) |

---

```cpp
void loadTestData();
```

Loads test data from test_batch.bin (10,000 images).

---

```cpp
const std::vector<float>& getTrainImages() const;
```

Returns normalized training images (float values in [0,1]).

**Returns**: Vector of size (N * 3072) where N is number of images.

---

```cpp
const std::vector<uint8_t>& getTrainLabels() const;
```

Returns training labels (values 0-9).

---

```cpp
const std::vector<float>& getTestImages() const;
```

Returns normalized test images.

---

```cpp
const std::vector<uint8_t>& getTestLabels() const;
```

Returns test labels.

---

```cpp
void shuffleTrainData();
```

Randomizes training data order for epoch randomization.

---

```cpp
std::vector<float> getBatch(int batchIdx, int batchSize);
```

Returns a batch of training images.

| Parameter   | Type  | Description              |
|-------------|-------|--------------------------|
| `batchIdx`  | `int` | Batch index              |
| `batchSize` | `int` | Number of images in batch |

**Returns**: Vector of size (batchSize * 3072).

---

```cpp
int getTrainSize() const;
int getTestSize() const;
```

Returns number of training/test images.

---

## CPU Layers

Neural network layer implementations for the CPU autoencoder.

### Conv2D

**Header**: `src/cpu/layers/conv2d.h`

2D Convolution layer with backpropagation support.

#### Constructor

```cpp
Conv2D(int inChannels, int outChannels, 
       int kernelSize = 3, int stride = 1, int padding = 1);
```

| Parameter     | Type  | Description                           |
|---------------|-------|---------------------------------------|
| `inChannels`  | `int` | Number of input channels              |
| `outChannels` | `int` | Number of output channels (filters)   |
| `kernelSize`  | `int` | Kernel size (default: 3)              |
| `stride`      | `int` | Stride (default: 1)                   |
| `padding`     | `int` | Padding (default: 1 for same conv)    |

#### Methods

```cpp
std::vector<float> forward(const std::vector<float>& input, int H, int W);
```

Forward pass. Input: (H *W* C_in) flattened. Output: (H_out *W_out* C_out) flattened.

---

```cpp
std::vector<float> backward(const std::vector<float>& gradOutput);
```

Backward pass. Computes gradients for weights and input.

---

```cpp
void updateWeights(float learningRate);
```

SGD weight update: w -= lr * grad_w

---

```cpp
void initializeWeights();
```

Weight initialization (He initialization).

---

### ReLU

**Header**: `src/cpu/layers/relu.h`

ReLU activation layer. Forward: max(0, x). Backward: grad * (x > 0 ? 1 : 0).

#### Constructor

```cpp
ReLU() = default;
```

#### Methods

```cpp
std::vector<float> forward(const std::vector<float>& input);
std::vector<float> backward(const std::vector<float>& gradOutput);
```

---

### MaxPool2D

**Header**: `src/cpu/layers/maxpool.h`

2D Max Pooling layer. Downsamples spatial dimensions by taking max value in each window.

#### Constructor

```cpp
MaxPool2D(int poolSize = 2, int stride = 2);
```

| Parameter  | Type  | Description                    |
|------------|-------|--------------------------------|
| `poolSize` | `int` | Pool window size (default: 2)  |
| `stride`   | `int` | Stride (default: 2)            |

#### Methods

```cpp
std::vector<float> forward(const std::vector<float>& input, int H, int W, int C);
std::vector<float> backward(const std::vector<float>& gradOutput);
```

Input: (H *W* C) flattened. Output: ((H/stride) *(W/stride)* C).

---

### Upsample2D

**Header**: `src/cpu/layers/upsample.h`

2D Upsampling using nearest neighbor interpolation. Replicates each pixel into scale x scale block.

#### Constructor

```cpp
Upsample2D(int scaleFactor = 2);
```

| Parameter     | Type  | Description                      |
|---------------|-------|----------------------------------|
| `scaleFactor` | `int` | Scale factor (default: 2)        |

#### Methods

```cpp
std::vector<float> forward(const std::vector<float>& input, int H, int W, int C);
std::vector<float> backward(const std::vector<float>& gradOutput);
```

Input: (H *W* C). Output: ((H*scale) * (W*scale) * C).

---

### MSELoss

**Header**: `src/cpu/layers/mse_loss.h`

Mean Squared Error loss. Loss = (1/N) * sum((target - prediction)^2).

#### Constructor

```cpp
MSELoss() = default;
```

#### Methods

```cpp
float forward(const std::vector<float>& predictions, const std::vector<float>& targets);
```

Returns scalar loss value.

---

```cpp
std::vector<float> backward();
```

Returns gradient: 2 * (prediction - target) / N.

---

## CPU Autoencoder

### Autoencoder

**Header**: `src/cpu/model/autoencoder.h`

CPU implementation of the convolutional autoencoder.

#### Constructor

```cpp
Autoencoder();
```

Initializes model with random weights (He initialization).

#### Methods

```cpp
std::vector<float> forward(const std::vector<float>& input);
```

Full forward pass (encoder + decoder).

| Parameter | Type                   | Description                    |
|-----------|------------------------|--------------------------------|
| `input`   | `std::vector<float>&`  | Flattened image (3072 floats)  |

**Returns**: Reconstructed image (3072 floats).

---

```cpp
std::vector<float> encode(const std::vector<float>& input);
```

Encoder-only forward pass for feature extraction.

| Parameter | Type                   | Description                    |
|-----------|------------------------|--------------------------------|
| `input`   | `std::vector<float>&`  | Flattened image (3072 floats)  |

**Returns**: Feature vector (8192 floats).

---

```cpp
void backward(const std::vector<float>& gradOutput);
```

Backpropagation through entire network.

| Parameter    | Type                   | Description                      |
|--------------|------------------------|----------------------------------|
| `gradOutput` | `std::vector<float>&`  | Gradient of loss w.r.t. output   |

---

```cpp
void updateWeights(float learningRate);
```

Updates all layer weights using SGD.

| Parameter      | Type    | Description          |
|----------------|---------|----------------------|
| `learningRate` | `float` | Learning rate (e.g., 0.001) |

---

```cpp
void saveModel(const std::string& filepath);
void loadModel(const std::string& filepath);
```

Model persistence (binary format).

---

## GPU Autoencoder

### GPUAutoencoder

**Header**: `src/gpu/model/autoencoder.h`

GPU-accelerated autoencoder with multiple optimization levels.

#### Constructor

```cpp
GPUAutoencoder(int batchSize = 64, float learningRate = 0.001f);
```

| Parameter      | Type    | Description                           |
|----------------|---------|---------------------------------------|
| `batchSize`    | `int`   | Batch size for training (default: 64) |
| `learningRate` | `float` | Learning rate (default: 0.001)        |

#### Destructor

```cpp
~GPUAutoencoder();
```

Frees all GPU memory.

#### Methods

```cpp
void trainStep(const std::vector<float>& h_batch);
```

Performs one training step: forward -> loss -> backward -> update.

| Parameter | Type                   | Description                              |
|-----------|------------------------|------------------------------------------|
| `h_batch` | `std::vector<float>&`  | Batch of images (batchSize * 3072 floats) |

---

```cpp
std::vector<float> getFeatures(const std::vector<float>& h_input);
```

Extracts features for a single image.

| Parameter  | Type                   | Description              |
|------------|------------------------|--------------------------|
| `h_input`  | `std::vector<float>&`  | Single image (3072 floats) |

**Returns**: Feature vector (8192 floats).

---

```cpp
std::vector<float> extractBatchFeatures(
    const std::vector<float>& images, 
    int numImages);
```

Extracts features for multiple images in a single GPU operation.

| Parameter   | Type                   | Description                    |
|-------------|------------------------|--------------------------------|
| `images`    | `std::vector<float>&`  | Multiple images (numImages * 3072) |
| `numImages` | `int`                  | Number of images               |

**Returns**: Features (numImages * 8192 floats).

---

```cpp
float getLoss() const;
```

Returns MSE loss from the last training step.

---

```cpp
void saveModel(const std::string& filepath);
void loadModel(const std::string& filepath);
```

Model persistence (binary format).

---

## Feature Extraction

### FeatureExtractor

**Header**: `src/gpu/inference/feature_extractor.h`

Batched GPU feature extraction for classification.

#### Constructor

```cpp
explicit FeatureExtractor(
    const std::string& encoderWeightsPath, 
    int batchSize = 128);
```

| Parameter            | Type          | Description                   |
|----------------------|---------------|-------------------------------|
| `encoderWeightsPath` | `std::string` | Path to saved encoder weights |
| `batchSize`          | `int`         | Max batch size for processing |

#### Destructor

```cpp
~FeatureExtractor();
```

Frees GPU memory.

#### Methods

```cpp
std::vector<float> extract_all(
    const std::vector<float>& images,
    int numImages,
    int batchSize = -1);
```

Extracts features for entire dataset.

| Parameter   | Type                   | Description                    |
|-------------|------------------------|--------------------------------|
| `images`    | `std::vector<float>&`  | All images (numImages * 3072)  |
| `numImages` | `int`                  | Number of images               |
| `batchSize` | `int`                  | Batch size (-1 = use default)  |

**Returns**: Features (numImages * 8192 floats).

---

```cpp
int get_feature_dim() const;
```

Returns feature dimension (always 8192).

---

## SVM Components

### ThunderSVMTrainer

**Header**: `src/gpu/svm/svm.h`  
**Namespace**: `gpu_svm`

GPU-accelerated SVM trainer using ThunderSVM for multi-class classification with RBF kernel.

#### Constructor

```cpp
explicit ThunderSVMTrainer(double C = 10.0, double gamma = -1.0);
```

| Parameter | Type     | Description                               |
|-----------|----------|-------------------------------------------|
| `C`       | `double` | Regularization parameter (default: 10.0)  |
| `gamma`   | `double` | RBF kernel parameter (-1 = auto: 1/num_features) |

#### Destructor

```cpp
~ThunderSVMTrainer();
```

Frees ThunderSVM model.

#### Methods

```cpp
void train(
    const std::vector<float>& train_features,
    const std::vector<uint8_t>& train_labels,
    int num_samples,
    int feature_dim);
```

Trains GPU-accelerated SVM model on extracted features.

| Parameter         | Type                    | Description                    |
|-------------------|-------------------------|--------------------------------|
| `train_features`  | `std::vector<float>&`   | Features (num_samples * feature_dim) |
| `train_labels`    | `std::vector<uint8_t>&` | Labels (0-9)                   |
| `num_samples`     | `int`                   | Number of training samples     |
| `feature_dim`     | `int`                   | Feature dimension (8192)       |

---

```cpp
void save_model(const std::string& filepath) const;
void load_model(const std::string& filepath);
```

Model persistence.

---

```cpp
std::vector<int> predict_batch(
    const std::vector<float>& features,
    int num_samples,
    int feature_dim) const;
```

Predicts labels for batch of feature vectors (GPU-accelerated).

| Parameter     | Type                   | Description                    |
|---------------|------------------------|--------------------------------|
| `features`    | `std::vector<float>&`  | Features (num_samples * feature_dim) |
| `num_samples` | `int`                  | Number of samples              |
| `feature_dim` | `int`                  | Feature dimension              |

**Returns**: Predicted labels (0-9) for each sample.

---

## Evaluation

### ClassificationMetrics

**Header**: `src/cpu/evaluation/metrics.h`  
**Namespace**: `evaluation`

Structure holding classification evaluation results.

#### Fields

| Field                | Type                        | Description              |
|----------------------|-----------------------------|--------------------------|
| `overall_accuracy`   | `float`                     | Overall accuracy (%)     |
| `per_class_accuracy` | `std::vector<float>`        | Per-class accuracy (%)   |
| `confusion_matrix`   | `std::vector<std::vector<int>>` | NxN confusion matrix |
| `class_counts`       | `std::vector<int>`          | Samples per class        |

#### Methods

```cpp
void print_summary() const;
```

---

### MetricsCalculator

**Header**: `src/cpu/evaluation/metrics.h`  
**Namespace**: `evaluation`

Static utility class for computing classification metrics.

#### Methods

```cpp
static ClassificationMetrics calculate(
    const std::vector<int>& predictions,
    const std::vector<uint8_t>& ground_truth,
    int num_classes = 10);
```

| Parameter      | Type                    | Description              |
|----------------|-------------------------|--------------------------|
| `predictions`  | `std::vector<int>&`     | Predicted labels         |
| `ground_truth` | `std::vector<uint8_t>&` | True labels              |
| `num_classes`  | `int`                   | Number of classes (10)   |

**Returns**: `ClassificationMetrics` structure with all metrics.

---

### ResultsVisualizer

**Header**: `src/cpu/evaluation/visualizer.h`  
**Namespace**: `evaluation`

Static utility class for outputting results.

#### Constants

```cpp
static const std::vector<std::string> CIFAR10_CLASSES;
// {"airplane", "automobile", "bird", "cat", "deer",
//  "dog", "frog", "horse", "ship", "truck"}
```

#### Methods

```cpp
static void print_confusion_matrix(
    const std::vector<std::vector<int>>& matrix,
    const std::vector<std::string>& class_names);

static void generate_report(
    const ClassificationMetrics& metrics,
    const std::vector<std::string>& class_names,
    const std::string& filepath);

static void save_confusion_matrix_csv(
    const std::vector<std::vector<int>>& matrix,
    const std::string& filepath);
```

---

## Configuration

### GPUVersion

**Header**: `src/config/gpu_config.h`

Enumeration of GPU implementation versions.

```cpp
enum class GPUVersion {
    CPU_BASELINE = 0,  // CPU-only (for reference)
    GPU_BASIC = 1,     // Naive GPU parallelization
    GPU_OPT_V1 = 2,    // Shared memory optimizations
    GPU_OPT_V2 = 3,    // Kernel fusion
    GPU_OPT_V3 = 4,    // NCHW + 2D grid + warp shuffle
    GPU_OPT_V4 = 5     // im2col + cuBLAS GEMM
};
```

---

### GPUConfig

**Header**: `src/config/gpu_config.h`

Singleton for GPU version selection and configuration.

#### Methods

```cpp
static GPUConfig& getInstance();
```

Returns singleton instance.

---

```cpp
void parseCommandLine(int argc, char** argv);
```

Parses `--gpu-version N` from command line.

---

```cpp
GPUVersion getVersion() const;
std::string getVersionName() const;
```

Returns current version and human-readable name.

---

```cpp
int getBatchSize() const;
```

Returns recommended batch size (32 for CPU, 64 for GPU).

---

## Utilities

### Timer

**Header**: `src/utils/timer.h`

High-resolution wall-clock timer.

#### Methods

```cpp
void start();
double stop();  // Returns elapsed time in seconds
```

---

### Logger

**Header**: `src/utils/logger.h`

Simple logging utility.

#### Methods

```cpp
void logEpoch(int epoch, int totalEpochs, float loss, double timeSeconds);
```

---

### GPUProfiler

**Header**: `src/benchmarking/profiler.h`

GPU training performance profiler.

#### Nested Types

```cpp
struct Metrics {
    double trainingTimeSec;
    double timePerEpochSec;
    float finalLoss;
    size_t gpuMemoryUsedBytes;
    float kernelOccupancy;
    float memoryBandwidthUtil;
};
```

#### Methods

```cpp
Metrics profileTraining(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    int epochs);

void printMetrics(
    const Metrics& metrics,
    const std::string& versionName) const;
```

---

## CUDA Kernel Functions

### Forward Pass Kernels

Kernels in `src/gpu/kernels/forward/` are primarily called directly via `extern` declarations for GPU_BASIC, or through NCHW host wrappers for GPU_OPT_V1+.

**NCHW Kernels (GPU Opt V1/V2/V3):**

```cpp
// NCHW convolution with optional ReLU fusion
void launchConv2dNCHW(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride);

void launchConv2dNCHWRelu(
    const float* d_input, const float* d_weights, const float* d_bias,
    float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride);

// NCHW max pooling
void launchMaxPool2dNCHW(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW,
    int k, int stride);

// NCHW upsample
void launchUpsample2dNCHW(
    const float* d_input, float* d_output,
    int batch, int channels, int inH, int inW,
    int scale);

// Stream-aware versions (GPU Opt V3)
void launchMaxPool2dNCHWStream(/* same params */ cudaStream_t stream);
void launchUpsample2dNCHWStream(/* same params */ cudaStream_t stream);
```

### Backward Pass Kernels

Backward kernels are located in `src/gpu/kernels/backward/`. For GPU_BASIC, kernels are called directly via `extern` declarations. For GPU_OPT_V1+, NCHW host wrappers are used.

**NCHW Wrappers (GPU Opt V1/V2/V3):**

```cpp
// NCHW Conv2D gradients
void launchConv2dBackwardInputNCHW(
    const float* d_gradOutput, const float* d_weights, 
    float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride);

void launchConv2dBackwardWeightsNCHW(
    const float* d_input, const float* d_gradOutput, 
    float* d_gradWeights,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride);

void launchConv2dBackwardBiasNCHW(
    const float* d_gradOutput, float* d_gradBias,
    int batch, int outC, int outH, int outW);

// NCHW MaxPool gradient
void launchMaxPool2dBackwardNCHW(
    const float* d_gradOutput, const int* d_indices,
    float* d_gradInput,
    int batch, int channels, int inH, int inW,
    int k, int stride);

// NCHW Upsample gradient
void launchUpsample2dBackwardNCHW(
    const float* d_gradOutput, float* d_gradInput,
    int batch, int channels, int inH, int inW,
    int scale);

// Stream-aware versions (GPU Opt V3)
void launchMaxPool2dBackwardNCHWStream(/* same params */ cudaStream_t stream);
void launchUpsample2dBackwardNCHWStream(/* same params */ cudaStream_t stream);
```

### CUDA Utilities

**Header**: `src/gpu/core/cuda_utils.h`

```cpp
// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Print GPU device info
void printGPUInfo();
```

### im2col + GEMM Kernels (V4)

Host wrapper functions for im2col + cuBLAS GEMM optimization:

```cpp
// im2col transformation: extract patches to column matrix
void launchIm2colNCHW(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

// col2im transformation: accumulate gradients back to input
void launchCol2imNCHW(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

// GEMM-based convolution forward pass
void launchConvGemmForward(
    const float* d_weights, const float* d_im2col, const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    bool applyRelu);

// GEMM-based backward pass for input gradients
void launchConvGemmBackwardInput(
    const float* d_weights, const float* d_gradOutput,
    float* d_col,
    int batch, int outC, int inC_k_k, int outHW);

// GEMM-based backward pass for weight gradients
void launchConvGemmBackwardWeights(
    const float* d_gradOutput, const float* d_im2col,
    float* d_gradWeights,
    int batch, int outC, int inC_k_k, int outHW);
```
