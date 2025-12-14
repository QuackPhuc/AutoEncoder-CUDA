# Network Architecture

Technical specification of the convolutional autoencoder architecture.

---

## Overview

The autoencoder is a symmetric encoder-decoder network that compresses 32x32x3 CIFAR-10 images into an 8,192-dimensional latent representation, then reconstructs the original image.

```
INPUT (32x32x3) -> ENCODER -> LATENT (8x8x128) -> DECODER -> OUTPUT (32x32x3)
         3072 dims           8192 dims                   3072 dims
```

---

## Architecture Diagram

```
                        ENCODER
                           |
     +---------------------+---------------------+
     |                     |                     |
  Input Image         Conv Layer 1           Pool Layer 1
  32 x 32 x 3   ->   32 x 32 x 256   ->   16 x 16 x 256
     (3072)            + ReLU                (65536)
                           |
                     Conv Layer 2           Pool Layer 2
                   16 x 16 x 128   ->    8 x 8 x 128
                       + ReLU                (8192)
                           |
                      LATENT SPACE
                       8 x 8 x 128
                         (8192)
                           |
                        DECODER
                           |
     +---------------------+---------------------+
     |                     |                     |
                     Conv Layer 3           Upsample 1
                    8 x 8 x 128    ->   16 x 16 x 128
                       + ReLU               (32768)
                           |
                     Conv Layer 4           Upsample 2
                   16 x 16 x 256   ->   32 x 32 x 256
                       + ReLU              (262144)
                           |
                     Conv Layer 5
                   32 x 32 x 3
                      (3072)
                           |
                    Output Image
                     32 x 32 x 3
```

---

## Layer Specifications

### Input

| Property     | Value          |
|--------------|----------------|
| Height       | 32 pixels      |
| Width        | 32 pixels      |
| Channels     | 3 (RGB)        |
| Data type    | float32        |
| Value range  | [0.0, 1.0]     |
| Total size   | 3,072 values   |

---

### Encoder Layer 1

**Convolution**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 32, 32, 3)         |
| Kernel size  | 3 x 3                      |
| Filters      | 256                        |
| Padding      | 1 (same)                   |
| Stride       | 1                          |
| Output shape | (batch, 32, 32, 256)       |

**Weight dimensions**

| Tensor   | Shape              | Parameters       |
|----------|-------------------|------------------|
| Weights  | (256, 3, 3, 3)    | 6,912            |
| Biases   | (256)             | 256              |
| Total    |                   | 7,168            |

**Activation**: ReLU (element-wise max(0, x))

**Max Pooling**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 32, 32, 256)       |
| Pool size    | 2 x 2                      |
| Stride       | 2                          |
| Output shape | (batch, 16, 16, 256)       |

---

### Encoder Layer 2

**Convolution**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 16, 16, 256)       |
| Kernel size  | 3 x 3                      |
| Filters      | 128                        |
| Padding      | 1 (same)                   |
| Stride       | 1                          |
| Output shape | (batch, 16, 16, 128)       |

**Weight dimensions**

| Tensor   | Shape              | Parameters       |
|----------|-------------------|------------------|
| Weights  | (128, 3, 3, 256)  | 294,912          |
| Biases   | (128)             | 128              |
| Total    |                   | 295,040          |

**Activation**: ReLU

**Max Pooling**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 16, 16, 128)       |
| Pool size    | 2 x 2                      |
| Stride       | 2                          |
| Output shape | (batch, 8, 8, 128)         |

---

### Latent Space

| Property          | Value                |
|-------------------|----------------------|
| Shape             | (batch, 8, 8, 128)   |
| Dimensions        | 8,192                |
| Compression ratio | 3072 / 8192 = 0.375  |

The latent representation captures essential visual features while maintaining spatial structure.

---

### Decoder Layer 3

**Convolution**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 8, 8, 128)         |
| Kernel size  | 3 x 3                      |
| Filters      | 128                        |
| Padding      | 1 (same)                   |
| Stride       | 1                          |
| Output shape | (batch, 8, 8, 128)         |

**Weight dimensions**

| Tensor   | Shape              | Parameters       |
|----------|-------------------|------------------|
| Weights  | (128, 3, 3, 128)  | 147,456          |
| Biases   | (128)             | 128              |
| Total    |                   | 147,584          |

**Activation**: ReLU

**Upsampling**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 8, 8, 128)         |
| Scale factor | 2x                         |
| Method       | Nearest neighbor           |
| Output shape | (batch, 16, 16, 128)       |

---

### Decoder Layer 4

**Convolution**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 16, 16, 128)       |
| Kernel size  | 3 x 3                      |
| Filters      | 256                        |
| Padding      | 1 (same)                   |
| Stride       | 1                          |
| Output shape | (batch, 16, 16, 256)       |

**Weight dimensions**

| Tensor   | Shape              | Parameters       |
|----------|-------------------|------------------|
| Weights  | (256, 3, 3, 128)  | 294,912          |
| Biases   | (256)             | 256              |
| Total    |                   | 295,168          |

**Activation**: ReLU

**Upsampling**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 16, 16, 256)       |
| Scale factor | 2x                         |
| Method       | Nearest neighbor           |
| Output shape | (batch, 32, 32, 256)       |

---

### Decoder Layer 5

**Convolution**

| Property     | Value                      |
|--------------|----------------------------|
| Input shape  | (batch, 32, 32, 256)       |
| Kernel size  | 3 x 3                      |
| Filters      | 3                          |
| Padding      | 1 (same)                   |
| Stride       | 1                          |
| Output shape | (batch, 32, 32, 3)         |

**Weight dimensions**

| Tensor   | Shape              | Parameters       |
|----------|-------------------|------------------|
| Weights  | (3, 3, 3, 256)    | 6,912            |
| Biases   | (3)               | 3                |
| Total    |                   | 6,915            |

**Activation**: None (linear output)

---

## Parameter Summary

| Layer                  | Output Shape        | Parameters |
|------------------------|---------------------|------------|
| Input                  | (32, 32, 3)         | 0          |
| Conv1                  | (32, 32, 256)       | 7,168      |
| MaxPool1               | (16, 16, 256)       | 0          |
| Conv2                  | (16, 16, 128)       | 295,040    |
| MaxPool2 (Latent)      | (8, 8, 128)         | 0          |
| Conv3                  | (8, 8, 128)         | 147,584    |
| Upsample1              | (16, 16, 128)       | 0          |
| Conv4                  | (16, 16, 256)       | 295,168    |
| Upsample2              | (32, 32, 256)       | 0          |
| Conv5                  | (32, 32, 3)         | 6,915      |
| **Total**              |                     | **751,875** |

---

## Weight Initialization

Weights are initialized using He initialization (appropriate for ReLU activations):

```cpp
std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fanIn));
```

Where `fanIn = inputChannels * kernelHeight * kernelWidth`.

Biases are initialized to zero.

---

## Loss Function

**Mean Squared Error (MSE)**

```
Loss = (1/N) * sum((prediction - target)^2)
```

Where N is the total number of pixels (batch_size *32* 32 * 3).

**Gradient**

```
dLoss/dPrediction = (2/N) * (prediction - target)
```

---

## Training Hyperparameters

| Parameter      | Default Value |
|----------------|---------------|
| Learning rate  | 0.001         |
| Batch size     | 64 (GPU)      |
| Epochs         | 20            |
| Optimizer      | SGD           |

---

## Memory Requirements

### Per-Image Memory

| Buffer                | Size (bytes)        |
|-----------------------|---------------------|
| Input                 | 3,072 * 4 = 12 KB   |
| Output                | 3,072 * 4 = 12 KB   |
| Encoder activations   | ~2.1 MB             |
| Decoder activations   | ~2.4 MB             |
| Gradients             | ~4.5 MB             |

### Total GPU Memory (batch 64)

| Component             | Size          |
|-----------------------|---------------|
| Weights               | ~3 MB         |
| Weight gradients      | ~3 MB         |
| Activations           | ~150 MB       |
| Gradient buffers      | ~290 MB       |
| Intermediate buffers  | ~100 MB       |
| im2col workspace (V4) | ~75 MB        |
| **Total**             | **~625 MB**   |

---

## Receptive Field Analysis

| Layer    | Receptive Field | Notes                      |
|----------|-----------------|----------------------------|
| Conv1    | 3 x 3           | Captures local edges       |
| Pool1    | 6 x 6           | 2x downsampling            |
| Conv2    | 10 x 10         | Larger patterns            |
| Pool2    | 14 x 14         | At latent space            |
| Conv3    | 14 x 14         | Same as input              |
| Up1      | 14 x 14         | Maintains receptive field  |
| Conv4    | 18 x 18         | Expands coverage           |
| Up2      | 18 x 18         | Maintains receptive field  |
| Conv5    | 22 x 22         | Nearly full image          |

### Memory Layout

**V1/V2**: NHWC (batch, height, width, channels) - optimized for channel-wise coalescing.

**V3/V4**: NCHW (batch, channels, height, width) - optimized for spatial operations and cuDNN compatibility.

The receptive field at the output covers most of the 32x32 input, allowing the network to capture global context.
