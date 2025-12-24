# CUDA Optimization Guide

In-depth documentation of the GPU optimization techniques implemented in this project.

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 2: Naive GPU Implementation](#phase-2-naive-gpu-implementation)
3. [Phase 3 Opt v1: NCHW Layout + Warp Shuffle](#phase-3-opt-v1-nchw-layout--warp-shuffle)
4. [Phase 3 Opt v2: im2col + cuBLAS GEMM](#phase-3-opt-v2-im2col--cublas-gemm)
5. [Performance Analysis](#performance-analysis)
6. [Profiling Guide](#profiling-guide)
7. [How to Measure Metrics (Colab)](#how-to-measure-metrics-colab)
8. [Optimization Checklist](#optimization-checklist)

---

## Overview

The project implements GPU optimization levels, each building on the previous:

| Version  | Key Techniques                           | Time/Epoch (s) | Speedup vs CPU |
|----------|------------------------------------------|----------------|----------------|
| CPU      | Sequential baseline                      | 169.18*        | 1.0x           |
| Naive    | Per-pixel parallelization                | 500.57         | 169x           |
| Opt v1   | NCHW layout, 2D grid, warp shuffle       | 247.15         | 342x           |
| Opt v2   | im2col + cuBLAS GEMM                     | 50.52          | 1690x           |

> [!NOTE]
> *CPU baseline measured on 100 samples only (~23.5 hours/epoch estimated for full 50,000 samples). GPU values are per-epoch averages from 50,000 samples, 3 epochs on T4.

---

## Phase 2: Naive GPU Implementation

### Design Principles

The naive implementation provides correctness first, performance second. Each operation maps directly to a CUDA kernel with straightforward parallelization.

### Thread Mapping

Each thread computes one output element:

```cpp
__global__ void conv2dForwardNaiveKernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = batch * outH * outW * outC;
    
    if (idx >= totalThreads) return;
    
    // Decode index (NHWC layout)
    int c = idx % outC;
    int w = (idx / outC) % outW;
    int h = (idx / (outC * outW)) % outH;
    int n = idx / (outC * outW * outH);
    
    // Compute convolution
    float sum = 0.0f;
    for (int ic = 0; ic < inC; ic++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                // Access global memory for each element
                sum += input[...] * weights[...];
            }
        }
    }
    
    sum += bias[c];
    output[idx] = sum;
}
```

### Launch Configuration

```cpp
int totalThreads = batch * outH * outW * outC;
int blockSize = 256;
int gridSize = (totalThreads + blockSize - 1) / blockSize;

kernel<<<gridSize, blockSize>>>(...);
```

### Performance Bottlenecks

| Issue                     | Impact                          |
|---------------------------|--------------------------------|
| Global memory accesses    | High latency (400-800 cycles)  |
| No data reuse             | Same input loaded multiple times |
| Memory coalescing issues  | Non-sequential access patterns |
| Low arithmetic intensity  | Memory-bound computation       |

---

## Phase 3 Opt v1: NCHW Layout + Warp Shuffle

### Memory Layout Change

V1 switches from NHWC to NCHW memory layout:

| Layout | Order | Description |
|--------|-------|-------------|
| NHWC | (batch, height, width, channels) | Channel-contiguous |
| NCHW | (batch, channels, height, width) | Spatial-contiguous |

**Benefits of NCHW for convolution**:

- Spatial locality for 2D operations
- Better for 2D grid indexing
- Aligned with cuDNN conventions

### 2D Grid Indexing

2D thread blocks map naturally to spatial output:

```cpp
dim3 block(16, 16);  // 256 threads per block
dim3 grid(
    (outW + block.x - 1) / block.x,
    (outH + block.y - 1) / block.y,
    batch * outC
);
```

Each thread computes one output pixel (oh, ow) for one (batch, channel) pair.

### 3x3 Kernel Special Casing

Compile-time unrolling for known 3x3 kernels:

```cpp
if (k == 3) {
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            int ih = ih_base + kh;
            int iw = iw_base + kw;
            if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                sum += input[...] * weights[...];
            }
        }
    }
}
```

### Warp Shuffle Reduction

Fast bias gradient reduction using warp primitives:

```cpp
// Warp shuffle instead of shared memory reduction
for (int offset = 16; offset > 0; offset /= 2) {
    myVal += __shfl_down_sync(0xffffffff, myVal, offset);
}
if (tid == 0) {
    gradBias[oc] = myVal;
}
```

**Understanding `0xffffffff` (Warp Mask)**:

The first parameter of `__shfl_down_sync` is a **32-bit thread mask** that specifies which threads participate in the shuffle:

| Bit Pattern | Meaning |
|-------------|---------|
| `0xffffffff` | All 32 bits = 1 → All 32 threads in the warp participate |
| `0x0000ffff` | Lower 16 bits = 1 → Only threads 0-15 participate |
| `0x00000001` | Only bit 0 = 1 → Only thread 0 participates |

Since a CUDA warp contains exactly 32 threads (indexed 0-31), `0xffffffff` (all 32 bits set to 1) indicates **full warp participation**. This is required for correctness: if some threads don't participate but are expected to contribute values, the result will be undefined.

**Benefits**:

- No shared memory required
- Lower latency than shared memory
- Full warp participation

---

## Phase 3 Opt v2: im2col + cuBLAS GEMM

### im2col Concept

Converts convolution to matrix multiplication:

```
Input:   [batch, inC, inH, inW]
            ↓ im2col
Col:     [batch, inC*k*k, outH*outW]
            × Weights: [outC, inC*k*k]
            ↓ cuBLAS SGEMM
Output:  [batch, outC, outH*outW]
```

### im2col Kernel

Each thread extracts one patch element:

```cpp
__global__ void im2colNCHWKernel(
    const float* input,      // [batch, inC, inH, inW]
    float* col,              // [batch, inC*k*k, outH*outW]
    int batch, int inC, int inH, int inW,
    int outH, int outW, int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Decode to (batch, ic, kh, kw, oh, ow)
    int w_col = idx % (outH * outW);
    int h_col = (idx / (outH * outW)) % (inC * k * k);
    int n = idx / ((outH * outW) * (inC * k * k));
    
    // Map to input position
    int iw = (w_col % outW) * stride + (h_col % k) - padding;
    int ih = (w_col / outW) * stride + ((h_col / k) % k) - padding;
    int ic = h_col / (k * k);
    
    float val = (ih >= 0 && ih < inH && iw >= 0 && iw < inW) 
                ? input[n * inC * inH * inW + ic * inH * inW + ih * inW + iw]
                : 0.0f;
    col[idx] = val;
}
```

### cuBLAS SGEMM Integration

Uses cuBLAS for optimized matrix multiplication:

```cpp
void launchConvGemmForward(
    const float* d_weights,   // [outC, inC*k*k]
    const float* d_im2col,    // [batch, inC*k*k, outHW]
    const float* d_bias,
    float* d_output,          // [batch, outC, outHW]
    int batch, int outC, int inC_k_k, int outHW,
    bool applyRelu
) {
    cublasHandle_t handle = CublasHandle::get();
    
    for (int n = 0; n < batch; ++n) {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            outHW, outC, inC_k_k,     // M, N, K
            &alpha,
            im2col_n, outHW,           // B
            d_weights, inC_k_k,        // A
            &beta,
            output_n, outHW            // C
        );
    }
    
    // Add bias + optional ReLU
    launchAddBias(d_output, d_bias, batch, outC, outHW, applyRelu);
}
```

### col2im Backward Pass

Gradient accumulation from column matrix back to input:

```cpp
// For each input pixel
for (int kh = 0; kh < k; ++kh) {
    for (int kw = 0; kw < k; ++kw) {
        // Find all output positions that used this input
        // Accumulate gradients from col matrix
        sum += col[col_idx];
    }
}
gradInput[idx] = sum;
```

### Memory Workspace

Opt v2 requires additional workspace for im2col matrix:

```cpp
// Workspace allocation
size_t workspace_size = batch * maxChannels * 9 * maxSpatial * sizeof(float);
cudaMalloc(&d_im2col_workspace, workspace_size);
```

Typical workspace size depends on batch size. Measure on your hardware.

---

## Performance Analysis

> [!IMPORTANT]
> The values below are **placeholders**. You must measure these on your own hardware using the [measurement guide](#how-to-measure-metrics-colab).

### Memory Traffic (Per Forward Pass)

**Configuration: batch size 64, layer 1: 32x32x3 → 32x32x256**

| Version | Read (MB)   | Write (MB) | Total (MB) | Notes |
|---------|-------------|------------|------------|-------|
| Naive   | `<TBD>`     | `<TBD>`    | `<TBD>`    | NHWC layout |
| Opt v1  | `<TBD>`     | `<TBD>`    | `<TBD>`    | NCHW layout |
| Opt v2  | `<TBD>`     | `<TBD>`    | `<TBD>`    | im2col expansion |

### Achieved Occupancy

| Version | Theoretical Occupancy | Achieved Occupancy |
|---------|----------------------|-------------------|
| Naive   | `<TBD>`              | `<TBD>`           |
| Opt v1  | `<TBD>`              | `<TBD>`           |
| Opt v2  | `<TBD>`              | `<TBD>`           |

### Kernel Execution Time

**GPU: `<YOUR_GPU>`, batch size: 64**

| Layer    | Naive (ms) | Opt v1 (ms) | Opt v2 (ms) |
|----------|------------|-------------|-------------|
| Conv1    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Pool1    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Conv2    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Pool2    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Conv3    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Up1      | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Conv4    | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Up2      | `<TBD>`    | `<TBD>`     | `<TBD>`     |
| Conv5    | `<TBD>`    | `<TBD>`     | `<TBD>`     |

---

## Profiling Guide

### Using NVIDIA Nsight Compute

```bash
# Full kernel analysis
ncu --set full ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1 --samples 1000

# Memory throughput analysis
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1
```

### Using NVIDIA Nsight Systems

```bash
# Timeline profiling
nsys profile --stats=true \
    ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 5
```

### Key Metrics to Monitor

| Metric                        | Target     | Indication                    |
|-------------------------------|------------|-------------------------------|
| Achieved Occupancy            | > 80%      | Sufficient parallelism        |
| Memory Throughput             | > 70%      | Efficient memory access       |
| SM Efficiency                 | > 90%      | Good workload distribution    |
| Warp Execution Efficiency     | > 90%      | Minimal divergence            |

### Identifying Bottlenecks

**Memory-bound kernels**

- High memory throughput but low compute throughput
- Solution: Increase arithmetic intensity, use shared memory

**Compute-bound kernels**

- High compute throughput but low memory throughput
- Solution: Increase parallelism, reduce register pressure

**Latency-bound kernels**

- Low both compute and memory throughput
- Solution: Increase occupancy, hide latency with more threads

---

## How to Measure Metrics (Colab)

This section provides step-by-step instructions to measure performance metrics on Google Colab.

### Prerequisites

1. Use a Colab notebook with GPU runtime (Runtime → Change runtime type → T4 GPU)
2. Clone and build the project

### Step 1: Setup Environment

```python
# In a Colab cell
!nvidia-smi  # Check GPU type

# Clone repo (adjust URL as needed)
!git clone https://github.com/<your-repo>/AutoEncoder-CUDA.git
%cd AutoEncoder-CUDA

# Install dependencies
!apt-get update && apt-get install -y cmake

# Build
!chmod +x build.sh
!./build.sh
```

### Step 2: Measure Speedup vs CPU

```python
import subprocess
import re
import time

def run_and_get_time(cmd):
    """Run command and extract training time from output"""
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    return elapsed, result.stdout + result.stderr

# CPU baseline
cpu_time, cpu_out = run_and_get_time(
    "./build/bin/autoencoder_cpu --epochs 1 --samples 5000"
)
print(f"CPU Time: {cpu_time:.2f}s")

# GPU versions
for version in [1, 2, 3, 4]:
    gpu_time, gpu_out = run_and_get_time(
        f"./build/bin/autoencoder_gpu --gpu-version {version} --epochs 1 --samples 5000"
    )
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"GPU V{version}: {gpu_time:.2f}s (Speedup: {speedup:.1f}x)")
```

### Step 3: Measure Memory Traffic with nvprof

```python
# Note: nvprof is deprecated, use ncu on newer systems
# For Colab T4, nvprof may still work

!nvprof --metrics dram_read_throughput,dram_write_throughput \
    ./build/bin/autoencoder_gpu --gpu-version 1 --epochs 1 --samples 1000 2>&1 | head -50
```

### Step 4: Measure Occupancy

```python
# Using nvprof (Colab compatible)
!nvprof --metrics achieved_occupancy \
    ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1 --samples 1000 2>&1 | head -50
```

### Step 5: Measure Kernel Execution Time

```python
# Profile with nvprof to get kernel times
!nvprof --print-gpu-trace \
    ./build/bin/autoencoder_gpu --gpu-version 3 --epochs 1 --samples 1000 2>&1 | \
    grep -E "(conv|pool|relu|upsample)" | head -20
```

### Step 6: Calculate Memory Traffic Theoretically

For a convolution layer, you can estimate memory traffic:

```python
def calc_conv_memory_traffic(batch, in_c, in_h, in_w, out_c, k=3):
    """
    Calculate theoretical memory traffic for a conv layer
    
    Read: input tensor + weights + bias
    Write: output tensor
    """
    # Input read
    input_size = batch * in_c * in_h * in_w * 4  # float32 = 4 bytes
    
    # Weights read (once, may be cached)
    weights_size = out_c * in_c * k * k * 4
    
    # Bias read
    bias_size = out_c * 4
    
    # Output write
    out_h = in_h  # assuming stride=1, padding=1
    out_w = in_w
    output_size = batch * out_c * out_h * out_w * 4
    
    total_read = input_size + weights_size + bias_size
    total_write = output_size
    
    return {
        'read_mb': total_read / (1024 * 1024),
        'write_mb': total_write / (1024 * 1024),
        'total_mb': (total_read + total_write) / (1024 * 1024)
    }

# Example: Layer 1 (32x32x3 -> 32x32x256), batch=64
result = calc_conv_memory_traffic(64, 3, 32, 32, 256)
print(f"Read: {result['read_mb']:.2f} MB")
print(f"Write: {result['write_mb']:.2f} MB")
print(f"Total: {result['total_mb']:.2f} MB")
```

### Step 7: im2col Memory Overhead

```python
def calc_im2col_overhead(batch, in_c, in_h, in_w, k=3, stride=1, padding=1):
    """
    Calculate additional memory for im2col transformation
    """
    out_h = (in_h + 2*padding - k) // stride + 1
    out_w = (in_w + 2*padding - k) // stride + 1
    
    # im2col creates: [batch, in_c * k * k, out_h * out_w]
    col_size = batch * in_c * k * k * out_h * out_w * 4
    
    return col_size / (1024 * 1024)  # MB

overhead = calc_im2col_overhead(64, 3, 32, 32)
print(f"im2col workspace for layer 1: {overhead:.2f} MB")
```

### Complete Measurement Script

```python
#!/usr/bin/env python3
"""
Complete measurement script for AutoEncoder-CUDA performance metrics.
Run this in Google Colab after building the project.
"""

import subprocess
import re
import time

def measure_all():
    results = {}
    
    # 1. Measure CPU baseline
    print("Measuring CPU baseline...")
    start = time.time()
    subprocess.run(
        "./build/bin/autoencoder_cpu --epochs 1 --samples 5000",
        shell=True, capture_output=True
    )
    results['cpu_time'] = time.time() - start
    
    # 2. Measure GPU versions
    for v in [1, 2, 3, 4]:
        print(f"Measuring GPU V{v}...")
        start = time.time()
        subprocess.run(
            f"./build/bin/autoencoder_gpu --gpu-version {v} --epochs 1 --samples 5000",
            shell=True, capture_output=True
        )
        results[f'gpu_v{v}_time'] = time.time() - start
        results[f'gpu_v{v}_speedup'] = results['cpu_time'] / results[f'gpu_v{v}_time']
    
    # 3. Print results
    print("\n" + "="*50)
    print("PERFORMANCE RESULTS")
    print("="*50)
    print(f"CPU Time: {results['cpu_time']:.2f}s")
    for v in [1, 2, 3, 4]:
        print(f"GPU V{v}: {results[f'gpu_v{v}_time']:.2f}s (Speedup: {results[f'gpu_v{v}_speedup']:.1f}x)")
    
    return results

if __name__ == "__main__":
    measure_all()
```

---

## Optimization Checklist

### Before Optimization

- [ ] Verify correctness against CPU implementation
- [ ] Profile to identify bottlenecks
- [ ] Measure baseline performance

### Memory Optimizations

- [ ] Use NCHW layout for spatial operations (v1/v2)
- [ ] Implement shared memory tiling
- [ ] Use constant memory for broadcast data
- [ ] Consider vectorized loads (float4)

### Kernel Optimizations

- [ ] Fuse consecutive operations
- [ ] Unroll loops with known bounds
- [ ] Minimize register pressure
- [ ] Tune block dimensions for occupancy
- [ ] Consider im2col + GEMM for convolutions (v2)

### After Optimization

- [ ] Verify correctness again
- [ ] Profile to measure improvement
- [ ] Check for regressions
