#ifndef GPU_CORE_MIXED_PRECISION_H
#define GPU_CORE_MIXED_PRECISION_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Mixed Precision Utilities for Tensor Core Optimization
// ============================================================

// Check if device supports Tensor Cores (SM >= 7.0)
bool deviceSupportsTensorCores();

// Convert FP32 array to FP16
void launchFP32ToFP16(const float* fp32, half* fp16, int count, cudaStream_t stream = nullptr);

// Convert FP16 array to FP32
void launchFP16ToFP32(const half* fp16, float* fp32, int count, cudaStream_t stream = nullptr);

// Fused convert FP32 to FP16 + im2col (for input preparation)
void launchIm2colFP16(
    const float* d_input_fp32,
    half* d_col_fp16,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream = nullptr
);

#endif // GPU_CORE_MIXED_PRECISION_H
