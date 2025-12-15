#ifndef GPU_GEMM_IM2COL_H
#define GPU_GEMM_IM2COL_H

#include <cuda_runtime.h>

// im2col for forward pass: convert input patches to column matrix
// Input:  [batch, inC, inH, inW]
// Output: [batch, inC*k*k, outH*outW]
void launchIm2colNCHW(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

// col2im for backward pass: accumulate gradients back to input
// Input:  [batch, inC*k*k, outH*outW]
// Output: [batch, inC, inH, inW]
void launchCol2imNCHW(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

// ============================================================
// Stream-aware versions for GPU Opt V3
// ============================================================

// Stream-aware im2col for forward pass
void launchIm2colNCHWStream(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream);

// Stream-aware col2im for backward pass
void launchCol2imNCHWStream(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream);

#endif // GPU_GEMM_IM2COL_H
