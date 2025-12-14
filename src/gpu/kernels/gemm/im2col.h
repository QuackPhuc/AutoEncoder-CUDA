#ifndef GPU_GEMM_IM2COL_H
#define GPU_GEMM_IM2COL_H

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

#endif // GPU_GEMM_IM2COL_H
