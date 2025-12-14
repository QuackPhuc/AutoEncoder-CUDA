#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"

// im2col kernel for NCHW format
__global__ void im2colNCHWKernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int k, int stride, int padding
) {
    // Each thread handles one element in the col matrix
    int col_h = inC * k * k;        // Height of col matrix (per batch)
    int col_w = outH * outW;        // Width of col matrix
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * col_h * col_w;
    
    if (idx >= total) return;
    
    // Decode indices
    int w_col = idx % col_w;        // Column index in col matrix
    int h_col = (idx / col_w) % col_h;  // Row index in col matrix  
    int n = idx / (col_w * col_h);  // Batch index
    
    // Decode h_col to (ic, kh, kw)
    int kw = h_col % k;
    int kh = (h_col / k) % k;
    int ic = h_col / (k * k);
    
    // Decode w_col to (oh, ow)
    int ow = w_col % outW;
    int oh = w_col / outW;
    
    // Calculate input position
    int iw = ow * stride + kw - padding;
    int ih = oh * stride + kh - padding;
    
    float val = 0.0f;
    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
        size_t in_idx = ((static_cast<size_t>(n) * inC + ic) * inH + ih) * inW + iw;
        val = input[in_idx];
    }
    
    // Write to col matrix: [n, h_col, w_col]
    size_t col_idx = (static_cast<size_t>(n) * col_h + h_col) * col_w + w_col;
    col[col_idx] = val;
}

// col2im kernel for NCHW backward pass
__global__ void col2imNCHWKernel(
    const float* __restrict__ col,
    float* __restrict__ gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int k, int stride, int padding
) {
    // Each thread handles one input pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * inC * inH * inW;
    
    if (idx >= total) return;
    
    // Decode to (n, ic, ih, iw)
    int iw = idx % inW;
    int ih = (idx / inW) % inH;
    int ic = (idx / (inW * inH)) % inC;
    int n = idx / (inW * inH * inC);
    
    float sum = 0.0f;
    int col_h = inC * k * k;
    int col_w = outH * outW;
    
    // Iterate over all kernel positions that could have used this input pixel
    for (int kh = 0; kh < k; ++kh) {
        for (int kw = 0; kw < k; ++kw) {
            // Output position that would have used (ih, iw) at (kh, kw)
            int oh_check = ih + padding - kh;
            int ow_check = iw + padding - kw;
            
            if (oh_check % stride == 0 && ow_check % stride == 0) {
                int oh = oh_check / stride;
                int ow = ow_check / stride;
                
                if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                    // Position in col matrix
                    int h_col = ic * k * k + kh * k + kw;
                    int w_col = oh * outW + ow;
                    
                    size_t col_idx = (static_cast<size_t>(n) * col_h + h_col) * col_w + w_col;
                    sum += col[col_idx];
                }
            }
        }
    }
    
    gradInput[idx] = sum;
}

// Host wrapper: im2col for forward pass
void launchIm2colNCHW(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding
) {
    int col_h = inC * kernelSize * kernelSize;
    int col_w = outH * outW;
    int total = batch * col_h * col_w;
    
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    im2colNCHWKernel<<<gridSize, blockSize>>>(
        d_input, d_col,
        batch, inC, inH, inW, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: col2im for backward pass
void launchCol2imNCHW(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding
) {
    int total = batch * inC * inH * inW;
    
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    col2imNCHWKernel<<<gridSize, blockSize>>>(
        d_col, d_gradInput,
        batch, inC, inH, inW, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Stream-aware versions for GPU Opt V3
// ============================================================

// Stream-aware im2col
void launchIm2colNCHWStream(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream
) {
    int col_h = inC * kernelSize * kernelSize;
    int col_w = outH * outW;
    int total = batch * col_h * col_w;
    
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    im2colNCHWKernel<<<gridSize, blockSize, 0, stream>>>(
        d_input, d_col,
        batch, inC, inH, inW, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Stream-aware col2im
void launchCol2imNCHWStream(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream
) {
    int total = batch * inC * inH * inW;
    
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    col2imNCHWKernel<<<gridSize, blockSize, 0, stream>>>(
        d_col, d_gradInput,
        batch, inC, inH, inW, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

