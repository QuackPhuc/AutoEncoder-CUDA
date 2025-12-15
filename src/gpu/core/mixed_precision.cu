#include "mixed_precision.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================
// Tensor Core Detection
// ============================================================

bool deviceSupportsTensorCores() {
    int device;
    cudaGetDevice(&device);
    
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    
    // Tensor Cores available on SM >= 7.0 (Volta, Turing, Ampere, Ada, Hopper)
    return (major > 7) || (major == 7 && minor >= 0);
}

// ============================================================
// FP32 <-> FP16 Conversion Kernels
// ============================================================

__global__ void fp32ToFp16Kernel(const float* __restrict__ fp32, 
                                  half* __restrict__ fp16, 
                                  int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        fp16[idx] = __float2half(fp32[idx]);
    }
}

__global__ void fp16ToFp32Kernel(const half* __restrict__ fp16, 
                                  float* __restrict__ fp32, 
                                  int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        fp32[idx] = __half2float(fp16[idx]);
    }
}

void launchFP32ToFP16(const float* fp32, half* fp16, int count, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    
    if (stream) {
        fp32ToFp16Kernel<<<gridSize, blockSize, 0, stream>>>(fp32, fp16, count);
    } else {
        fp32ToFp16Kernel<<<gridSize, blockSize>>>(fp32, fp16, count);
    }
    CHECK_CUDA(cudaGetLastError());
}

void launchFP16ToFP32(const half* fp16, float* fp32, int count, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    
    if (stream) {
        fp16ToFp32Kernel<<<gridSize, blockSize, 0, stream>>>(fp16, fp32, count);
    } else {
        fp16ToFp32Kernel<<<gridSize, blockSize>>>(fp16, fp32, count);
    }
    CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Fused im2col with FP16 conversion
// ============================================================

__global__ void im2colFP16Kernel(
    const float* __restrict__ input,
    half* __restrict__ col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int k, int stride, int padding
) {
    int col_h = inC * k * k;
    int col_w = outH * outW;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * col_h * col_w;
    
    if (idx >= total) return;
    
    // Decode indices
    int w_col = idx % col_w;
    int h_col = (idx / col_w) % col_h;
    int n = idx / (col_w * col_h);
    
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
    
    // Convert to FP16 and store
    col[idx] = __float2half(val);
}

void launchIm2colFP16(
    const float* d_input_fp32,
    half* d_col_fp16,
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
    
    if (stream) {
        im2colFP16Kernel<<<gridSize, blockSize, 0, stream>>>(
            d_input_fp32, d_col_fp16,
            batch, inC, inH, inW, outH, outW,
            kernelSize, stride, padding
        );
    } else {
        im2colFP16Kernel<<<gridSize, blockSize>>>(
            d_input_fp32, d_col_fp16,
            batch, inC, inH, inW, outH, outW,
            kernelSize, stride, padding
        );
    }
    CHECK_CUDA(cudaGetLastError());
}
