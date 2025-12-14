#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_utils.h"
#include "layout_convert.h"

// NHWC to NCHW conversion kernel
// Input: (batch, H, W, C) -> Output: (batch, C, H, W)
__global__ void nhwcToNchwKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int height, int width, int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * height * width * channels;
    
    if (idx >= total) return;
    
    // Decode NHWC index
    int c = idx % channels;
    int w = (idx / channels) % width;
    int h = (idx / (channels * width)) % height;
    int n = idx / (channels * width * height);
    
    // Compute NCHW index
    int nchw_idx = ((n * channels + c) * height + h) * width + w;
    
    output[nchw_idx] = input[idx];
}

// NCHW to NHWC conversion kernel  
// Input: (batch, C, H, W) -> Output: (batch, H, W, C)
__global__ void nchwToNhwcKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    
    if (idx >= total) return;
    
    // Decode NCHW index
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int n = idx / (width * height * channels);
    
    // Compute NHWC index
    int nhwc_idx = ((n * height + h) * width + w) * channels + c;
    
    output[nhwc_idx] = input[idx];
}

// Host wrapper: NHWC to NCHW conversion
void launchNhwcToNchw(
    const float* d_nhwc, float* d_nchw,
    int batch, int height, int width, int channels
) {
    int total = batch * height * width * channels;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    nhwcToNchwKernel<<<gridSize, blockSize>>>(
        d_nhwc, d_nchw, batch, height, width, channels
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW to NHWC conversion
void launchNchwToNhwc(
    const float* d_nchw, float* d_nhwc,
    int batch, int channels, int height, int width
) {
    int total = batch * channels * height * width;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    nchwToNhwcKernel<<<gridSize, blockSize>>>(
        d_nchw, d_nhwc, batch, channels, height, width
    );
    CHECK_CUDA(cudaGetLastError());
}
