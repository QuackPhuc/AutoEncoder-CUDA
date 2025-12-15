#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive MaxPool2D forward kernel (2x2 pooling, stride 2)
__global__ void maxpool2dForwardKernel(
    const float* input,    // (batch, inH, inW, channels)
    float* output,         // (batch, outH, outW, channels)
    int* indices,          // (batch, outH, outW, channels)
    int batch, int inH, int inW, int channels
) {
    int outH = inH / 2;
    int outW = inW / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = batch * outH * outW * channels;
    
    if (idx >= totalThreads) return;
    
    // Decode index (NHWC)
    int c = idx % channels;
    int w = (idx / channels) % outW;
    int h = (idx / (channels * outW)) % outH;
    int n = idx / (channels * outW * outH);
    
    // Find max in 2x2 window
    float maxVal = -1e38f;
    int maxIdx = 0;
    
    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int in_h = h * 2 + kh;
            int in_w = w * 2 + kw;
            int inputIdx = ((n * inH + in_h) * inW + in_w) * channels + c;
            float val = input[inputIdx];
            if (val > maxVal) {
                maxVal = val;
                maxIdx = kh * 2 + kw;
            }
        }
    }
    
    output[idx] = maxVal;
    indices[idx] = maxIdx;
}


// Naive Upsample2D forward kernel (nearest neighbor, 2x scale)
__global__ void upsample2dForwardKernel(
    const float* input,    // (batch, inH, inW, channels)
    float* output,         // (batch, outH, outW, channels)
    int batch, int inH, int inW, int channels
) {
    int outH = inH * 2;
    int outW = inW * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = batch * outH * outW * channels;
    
    if (idx >= totalThreads) return;
    
    // Decode index
    int c = idx % channels;
    int w = (idx / channels) % outW;
    int h = (idx / (channels * outW)) % outH;
    int n = idx / (channels * outW * outH);
    
    // Nearest neighbor
    int in_h = h / 2;
    int in_w = w / 2;
    int inputIdx = ((n * inH + in_h) * inW + in_w) * channels + c;
    
    output[idx] = input[inputIdx];
}

// NCHW MaxPool2D kernel with 2D grid indexing (2x2 pooling, stride 2)
__global__ void maxpool2dForwardNCHWKernel(
    const float* __restrict__ input,   // (batch, channels, inH, inW)
    float* __restrict__ output,        // (batch, channels, outH, outW)
    int* __restrict__ indices,         // (batch, channels, outH, outW)
    int batch, int channels, int inH, int inW,
    int outH, int outW, int k, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= outW || oh >= outH || n >= batch) return;
    
    const size_t in_base = ((static_cast<size_t>(n) * channels + c) * inH) * inW;
    const int ih_base = oh * stride;
    const int iw_base = ow * stride;
    
    float max_val = -1e38f;
    int max_idx = 0;
    
    // Unrolled 2x2 pooling (most common case)
    if (k == 2 && stride == 2) {
        const size_t idx00 = in_base + ih_base * inW + iw_base;
        const size_t idx01 = idx00 + 1;
        const size_t idx10 = idx00 + inW;
        const size_t idx11 = idx10 + 1;
        
        float v00 = input[idx00];
        float v01 = input[idx01];
        float v10 = input[idx10];
        float v11 = input[idx11];
        
        max_val = v00; max_idx = 0;
        if (v01 > max_val) { max_val = v01; max_idx = 1; }
        if (v10 > max_val) { max_val = v10; max_idx = 2; }
        if (v11 > max_val) { max_val = v11; max_idx = 3; }
    } else {
        #pragma unroll 4
        for (int kh = 0; kh < k; ++kh) {
            #pragma unroll 4
            for (int kw = 0; kw < k; ++kw) {
                int ih = ih_base + kh;
                int iw = iw_base + kw;
                
                if (ih < inH && iw < inW) {
                    size_t in_idx = in_base + ih * inW + iw;
                    float val = input[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = kh * k + kw;
                    }
                }
            }
        }
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * outH + oh) * outW + ow;
    output[out_idx] = max_val;
    indices[out_idx] = max_idx;
}

// NCHW Upsample2D kernel with 2D grid indexing (nearest neighbor, 2x scale)
__global__ void upsample2dForwardNCHWKernel(
    const float* __restrict__ input,   // (batch, channels, inH, inW)
    float* __restrict__ output,        // (batch, channels, outH, outW)
    int batch, int channels, int inH, int inW,
    int outH, int outW, int scale
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= outW || oh >= outH || n >= batch) return;
    
    int ih = oh / scale;
    int iw = ow / scale;
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * inH + ih) * inW + iw;
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * outH + oh) * outW + ow;
    output[out_idx] = input[in_idx];
}

// Host wrapper: NCHW MaxPool2D forward (GPU Opt V3)
void launchMaxPool2dNCHW(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW,
    int k, int stride
) {
    int outH = (inH - k) / stride + 1;
    int outW = (inW - k) / stride + 1;
    
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * channels
    );
    
    maxpool2dForwardNCHWKernel<<<grid, block>>>(
        d_input, d_output, d_indices,
        batch, channels, inH, inW, outH, outW, k, stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Upsample2D forward (GPU Opt V3)
void launchUpsample2dNCHW(
    const float* d_input, float* d_output,
    int batch, int channels, int inH, int inW,
    int scale
) {
    int outH = inH * scale;
    int outW = inW * scale;
    
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * channels
    );
    
    upsample2dForwardNCHWKernel<<<grid, block>>>(
        d_input, d_output,
        batch, channels, inH, inW, outH, outW, scale
    );
    CHECK_CUDA(cudaGetLastError());
}

// ============================================================
// Stream-aware versions for GPU Opt V3
// ============================================================

// Stream-aware NCHW MaxPool2D forward
void launchMaxPool2dNCHWStream(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW,
    int k, int stride,
    cudaStream_t stream
) {
    int outH = (inH - k) / stride + 1;
    int outW = (inW - k) / stride + 1;
    
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * channels
    );
    
    maxpool2dForwardNCHWKernel<<<grid, block, 0, stream>>>(
        d_input, d_output, d_indices,
        batch, channels, inH, inW, outH, outW, k, stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Stream-aware NCHW Upsample2D forward
void launchUpsample2dNCHWStream(
    const float* d_input, float* d_output,
    int batch, int channels, int inH, int inW,
    int scale,
    cudaStream_t stream
) {
    int outH = inH * scale;
    int outW = inW * scale;
    
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * channels
    );
    
    upsample2dForwardNCHWKernel<<<grid, block, 0, stream>>>(
        d_input, d_output,
        batch, channels, inH, inW, outH, outW, scale
    );
    CHECK_CUDA(cudaGetLastError());
}

