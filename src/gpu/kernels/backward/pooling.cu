#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// MaxPool2D backward - zero initialize
__global__ void maxpool2dBackwardZeroKernel(
    float* gradInput,
    int batch, int inH, int inW, int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInputs = batch * inH * inW * channels;
    
    if (idx >= totalInputs) return;
    gradInput[idx] = 0.0f;
}

// MaxPool2D backward - scatter gradients
__global__ void maxpool2dBackwardScatterKernel(
    const float* gradOutput,  // (batch, outH, outW, channels)
    const int* indices,       // (batch, outH, outW, channels)
    float* gradInput,         // (batch, inH, inW, channels)
    int batch, int inH, int inW, int channels
) {
    int outH = inH / 2;
    int outW = inW / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = batch * outH * outW * channels;
    
    if (idx >= totalOutputs) return;
    
    int c = idx % channels;
    int w = (idx / channels) % outW;
    int h = (idx / (channels * outW)) % outH;
    int n = idx / (channels * outW * outH);
    
    int maxIdx = indices[idx];
    int kh = maxIdx / 2;
    int kw = maxIdx % 2;
    
    int in_h = h * 2 + kh;
    int in_w = w * 2 + kw;
    int inputIdx = ((n * inH + in_h) * inW + in_w) * channels + c;
    
    atomicAdd(&gradInput[inputIdx], gradOutput[idx]);
}

// Upsample2D backward (sum 2x2 gradients)
__global__ void upsample2dBackwardKernel(
    const float* gradOutput,  // (batch, outH, outW, channels)
    float* gradInput,         // (batch, inH, inW, channels)
    int batch, int inH, int inW, int channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalInputs = batch * inH * inW * channels;
    
    if (idx >= totalInputs) return;
    
    int c = idx % channels;
    int w = (idx / channels) % inW;
    int h = (idx / (channels * inW)) % inH;
    int n = idx / (channels * inW * inH);
    
    int outH = inH * 2;
    int outW = inW * 2;
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int out_h = h * 2 + kh;
            int out_w = w * 2 + kw;
            int outputIdx = ((n * outH + out_h) * outW + out_w) * channels + c;
            sum += gradOutput[outputIdx];
        }
    }
    
    gradInput[idx] = sum;
}


// NCHW MaxPool2D backward with 2D grid indexing
__global__ void maxpool2dBackwardNCHWKernel(
    const float* __restrict__ gradOutput,  // (batch, channels, outH, outW)
    const int* __restrict__ indices,       // (batch, channels, outH, outW) 
    float* __restrict__ gradInput,         // (batch, channels, inH, inW)
    int batch, int channels, int inH, int inW,
    int outH, int outW, int k, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= outW || oh >= outH || n >= batch) return;
    
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * outH + oh) * outW + ow;
    float grad = gradOutput[out_idx];
    int maxIdx = indices[out_idx];
    
    int kh = maxIdx / k;
    int kw = maxIdx % k;
    int ih = oh * stride + kh;
    int iw = ow * stride + kw;
    
    if (ih < inH && iw < inW) {
        size_t in_idx = ((static_cast<size_t>(n) * channels + c) * inH + ih) * inW + iw;
        atomicAdd(&gradInput[in_idx], grad);
    }
}

// NCHW Upsample2D backward with 2D grid indexing
__global__ void upsample2dBackwardNCHWKernel(
    const float* __restrict__ gradOutput,  // (batch, channels, outH, outW)
    float* __restrict__ gradInput,         // (batch, channels, inH, inW)
    int batch, int channels, int inH, int inW,
    int outH, int outW, int scale
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (iw >= inW || ih >= inH || n >= batch) return;
    
    float sum = 0.0f;
    int base_oh = ih * scale;
    int base_ow = iw * scale;
    
    #pragma unroll
    for (int kh = 0; kh < scale; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < scale; ++kw) {
            int oh = base_oh + kh;
            int ow = base_ow + kw;
            size_t out_idx = ((static_cast<size_t>(n) * channels + c) * outH + oh) * outW + ow;
            sum += gradOutput[out_idx];
        }
    }
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * inH + ih) * inW + iw;
    gradInput[in_idx] = sum;
}

// Host wrapper: NCHW MaxPool2D backward (GPU Opt V3)
void launchMaxPool2dBackwardNCHW(
    const float* d_gradOutput, const int* d_indices, float* d_gradInput,
    int batch, int channels, int inH, int inW,
    int k, int stride
) {
    int outH = (inH - k) / stride + 1;
    int outW = (inW - k) / stride + 1;
    
    // First zero-initialize gradInput
    size_t inputSize = static_cast<size_t>(batch) * channels * inH * inW;
    CHECK_CUDA(cudaMemset(d_gradInput, 0, inputSize * sizeof(float)));
    
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * channels
    );
    
    maxpool2dBackwardNCHWKernel<<<grid, block>>>(
        d_gradOutput, d_indices, d_gradInput,
        batch, channels, inH, inW, outH, outW, k, stride
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Upsample2D backward (GPU Opt V3)
void launchUpsample2dBackwardNCHW(
    const float* d_gradOutput, float* d_gradInput,
    int batch, int channels, int inH, int inW,
    int scale
) {
    int outH = inH * scale;
    int outW = inW * scale;
    
    dim3 block(16, 16);
    dim3 grid(
        (inW + block.x - 1) / block.x,
        (inH + block.y - 1) / block.y,
        batch * channels
    );
    
    upsample2dBackwardNCHWKernel<<<grid, block>>>(
        d_gradOutput, d_gradInput,
        batch, channels, inH, inW, outH, outW, scale
    );
    CHECK_CUDA(cudaGetLastError());
}

