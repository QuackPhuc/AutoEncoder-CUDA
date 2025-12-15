#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive Conv2D backward w.r.t. input
__global__ void conv2dBackwardInputKernel(
    const float* gradOutput,  // (batch, outH, outW, outC)
    const float* weights,     // (outC, kernelSize, kernelSize, inC)
    float* gradInput,         // (batch, inH, inW, inC)
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = batch * inH * inW * inC;
    
    if (idx >= totalThreads) return;
    
    int ic = idx % inC;
    int w = (idx / inC) % inW;
    int h = (idx / (inC * inW)) % inH;
    int n = idx / (inC * inW * inH);
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < outC; oc++) {
        for (int kh = 0; kh < kernelSize; kh++) {
            for (int kw = 0; kw < kernelSize; kw++) {
                int out_h = (h + padding - kh) / stride;
                int out_w = (w + padding - kw) / stride;
                
                if (out_h >= 0 && out_h < outH && out_w >= 0 && out_w < outW) {
                    if ((h + padding - kh) % stride == 0 && (w + padding - kw) % stride == 0) {
                        int gradOutIdx = ((n * outH + out_h) * outW + out_w) * outC + oc;
                        int weightIdx = (((oc * kernelSize + kh) * kernelSize + kw) * inC + ic);
                        sum += gradOutput[gradOutIdx] * weights[weightIdx];
                    }
                }
            }
        }
    }
    
    gradInput[idx] = sum;
}

// Naive Conv2D backward w.r.t. weights
__global__ void conv2dBackwardWeightsKernel(
    const float* gradOutput,  // (batch, outH, outW, outC)
    const float* input,       // (batch, inH, inW, inC)
    float* gradWeights,       // (outC, kernelSize, kernelSize, inC)
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = outC * kernelSize * kernelSize * inC;
    
    if (idx >= totalThreads) return;
    
    int ic = idx % inC;
    int kw = (idx / inC) % kernelSize;
    int kh = (idx / (inC * kernelSize)) % kernelSize;
    int oc = idx / (inC * kernelSize * kernelSize);
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch; n++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int in_h = oh * stride + kh - padding;
                int in_w = ow * stride + kw - padding;
                
                if (in_h >= 0 && in_h < inH && in_w >= 0 && in_w < inW) {
                    int inputIdx = ((n * inH + in_h) * inW + in_w) * inC + ic;
                    int gradOutIdx = ((n * outH + oh) * outW + ow) * outC + oc;
                    sum += input[inputIdx] * gradOutput[gradOutIdx];
                }
            }
        }
    }
    
    gradWeights[idx] = sum;
}

// Naive Conv2D backward w.r.t. bias
__global__ void conv2dBackwardBiasKernel(
    const float* gradOutput,  // (batch, outH, outW, outC)
    float* gradBias,          // (outC)
    int batch, int outH, int outW, int outC
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oc >= outC) return;
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch; n++) {
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                int gradOutIdx = ((n * outH + h) * outW + w) * outC + oc;
                sum += gradOutput[gradOutIdx];
            }
        }
    }
    
    gradBias[oc] = sum;
}

// NCHW Conv2D backward w.r.t. input with 2D grid indexing
__global__ void conv2dBackwardInputNCHWKernel(
    const float* __restrict__ gradOutput,  // (batch, outC, outH, outW)
    const float* __restrict__ weights,     // (outC, inC, k, k)
    float* __restrict__ gradInput,         // (batch, inC, inH, inW)
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int ic_n = blockIdx.z;
    int ic = ic_n % inC;
    int n = ic_n / inC;
    
    if (iw >= inW || ih >= inH || n >= batch) return;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < outC; ++oc) {
        if (k == 3) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    int oh_check = ih + padding - kh;
                    int ow_check = iw + padding - kw;
                    
                    if (oh_check % stride == 0 && ow_check % stride == 0) {
                        int oh = oh_check / stride;
                        int ow = ow_check / stride;
                        
                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                            size_t go_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
                            size_t w_idx = ((static_cast<size_t>(oc) * inC + ic) * k + kh) * k + kw;
                            sum += gradOutput[go_idx] * weights[w_idx];
                        }
                    }
                }
            }
        } else {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int oh_check = ih + padding - kh;
                    int ow_check = iw + padding - kw;
                    
                    if (oh_check % stride == 0 && ow_check % stride == 0) {
                        int oh = oh_check / stride;
                        int ow = ow_check / stride;
                        
                        if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                            size_t go_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
                            size_t w_idx = ((static_cast<size_t>(oc) * inC + ic) * k + kh) * k + kw;
                            sum += gradOutput[go_idx] * weights[w_idx];
                        }
                    }
                }
            }
        }
    }
    
    size_t gi_idx = ((static_cast<size_t>(n) * inC + ic) * inH + ih) * inW + iw;
    gradInput[gi_idx] = sum;
}

// NCHW Conv2D backward w.r.t. weights
__global__ void conv2dBackwardWeightsNCHWKernel(
    const float* __restrict__ input,       // (batch, inC, inH, inW)
    const float* __restrict__ gradOutput,  // (batch, outC, outH, outW)
    float* __restrict__ gradWeights,       // (outC, inC, k, k)
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWeights = outC * inC * k * k;
    
    if (idx >= totalWeights) return;
    
    int kw = idx % k;
    int temp = idx / k;
    int kh = temp % k;
    temp = temp / k;
    int ic = temp % inC;
    int oc = temp / inC;
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < outH; ++oh) {
            for (int ow = 0; ow < outW; ++ow) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    size_t in_idx = ((static_cast<size_t>(n) * inC + ic) * inH + ih) * inW + iw;
                    size_t go_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
                    sum += input[in_idx] * gradOutput[go_idx];
                }
            }
        }
    }
    
    gradWeights[idx] = sum;
}

// NCHW Conv2D backward w.r.t. bias with warp shuffle reduction
__global__ void conv2dBackwardBiasNCHWKernel(
    const float* __restrict__ gradOutput,  // (batch, outC, outH, outW)
    float* __restrict__ gradBias,          // (outC)
    int batch, int outC, int outH, int outW
) {
    int oc = blockIdx.x;
    if (oc >= outC) return;
    
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int spatialSize = outH * outW;
    int totalElements = batch * spatialSize;
    
    float localSum = 0.0f;
    for (int i = tid; i < totalElements; i += blockDim.x) {
        int n = i / spatialSize;
        int spatial = i % spatialSize;
        int oh = spatial / outW;
        int ow = spatial % outW;
        size_t idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
        localSum += gradOutput[idx];
    }
    
    sdata[tid] = localSum;
    __syncthreads();
    
    // Warp shuffle reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        float myVal = vsmem[tid];
        
        for (int offset = 16; offset > 0; offset /= 2) {
            myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        }
        
        if (tid == 0) {
            gradBias[oc] = myVal;
        }
    }
}

// Host wrapper: NCHW Conv2D backward w.r.t. input (GPU Opt V3)
void launchConv2dBackwardInputNCHW(
    const float* d_gradOutput, const float* d_weights, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    dim3 block(16, 16);
    dim3 grid(
        (inW + block.x - 1) / block.x,
        (inH + block.y - 1) / block.y,
        batch * inC
    );
    
    conv2dBackwardInputNCHWKernel<<<grid, block>>>(
        d_gradOutput, d_weights, d_gradInput,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Conv2D backward w.r.t. weights (GPU Opt V3)
void launchConv2dBackwardWeightsNCHW(
    const float* d_input, const float* d_gradOutput, float* d_gradWeights,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    int totalWeights = outC * inC * kernelSize * kernelSize;
    int blockSize = 256;
    int gridSize = (totalWeights + blockSize - 1) / blockSize;
    
    conv2dBackwardWeightsNCHWKernel<<<gridSize, blockSize>>>(
        d_input, d_gradOutput, d_gradWeights,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Conv2D backward w.r.t. bias with warp shuffle (GPU Opt V3)
void launchConv2dBackwardBiasNCHW(
    const float* d_gradOutput, float* d_gradBias,
    int batch, int outC, int outH, int outW
) {
    dim3 grid(outC);
    dim3 block(256);
    size_t sharedMem = 256 * sizeof(float);
    
    conv2dBackwardBiasNCHWKernel<<<grid, block, sharedMem>>>(
        d_gradOutput, d_gradBias, batch, outC, outH, outW
    );
    CHECK_CUDA(cudaGetLastError());
}
