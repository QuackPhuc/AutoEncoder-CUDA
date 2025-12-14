#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive Conv2D forward kernel
// Grid: ((totalThreads + 255) / 256), Block: 256
// Memory layout: NHWC
__global__ void conv2dForwardKernel(
    const float* input,    // (batch, inH, inW, inC)
    const float* weights,  // (outC, kernelSize, kernelSize, inC)
    const float* bias,     // (outC)
    float* output,         // (batch, outH, outW, outC)
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = batch * outH * outW * outC;
    
    if (idx >= totalThreads) return;
    
    // Decode index (NHWC layout)
    int c = idx % outC;
    int w = (idx / outC) % outW;
    int h = (idx / (outC * outW)) % outH;
    int n = idx / (outC * outW * outH);
    
    float sum = 0.0f;
    
    for (int kh = 0; kh < kernelSize; kh++) {
        for (int kw = 0; kw < kernelSize; kw++) {
            for (int ic = 0; ic < inC; ic++) {
                int in_h = h * stride + kh - padding;
                int in_w = w * stride + kw - padding;
                
                float inputVal = 0.0f;
                if (in_h >= 0 && in_h < inH && in_w >= 0 && in_w < inW) {
                    int inputIdx = ((n * inH + in_h) * inW + in_w) * inC + ic;
                    inputVal = input[inputIdx];
                }
                
                int weightIdx = (((c * kernelSize + kh) * kernelSize + kw) * inC + ic);
                sum += inputVal * weights[weightIdx];
            }
        }
    }
    
    sum += bias[c];
    output[idx] = sum;
}

// Host wrapper: Naive Conv2D forward
void launchConv2dForward(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inH, int inW, int inC,
    int outH, int outW, int outC,
    int kernelSize, int padding, int stride
) {
    int totalThreads = batch * outH * outW * outC;
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(totalThreads);
    
    conv2dForwardKernel<<<gridSize, blockSize>>>(
        d_input, d_weights, d_bias, d_output,
        batch, inH, inW, inC, outH, outW, outC,
        kernelSize, padding, stride
    );
    CHECK_CUDA(cudaGetLastError());
}


// NCHW Layout Kernels
__global__ void conv2dForwardNCHWKernel(
    const float* __restrict__ input,    // (batch, inC, inH, inW)
    const float* __restrict__ weights,  // (outC, inC, kernelSize, kernelSize)
    const float* __restrict__ bias,     // (outC)
    float* __restrict__ output,         // (batch, outC, outH, outW)
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc_n = blockIdx.z;
    int oc = oc_n % outC;
    int n = oc_n / outC;
    
    if (ow >= outW || oh >= outH || n >= batch) return;
    
    float sum = bias[oc];
    
    const size_t in_n_offset = static_cast<size_t>(n) * inC * inH * inW;
    const size_t w_oc_offset = static_cast<size_t>(oc) * inC * k * k;
    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;
    
    // Special case for 3x3 kernel with compile-time unrolling
    if (k == 3) {
        #pragma unroll
        for (int ic = 0; ic < inC; ++ic) {
            const size_t in_ic_offset = in_n_offset + static_cast<size_t>(ic) * inH * inW;
            const size_t w_ic_offset = w_oc_offset + static_cast<size_t>(ic) * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih_base + kh;
                if (ih >= 0 && ih < inH) {
                    const size_t in_row = in_ic_offset + ih * inW;
                    const size_t w_kh = w_ic_offset + kh * 3;
                    
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw_base + kw;
                        if (iw >= 0 && iw < inW) {
                            sum += input[in_row + iw] * weights[w_kh + kw];
                        }
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < inC; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = ih_base + kh;
                    int iw = iw_base + kw;
                    
                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                        size_t in_idx = in_n_offset + (static_cast<size_t>(ic) * inH + ih) * inW + iw;
                        size_t w_idx = w_oc_offset + (static_cast<size_t>(ic) * k + kh) * k + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
    output[out_idx] = sum;
}

// NCHW Conv2D + ReLU fused kernel
__global__ void conv2dForwardNCHWReluKernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc_n = blockIdx.z;
    int oc = oc_n % outC;
    int n = oc_n / outC;
    
    if (ow >= outW || oh >= outH || n >= batch) return;
    
    float sum = bias[oc];
    
    const size_t in_n_offset = static_cast<size_t>(n) * inC * inH * inW;
    const size_t w_oc_offset = static_cast<size_t>(oc) * inC * k * k;
    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;
    
    if (k == 3) {
        #pragma unroll
        for (int ic = 0; ic < inC; ++ic) {
            const size_t in_ic_offset = in_n_offset + static_cast<size_t>(ic) * inH * inW;
            const size_t w_ic_offset = w_oc_offset + static_cast<size_t>(ic) * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih_base + kh;
                if (ih >= 0 && ih < inH) {
                    const size_t in_row = in_ic_offset + ih * inW;
                    const size_t w_kh = w_ic_offset + kh * 3;
                    
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw_base + kw;
                        if (iw >= 0 && iw < inW) {
                            sum += input[in_row + iw] * weights[w_kh + kw];
                        }
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < inC; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = ih_base + kh;
                    int iw = iw_base + kw;
                    
                    if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                        size_t in_idx = in_n_offset + (static_cast<size_t>(ic) * inH + ih) * inW + iw;
                        size_t w_idx = w_oc_offset + (static_cast<size_t>(ic) * k + kh) * k + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    
    // Apply ReLU
    sum = fmaxf(0.0f, sum);
    
    size_t out_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
    output[out_idx] = sum;
}

// Host wrapper: NCHW Conv2D forward (GPU Opt V1)
void launchConv2dNCHW(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * outC
    );
    
    conv2dForwardNCHWKernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Conv2D + ReLU forward (GPU Opt V1)
void launchConv2dNCHWRelu(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    dim3 block(16, 16);
    dim3 grid(
        (outW + block.x - 1) / block.x,
        (outH + block.y - 1) / block.y,
        batch * outC
    );
    
    conv2dForwardNCHWReluKernel<<<grid, block>>>(
        d_input, d_weights, d_bias, d_output,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// GPU OPT V2: NCHW + Shared Memory Tiling Kernels

// Tile sizes for NCHW shared memory convolution
#define NCHW_TILE_SIZE 16
#define NCHW_HALO 1
#define NCHW_SHARED_SIZE (NCHW_TILE_SIZE + 2 * NCHW_HALO)

// NCHW Conv2D with Shared Memory Tiling
// Each thread block processes a TILE_SIZE x TILE_SIZE output tile
// Shared memory holds input tile + halo for the current input channel
__global__ void conv2dForwardNCHWSharedKernel(
    const float* __restrict__ input,    // (batch, inC, inH, inW)
    const float* __restrict__ weights,  // (outC, inC, k, k)
    const float* __restrict__ bias,     // (outC)
    float* __restrict__ output,         // (batch, outC, outH, outW)
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    // Shared memory for input tile (with halo)
    extern __shared__ float s_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * NCHW_TILE_SIZE + tx;
    int oh = blockIdx.y * NCHW_TILE_SIZE + ty;
    int oc_n = blockIdx.z;
    int oc = oc_n % outC;
    int n = oc_n / outC;
    
    if (n >= batch) return;
    
    float sum = bias[oc];
    
    // Process each input channel
    for (int ic = 0; ic < inC; ++ic) {
        // Cooperative loading of input tile with halo into shared memory
        const size_t in_base = static_cast<size_t>(n) * inC * inH * inW + 
                               static_cast<size_t>(ic) * inH * inW;
        
        // Each thread may need to load multiple elements
        int threadId = ty * blockDim.x + tx;
        int totalLoads = NCHW_SHARED_SIZE * NCHW_SHARED_SIZE;
        int threadsPerBlock = blockDim.x * blockDim.y;
        
        for (int loadIdx = threadId; loadIdx < totalLoads; loadIdx += threadsPerBlock) {
            int sy = loadIdx / NCHW_SHARED_SIZE;
            int sx = loadIdx % NCHW_SHARED_SIZE;
            
            // Map to input coordinates
            int globalY = blockIdx.y * NCHW_TILE_SIZE * stride + sy - padding;
            int globalX = blockIdx.x * NCHW_TILE_SIZE * stride + sx - padding;
            
            float val = 0.0f;
            if (globalY >= 0 && globalY < inH && globalX >= 0 && globalX < inW) {
                val = input[in_base + globalY * inW + globalX];
            }
            s_input[sy * NCHW_SHARED_SIZE + sx] = val;
        }
        
        __syncthreads();
        
        // Compute convolution using shared memory
        if (ow < outW && oh < outH) {
            const size_t w_base = (static_cast<size_t>(oc) * inC + ic) * k * k;
            
            if (k == 3) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int sy = ty * stride + kh;
                        int sx = tx * stride + kw;
                        float inputVal = s_input[sy * NCHW_SHARED_SIZE + sx];
                        float weightVal = weights[w_base + kh * 3 + kw];
                        sum += inputVal * weightVal;
                    }
                }
            } else {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {
                        int sy = ty * stride + kh;
                        int sx = tx * stride + kw;
                        float inputVal = s_input[sy * NCHW_SHARED_SIZE + sx];
                        float weightVal = weights[w_base + kh * k + kw];
                        sum += inputVal * weightVal;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (ow < outW && oh < outH) {
        size_t out_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
        output[out_idx] = sum;
    }
}

// NCHW Conv2D + ReLU fused with Shared Memory Tiling
__global__ void conv2dForwardNCHWSharedReluKernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int k, int stride, int padding
) {
    extern __shared__ float s_input[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * NCHW_TILE_SIZE + tx;
    int oh = blockIdx.y * NCHW_TILE_SIZE + ty;
    int oc_n = blockIdx.z;
    int oc = oc_n % outC;
    int n = oc_n / outC;
    
    if (n >= batch) return;
    
    float sum = bias[oc];
    
    for (int ic = 0; ic < inC; ++ic) {
        const size_t in_base = static_cast<size_t>(n) * inC * inH * inW + 
                               static_cast<size_t>(ic) * inH * inW;
        
        int threadId = ty * blockDim.x + tx;
        int totalLoads = NCHW_SHARED_SIZE * NCHW_SHARED_SIZE;
        int threadsPerBlock = blockDim.x * blockDim.y;
        
        for (int loadIdx = threadId; loadIdx < totalLoads; loadIdx += threadsPerBlock) {
            int sy = loadIdx / NCHW_SHARED_SIZE;
            int sx = loadIdx % NCHW_SHARED_SIZE;
            
            int globalY = blockIdx.y * NCHW_TILE_SIZE * stride + sy - padding;
            int globalX = blockIdx.x * NCHW_TILE_SIZE * stride + sx - padding;
            
            float val = 0.0f;
            if (globalY >= 0 && globalY < inH && globalX >= 0 && globalX < inW) {
                val = input[in_base + globalY * inW + globalX];
            }
            s_input[sy * NCHW_SHARED_SIZE + sx] = val;
        }
        
        __syncthreads();
        
        if (ow < outW && oh < outH) {
            const size_t w_base = (static_cast<size_t>(oc) * inC + ic) * k * k;
            
            if (k == 3) {
                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int sy = ty * stride + kh;
                        int sx = tx * stride + kw;
                        float inputVal = s_input[sy * NCHW_SHARED_SIZE + sx];
                        float weightVal = weights[w_base + kh * 3 + kw];
                        sum += inputVal * weightVal;
                    }
                }
            } else {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {
                        int sy = ty * stride + kh;
                        int sx = tx * stride + kw;
                        float inputVal = s_input[sy * NCHW_SHARED_SIZE + sx];
                        float weightVal = weights[w_base + kh * k + kw];
                        sum += inputVal * weightVal;
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Apply ReLU and write output
    if (ow < outW && oh < outH) {
        sum = fmaxf(0.0f, sum);
        size_t out_idx = ((static_cast<size_t>(n) * outC + oc) * outH + oh) * outW + ow;
        output[out_idx] = sum;
    }
}

// Host wrapper: NCHW Conv2D with Shared Memory (GPU Opt V2)
void launchConv2dNCHWShared(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    dim3 block(NCHW_TILE_SIZE, NCHW_TILE_SIZE);
    dim3 grid(
        (outW + NCHW_TILE_SIZE - 1) / NCHW_TILE_SIZE,
        (outH + NCHW_TILE_SIZE - 1) / NCHW_TILE_SIZE,
        batch * outC
    );
    
    size_t sharedMemSize = NCHW_SHARED_SIZE * NCHW_SHARED_SIZE * sizeof(float);
    
    conv2dForwardNCHWSharedKernel<<<grid, block, sharedMemSize>>>(
        d_input, d_weights, d_bias, d_output,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: NCHW Conv2D + ReLU with Shared Memory (GPU Opt V2)
void launchConv2dNCHWSharedRelu(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW,
    int outC, int outH, int outW,
    int kernelSize, int padding, int stride
) {
    dim3 block(NCHW_TILE_SIZE, NCHW_TILE_SIZE);
    dim3 grid(
        (outW + NCHW_TILE_SIZE - 1) / NCHW_TILE_SIZE,
        (outH + NCHW_TILE_SIZE - 1) / NCHW_TILE_SIZE,
        batch * outC
    );
    
    size_t sharedMemSize = NCHW_SHARED_SIZE * NCHW_SHARED_SIZE * sizeof(float);
    
    conv2dForwardNCHWSharedReluKernel<<<grid, block, sharedMemSize>>>(
        d_input, d_weights, d_bias, d_output,
        batch, inC, inH, inW, outC, outH, outW,
        kernelSize, stride, padding
    );
    CHECK_CUDA(cudaGetLastError());
}

