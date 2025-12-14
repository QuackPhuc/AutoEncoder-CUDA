#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive MSE loss kernel with atomic reduction
__global__ void mseLossKernel(
    const float* pred,
    const float* target,
    float* loss,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        atomicAdd(loss, diff * diff);
    }
}

// Optimized MSE loss with shared memory reduction (GPU Opt v1)
__global__ void mseLossOptKernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ loss,
    int size
) {
    __shared__ float s_sum[CUDA_SHARED_MEM_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float localSum = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        localSum = diff * diff;
    }
    
    s_sum[tid] = localSum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, s_sum[0]);
    }
}

// MSE loss gradient kernel
// gradOutput = 2 * (pred - target) / size
__global__ void mseLossGradKernel(
    const float* pred,
    const float* target,
    float* gradOutput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        gradOutput[idx] = 2.0f * (pred[idx] - target[idx]) / size;
    }
}

// Host wrapper: Naive MSE loss
void launchMSELoss(
    const float* d_pred, const float* d_target, float* d_loss, int size
) {
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    mseLossKernel<<<gridSize, blockSize>>>(d_pred, d_target, d_loss, size);
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: Optimized MSE loss (GPU Opt v1)
void launchMSELossOpt(
    const float* d_pred, const float* d_target, float* d_loss, int size
) {
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    mseLossOptKernel<<<gridSize, blockSize>>>(d_pred, d_target, d_loss, size);
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: MSE loss gradient
void launchMSELossGrad(
    const float* d_pred, const float* d_target, float* d_gradOutput, int size
) {
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    mseLossGradKernel<<<gridSize, blockSize>>>(d_pred, d_target, d_gradOutput, size);
    CHECK_CUDA(cudaGetLastError());
}
