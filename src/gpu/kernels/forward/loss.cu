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
