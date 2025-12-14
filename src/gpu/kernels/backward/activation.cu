#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive ReLU backward kernel
__global__ void reluBackwardKernel(
    const float* gradOutput,
    const float* input,
    float* gradInput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = (input[idx] > 0.0f) ? gradOutput[idx] : 0.0f;
    }
}

// Optimized ReLU backward (same logic, cleaner code)
__global__ void reluBackwardOptKernel(
    const float* __restrict__ gradOutput,
    const float* __restrict__ input,
    float* __restrict__ gradInput,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = (input[idx] > 0.0f) ? gradOutput[idx] : 0.0f;
    }
}

// Host wrapper: Naive ReLU backward
void launchReluBackward(
    const float* d_gradOutput, const float* d_input, float* d_gradInput, int size
) {
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    reluBackwardKernel<<<gridSize, blockSize>>>(d_gradOutput, d_input, d_gradInput, size);
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: Optimized ReLU backward
void launchReluBackwardOpt(
    const float* d_gradOutput, const float* d_input, float* d_gradInput, int size
) {
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    reluBackwardOptKernel<<<gridSize, blockSize>>>(d_gradOutput, d_input, d_gradInput, size);
    CHECK_CUDA(cudaGetLastError());
}
