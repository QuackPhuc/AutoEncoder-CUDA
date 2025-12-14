#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive ReLU forward kernel
__global__ void reluForwardKernel(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Host wrapper: Naive ReLU forward
void launchReluForward(const float* d_input, float* d_output, int size) {
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    reluForwardKernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CHECK_CUDA(cudaGetLastError());
}