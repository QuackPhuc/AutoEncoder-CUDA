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