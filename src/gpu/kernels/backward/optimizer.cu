#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gpu/core/cuda_utils.h"
#include "gpu/core/kernel_config.h"

// Naive SGD weight update kernel
__global__ void sgdUpdateKernel(
    float* weights,
    const float* gradients,
    float learningRate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        weights[idx] -= learningRate * gradients[idx];
    }
}
