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

// Vectorized SGD update using float4 (GPU Opt v1)
// Processes 4 weights per thread for better memory bandwidth
__global__ void sgdUpdateVectorizedKernel(
    float* __restrict__ weights,
    const float* __restrict__ gradients,
    float learningRate,
    int size
) {
    int idx4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx4 + 3 < size) {
        float4 w = reinterpret_cast<float4*>(weights)[idx4 / 4];
        float4 g = reinterpret_cast<const float4*>(gradients)[idx4 / 4];
        
        w.x -= learningRate * g.x;
        w.y -= learningRate * g.y;
        w.z -= learningRate * g.z;
        w.w -= learningRate * g.w;
        
        reinterpret_cast<float4*>(weights)[idx4 / 4] = w;
    } else if (idx4 < size) {
        // Handle remaining elements
        for (int i = idx4; i < size && i < idx4 + 4; i++) {
            weights[i] -= learningRate * gradients[i];
        }
    }
}

// Host wrapper: Naive SGD update
void launchSGDUpdate(float* d_weights, const float* d_gradients, float learningRate, int size) {
    int blockSize = CUDA_BLOCK_SIZE;
    int gridSize = CUDA_GRID_SIZE(size);
    
    sgdUpdateKernel<<<gridSize, blockSize>>>(d_weights, d_gradients, learningRate, size);
    CHECK_CUDA(cudaGetLastError());
}

// Host wrapper: Vectorized SGD update (GPU Opt v1)
void launchSGDUpdateVectorized(float* d_weights, const float* d_gradients, float learningRate, int size) {
    int blockSize = CUDA_BLOCK_SIZE;
    // Use ceiling division and ensure at least 1 block for small sizes (e.g., bias vectors)
    int numFloat4 = (size + 3) / 4;
    int gridSize = (numFloat4 + blockSize - 1) / blockSize;
    if (gridSize < 1) gridSize = 1;
    
    sgdUpdateVectorizedKernel<<<gridSize, blockSize>>>(d_weights, d_gradients, learningRate, size);
    CHECK_CUDA(cudaGetLastError());
}
