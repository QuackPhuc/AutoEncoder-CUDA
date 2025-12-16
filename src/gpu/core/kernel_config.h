#ifndef GPU_CORE_KERNEL_CONFIG_H
#define GPU_CORE_KERNEL_CONFIG_H

// Default block size for 1D kernel launches
// Note: 2D tiled convolutions use explicit dim3(16,16) blocks
#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 256
#endif

#endif // GPU_CORE_KERNEL_CONFIG_H

