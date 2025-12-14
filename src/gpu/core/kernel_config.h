#ifndef GPU_CORE_KERNEL_CONFIG_H
#define GPU_CORE_KERNEL_CONFIG_H

// Note: Block sizes for 2D tiled convolutions use TILE_SIZE (16x16=256)

#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 256
#endif

// Helper macro for grid size calculation
#define CUDA_GRID_SIZE(total_threads) (((total_threads) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)

// Maximum shared memory block size (for reduction kernels)
#define CUDA_SHARED_MEM_SIZE CUDA_BLOCK_SIZE

#endif // GPU_CORE_KERNEL_CONFIG_H
