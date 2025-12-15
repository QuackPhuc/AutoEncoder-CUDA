// GPU Core Utilities
// CUDA error checking macros and helper functions

#ifndef GPU_CORE_CUDA_UTILS_H
#define GPU_CORE_CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA cleanup macro - logs error but doesn't exit (use in destructors)
#define CHECK_CUDA_CLEANUP(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Cleanup Warning: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            cudaGetLastError(); \
        } \
    } while(0)


// Print GPU device information (compact)
inline void printGPUInfo() {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "GPU: " << prop.name << " (" 
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
}

#endif // GPU_CORE_CUDA_UTILS_H
