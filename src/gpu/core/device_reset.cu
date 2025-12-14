#include <cuda_runtime.h>

void resetGPUDevice() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        cudaGetLastError();
    }
}
