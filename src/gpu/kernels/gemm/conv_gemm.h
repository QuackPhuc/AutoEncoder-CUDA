#ifndef GPU_GEMM_CONV_GEMM_H
#define GPU_GEMM_CONV_GEMM_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// cuBLAS error checking macro
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d code=%d\n", \
                    __FILE__, __LINE__, static_cast<int>(status)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Singleton cuBLAS handle manager
class CublasHandle {
public:
    static cublasHandle_t& get() {
        static CublasHandle instance;
        return instance.handle;
    }
    
    ~CublasHandle() {
        if (initialized) {
            cublasDestroy(handle);
        }
    }
    
private:
    CublasHandle() {
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS handle creation failed: %d\n", static_cast<int>(status));
            exit(EXIT_FAILURE);
        }
        initialized = true;
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    
    cublasHandle_t handle;
    bool initialized = false;
};

// GEMM wrapper for convolution forward using batched GEMM
// Weights: [outC, inC_k_k]  (M x K) - shared across batches
// Im2col:  [batch, inC_k_k, outHW] (batch x K x N)
// Output:  [batch, outC, outHW]    (batch x M x N)
void launchConvGemmForward(
    const float* d_weights,  // [outC, inC*k*k]
    const float* d_im2col,   // [batch, inC*k*k, outH*outW]
    const float* d_bias,     // [outC]
    float* d_output,         // [batch, outC, outH*outW]
    int batch, int outC, int inC_k_k, int outHW,
    bool applyRelu = false
);

// GEMM wrapper for convolution backward w.r.t input
// Computes: d_col = Weights^T × d_output
void launchConvGemmBackwardInput(
    const float* d_weights,    // [outC, inC*k*k]
    const float* d_gradOutput, // [batch, outC, outH*outW]
    float* d_col,              // [batch, inC*k*k, outH*outW]
    int batch, int outC, int inC_k_k, int outHW
);

// GEMM wrapper for convolution backward w.r.t weights
// Computes: d_weights = d_output × Im2col^T
void launchConvGemmBackwardWeights(
    const float* d_gradOutput, // [batch, outC, outH*outW]
    const float* d_im2col,     // [batch, inC*k*k, outH*outW]
    float* d_gradWeights,      // [outC, inC*k*k]
    int batch, int outC, int inC_k_k, int outHW
);

#endif // GPU_GEMM_CONV_GEMM_H
