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

// ============================================================
// Stream-aware versions for GPU Opt V3
// ============================================================

// Stream-aware GEMM forward with custom cuBLAS handle
void launchConvGemmForwardStream(
    const float* d_weights,
    const float* d_im2col,
    const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream,
    bool applyRelu = false
);

// Stream-aware GEMM backward w.r.t. input
void launchConvGemmBackwardInputStream(
    const float* d_weights,
    const float* d_gradOutput,
    float* d_col,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
);

// Stream-aware GEMM backward w.r.t. weights 
void launchConvGemmBackwardWeightsStream(
    const float* d_gradOutput,
    const float* d_im2col,
    float* d_gradWeights,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
);

// ============================================================
// Mixed Precision (FP16 + Tensor Cores) for GPU Opt V3
// ============================================================

#include <cuda_fp16.h>

// FP16 GEMM forward with Tensor Cores (cublasGemmEx)
// Input weights and im2col in FP16, output in FP32
void launchConvGemmForwardFP16(
    const half* d_weights_fp16,  // [outC, inC*k*k] in FP16
    const half* d_im2col_fp16,   // [batch, inC*k*k, outH*outW] in FP16
    const float* d_bias,         // [outC] in FP32
    float* d_output,             // [batch, outC, outH*outW] in FP32
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream,
    bool applyRelu = false
);

// FP16 GEMM backward w.r.t. input (for gradient computation)
void launchConvGemmBackwardInputFP16(
    const half* d_weights_fp16,
    const half* d_gradOutput_fp16,
    half* d_col_fp16,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
);

#endif // GPU_GEMM_CONV_GEMM_H
