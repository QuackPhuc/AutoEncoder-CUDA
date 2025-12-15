#include "conv_gemm.h"
#include "gpu/core/cuda_utils.h"
#include <cuda_runtime.h>
#include <cstdio>

// Add bias kernel
__global__ void addBiasNCHWKernel(
    float* output,        // [batch, outC, outHW]
    const float* bias,    // [outC]
    int batch, int outC, int outHW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * outC * outHW;
    
    if (idx >= total) return;
    
    int oc = (idx / outHW) % outC;
    output[idx] += bias[oc];
}

// Add bias + ReLU kernel
__global__ void addBiasReluNCHWKernel(
    float* output,
    const float* bias,
    int batch, int outC, int outHW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * outC * outHW;
    
    if (idx >= total) return;
    
    int oc = (idx / outHW) % outC;
    float val = output[idx] + bias[oc];
    output[idx] = fmaxf(0.0f, val);
}

// Helper: Add bias to output
void launchAddBias(float* d_output, const float* d_bias, 
                   int batch, int outC, int outHW, bool applyRelu) {
    int total = batch * outC * outHW;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    if (applyRelu) {
        addBiasReluNCHWKernel<<<gridSize, blockSize>>>(
            d_output, d_bias, batch, outC, outHW
        );
    } else {
        addBiasNCHWKernel<<<gridSize, blockSize>>>(
            d_output, d_bias, batch, outC, outHW
        );
    }
    CHECK_CUDA(cudaGetLastError());
}

// GEMM wrapper for convolution forward using batched GEMM
// Weights: [outC, inC_k_k]  (M x K) - shared across batches
// Im2col:  [batch, inC_k_k, outHW] (batch x K x N)
// Output:  [batch, outC, outHW]    (batch x M x N)
void launchConvGemmForward(
    const float* d_weights,
    const float* d_im2col,
    const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    bool applyRelu
) {
    cublasHandle_t handle = CublasHandle::get();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = outC;      // Rows of output
    int N = outHW;     // Cols of output  
    int K = inC_k_k;   // Inner dimension
    
    // Strides for batched operation
    long long int strideA = 0;           // Weights shared across batches
    long long int strideB = (long long int)K * N;  // im2col stride per batch
    long long int strideC = (long long int)M * N;  // output stride per batch
    
    // cuBLAS column-major: C^T = B^T × A^T for row-major matrices
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_im2col, N, strideB,      // B: [K, N] per batch
        d_weights, K, strideA,      // A: [M, K] shared
        &beta,
        d_output, N, strideC,       // C: [M, N] per batch
        batch
    ));
    
    // Add bias (and optionally ReLU)
    launchAddBias(d_output, d_bias, batch, outC, outHW, applyRelu);
}

// GEMM wrapper for backward w.r.t. input using batched GEMM
// d_col = Weights^T × d_output
// Weights: [outC, inC_k_k] (K x M) - shared across batches
// d_output: [batch, outC, outHW] (batch x K x N)
// d_col:    [batch, inC_k_k, outHW] (batch x M x N)
void launchConvGemmBackwardInput(
    const float* d_weights,
    const float* d_gradOutput,
    float* d_col,
    int batch, int outC, int inC_k_k, int outHW
) {
    cublasHandle_t handle = CublasHandle::get();
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = inC_k_k;   // Rows of output (d_col)
    int N = outHW;     // Cols of output
    int K = outC;      // Inner dimension
    
    // Strides for batched operation
    long long int strideA = 0;           // Weights shared across batches
    long long int strideB = (long long int)K * N;  // gradOutput stride per batch
    long long int strideC = (long long int)M * N;  // col stride per batch
    
    // d_col = W^T × gradOut using batched GEMM
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_gradOutput, N, strideB,   // gradOut: [K, N] per batch
        d_weights, M, strideA,      // W: [K, M] shared -> W^T gives [M, K]
        &beta,
        d_col, N, strideC,          // d_col: [M, N] per batch
        batch
    ));
}

// GEMM wrapper for backward w.r.t. weights
// d_weights += sum over batch of (d_output × im2col^T)
// Note: Uses loop because gradients must accumulate into single weight matrix
// d_output: [batch, outC, outHW]  (batch x M x K)
// im2col:   [batch, inC_k_k, outHW] (batch x N x K)
// d_weights: [outC, inC_k_k] (M x N) - accumulated
void launchConvGemmBackwardWeights(
    const float* d_gradOutput,
    const float* d_im2col,
    float* d_gradWeights,
    int batch, int outC, int inC_k_k, int outHW
) {
    cublasHandle_t handle = CublasHandle::get();
    
    const float alpha = 1.0f;
    float beta = 0.0f;  // First batch: overwrite, rest: accumulate
    
    int M = outC;
    int N = inC_k_k;
    int K = outHW;
    
    for (int n = 0; n < batch; ++n) {
        const float* gradOut_n = d_gradOutput + n * M * K;
        const float* im2col_n = d_im2col + n * N * K;
        
        // d_W = gradOut × im2col^T
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            im2col_n, K,        // im2col: [N, K] -> transpose gives [K, N]
            gradOut_n, K,       // gradOut: [M, K]
            &beta,
            d_gradWeights, N    // d_W: [M, N]
        ));
        
        beta = 1.0f;  // Accumulate for remaining batches
    }
}

// ============================================================
// Stream-aware versions for GPU Opt V3
// ============================================================

// Helper: Add bias on specified stream
static void launchAddBiasStream(float* d_output, const float* d_bias, 
                                int batch, int outC, int outHW, 
                                bool applyRelu, cudaStream_t stream) {
    int total = batch * outC * outHW;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    if (applyRelu) {
        addBiasReluNCHWKernel<<<gridSize, blockSize, 0, stream>>>(
            d_output, d_bias, batch, outC, outHW
        );
    } else {
        addBiasNCHWKernel<<<gridSize, blockSize, 0, stream>>>(
            d_output, d_bias, batch, outC, outHW
        );
    }
    CHECK_CUDA(cudaGetLastError());
}

// Stream-aware GEMM forward
void launchConvGemmForwardStream(
    const float* d_weights,
    const float* d_im2col,
    const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream,
    bool applyRelu
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = outC;
    int N = outHW;
    int K = inC_k_k;
    
    long long int strideA = 0;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_im2col, N, strideB,
        d_weights, K, strideA,
        &beta,
        d_output, N, strideC,
        batch
    ));
    
    // Add bias on the same stream
    launchAddBiasStream(d_output, d_bias, batch, outC, outHW, applyRelu, stream);
}

// Stream-aware GEMM backward w.r.t. input
void launchConvGemmBackwardInputStream(
    const float* d_weights,
    const float* d_gradOutput,
    float* d_col,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    (void)stream; // Stream is already set on handle
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = inC_k_k;
    int N = outHW;
    int K = outC;
    
    long long int strideA = 0;
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;
    
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_gradOutput, N, strideB,
        d_weights, M, strideA,
        &beta,
        d_col, N, strideC,
        batch
    ));
}

// Stream-aware GEMM backward w.r.t. weights
void launchConvGemmBackwardWeightsStream(
    const float* d_gradOutput,
    const float* d_im2col,
    float* d_gradWeights,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    (void)stream; // Stream is already set on handle
    
    const float alpha = 1.0f;
    float beta = 0.0f;
    
    int M = outC;
    int N = inC_k_k;
    int K = outHW;
    
    for (int n = 0; n < batch; ++n) {
        const float* gradOut_n = d_gradOutput + n * M * K;
        const float* im2col_n = d_im2col + n * N * K;
        
        CHECK_CUBLAS(cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            im2col_n, K,
            gradOut_n, K,
            &beta,
            d_gradWeights, N
        ));
        
        beta = 1.0f;
    }
}

// ============================================================
// Mixed Precision (FP16 + Tensor Cores) Implementations
// ============================================================

#include <cuda_fp16.h>

// Add bias kernel with FP32 output
__global__ void addBiasFP32Kernel(
    float* output,
    const float* bias,
    int batch, int outC, int outHW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * outC * outHW;
    
    if (idx >= total) return;
    
    int oc = (idx / outHW) % outC;
    output[idx] += bias[oc];
}

// Add bias + ReLU kernel with FP32 output
__global__ void addBiasReluFP32Kernel(
    float* output,
    const float* bias,
    int batch, int outC, int outHW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * outC * outHW;
    
    if (idx >= total) return;
    
    int oc = (idx / outHW) % outC;
    float val = output[idx] + bias[oc];
    output[idx] = fmaxf(0.0f, val);
}

// FP16 GEMM forward with Tensor Cores
void launchConvGemmForwardFP16(
    const half* d_weights_fp16,
    const half* d_im2col_fp16,
    const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream,
    bool applyRelu
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = outC;
    int N = outHW;
    int K = inC_k_k;
    
    long long int strideA = 0;  // Weights shared across batches
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;
    
    // Use cublasGemmStridedBatchedEx for Tensor Core acceleration
    // A (weights): [M, K] in FP16 - shared
    // B (im2col):  [K, N] per batch in FP16
    // C (output):  [M, N] per batch in FP32
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_im2col_fp16, CUDA_R_16F, N, strideB,
        d_weights_fp16, CUDA_R_16F, K, strideA,
        &beta,
        d_output, CUDA_R_32F, N, strideC,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // Add bias (and optionally ReLU) - use FP32 kernels
    int total = batch * outC * outHW;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    if (applyRelu) {
        addBiasReluFP32Kernel<<<gridSize, blockSize, 0, stream>>>(
            d_output, d_bias, batch, outC, outHW
        );
    } else {
        addBiasFP32Kernel<<<gridSize, blockSize, 0, stream>>>(
            d_output, d_bias, batch, outC, outHW
        );
    }
    CHECK_CUDA(cudaGetLastError());
}

// FP16 GEMM backward w.r.t. input
void launchConvGemmBackwardInputFP16(
    const half* d_weights_fp16,
    const half* d_gradOutput_fp16,
    half* d_col_fp16,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle,
    cudaStream_t stream
) {
    (void)stream; // Stream already set on handle
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = inC_k_k;
    int N = outHW;
    int K = outC;
    
    long long int strideA = 0;  // Weights shared
    long long int strideB = (long long int)K * N;
    long long int strideC = (long long int)M * N;
    
    // d_col = W^T × gradOut
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        N, M, K,
        &alpha,
        d_gradOutput_fp16, CUDA_R_16F, N, strideB,
        d_weights_fp16, CUDA_R_16F, M, strideA,
        &beta,
        d_col_fp16, CUDA_R_16F, N, strideC,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

