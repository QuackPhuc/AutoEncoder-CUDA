#include "autoencoder.h"
#include "gpu/core/cuda_utils.h"
#include "gpu/core/layout_convert.h"
#include "config/gpu_config.h"
#include <cuda_runtime.h>

// External kernel declarations (from kernels/forward/)
extern __global__ void conv2dForwardKernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch, int inH, int inW, int inC, int outH, int outW, int outC,
    int kernelSize, int padding, int stride);

extern __global__ void reluForwardKernel(const float* input, float* output, int size);

extern __global__ void maxpool2dForwardKernel(
    const float* input, float* output, int* indices,
    int batch, int inH, int inW, int channels);

extern __global__ void upsample2dForwardKernel(
    const float* input, float* output, int batch, int inH, int inW, int channels);

// Kernel wrapper implementations for GPUAutoencoder

void GPUAutoencoder::conv2dForward(
    const float* d_in, const float* d_w, const float* d_b, float* d_out,
    int batch, int inH, int inW, int inC, int outH, int outW, int outC
) {
    int totalThreads = batch * outH * outW * outC;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    
    conv2dForwardKernel<<<gridSize, blockSize>>>(
        d_in, d_w, d_b, d_out,
        batch, inH, inW, inC, outH, outW, outC,
        3, 1, 1  // kernel=3, pad=1, stride=1
    );
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::reluForward(const float* d_in, float* d_out, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    reluForwardKernel<<<gridSize, blockSize>>>(d_in, d_out, size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::maxpool2dForward(
    const float* d_in, float* d_out, int* d_indices,
    int batch, int inH, int inW, int channels
) {
    int outH = inH / 2;
    int outW = inW / 2;
    int totalThreads = batch * outH * outW * channels;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    
    maxpool2dForwardKernel<<<gridSize, blockSize>>>(d_in, d_out, d_indices, batch, inH, inW, channels);
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::upsample2dForward(
    const float* d_in, float* d_out, int batch, int inH, int inW, int channels
) {
    int outH = inH * 2;
    int outW = inW * 2;
    int totalThreads = batch * outH * outW * channels;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    
    upsample2dForwardKernel<<<gridSize, blockSize>>>(d_in, d_out, batch, inH, inW, channels);
    CHECK_CUDA(cudaGetLastError());
}

// Basic forward pass (naive kernels) - NHWC layout
void GPUAutoencoder::forwardBasic() {
    // Convert input from NCHW (loader format) to NHWC (V1/V2 kernel format)
    launchNchwToNhwc(d_input, d_input_nhwc, m_batchSize, 3, 32, 32);
    
    // Encoder Layer 1: Conv -> ReLU -> MaxPool
    conv2dForward(d_input_nhwc, d_enc_conv1_w, d_enc_conv1_b, d_enc_conv1_out,
                  m_batchSize, 32, 32, 3, 32, 32, 256);
    reluForward(d_enc_conv1_out, d_enc_relu1_out, m_batchSize * 32 * 32 * 256);
    maxpool2dForward(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                     m_batchSize, 32, 32, 256);
    
    // Encoder Layer 2: Conv -> ReLU -> MaxPool
    conv2dForward(d_enc_pool1_out, d_enc_conv2_w, d_enc_conv2_b, d_enc_conv2_out,
                  m_batchSize, 16, 16, 256, 16, 16, 128);
    reluForward(d_enc_conv2_out, d_enc_relu2_out, m_batchSize * 16 * 16 * 128);
    maxpool2dForward(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                     m_batchSize, 16, 16, 128);
    
    // Decoder Layer 3: Conv -> ReLU -> Upsample
    conv2dForward(d_enc_pool2_out, d_dec_conv3_w, d_dec_conv3_b, d_dec_conv3_out,
                  m_batchSize, 8, 8, 128, 8, 8, 128);
    reluForward(d_dec_conv3_out, d_dec_relu3_out, m_batchSize * 8 * 8 * 128);
    upsample2dForward(d_dec_relu3_out, d_dec_up1_out, m_batchSize, 8, 8, 128);
    
    // Decoder Layer 4: Conv -> ReLU -> Upsample
    conv2dForward(d_dec_up1_out, d_dec_conv4_w, d_dec_conv4_b, d_dec_conv4_out,
                  m_batchSize, 16, 16, 128, 16, 16, 256);
    reluForward(d_dec_conv4_out, d_dec_relu4_out, m_batchSize * 16 * 16 * 256);
    upsample2dForward(d_dec_relu4_out, d_dec_up2_out, m_batchSize, 16, 16, 256);
    
    // Decoder Layer 5: Conv (no activation) -> output to NHWC temp buffer
    conv2dForward(d_dec_up2_out, d_dec_conv5_w, d_dec_conv5_b, d_output_nhwc,
                  m_batchSize, 32, 32, 256, 32, 32, 3);
    
    // Convert output from NHWC back to NCHW (for loss computation with NCHW target)
    launchNhwcToNchw(d_output_nhwc, d_output, m_batchSize, 32, 32, 3);
}

// GPU Opt V1: NCHW Layout Forward Pass
// External NCHW kernel declarations
extern void launchConv2dNCHW(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kernelSize, int padding, int stride);
extern void launchConv2dNCHWRelu(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kernelSize, int padding, int stride);
extern void launchMaxPool2dNCHW(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW, int k, int stride);
extern void launchUpsample2dNCHW(
    const float* d_input, float* d_output,
    int batch, int channels, int inH, int inW, int scale);

// Optimized forward pass v1 (NCHW layout)
void GPUAutoencoder::forwardOptV1() {
    
    // Encoder Layer 1: Conv+ReLU (fused) + MaxPool
    launchConv2dNCHWRelu(d_input, d_enc_conv1_w, d_enc_conv1_b, d_enc_relu1_out,
                         m_batchSize, 3, 32, 32, 256, 32, 32, 3, 1, 1);
    launchMaxPool2dNCHW(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                        m_batchSize, 256, 32, 32, 2, 2);
    
    // Encoder Layer 2: Conv+ReLU (fused) + MaxPool
    launchConv2dNCHWRelu(d_enc_pool1_out, d_enc_conv2_w, d_enc_conv2_b, d_enc_relu2_out,
                         m_batchSize, 256, 16, 16, 128, 16, 16, 3, 1, 1);
    launchMaxPool2dNCHW(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                        m_batchSize, 128, 16, 16, 2, 2);
    
    // Decoder Layer 3: Conv+ReLU (fused) + Upsample
    launchConv2dNCHWRelu(d_enc_pool2_out, d_dec_conv3_w, d_dec_conv3_b, d_dec_relu3_out,
                         m_batchSize, 128, 8, 8, 128, 8, 8, 3, 1, 1);
    launchUpsample2dNCHW(d_dec_relu3_out, d_dec_up1_out, m_batchSize, 128, 8, 8, 2);
    
    // Decoder Layer 4: Conv+ReLU (fused) + Upsample
    launchConv2dNCHWRelu(d_dec_up1_out, d_dec_conv4_w, d_dec_conv4_b, d_dec_relu4_out,
                         m_batchSize, 128, 16, 16, 256, 16, 16, 3, 1, 1);
    launchUpsample2dNCHW(d_dec_relu4_out, d_dec_up2_out, m_batchSize, 256, 16, 16, 2);
    
    // Decoder Layer 5: Conv only (no ReLU)
    launchConv2dNCHW(d_dec_up2_out, d_dec_conv5_w, d_dec_conv5_b, d_output,
                     m_batchSize, 256, 32, 32, 3, 32, 32, 3, 1, 1);
}
// GPU Opt V2: im2col + cuBLAS GEMM Forward Pass
// External kernel declarations for im2col and GEMM
extern void launchIm2colNCHW(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

extern void launchConvGemmForward(
    const float* d_weights, const float* d_im2col, const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    bool applyRelu);

// Optimized forward pass v2 (im2col + cuBLAS GEMM)
void GPUAutoencoder::forwardOptV2() {
    
    // Layer 1: Input [batch,3,32,32] -> Conv -> ReLU -> [batch,256,32,32] -> Pool -> [batch,256,16,16]
    // im2col: [batch, 3*9, 32*32] = [batch, 27, 1024]
    launchIm2colNCHW(d_input, d_im2col_workspace,
                     m_batchSize, 3, 32, 32, 32, 32, 3, 1, 1);
    // GEMM: W[256,27] × im2col[27,1024] = out[256,1024] -> reshape to [256,32,32]
    launchConvGemmForward(d_enc_conv1_w, d_im2col_workspace, d_enc_conv1_b,
                          d_enc_relu1_out,
                          m_batchSize, 256, 27, 1024, true);  // applyRelu=true
    launchMaxPool2dNCHW(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                        m_batchSize, 256, 32, 32, 2, 2);
    
    // Layer 2: [batch,256,16,16] -> Conv -> ReLU -> [batch,128,16,16] -> Pool -> [batch,128,8,8]  
    // im2col: [batch, 256*9, 16*16] = [batch, 2304, 256]
    launchIm2colNCHW(d_enc_pool1_out, d_im2col_workspace,
                     m_batchSize, 256, 16, 16, 16, 16, 3, 1, 1);
    // GEMM: W[128,2304] × im2col[2304,256] = out[128,256]
    launchConvGemmForward(d_enc_conv2_w, d_im2col_workspace, d_enc_conv2_b,
                          d_enc_relu2_out,
                          m_batchSize, 128, 2304, 256, true);
    launchMaxPool2dNCHW(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                        m_batchSize, 128, 16, 16, 2, 2);
    
    // Layer 3: [batch,128,8,8] -> Conv -> ReLU -> [batch,128,8,8] -> Upsample -> [batch,128,16,16]
    // im2col: [batch, 128*9, 8*8] = [batch, 1152, 64]
    launchIm2colNCHW(d_enc_pool2_out, d_im2col_workspace,
                     m_batchSize, 128, 8, 8, 8, 8, 3, 1, 1);
    launchConvGemmForward(d_dec_conv3_w, d_im2col_workspace, d_dec_conv3_b,
                          d_dec_relu3_out,
                          m_batchSize, 128, 1152, 64, true);
    launchUpsample2dNCHW(d_dec_relu3_out, d_dec_up1_out, m_batchSize, 128, 8, 8, 2);
    
    // Layer 4: [batch,128,16,16] -> Conv -> ReLU -> [batch,256,16,16] -> Upsample -> [batch,256,32,32]
    // im2col: [batch, 128*9, 16*16] = [batch, 1152, 256]
    launchIm2colNCHW(d_dec_up1_out, d_im2col_workspace,
                     m_batchSize, 128, 16, 16, 16, 16, 3, 1, 1);
    launchConvGemmForward(d_dec_conv4_w, d_im2col_workspace, d_dec_conv4_b,
                          d_dec_relu4_out,
                          m_batchSize, 256, 1152, 256, true);
    launchUpsample2dNCHW(d_dec_relu4_out, d_dec_up2_out, m_batchSize, 256, 16, 16, 2);
    
    // Layer 5: [batch,256,32,32] -> Conv -> [batch,3,32,32] (no ReLU)
    // im2col: [batch, 256*9, 32*32] = [batch, 2304, 1024]
    launchIm2colNCHW(d_dec_up2_out, d_im2col_workspace,
                     m_batchSize, 256, 32, 32, 32, 32, 3, 1, 1);
    launchConvGemmForward(d_dec_conv5_w, d_im2col_workspace, d_dec_conv5_b,
                          d_output,
                          m_batchSize, 3, 2304, 1024, false);  // applyRelu=false
}

// ============================================================
// GPU Opt V3: im2col + cuBLAS GEMM + CUDA Streams Forward Pass
// ============================================================

// External stream-aware kernel declarations
extern void launchIm2colNCHWStream(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding,
    cudaStream_t stream);

extern void launchConvGemmForwardStream(
    const float* d_weights, const float* d_im2col, const float* d_bias,
    float* d_output,
    int batch, int outC, int inC_k_k, int outHW,
    cublasHandle_t handle, cudaStream_t stream,
    bool applyRelu);

extern void launchMaxPool2dNCHWStream(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW,
    int k, int stride, cudaStream_t stream);

extern void launchUpsample2dNCHWStream(
    const float* d_input, float* d_output,
    int batch, int channels, int inH, int inW,
    int scale, cudaStream_t stream);

void GPUAutoencoder::forwardOptV3() {
    // V3 uses two streams:
    // - m_stream_compute: for GEMM (cuBLAS)
    // - m_stream_transfer: for im2col, pooling, upsample
    // The cuBLAS handle is already set to m_stream_compute
    
    cudaStream_t s_comp = m_stream_compute;
    cudaStream_t s_aux = m_stream_transfer;
    
    // ========================
    // Layer 1: Conv+ReLU+Pool
    // ========================
    // im2col on aux stream
    launchIm2colNCHWStream(d_input, d_im2col_workspace,
                           m_batchSize, 3, 32, 32, 32, 32, 3, 1, 1, s_aux);
    // Wait for im2col to complete before GEMM reads it
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    
    // GEMM+bias+ReLU on compute stream
    launchConvGemmForwardStream(d_enc_conv1_w, d_im2col_workspace, d_enc_conv1_b,
                                d_enc_relu1_out,
                                m_batchSize, 256, 27, 1024,
                                m_cublas_handle, s_comp, true);
    // Wait for GEMM before pooling
    CHECK_CUDA(cudaStreamSynchronize(s_comp));
    
    // MaxPool on aux stream
    launchMaxPool2dNCHWStream(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                              m_batchSize, 256, 32, 32, 2, 2, s_aux);
    
    // ========================
    // Layer 2: Conv+ReLU+Pool
    // ========================
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    launchIm2colNCHWStream(d_enc_pool1_out, d_im2col_workspace,
                           m_batchSize, 256, 16, 16, 16, 16, 3, 1, 1, s_aux);
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    
    launchConvGemmForwardStream(d_enc_conv2_w, d_im2col_workspace, d_enc_conv2_b,
                                d_enc_relu2_out,
                                m_batchSize, 128, 2304, 256,
                                m_cublas_handle, s_comp, true);
    CHECK_CUDA(cudaStreamSynchronize(s_comp));
    
    launchMaxPool2dNCHWStream(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                              m_batchSize, 128, 16, 16, 2, 2, s_aux);
    
    // ========================
    // Layer 3: Conv+ReLU+Upsample
    // ========================
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    launchIm2colNCHWStream(d_enc_pool2_out, d_im2col_workspace,
                           m_batchSize, 128, 8, 8, 8, 8, 3, 1, 1, s_aux);
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    
    launchConvGemmForwardStream(d_dec_conv3_w, d_im2col_workspace, d_dec_conv3_b,
                                d_dec_relu3_out,
                                m_batchSize, 128, 1152, 64,
                                m_cublas_handle, s_comp, true);
    CHECK_CUDA(cudaStreamSynchronize(s_comp));
    
    launchUpsample2dNCHWStream(d_dec_relu3_out, d_dec_up1_out,
                               m_batchSize, 128, 8, 8, 2, s_aux);
    
    // ========================
    // Layer 4: Conv+ReLU+Upsample
    // ========================
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    launchIm2colNCHWStream(d_dec_up1_out, d_im2col_workspace,
                           m_batchSize, 128, 16, 16, 16, 16, 3, 1, 1, s_aux);
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    
    launchConvGemmForwardStream(d_dec_conv4_w, d_im2col_workspace, d_dec_conv4_b,
                                d_dec_relu4_out,
                                m_batchSize, 256, 1152, 256,
                                m_cublas_handle, s_comp, true);
    CHECK_CUDA(cudaStreamSynchronize(s_comp));
    
    launchUpsample2dNCHWStream(d_dec_relu4_out, d_dec_up2_out,
                               m_batchSize, 256, 16, 16, 2, s_aux);
    
    // ========================
    // Layer 5: Conv only (no ReLU)
    // ========================
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    launchIm2colNCHWStream(d_dec_up2_out, d_im2col_workspace,
                           m_batchSize, 256, 32, 32, 32, 32, 3, 1, 1, s_aux);
    CHECK_CUDA(cudaStreamSynchronize(s_aux));
    
    launchConvGemmForwardStream(d_dec_conv5_w, d_im2col_workspace, d_dec_conv5_b,
                                d_output,
                                m_batchSize, 3, 2304, 1024,
                                m_cublas_handle, s_comp, false);
    
    // Ensure all operations complete before returning
    CHECK_CUDA(cudaStreamSynchronize(s_comp));
}

