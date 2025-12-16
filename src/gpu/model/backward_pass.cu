#include "autoencoder.h"
#include "gpu/core/cuda_utils.h"
#include "config/gpu_config.h"
#include <cuda_runtime.h>

// External kernel declarations
extern __global__ void mseLossKernel(const float* pred, const float* target, float* loss, int size);
extern __global__ void mseLossGradKernel(const float* pred, const float* target, float* gradOutput, int size);
extern __global__ void sgdUpdateKernel(float* weights, const float* gradients, float learningRate, int size);

extern __global__ void conv2dBackwardInputKernel(
    const float* gradOutput, const float* weights, float* gradInput,
    int batch, int inH, int inW, int inC, int outH, int outW, int outC,
    int kernelSize, int padding, int stride);
extern __global__ void conv2dBackwardWeightsKernel(
    const float* gradOutput, const float* input, float* gradWeights,
    int batch, int inH, int inW, int inC, int outH, int outW, int outC,
    int kernelSize, int padding, int stride);
extern __global__ void conv2dBackwardBiasKernel(
    const float* gradOutput, float* gradBias, int batch, int outH, int outW, int outC);

extern __global__ void reluBackwardKernel(
    const float* gradOutput, const float* input, float* gradInput, int size);

extern __global__ void maxpool2dBackwardZeroKernel(
    float* gradInput, int batch, int inH, int inW, int channels);
extern __global__ void maxpool2dBackwardScatterKernel(
    const float* gradOutput, const int* indices, float* gradInput,
    int batch, int inH, int inW, int channels);

extern __global__ void upsample2dBackwardKernel(
    const float* gradOutput, float* gradInput, int batch, int inH, int inW, int channels);

// Kernel wrappers

void GPUAutoencoder::computeMSELoss(const float* d_pred, const float* d_target, float* d_loss, int size) {
    CHECK_CUDA(cudaMemset(d_loss, 0, sizeof(float)));
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    mseLossKernel<<<gridSize, blockSize>>>(d_pred, d_target, d_loss, size);
    
    float h_loss;
    CHECK_CUDA(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    h_loss /= size;
    CHECK_CUDA(cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::sgdUpdate(float* d_weights, const float* d_gradients, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sgdUpdateKernel<<<gridSize, blockSize>>>(d_weights, d_gradients, m_learningRate, size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::conv2dBackward(
    const float* d_gradOut, const float* d_in, const float* d_w,
    float* d_gradIn, float* d_gradW, float* d_gradB,
    int batch, int inH, int inW, int inC, int outH, int outW, int outC
) {
    int blockSize = 256;
    
    // Backward w.r.t. input
    int inputThreads = batch * inH * inW * inC;
    int gridInput = (inputThreads + blockSize - 1) / blockSize;
    conv2dBackwardInputKernel<<<gridInput, blockSize>>>(
        d_gradOut, d_w, d_gradIn,
        batch, inH, inW, inC, outH, outW, outC, 3, 1, 1);
    
    // Backward w.r.t. weights
    int weightThreads = outC * 3 * 3 * inC;
    int gridWeights = (weightThreads + blockSize - 1) / blockSize;
    conv2dBackwardWeightsKernel<<<gridWeights, blockSize>>>(
        d_gradOut, d_in, d_gradW,
        batch, inH, inW, inC, outH, outW, outC, 3, 1, 1);
    
    // Backward w.r.t. bias
    int gridBias = (outC + blockSize - 1) / blockSize;
    conv2dBackwardBiasKernel<<<gridBias, blockSize>>>(d_gradOut, d_gradB, batch, outH, outW, outC);
    
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::reluBackward(const float* d_gradOut, const float* d_in, float* d_gradIn, int size) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    reluBackwardKernel<<<gridSize, blockSize>>>(d_gradOut, d_in, d_gradIn, size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::maxpool2dBackward(
    const float* d_gradOut, const int* d_indices, float* d_gradIn,
    int batch, int inH, int inW, int channels
) {
    int blockSize = 256;
    
    int inputThreads = batch * inH * inW * channels;
    int gridInput = (inputThreads + blockSize - 1) / blockSize;
    maxpool2dBackwardZeroKernel<<<gridInput, blockSize>>>(d_gradIn, batch, inH, inW, channels);
    
    int outH = inH / 2;
    int outW = inW / 2;
    int outputThreads = batch * outH * outW * channels;
    int gridOutput = (outputThreads + blockSize - 1) / blockSize;
    maxpool2dBackwardScatterKernel<<<gridOutput, blockSize>>>(d_gradOut, d_indices, d_gradIn, batch, inH, inW, channels);
    
    CHECK_CUDA(cudaGetLastError());
}

void GPUAutoencoder::upsample2dBackward(const float* d_gradOut, float* d_gradIn, int batch, int inH, int inW, int channels) {
    int totalThreads = batch * inH * inW * channels;
    int blockSize = 256;
    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    upsample2dBackwardKernel<<<gridSize, blockSize>>>(d_gradOut, d_gradIn, batch, inH, inW, channels);
    CHECK_CUDA(cudaGetLastError());
}

// Basic backward pass
void GPUAutoencoder::backwardBasic() {
    int blockSize = 256;
    int outputSize = m_batchSize * 32 * 32 * 3;
    
    // MSE gradient
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    mseLossGradKernel<<<gridSize, blockSize>>>(d_output, d_target, d_grad_output, outputSize);
    CHECK_CUDA(cudaGetLastError());
    
    // Decoder backward
    conv2dBackward(d_grad_output, d_dec_up2_out, d_dec_conv5_w,
                   d_grad_dec_up2, d_dec_conv5_grad_w, d_dec_conv5_grad_b,
                   m_batchSize, 32, 32, 256, 32, 32, 3);
    
    upsample2dBackward(d_grad_dec_up2, d_grad_dec_relu4, m_batchSize, 16, 16, 256);
    reluBackward(d_grad_dec_relu4, d_dec_conv4_out, d_grad_dec_conv4, m_batchSize * 16 * 16 * 256);
    
    conv2dBackward(d_grad_dec_conv4, d_dec_up1_out, d_dec_conv4_w,
                   d_grad_dec_up1, d_dec_conv4_grad_w, d_dec_conv4_grad_b,
                   m_batchSize, 16, 16, 128, 16, 16, 256);
    
    upsample2dBackward(d_grad_dec_up1, d_grad_dec_relu3, m_batchSize, 8, 8, 128);
    reluBackward(d_grad_dec_relu3, d_dec_conv3_out, d_grad_dec_conv3, m_batchSize * 8 * 8 * 128);
    
    conv2dBackward(d_grad_dec_conv3, d_enc_pool2_out, d_dec_conv3_w,
                   d_grad_enc_pool2, d_dec_conv3_grad_w, d_dec_conv3_grad_b,
                   m_batchSize, 8, 8, 128, 8, 8, 128);
    
    // Encoder backward
    maxpool2dBackward(d_grad_enc_pool2, d_enc_pool2_indices, d_grad_enc_relu2, m_batchSize, 16, 16, 128);
    reluBackward(d_grad_enc_relu2, d_enc_conv2_out, d_grad_enc_conv2, m_batchSize * 16 * 16 * 128);
    
    conv2dBackward(d_grad_enc_conv2, d_enc_pool1_out, d_enc_conv2_w,
                   d_grad_enc_pool1, d_enc_conv2_grad_w, d_enc_conv2_grad_b,
                   m_batchSize, 16, 16, 256, 16, 16, 128);
    
    maxpool2dBackward(d_grad_enc_pool1, d_enc_pool1_indices, d_grad_enc_relu1, m_batchSize, 32, 32, 256);
    reluBackward(d_grad_enc_relu1, d_enc_conv1_out, d_grad_enc_conv1, m_batchSize * 32 * 32 * 256);
    
    conv2dBackward(d_grad_enc_conv1, d_input, d_enc_conv1_w,
                   d_target, d_enc_conv1_grad_w, d_enc_conv1_grad_b,
                   m_batchSize, 32, 32, 3, 32, 32, 256);
}

void GPUAutoencoder::updateWeights() {
    sgdUpdate(d_enc_conv1_w, d_enc_conv1_grad_w, 256 * 3 * 3 * 3);
    sgdUpdate(d_enc_conv1_b, d_enc_conv1_grad_b, 256);
    
    sgdUpdate(d_enc_conv2_w, d_enc_conv2_grad_w, 128 * 3 * 3 * 256);
    sgdUpdate(d_enc_conv2_b, d_enc_conv2_grad_b, 128);
    
    sgdUpdate(d_dec_conv3_w, d_dec_conv3_grad_w, 128 * 3 * 3 * 128);
    sgdUpdate(d_dec_conv3_b, d_dec_conv3_grad_b, 128);
    
    sgdUpdate(d_dec_conv4_w, d_dec_conv4_grad_w, 256 * 3 * 3 * 128);
    sgdUpdate(d_dec_conv4_b, d_dec_conv4_grad_b, 256);
    
    sgdUpdate(d_dec_conv5_w, d_dec_conv5_grad_w, 3 * 3 * 3 * 256);
    sgdUpdate(d_dec_conv5_b, d_dec_conv5_grad_b, 3);
}

// External NCHW backward kernel declarations
extern void launchConv2dBackwardInputNCHW(
    const float* d_gradOutput, const float* d_weights, float* d_gradInput,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kernelSize, int padding, int stride);
extern void launchConv2dBackwardWeightsNCHW(
    const float* d_input, const float* d_gradOutput, float* d_gradWeights,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kernelSize, int padding, int stride);
extern void launchConv2dBackwardBiasNCHW(
    const float* d_gradOutput, float* d_gradBias,
    int batch, int outC, int outH, int outW);
extern void launchMaxPool2dBackwardNCHW(
    const float* d_gradOutput, const int* d_indices, float* d_gradInput,
    int batch, int channels, int inH, int inW, int k, int stride);
extern void launchUpsample2dBackwardNCHW(
    const float* d_gradOutput, float* d_gradInput,
    int batch, int channels, int inH, int inW, int scale);

// Optimized backward pass v1 (NCHW layout)
void GPUAutoencoder::backwardOptV1() {
    int blockSize = 256;
    int outputSize = m_batchSize * 32 * 32 * 3;
    
    // MSE gradient (same for any layout, element-wise)
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    mseLossGradKernel<<<gridSize, blockSize>>>(d_output, d_target, d_grad_output, outputSize);
    CHECK_CUDA(cudaGetLastError());
    
    // Decoder Layer 5 backward: Conv only
    launchConv2dBackwardInputNCHW(d_grad_output, d_dec_conv5_w, d_grad_dec_up2,
                                   m_batchSize, 256, 32, 32, 3, 32, 32, 3, 1, 1);
    launchConv2dBackwardWeightsNCHW(d_dec_up2_out, d_grad_output, d_dec_conv5_grad_w,
                                     m_batchSize, 256, 32, 32, 3, 32, 32, 3, 1, 1);
    launchConv2dBackwardBiasNCHW(d_grad_output, d_dec_conv5_grad_b, m_batchSize, 3, 32, 32);
    
    // Decoder Layer 4 backward: Upsample + ReLU + Conv
    launchUpsample2dBackwardNCHW(d_grad_dec_up2, d_grad_dec_relu4, m_batchSize, 256, 16, 16, 2);
    
    // ReLU backward
    int relu4Size = m_batchSize * 256 * 16 * 16;
    reluBackwardKernel<<<(relu4Size + 255) / 256, 256>>>(
        d_grad_dec_relu4, d_dec_relu4_out, d_grad_dec_conv4, relu4Size);
    
    launchConv2dBackwardInputNCHW(d_grad_dec_conv4, d_dec_conv4_w, d_grad_dec_up1,
                                   m_batchSize, 128, 16, 16, 256, 16, 16, 3, 1, 1);
    launchConv2dBackwardWeightsNCHW(d_dec_up1_out, d_grad_dec_conv4, d_dec_conv4_grad_w,
                                     m_batchSize, 128, 16, 16, 256, 16, 16, 3, 1, 1);
    launchConv2dBackwardBiasNCHW(d_grad_dec_conv4, d_dec_conv4_grad_b, m_batchSize, 256, 16, 16);
    
    // Decoder Layer 3 backward: Upsample + ReLU + Conv
    launchUpsample2dBackwardNCHW(d_grad_dec_up1, d_grad_dec_relu3, m_batchSize, 128, 8, 8, 2);
    
    int relu3Size = m_batchSize * 128 * 8 * 8;
    reluBackwardKernel<<<(relu3Size + 255) / 256, 256>>>(
        d_grad_dec_relu3, d_dec_relu3_out, d_grad_dec_conv3, relu3Size);
    
    launchConv2dBackwardInputNCHW(d_grad_dec_conv3, d_dec_conv3_w, d_grad_enc_pool2,
                                   m_batchSize, 128, 8, 8, 128, 8, 8, 3, 1, 1);
    launchConv2dBackwardWeightsNCHW(d_enc_pool2_out, d_grad_dec_conv3, d_dec_conv3_grad_w,
                                     m_batchSize, 128, 8, 8, 128, 8, 8, 3, 1, 1);
    launchConv2dBackwardBiasNCHW(d_grad_dec_conv3, d_dec_conv3_grad_b, m_batchSize, 128, 8, 8);
    
    // Encoder Layer 2 backward: MaxPool + ReLU + Conv
    launchMaxPool2dBackwardNCHW(d_grad_enc_pool2, d_enc_pool2_indices, d_grad_enc_relu2,
                                 m_batchSize, 128, 16, 16, 2, 2);
    
    int relu2Size = m_batchSize * 128 * 16 * 16;
    reluBackwardKernel<<<(relu2Size + 255) / 256, 256>>>(
        d_grad_enc_relu2, d_enc_relu2_out, d_grad_enc_conv2, relu2Size);
    
    launchConv2dBackwardInputNCHW(d_grad_enc_conv2, d_enc_conv2_w, d_grad_enc_pool1,
                                   m_batchSize, 256, 16, 16, 128, 16, 16, 3, 1, 1);
    launchConv2dBackwardWeightsNCHW(d_enc_pool1_out, d_grad_enc_conv2, d_enc_conv2_grad_w,
                                     m_batchSize, 256, 16, 16, 128, 16, 16, 3, 1, 1);
    launchConv2dBackwardBiasNCHW(d_grad_enc_conv2, d_enc_conv2_grad_b, m_batchSize, 128, 16, 16);
    
    // Encoder Layer 1 backward: MaxPool + ReLU + Conv
    launchMaxPool2dBackwardNCHW(d_grad_enc_pool1, d_enc_pool1_indices, d_grad_enc_relu1,
                                 m_batchSize, 256, 32, 32, 2, 2);
    
    int relu1Size = m_batchSize * 256 * 32 * 32;
    reluBackwardKernel<<<(relu1Size + 255) / 256, 256>>>(
        d_grad_enc_relu1, d_enc_relu1_out, d_grad_enc_conv1, relu1Size);
    
    // Note: d_target is not used for input gradient, we skip computing grad_input for first layer
    launchConv2dBackwardWeightsNCHW(d_input, d_grad_enc_conv1, d_enc_conv1_grad_w,
                                     m_batchSize, 3, 32, 32, 256, 32, 32, 3, 1, 1);
    launchConv2dBackwardBiasNCHW(d_grad_enc_conv1, d_enc_conv1_grad_b, m_batchSize, 256, 32, 32);
}
// External V2 backward kernel declarations (im2col + GEMM)
extern void launchIm2colNCHW(
    const float* d_input, float* d_col,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);
    
extern void launchCol2imNCHW(
    const float* d_col, float* d_gradInput,
    int batch, int inC, int inH, int inW,
    int outH, int outW,
    int kernelSize, int stride, int padding);

extern void launchConvGemmBackwardInput(
    const float* d_weights, const float* d_gradOutput, float* d_col,
    int batch, int outC, int inC_k_k, int outHW);

extern void launchConvGemmBackwardWeights(
    const float* d_gradOutput, const float* d_im2col, float* d_gradWeights,
    int batch, int outC, int inC_k_k, int outHW);

// Optimized backward pass v2 (im2col + cuBLAS GEMM)
void GPUAutoencoder::backwardOptV2() {
    int blockSize = 256;
    int outputSize = m_batchSize * 32 * 32 * 3;
    
    // MSE gradient
    int gridSize = (outputSize + blockSize - 1) / blockSize;
    mseLossGradKernel<<<gridSize, blockSize>>>(d_output, d_target, d_grad_output, outputSize);
    CHECK_CUDA(cudaGetLastError());
    
    // Layer 5 backward: Conv only (no ReLU)
    // Backward input: dX = col2im(W^T × dY)
    launchConvGemmBackwardInput(d_dec_conv5_w, d_grad_output, d_im2col_workspace,
                                 m_batchSize, 3, 2304, 1024);
    launchCol2imNCHW(d_im2col_workspace, d_grad_dec_up2,
                     m_batchSize, 256, 32, 32, 32, 32, 3, 1, 1);
    // Backward weights: dW = dY × im2col(X)^T
    launchIm2colNCHW(d_dec_up2_out, d_im2col_workspace,
                     m_batchSize, 256, 32, 32, 32, 32, 3, 1, 1);
    launchConvGemmBackwardWeights(d_grad_output, d_im2col_workspace, d_dec_conv5_grad_w,
                                   m_batchSize, 3, 2304, 1024);
    launchConv2dBackwardBiasNCHW(d_grad_output, d_dec_conv5_grad_b, m_batchSize, 3, 32, 32);
    
    // Layer 4 backward: Upsample + ReLU + Conv
    launchUpsample2dBackwardNCHW(d_grad_dec_up2, d_grad_dec_relu4, m_batchSize, 256, 16, 16, 2);
    
    int relu4Size = m_batchSize * 256 * 16 * 16;
    reluBackwardKernel<<<(relu4Size + 255) / 256, 256>>>(
        d_grad_dec_relu4, d_dec_relu4_out, d_grad_dec_conv4, relu4Size);
    
    launchConvGemmBackwardInput(d_dec_conv4_w, d_grad_dec_conv4, d_im2col_workspace,
                                 m_batchSize, 256, 1152, 256);
    launchCol2imNCHW(d_im2col_workspace, d_grad_dec_up1,
                     m_batchSize, 128, 16, 16, 16, 16, 3, 1, 1);
    launchIm2colNCHW(d_dec_up1_out, d_im2col_workspace,
                     m_batchSize, 128, 16, 16, 16, 16, 3, 1, 1);
    launchConvGemmBackwardWeights(d_grad_dec_conv4, d_im2col_workspace, d_dec_conv4_grad_w,
                                   m_batchSize, 256, 1152, 256);
    launchConv2dBackwardBiasNCHW(d_grad_dec_conv4, d_dec_conv4_grad_b, m_batchSize, 256, 16, 16);
    
    // Layer 3 backward: Upsample + ReLU + Conv
    launchUpsample2dBackwardNCHW(d_grad_dec_up1, d_grad_dec_relu3, m_batchSize, 128, 8, 8, 2);
    
    int relu3Size = m_batchSize * 128 * 8 * 8;
    reluBackwardKernel<<<(relu3Size + 255) / 256, 256>>>(
        d_grad_dec_relu3, d_dec_relu3_out, d_grad_dec_conv3, relu3Size);
    
    launchConvGemmBackwardInput(d_dec_conv3_w, d_grad_dec_conv3, d_im2col_workspace,
                                 m_batchSize, 128, 1152, 64);
    launchCol2imNCHW(d_im2col_workspace, d_grad_enc_pool2,
                     m_batchSize, 128, 8, 8, 8, 8, 3, 1, 1);
    launchIm2colNCHW(d_enc_pool2_out, d_im2col_workspace,
                     m_batchSize, 128, 8, 8, 8, 8, 3, 1, 1);
    launchConvGemmBackwardWeights(d_grad_dec_conv3, d_im2col_workspace, d_dec_conv3_grad_w,
                                   m_batchSize, 128, 1152, 64);
    launchConv2dBackwardBiasNCHW(d_grad_dec_conv3, d_dec_conv3_grad_b, m_batchSize, 128, 8, 8);
    
    // Layer 2 backward: MaxPool + ReLU + Conv
    launchMaxPool2dBackwardNCHW(d_grad_enc_pool2, d_enc_pool2_indices, d_grad_enc_relu2,
                                 m_batchSize, 128, 16, 16, 2, 2);
    
    int relu2Size = m_batchSize * 128 * 16 * 16;
    reluBackwardKernel<<<(relu2Size + 255) / 256, 256>>>(
        d_grad_enc_relu2, d_enc_relu2_out, d_grad_enc_conv2, relu2Size);
    
    launchConvGemmBackwardInput(d_enc_conv2_w, d_grad_enc_conv2, d_im2col_workspace,
                                 m_batchSize, 128, 2304, 256);
    launchCol2imNCHW(d_im2col_workspace, d_grad_enc_pool1,
                     m_batchSize, 256, 16, 16, 16, 16, 3, 1, 1);
    launchIm2colNCHW(d_enc_pool1_out, d_im2col_workspace,
                     m_batchSize, 256, 16, 16, 16, 16, 3, 1, 1);
    launchConvGemmBackwardWeights(d_grad_enc_conv2, d_im2col_workspace, d_enc_conv2_grad_w,
                                   m_batchSize, 128, 2304, 256);
    launchConv2dBackwardBiasNCHW(d_grad_enc_conv2, d_enc_conv2_grad_b, m_batchSize, 128, 16, 16);
    
    // Layer 1 backward: MaxPool + ReLU + Conv (no input gradient needed)
    launchMaxPool2dBackwardNCHW(d_grad_enc_pool1, d_enc_pool1_indices, d_grad_enc_relu1,
                                 m_batchSize, 256, 32, 32, 2, 2);
    
    int relu1Size = m_batchSize * 256 * 32 * 32;
    reluBackwardKernel<<<(relu1Size + 255) / 256, 256>>>(
        d_grad_enc_relu1, d_enc_relu1_out, d_grad_enc_conv1, relu1Size);
    
    // Only backward weights for first layer (no input gradient needed)
    launchIm2colNCHW(d_input, d_im2col_workspace,
                     m_batchSize, 3, 32, 32, 32, 32, 3, 1, 1);
    launchConvGemmBackwardWeights(d_grad_enc_conv1, d_im2col_workspace, d_enc_conv1_grad_w,
                                   m_batchSize, 256, 27, 1024);
    launchConv2dBackwardBiasNCHW(d_grad_enc_conv1, d_enc_conv1_grad_b, m_batchSize, 256, 32, 32);
}
