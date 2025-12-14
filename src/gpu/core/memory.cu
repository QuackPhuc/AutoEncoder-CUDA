#include "memory.h"
#include "cuda_utils.h"
#include <iostream>
#include <random>
#include <vector>

size_t allocateGPUBuffers(GPUBuffers& buffers, int batchSize) {
    std::cout << "Allocating GPU memory..." << std::endl;
    
    size_t totalBytes = 0;
    
    // Input/Output buffers (batchSize, 32, 32, 3)
    size_t inputSize = batchSize * 32 * 32 * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&buffers.d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&buffers.d_output, inputSize));
    CHECK_CUDA(cudaMalloc(&buffers.d_target, inputSize));
    totalBytes += 3 * inputSize;
    
    // Encoder Layer 1: Conv2D (3 -> 256), output 32x32x256
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv1_w, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv1_b, 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv1_out, batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_relu1_out, batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_pool1_out, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_pool1_indices, batchSize * 16 * 16 * 256 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv1_grad_w, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv1_grad_b, 256 * sizeof(float)));
    totalBytes += (256*3*3*3 + 256) * 2 * sizeof(float);
    totalBytes += batchSize * (32*32*256*2 + 16*16*256) * sizeof(float);
    totalBytes += batchSize * 16*16*256 * sizeof(int);
    
    // Encoder Layer 2: Conv2D (256 -> 128), output 16x16x128
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv2_w, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv2_b, 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv2_out, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_relu2_out, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_pool2_out, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_pool2_indices, batchSize * 8 * 8 * 128 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv2_grad_w, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_enc_conv2_grad_b, 128 * sizeof(float)));
    totalBytes += (128*3*3*256 + 128) * 2 * sizeof(float);
    totalBytes += batchSize * (16*16*128*2 + 8*8*128) * sizeof(float);
    totalBytes += batchSize * 8*8*128 * sizeof(int);
    
    // Decoder Layer 3: Conv2D (128 -> 128), output 8x8x128
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv3_w, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv3_b, 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv3_out, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_relu3_out, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_up1_out, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv3_grad_w, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv3_grad_b, 128 * sizeof(float)));
    totalBytes += (128*3*3*128 + 128) * 2 * sizeof(float);
    totalBytes += batchSize * (8*8*128*2 + 16*16*128) * sizeof(float);
    
    // Decoder Layer 4: Conv2D (128 -> 256), output 16x16x256
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv4_w, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv4_b, 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv4_out, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_relu4_out, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_up2_out, batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv4_grad_w, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv4_grad_b, 256 * sizeof(float)));
    totalBytes += (256*3*3*128 + 256) * 2 * sizeof(float);
    totalBytes += batchSize * (16*16*256*2 + 32*32*256) * sizeof(float);
    
    // Decoder Layer 5: Conv2D (256 -> 3), output 32x32x3
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv5_w, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv5_b, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv5_grad_w, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_dec_conv5_grad_b, 3 * sizeof(float)));
    totalBytes += (3*3*3*256 + 3) * 2 * sizeof(float);
    
    // Loss
    CHECK_CUDA(cudaMalloc(&buffers.d_loss, sizeof(float)));
    totalBytes += sizeof(float);
    
    // Gradient buffers for backpropagation
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_output, batchSize * 32 * 32 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_up2, batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_relu4, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_conv4, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_up1, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_relu3, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_dec_conv3, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_pool2, batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_relu2, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_conv2, batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_pool1, batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_relu1, batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&buffers.d_grad_enc_conv1, batchSize * 32 * 32 * 256 * sizeof(float)));
    
    std::cout << "  Total GPU memory: " << (totalBytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    return totalBytes;
}

void freeGPUBuffers(GPUBuffers& buffers) {
    // Ensure all async operations complete and clear any pending errors
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    // Helper macro to free and nullify pointer (defensive programming)
    #define FREE_AND_NULL(ptr) do { CHECK_CUDA_CLEANUP(cudaFree(ptr)); ptr = nullptr; } while(0)
    
    // Input/Output
    FREE_AND_NULL(buffers.d_input);
    FREE_AND_NULL(buffers.d_output);
    FREE_AND_NULL(buffers.d_target);
    
    // Encoder Layer 1
    FREE_AND_NULL(buffers.d_enc_conv1_w);
    FREE_AND_NULL(buffers.d_enc_conv1_b);
    FREE_AND_NULL(buffers.d_enc_conv1_out);
    FREE_AND_NULL(buffers.d_enc_relu1_out);
    FREE_AND_NULL(buffers.d_enc_pool1_out);
    FREE_AND_NULL(buffers.d_enc_pool1_indices);
    FREE_AND_NULL(buffers.d_enc_conv1_grad_w);
    FREE_AND_NULL(buffers.d_enc_conv1_grad_b);
    
    // Encoder Layer 2
    FREE_AND_NULL(buffers.d_enc_conv2_w);
    FREE_AND_NULL(buffers.d_enc_conv2_b);
    FREE_AND_NULL(buffers.d_enc_conv2_out);
    FREE_AND_NULL(buffers.d_enc_relu2_out);
    FREE_AND_NULL(buffers.d_enc_pool2_out);
    FREE_AND_NULL(buffers.d_enc_pool2_indices);
    FREE_AND_NULL(buffers.d_enc_conv2_grad_w);
    FREE_AND_NULL(buffers.d_enc_conv2_grad_b);
    
    // Decoder Layer 3
    FREE_AND_NULL(buffers.d_dec_conv3_w);
    FREE_AND_NULL(buffers.d_dec_conv3_b);
    FREE_AND_NULL(buffers.d_dec_conv3_out);
    FREE_AND_NULL(buffers.d_dec_relu3_out);
    FREE_AND_NULL(buffers.d_dec_up1_out);
    FREE_AND_NULL(buffers.d_dec_conv3_grad_w);
    FREE_AND_NULL(buffers.d_dec_conv3_grad_b);
    
    // Decoder Layer 4
    FREE_AND_NULL(buffers.d_dec_conv4_w);
    FREE_AND_NULL(buffers.d_dec_conv4_b);
    FREE_AND_NULL(buffers.d_dec_conv4_out);
    FREE_AND_NULL(buffers.d_dec_relu4_out);
    FREE_AND_NULL(buffers.d_dec_up2_out);
    FREE_AND_NULL(buffers.d_dec_conv4_grad_w);
    FREE_AND_NULL(buffers.d_dec_conv4_grad_b);
    
    // Decoder Layer 5
    FREE_AND_NULL(buffers.d_dec_conv5_w);
    FREE_AND_NULL(buffers.d_dec_conv5_b);
    FREE_AND_NULL(buffers.d_dec_conv5_grad_w);
    FREE_AND_NULL(buffers.d_dec_conv5_grad_b);
    
    // Loss
    FREE_AND_NULL(buffers.d_loss);
    
    // Gradient buffers
    FREE_AND_NULL(buffers.d_grad_output);
    FREE_AND_NULL(buffers.d_grad_dec_up2);
    FREE_AND_NULL(buffers.d_grad_dec_relu4);
    FREE_AND_NULL(buffers.d_grad_dec_conv4);
    FREE_AND_NULL(buffers.d_grad_dec_up1);
    FREE_AND_NULL(buffers.d_grad_dec_relu3);
    FREE_AND_NULL(buffers.d_grad_dec_conv3);
    FREE_AND_NULL(buffers.d_grad_enc_pool2);
    FREE_AND_NULL(buffers.d_grad_enc_relu2);
    FREE_AND_NULL(buffers.d_grad_enc_conv2);
    FREE_AND_NULL(buffers.d_grad_enc_pool1);
    FREE_AND_NULL(buffers.d_grad_enc_relu1);
    FREE_AND_NULL(buffers.d_grad_enc_conv1);
    
    #undef FREE_AND_NULL
}

// Helper: Initialize conv layer weights with He initialization
static void initConvWeights(float* d_w, float* d_b, int outChannels, int kernelSize, int inChannels) {
    float stddev = std::sqrt(2.0f / (kernelSize * kernelSize * inChannels));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, stddev);
    
    int weightSize = outChannels * kernelSize * kernelSize * inChannels;
    std::vector<float> h_w(weightSize);
    std::vector<float> h_b(outChannels, 0.0f);
    
    for (int i = 0; i < weightSize; i++) {
        h_w[i] = dist(gen);
    }
    
    CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), outChannels * sizeof(float), cudaMemcpyHostToDevice));
}

void initializeWeights(GPUBuffers& buffers) {
    std::cout << "Initializing weights..." << std::endl;
    
    initConvWeights(buffers.d_enc_conv1_w, buffers.d_enc_conv1_b, 256, 3, 3);
    initConvWeights(buffers.d_enc_conv2_w, buffers.d_enc_conv2_b, 128, 3, 256);
    initConvWeights(buffers.d_dec_conv3_w, buffers.d_dec_conv3_b, 128, 3, 128);
    initConvWeights(buffers.d_dec_conv4_w, buffers.d_dec_conv4_b, 256, 3, 128);
    initConvWeights(buffers.d_dec_conv5_w, buffers.d_dec_conv5_b, 3, 3, 256);
}

void zeroGradients(GPUBuffers& buffers) {
    CHECK_CUDA(cudaMemset(buffers.d_enc_conv1_grad_w, 0, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_enc_conv1_grad_b, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_enc_conv2_grad_w, 0, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_enc_conv2_grad_b, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv3_grad_w, 0, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv3_grad_b, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv4_grad_w, 0, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv4_grad_b, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv5_grad_w, 0, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(buffers.d_dec_conv5_grad_b, 0, 3 * sizeof(float)));
}
