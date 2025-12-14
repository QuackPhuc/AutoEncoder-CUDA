#ifndef GPU_CORE_MEMORY_H
#define GPU_CORE_MEMORY_H

#include <cstddef>

// Layer buffer sizes for CIFAR-10 autoencoder
// Input: 32x32x3, Latent: 8x8x128 (8192 features)
struct LayerSizes {
    int batchSize;
    
    // Encoder layer 1: 32x32x3 -> 16x16x256
    static constexpr int ENC1_OUT_C = 256;
    static constexpr int ENC1_IN_C = 3;
    
    // Encoder layer 2: 16x16x256 -> 8x8x128
    static constexpr int ENC2_OUT_C = 128;
    static constexpr int ENC2_IN_C = 256;
    
    // Decoder layer 3: 8x8x128 -> 16x16x128
    static constexpr int DEC3_OUT_C = 128;
    static constexpr int DEC3_IN_C = 128;
    
    // Decoder layer 4: 16x16x128 -> 32x32x256
    static constexpr int DEC4_OUT_C = 256;
    static constexpr int DEC4_IN_C = 128;
    
    // Decoder layer 5: 32x32x256 -> 32x32x3
    static constexpr int DEC5_OUT_C = 3;
    static constexpr int DEC5_IN_C = 256;
    
    static constexpr int KERNEL_SIZE = 3;
};

// GPU memory buffer collection for autoencoder
struct GPUBuffers {
    // Input/Output
    float* d_input;
    float* d_output;
    float* d_target;
    
    // Encoder Layer 1
    float* d_enc_conv1_w;
    float* d_enc_conv1_b;
    float* d_enc_conv1_out;
    float* d_enc_relu1_out;
    float* d_enc_pool1_out;
    int* d_enc_pool1_indices;
    float* d_enc_conv1_grad_w;
    float* d_enc_conv1_grad_b;
    
    // Encoder Layer 2
    float* d_enc_conv2_w;
    float* d_enc_conv2_b;
    float* d_enc_conv2_out;
    float* d_enc_relu2_out;
    float* d_enc_pool2_out;
    int* d_enc_pool2_indices;
    float* d_enc_conv2_grad_w;
    float* d_enc_conv2_grad_b;
    
    // Decoder Layer 3
    float* d_dec_conv3_w;
    float* d_dec_conv3_b;
    float* d_dec_conv3_out;
    float* d_dec_relu3_out;
    float* d_dec_up1_out;
    float* d_dec_conv3_grad_w;
    float* d_dec_conv3_grad_b;
    
    // Decoder Layer 4
    float* d_dec_conv4_w;
    float* d_dec_conv4_b;
    float* d_dec_conv4_out;
    float* d_dec_relu4_out;
    float* d_dec_up2_out;
    float* d_dec_conv4_grad_w;
    float* d_dec_conv4_grad_b;
    
    // Decoder Layer 5
    float* d_dec_conv5_w;
    float* d_dec_conv5_b;
    float* d_dec_conv5_grad_w;
    float* d_dec_conv5_grad_b;
    
    // Loss
    float* d_loss;
    
    // Gradient buffers for backpropagation
    float* d_grad_output;
    float* d_grad_dec_up2;
    float* d_grad_dec_relu4;
    float* d_grad_dec_conv4;
    float* d_grad_dec_up1;
    float* d_grad_dec_relu3;
    float* d_grad_dec_conv3;
    float* d_grad_enc_pool2;
    float* d_grad_enc_relu2;
    float* d_grad_enc_conv2;
    float* d_grad_enc_pool1;
    float* d_grad_enc_relu1;
    float* d_grad_enc_conv1;
};

// Allocate all GPU memory for autoencoder
// Returns total bytes allocated
size_t allocateGPUBuffers(GPUBuffers& buffers, int batchSize);

// Free all GPU memory
void freeGPUBuffers(GPUBuffers& buffers);

// Initialize weight buffers with He initialization
void initializeWeights(GPUBuffers& buffers);

// Zero-initialize gradient buffers
void zeroGradients(GPUBuffers& buffers);

#endif // GPU_CORE_MEMORY_H
