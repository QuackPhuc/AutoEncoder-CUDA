#include "autoencoder.h"
#include "gpu/core/cuda_utils.h"
#include "config/gpu_config.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <iostream>

GPUAutoencoder::GPUAutoencoder(int batchSize, float learningRate)
    : m_batchSize(batchSize), m_learningRate(learningRate), m_lastLoss(0.0f),
      m_stream_compute(nullptr), m_stream_transfer(nullptr), 
      m_cublas_handle(nullptr), m_streams_initialized(false) {
    
    printGPUInfo();
    allocateMemory();
    initializeWeights();
    
    // Initialize CUDA streams for V3 mode
    if (GPUConfig::getInstance().getVersion() == GPUVersion::GPU_OPT_V3) {
        CHECK_CUDA(cudaStreamCreate(&m_stream_compute));
        CHECK_CUDA(cudaStreamCreate(&m_stream_transfer));
        
        cublasStatus_t status = cublasCreate(&m_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS handle creation failed for V3" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // Associate cuBLAS handle with compute stream
        cublasSetStream(m_cublas_handle, m_stream_compute);
        m_streams_initialized = true;
        std::cout << "CUDA Streams initialized for V3 mode" << std::endl;
    }
}

GPUAutoencoder::~GPUAutoencoder() {
    // Clean up streams and cuBLAS handle first
    if (m_streams_initialized) {
        if (m_cublas_handle) {
            cublasDestroy(m_cublas_handle);
        }
        if (m_stream_compute) {
            cudaStreamDestroy(m_stream_compute);
        }
        if (m_stream_transfer) {
            cudaStreamDestroy(m_stream_transfer);
        }
    }
    freeMemory();
}

void GPUAutoencoder::allocateMemory() {
    size_t totalBytes = 0;
    
    // Input/Output buffers (NCHW layout from loader)
    size_t inputSize = m_batchSize * 32 * 32 * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA(cudaMalloc(&d_output, inputSize));
    CHECK_CUDA(cudaMalloc(&d_target, inputSize));
    
    // Allocate NHWC temporary buffers
    CHECK_CUDA(cudaMalloc(&d_input_nhwc, inputSize));
    CHECK_CUDA(cudaMalloc(&d_output_nhwc, inputSize));
    
    totalBytes += 3 * inputSize;
    
    // Encoder Layer 1
    CHECK_CUDA(cudaMalloc(&d_enc_conv1_w, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv1_b, 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv1_out, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_relu1_out, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_pool1_out, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_pool1_indices, m_batchSize * 16 * 16 * 256 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv1_grad_w, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv1_grad_b, 256 * sizeof(float)));
    
    // Encoder Layer 2
    CHECK_CUDA(cudaMalloc(&d_enc_conv2_w, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv2_b, 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv2_out, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_relu2_out, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_pool2_out, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_pool2_indices, m_batchSize * 8 * 8 * 128 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv2_grad_w, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_enc_conv2_grad_b, 128 * sizeof(float)));
    
    // Decoder Layer 3
    CHECK_CUDA(cudaMalloc(&d_dec_conv3_w, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv3_b, 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv3_out, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_relu3_out, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_up1_out, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv3_grad_w, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv3_grad_b, 128 * sizeof(float)));
    
    // Decoder Layer 4
    CHECK_CUDA(cudaMalloc(&d_dec_conv4_w, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv4_b, 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv4_out, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_relu4_out, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_up2_out, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv4_grad_w, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv4_grad_b, 256 * sizeof(float)));
    
    // Decoder Layer 5
    CHECK_CUDA(cudaMalloc(&d_dec_conv5_w, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv5_b, 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv5_grad_w, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dec_conv5_grad_b, 3 * sizeof(float)));
    
    // Loss
    CHECK_CUDA(cudaMalloc(&d_loss, sizeof(float)));
    
    // Gradient buffers
    CHECK_CUDA(cudaMalloc(&d_grad_output, m_batchSize * 32 * 32 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_up2, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_relu4, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_conv4, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_up1, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_relu3, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_dec_conv3, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_pool2, m_batchSize * 8 * 8 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_relu2, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_conv2, m_batchSize * 16 * 16 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_pool1, m_batchSize * 16 * 16 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_relu1, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grad_enc_conv1, m_batchSize * 32 * 32 * 256 * sizeof(float)));
    
    // im2col workspace for V4 GEMM convolution
    // Max size: batch * 256 * 3 * 3 * 16 * 16 = batch * 2304 * 256
    // Largest layer: 256 channels * 9 kernel * 32*32 output = 2,359,296 per batch
    m_im2col_workspace_size = static_cast<size_t>(m_batchSize) * 256 * 9 * 32 * 32 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_im2col_workspace, m_im2col_workspace_size));
}

void GPUAutoencoder::freeMemory() {
    // Ensure all async operations complete and clear any pending errors
    cudaDeviceSynchronize();
    cudaGetLastError();
    
    CHECK_CUDA_CLEANUP(cudaFree(d_input));
    CHECK_CUDA_CLEANUP(cudaFree(d_output));
    CHECK_CUDA_CLEANUP(cudaFree(d_target));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_input_nhwc));
    CHECK_CUDA_CLEANUP(cudaFree(d_output_nhwc));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv1_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv1_b));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv1_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_relu1_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_pool1_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_pool1_indices));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv1_grad_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv1_grad_b));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv2_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv2_b));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv2_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_relu2_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_pool2_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_pool2_indices));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv2_grad_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_enc_conv2_grad_b));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv3_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv3_b));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv3_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_relu3_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_up1_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv3_grad_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv3_grad_b));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv4_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv4_b));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv4_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_relu4_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_up2_out));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv4_grad_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv4_grad_b));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv5_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv5_b));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv5_grad_w));
    CHECK_CUDA_CLEANUP(cudaFree(d_dec_conv5_grad_b));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_loss));
    
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_output));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_up2));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_relu4));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_conv4));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_up1));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_relu3));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_dec_conv3));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_pool2));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_relu2));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_conv2));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_pool1));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_relu1));
    CHECK_CUDA_CLEANUP(cudaFree(d_grad_enc_conv1));
    
    // Free im2col workspace
    if (d_im2col_workspace) {
        CHECK_CUDA_CLEANUP(cudaFree(d_im2col_workspace));
    }
}


void GPUAutoencoder::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    auto initConvWeights = [&](float* d_w, float* d_b, int outChannels, int kernelSize, int inChannels) {
        float std = std::sqrt(2.0f / (kernelSize * kernelSize * inChannels));
        std::normal_distribution<float> dist(0.0f, std);
        
        int weightSize = outChannels * kernelSize * kernelSize * inChannels;
        std::vector<float> h_w(weightSize);
        std::vector<float> h_b(outChannels, 0.0f);
        
        for (int i = 0; i < weightSize; i++) {
            h_w[i] = dist(gen);
        }
        
        CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), outChannels * sizeof(float), cudaMemcpyHostToDevice));
    };
    
    initConvWeights(d_enc_conv1_w, d_enc_conv1_b, 256, 3, 3);
    initConvWeights(d_enc_conv2_w, d_enc_conv2_b, 128, 3, 256);
    initConvWeights(d_dec_conv3_w, d_dec_conv3_b, 128, 3, 128);
    initConvWeights(d_dec_conv4_w, d_dec_conv4_b, 256, 3, 128);
    initConvWeights(d_dec_conv5_w, d_dec_conv5_b, 3, 3, 256);
    
    // Zero-initialize gradients
    CHECK_CUDA(cudaMemset(d_enc_conv1_grad_w, 0, 256 * 3 * 3 * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_enc_conv1_grad_b, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_enc_conv2_grad_w, 0, 128 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_enc_conv2_grad_b, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv3_grad_w, 0, 128 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv3_grad_b, 0, 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv4_grad_w, 0, 256 * 3 * 3 * 128 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv4_grad_b, 0, 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv5_grad_w, 0, 3 * 3 * 3 * 256 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_dec_conv5_grad_b, 0, 3 * sizeof(float)));
    
    CHECK_CUDA(cudaDeviceSynchronize());
}

void GPUAutoencoder::trainStep(const std::vector<float>& h_batch) {
    // Ensure clean state before starting step
    CHECK_CUDA(cudaDeviceSynchronize());
    
    size_t batchBytes = m_batchSize * 32 * 32 * 3 * sizeof(float);
    CHECK_CUDA(cudaMemcpy(d_input, h_batch.data(), batchBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target, h_batch.data(), batchBytes, cudaMemcpyHostToDevice));
    
    forward();
    
    int outputSize = m_batchSize * 32 * 32 * 3;
    computeMSELoss(d_output, d_target, d_loss, outputSize);
    
    CHECK_CUDA(cudaMemcpy(&m_lastLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    
    backward();
    updateWeights();
}

void GPUAutoencoder::forward() {
    GPUVersion version = GPUConfig::getInstance().getVersion();
    
    switch (version) {
        case GPUVersion::GPU_BASIC:
            forwardBasic();
            break;
        case GPUVersion::GPU_OPT_V1:
            forwardOptV1();
            break;
        case GPUVersion::GPU_OPT_V2:
            forwardOptV2();
            break;
        case GPUVersion::GPU_OPT_V3:
            forwardOptV3();
            break;
        default:
            forwardBasic();
            break;
    }
}

void GPUAutoencoder::backward() {
    GPUVersion version = GPUConfig::getInstance().getVersion();
    
    switch (version) {
        case GPUVersion::GPU_BASIC:
            backwardBasic();
            break;
        case GPUVersion::GPU_OPT_V1:
            backwardOptV1();
            break;
        case GPUVersion::GPU_OPT_V2:
            backwardOptV2();
            break;
        case GPUVersion::GPU_OPT_V3:
            backwardOptV3();
            break;
        default:
            backwardBasic();
            break;
    }
}

void GPUAutoencoder::saveModel(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file: " << filepath << std::endl;
        return;
    }
    
    auto saveWeights = [&](float* d_ptr, size_t count) {
        std::vector<float> h_weights(count);
        CHECK_CUDA(cudaMemcpy(h_weights.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
        file.write(reinterpret_cast<const char*>(h_weights.data()), count * sizeof(float));
    };
    
    saveWeights(d_enc_conv1_w, 256 * 3 * 3 * 3);
    saveWeights(d_enc_conv1_b, 256);
    saveWeights(d_enc_conv2_w, 128 * 3 * 3 * 256);
    saveWeights(d_enc_conv2_b, 128);
    saveWeights(d_dec_conv3_w, 128 * 3 * 3 * 128);
    saveWeights(d_dec_conv3_b, 128);
    saveWeights(d_dec_conv4_w, 256 * 3 * 3 * 128);
    saveWeights(d_dec_conv4_b, 256);
    saveWeights(d_dec_conv5_w, 3 * 3 * 3 * 256);
    saveWeights(d_dec_conv5_b, 3);
    
    file.close();
}

void GPUAutoencoder::loadModel(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: Model not found: " << filepath << ", using random weights." << std::endl;
        return;
    }
    
    auto loadWeights = [&](float* d_ptr, size_t count) {
        std::vector<float> h_weights(count);
        file.read(reinterpret_cast<char*>(h_weights.data()), count * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_ptr, h_weights.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    };
    
    loadWeights(d_enc_conv1_w, 256 * 3 * 3 * 3);
    loadWeights(d_enc_conv1_b, 256);
    loadWeights(d_enc_conv2_w, 128 * 3 * 3 * 256);
    loadWeights(d_enc_conv2_b, 128);
    loadWeights(d_dec_conv3_w, 128 * 3 * 3 * 128);
    loadWeights(d_dec_conv3_b, 128);
    loadWeights(d_dec_conv4_w, 256 * 3 * 3 * 128);
    loadWeights(d_dec_conv4_b, 256);
    loadWeights(d_dec_conv5_w, 3 * 3 * 3 * 256);
    loadWeights(d_dec_conv5_b, 3);
    
    file.close();
}

std::vector<float> GPUAutoencoder::getFeatures(const std::vector<float>& h_input) {
    // Delegate to extractBatchFeatures for batch size = 1
    return extractBatchFeatures(h_input, 1);
}

// NCHW kernel declarations
extern void launchConv2dNCHWRelu(
    const float* d_input, const float* d_weights, const float* d_bias, float* d_output,
    int batch, int inC, int inH, int inW, int outC, int outH, int outW,
    int kernelSize, int padding, int stride);
extern void launchMaxPool2dNCHW(
    const float* d_input, float* d_output, int* d_indices,
    int batch, int channels, int inH, int inW, int k, int stride);

// V4 im2col + GEMM kernel declarations
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

std::vector<float> GPUAutoencoder::extractBatchFeatures(const std::vector<float>& images, int numImages) {
    const int imgSize = 32 * 32 * 3;
    const int featureSize = 8 * 8 * 128;
    
    if (numImages > m_batchSize) {
        std::cerr << "Warning: numImages exceeds batch size. Truncating." << std::endl;
        numImages = m_batchSize;
    }
    
    size_t inputBytes = numImages * imgSize * sizeof(float);
    CHECK_CUDA(cudaMemcpy(d_input, images.data(), inputBytes, cudaMemcpyHostToDevice));
    
    GPUVersion version = GPUConfig::getInstance().getVersion();
    
    // Run encoder for batch - dispatch to appropriate optimized kernels
    switch (version) {
        case GPUVersion::GPU_OPT_V1:
            // V1: NCHW layout with fused Conv+ReLU
            launchConv2dNCHWRelu(d_input, d_enc_conv1_w, d_enc_conv1_b, d_enc_relu1_out,
                                 numImages, 3, 32, 32, 256, 32, 32, 3, 1, 1);
            launchMaxPool2dNCHW(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                                numImages, 256, 32, 32, 2, 2);
            
            launchConv2dNCHWRelu(d_enc_pool1_out, d_enc_conv2_w, d_enc_conv2_b, d_enc_relu2_out,
                                 numImages, 256, 16, 16, 128, 16, 16, 3, 1, 1);
            launchMaxPool2dNCHW(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                                numImages, 128, 16, 16, 2, 2);
            break;
            
        case GPUVersion::GPU_OPT_V2:
            // V2: im2col + cuBLAS GEMM
            // Layer 1: [batch,3,32,32] -> Conv+ReLU -> [batch,256,32,32] -> Pool -> [batch,256,16,16]
            launchIm2colNCHW(d_input, d_im2col_workspace,
                             numImages, 3, 32, 32, 32, 32, 3, 1, 1);
            launchConvGemmForward(d_enc_conv1_w, d_im2col_workspace, d_enc_conv1_b,
                                  d_enc_relu1_out,
                                  numImages, 256, 27, 1024, true);
            launchMaxPool2dNCHW(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices,
                                numImages, 256, 32, 32, 2, 2);
            
            // Layer 2: [batch,256,16,16] -> Conv+ReLU -> [batch,128,16,16] -> Pool -> [batch,128,8,8]
            launchIm2colNCHW(d_enc_pool1_out, d_im2col_workspace,
                             numImages, 256, 16, 16, 16, 16, 3, 1, 1);
            launchConvGemmForward(d_enc_conv2_w, d_im2col_workspace, d_enc_conv2_b,
                                  d_enc_relu2_out,
                                  numImages, 128, 2304, 256, true);
            launchMaxPool2dNCHW(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices,
                                numImages, 128, 16, 16, 2, 2);
            break;
            
        case GPUVersion::GPU_BASIC:
        default:
            // Naive/Basic: Use simple kernels
            conv2dForward(d_input, d_enc_conv1_w, d_enc_conv1_b, d_enc_conv1_out, 
                          numImages, 32, 32, 3, 32, 32, 256);
            reluForward(d_enc_conv1_out, d_enc_relu1_out, numImages * 32 * 32 * 256);
            maxpool2dForward(d_enc_relu1_out, d_enc_pool1_out, d_enc_pool1_indices, 
                             numImages, 32, 32, 256);
            
            conv2dForward(d_enc_pool1_out, d_enc_conv2_w, d_enc_conv2_b, d_enc_conv2_out, 
                          numImages, 16, 16, 256, 16, 16, 128);
            reluForward(d_enc_conv2_out, d_enc_relu2_out, numImages * 16 * 16 * 128);
            maxpool2dForward(d_enc_relu2_out, d_enc_pool2_out, d_enc_pool2_indices, 
                             numImages, 16, 16, 128);
            break;
    }
    
    std::vector<float> features(numImages * featureSize);
    size_t outputBytes = numImages * featureSize * sizeof(float);
    CHECK_CUDA(cudaMemcpy(features.data(), d_enc_pool2_out, outputBytes, cudaMemcpyDeviceToHost));
    
    return features;
}
