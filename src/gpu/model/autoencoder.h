#ifndef GPU_MODEL_AUTOENCODER_H
#define GPU_MODEL_AUTOENCODER_H

#include <vector>
#include <string>

// Forward-declare GPU config enum
enum class GPUVersion;

class GPUAutoencoder {
public:
    // Constructor: allocates GPU memory and initializes weights
    GPUAutoencoder(int batchSize = 64, float learningRate = 0.001f);
    
    // Destructor: frees GPU memory
    ~GPUAutoencoder();
    
    // Training step: forward + backward + update
    void trainStep(const std::vector<float>& h_batch);
    
    // Extract 8,192-dimensional features (single image)
    std::vector<float> getFeatures(const std::vector<float>& h_input);
    
    // Extract features for batch of images
    std::vector<float> extractBatchFeatures(const std::vector<float>& images, int numImages);
    
    // Get last computed loss
    float getLoss() const { return m_lastLoss; }
    
    // Model persistence
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);

private:
    int m_batchSize;
    float m_learningRate;
    float m_lastLoss;
    
    // Forward/backward pass implementations (in separate files)
    void forward();
    void forwardBasic();
    void forwardOptV1();  // NCHW layout optimized (formerly V3)
    void forwardOptV2();  // im2col + GEMM (formerly V4)
    
    void backward();
    void backwardBasic();
    void backwardOptV1();  // NCHW layout optimized (formerly V3)
    void backwardOptV2();  // im2col + GEMM (formerly V4)
    
    void updateWeights();
    
    // Memory management
    void allocateMemory();
    void freeMemory();
    void initializeWeights();
    
    // Kernel wrappers (basic versions)
    void conv2dForward(const float* d_in, const float* d_w, const float* d_b, float* d_out,
                       int batch, int inH, int inW, int inC, int outH, int outW, int outC);
    void reluForward(const float* d_in, float* d_out, int size);
    void maxpool2dForward(const float* d_in, float* d_out, int* d_indices,
                          int batch, int inH, int inW, int channels);
    void upsample2dForward(const float* d_in, float* d_out,
                           int batch, int inH, int inW, int channels);
    
    void conv2dBackward(const float* d_gradOut, const float* d_in, const float* d_w,
                        float* d_gradIn, float* d_gradW, float* d_gradB,
                        int batch, int inH, int inW, int inC, int outH, int outW, int outC);
    void reluBackward(const float* d_gradOut, const float* d_in, float* d_gradIn, int size);
    void maxpool2dBackward(const float* d_gradOut, const int* d_indices, float* d_gradIn,
                           int batch, int inH, int inW, int channels);
    void upsample2dBackward(const float* d_gradOut, float* d_gradIn,
                            int batch, int inH, int inW, int channels);
    
    void computeMSELoss(const float* d_pred, const float* d_target, float* d_loss, int size);
    void sgdUpdate(float* d_weights, const float* d_gradients, int size);
    
    // GPU Memory Buffers
    // Input/Output (d_input/d_output are NCHW, d_*_nhwc are for V1/V2 NHWC kernels)
    float* d_input;          // NCHW layout (from loader or converted)
    float* d_output;         // NCHW layout (for loss computation)
    float* d_target;

    // Temporary buffers for layout conversion (needed for Basic kernels which use NHWC)
    float* d_input_nhwc;     // NHWC layout
    float* d_output_nhwc;    // NHWC layout
    
    // Encoder Layer 1: 32x32x3 -> 16x16x256
    float* d_enc_conv1_w;
    float* d_enc_conv1_b;
    float* d_enc_conv1_out;
    float* d_enc_relu1_out;
    float* d_enc_pool1_out;
    int* d_enc_pool1_indices;
    float* d_enc_conv1_grad_w;
    float* d_enc_conv1_grad_b;
    
    // Encoder Layer 2: 16x16x256 -> 8x8x128 (latent)
    float* d_enc_conv2_w;
    float* d_enc_conv2_b;
    float* d_enc_conv2_out;
    float* d_enc_relu2_out;
    float* d_enc_pool2_out;
    int* d_enc_pool2_indices;
    float* d_enc_conv2_grad_w;
    float* d_enc_conv2_grad_b;
    
    // Decoder Layer 3: 8x8x128 -> 16x16x128
    float* d_dec_conv3_w;
    float* d_dec_conv3_b;
    float* d_dec_conv3_out;
    float* d_dec_relu3_out;
    float* d_dec_up1_out;
    float* d_dec_conv3_grad_w;
    float* d_dec_conv3_grad_b;
    
    // Decoder Layer 4: 16x16x128 -> 32x32x256
    float* d_dec_conv4_w;
    float* d_dec_conv4_b;
    float* d_dec_conv4_out;
    float* d_dec_relu4_out;
    float* d_dec_up2_out;
    float* d_dec_conv4_grad_w;
    float* d_dec_conv4_grad_b;
    
    // Decoder Layer 5: 32x32x256 -> 32x32x3
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
    
    // im2col workspace for V4 GEMM convolution
    // Size: batch * max(inC*k*k * outH*outW) for largest layer
    float* d_im2col_workspace;
    size_t m_im2col_workspace_size;
};

#endif // GPU_MODEL_AUTOENCODER_H
