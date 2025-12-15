#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <vector>
#include <string>
#include "cpu/layers/conv2d.h"
#include "cpu/layers/relu.h"
#include "cpu/layers/maxpool.h"
#include "cpu/layers/upsample.h"

// Convolutional Autoencoder for CIFAR-10
// Architecture: 8,192 latent features (8x8x128)
// Total parameters: ~751,875
class Autoencoder {
public:
    Autoencoder();
    
    // Full forward pass (encoder + decoder)
    // Input: (32*32*3) flattened -> Output: (32*32*3) reconstructed
    std::vector<float> forward(const std::vector<float>& input);
    
    // Encoder only for feature extraction
    // Input: (32*32*3) -> Output: (8*8*128) = 8,192 features
    std::vector<float> encode(const std::vector<float>& input);
    
    // Backward pass through entire network
    // NOTE: Gradients are ACCUMULATED. Call zeroGradients() at start of each batch.
    void backward(const std::vector<float>& gradOutput);
    
    // Zero out all layer gradients (call at start of each mini-batch)
    void zeroGradients();
    
    // Update all layer weights using SGD
    void updateWeights(float learningRate);
    
    // Save/load model weights
    void saveModel(const std::string& filepath);
    void loadModel(const std::string& filepath);

private:
    // Encoder layers
    Conv2D m_conv1;      // (3 -> 256)
    ReLU m_relu1;
    MaxPool2D m_pool1;
    Conv2D m_conv2;      // (256 -> 128)
    ReLU m_relu2;
    MaxPool2D m_pool2;
    
    // Decoder layers
    Conv2D m_conv3;      // (128 -> 128)
    ReLU m_relu3;
    Upsample2D m_up1;
    Conv2D m_conv4;      // (128 -> 256)
    ReLU m_relu4;
    Upsample2D m_up2;
    Conv2D m_conv5;      // (256 -> 3)
};

#endif // AUTOENCODER_H
