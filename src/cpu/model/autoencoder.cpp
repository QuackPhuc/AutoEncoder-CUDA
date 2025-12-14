#include "autoencoder.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

Autoencoder::Autoencoder()
    : m_conv1(3, 256),       // (32,32,3) -> (32,32,256)
      m_pool1(2, 2),
      m_conv2(256, 128),     // (16,16,256) -> (16,16,128)
      m_pool2(2, 2),
      m_conv3(128, 128),     // (8,8,128) -> (8,8,128)
      m_up1(2),
      m_conv4(128, 256),     // (16,16,128) -> (16,16,256)
      m_up2(2),
      m_conv5(256, 3) {      // (32,32,256) -> (32,32,3)
    
    std::cout << "Autoencoder initialized with ~751,875 parameters" << std::endl;
}

std::vector<float> Autoencoder::forward(const std::vector<float>& input) {
    // ENCODER - assignments copy from const refs (layer outputs are pre-allocated)
    const auto& conv1_out = m_conv1.forward(input, 32, 32);          // (32,32,3) -> (32,32,256)
    const auto& relu1_out = m_relu1.forward(conv1_out);
    const auto& pool1_out = m_pool1.forward(relu1_out, 32, 32, 256); // (32,32,256) -> (16,16,256)
    const auto& conv2_out = m_conv2.forward(pool1_out, 16, 16);      // (16,16,256) -> (16,16,128)
    const auto& relu2_out = m_relu2.forward(conv2_out);
    const auto& pool2_out = m_pool2.forward(relu2_out, 16, 16, 128); // (16,16,128) -> (8,8,128) LATENT
    
    // DECODER
    const auto& conv3_out = m_conv3.forward(pool2_out, 8, 8);        // (8,8,128) -> (8,8,128)
    const auto& relu3_out = m_relu3.forward(conv3_out);
    const auto& up1_out = m_up1.forward(relu3_out, 8, 8, 128);       // (8,8,128) -> (16,16,128)
    const auto& conv4_out = m_conv4.forward(up1_out, 16, 16);        // (16,16,128) -> (16,16,256)
    const auto& relu4_out = m_relu4.forward(conv4_out);
    const auto& up2_out = m_up2.forward(relu4_out, 16, 16, 256);     // (16,16,256) -> (32,32,256)
    
    // Final layer returns pre-allocated buffer, copy for return
    const auto& output = m_conv5.forward(up2_out, 32, 32);           // (32,32,256) -> (32,32,3)
    return std::vector<float>(output.begin(), output.end());
}

std::vector<float> Autoencoder::encode(const std::vector<float>& input) {
    const auto& act1 = m_conv1.forward(input, 32, 32);
    const auto& act2 = m_relu1.forward(act1);
    const auto& act3 = m_pool1.forward(act2, 32, 32, 256);
    const auto& act4 = m_conv2.forward(act3, 16, 16);
    const auto& act5 = m_relu2.forward(act4);
    const auto& latent = m_pool2.forward(act5, 16, 16, 128);  // (8,8,128) = 8,192 features
    return std::vector<float>(latent.begin(), latent.end());
}

void Autoencoder::backward(const std::vector<float>& gradOutput) {
    // NOTE: Gradients are ACCUMULATED in Conv layers.
    // Call zeroGradients() at the start of each mini-batch.
    
    // Decoder backward (reverse order)
    const auto& grad9 = m_conv5.backward(gradOutput);
    const auto& grad8 = m_up2.backward(grad9);
    const auto& grad7b = m_relu4.backward(grad8);
    const auto& grad7 = m_conv4.backward(grad7b);
    const auto& grad6 = m_up1.backward(grad7);
    const auto& grad5b = m_relu3.backward(grad6);
    const auto& grad5 = m_conv3.backward(grad5b);
    
    // Encoder backward
    const auto& grad4 = m_pool2.backward(grad5);
    const auto& grad3b = m_relu2.backward(grad4);
    const auto& grad3 = m_conv2.backward(grad3b);
    const auto& grad2 = m_pool1.backward(grad3);
    const auto& grad1b = m_relu1.backward(grad2);
    m_conv1.backward(grad1b);
}

void Autoencoder::zeroGradients() {
    m_conv1.zeroGradients();
    m_conv2.zeroGradients();
    m_conv3.zeroGradients();
    m_conv4.zeroGradients();
    m_conv5.zeroGradients();
}

void Autoencoder::updateWeights(float learningRate) {
    m_conv1.updateWeights(learningRate);
    m_conv2.updateWeights(learningRate);
    m_conv3.updateWeights(learningRate);
    m_conv4.updateWeights(learningRate);
    m_conv5.updateWeights(learningRate);
}

void Autoencoder::saveModel(const std::string& filepath) {
    std::cout << "Saving model to " << filepath << "..." << std::endl;
    
    // Note: Weight layout differs (CPU: HWIO, GPU: OIHW), so weights are not directly portable
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Save all conv layer weights and biases in order
    auto saveLayerWeights = [&file](const Conv2D& layer) {
        const auto& weights = layer.getWeights();
        const auto& bias = layer.getBias();
        file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
    };
    
    saveLayerWeights(m_conv1);  // enc_conv1: (256, 3, 3, 3) + bias(256)
    saveLayerWeights(m_conv2);  // enc_conv2: (128, 3, 3, 256) + bias(128)
    saveLayerWeights(m_conv3);  // dec_conv3: (128, 3, 3, 128) + bias(128)
    saveLayerWeights(m_conv4);  // dec_conv4: (256, 3, 3, 128) + bias(256)
    saveLayerWeights(m_conv5);  // dec_conv5: (3, 3, 3, 256) + bias(3)
    
    file.close();
    std::cout << "Model saved successfully" << std::endl;
}

void Autoencoder::loadModel(const std::string& filepath) {
    std::cout << "Loading model from " << filepath << "..." << std::endl;
    
    // Use single binary file format (matches GPU implementation structure)
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Load all conv layer weights and biases in order
    auto loadLayerWeights = [&file](Conv2D& layer) {
        auto& weights = layer.getWeights();
        auto& bias = layer.getBias();
        file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
    };
    
    loadLayerWeights(m_conv1);
    loadLayerWeights(m_conv2);
    loadLayerWeights(m_conv3);
    loadLayerWeights(m_conv4);
    loadLayerWeights(m_conv5);
    
    file.close();
    std::cout << "Model loaded successfully" << std::endl;
}
