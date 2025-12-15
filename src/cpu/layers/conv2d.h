#ifndef CONV2D_H
#define CONV2D_H

#include <vector>
#include <string>

// 2D Convolution layer with backpropagation support.
// Supports 3x3 kernels with stride=1, padding=1 (same convolution).
// Weights layout: (K, K, C_in, C_out) flattened.
// Best practices: Pre-allocated buffers to avoid heap allocation in hot paths.
class Conv2D {
public:
    Conv2D(int inChannels, int outChannels, 
           int kernelSize = 3, int stride = 1, int padding = 1);
    
    // Input: (H * W * C_in) flattened -> Output reference (H_out * W_out * C_out)
    // Returns reference to internal pre-allocated buffer (avoid copy)
    const std::vector<float>& forward(const std::vector<float>& input, int H, int W);
    
    // Gradient: (H_out * W_out * C_out) -> reference to (H * W * C_in)
    // Gradients are ACCUMULATED (not cleared). Call zeroGradients() before batch.
    const std::vector<float>& backward(const std::vector<float>& gradOutput);
    
    // Zero out accumulated gradients (call at start of each mini-batch)
    void zeroGradients();
    
    // SGD update: w -= lr * grad_w
    void updateWeights(float learningRate);
    
    void initializeWeights();
    
    // Accessors for unified save/load format
    const std::vector<float>& getWeights() const { return m_weights; }
    const std::vector<float>& getBias() const { return m_bias; }
    std::vector<float>& getWeights() { return m_weights; }
    std::vector<float>& getBias() { return m_bias; }

private:
    float getPaddedValue(const std::vector<float>& input, int h, int w, int c) const;
    void ensureOutputSize(int size);
    void ensureGradInputSize(int size);
    
    int m_inChannels;
    int m_outChannels;
    int m_kernelSize;
    int m_stride;
    int m_padding;
    
    int m_inH, m_inW;
    int m_outH, m_outW;
    
    // Weights: (K, K, C_in, C_out) flattened
    std::vector<float> m_weights;
    std::vector<float> m_bias;
    
    std::vector<float> m_gradWeights;
    std::vector<float> m_gradBias;
    
    std::vector<float> m_cachedInput;
    
    // Pre-allocated output buffers (avoid heap allocation in hot path)
    std::vector<float> m_output;
    std::vector<float> m_gradInput;
};

#endif // CONV2D_H
