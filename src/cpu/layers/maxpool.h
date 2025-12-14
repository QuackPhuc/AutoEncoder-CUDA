#ifndef MAXPOOL_H
#define MAXPOOL_H

#include <vector>

// 2D Max Pooling layer.
// Downsamples spatial dimensions by taking max value in each window.
class MaxPool2D {
public:
    MaxPool2D(int poolSize = 2, int stride = 2);
    
    // Input: (H * W * C) flattened -> Output reference: ((H/stride) * (W/stride) * C)
    const std::vector<float>& forward(const std::vector<float>& input, int H, int W, int C);
    const std::vector<float>& backward(const std::vector<float>& gradOutput);
    
private:
    int m_poolSize;
    int m_stride;
    
    int m_inH, m_inW, m_inC;
    int m_outH, m_outW;
    
    // Max indices for backward gradient routing
    std::vector<int> m_maxIndices;
    
    // Pre-allocated output buffers
    std::vector<float> m_output;
    std::vector<float> m_gradInput;
};
#endif // MAXPOOL_H
