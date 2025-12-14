#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include <vector>

// 2D Upsampling using nearest neighbor interpolation.
// Replicates each pixel into scale x scale block.
// Best practices: Pre-allocated buffers to avoid heap allocation in hot paths.
class Upsample2D {
public:
    Upsample2D(int scaleFactor = 2);
    
    // Input: (H * W * C) -> Output reference: ((H*scale) * (W*scale) * C)
    const std::vector<float>& forward(const std::vector<float>& input, int H, int W, int C);
    
    // Backward: sum gradients from replicated positions
    const std::vector<float>& backward(const std::vector<float>& gradOutput);

private:
    int m_scaleFactor;
    
    int m_inH, m_inW, m_inC;
    int m_outH, m_outW;
    
    // Pre-allocated output buffers
    std::vector<float> m_output;
    std::vector<float> m_gradInput;
};

#endif // UPSAMPLE_H
