#include "upsample.h"

Upsample2D::Upsample2D(int scaleFactor)
    : m_scaleFactor(scaleFactor), m_inH(0), m_inW(0), m_inC(0), m_outH(0), m_outW(0) {
}

const std::vector<float>& Upsample2D::forward(const std::vector<float>& input, int H, int W, int C) {
    m_inH = H;
    m_inW = W;
    m_inC = C;
    m_outH = H * m_scaleFactor;
    m_outW = W * m_scaleFactor;
    
    int outputSize = m_outH * m_outW * C;
    
    // Resize only if needed (avoid reallocation)
    if (static_cast<int>(m_output.size()) < outputSize) {
        m_output.resize(outputSize);
    }
    
    for (int c = 0; c < C; ++c) {
        for (int ih = 0; ih < H; ++ih) {
            for (int iw = 0; iw < W; ++iw) {
                int inIdx = (ih * W + iw) * C + c;
                float value = input[inIdx];
                
                // Replicate to scale x scale block
                for (int sh = 0; sh < m_scaleFactor; ++sh) {
                    for (int sw = 0; sw < m_scaleFactor; ++sw) {
                        int oh = ih * m_scaleFactor + sh;
                        int ow = iw * m_scaleFactor + sw;
                        int outIdx = (oh * m_outW + ow) * C + c;
                        m_output[outIdx] = value;
                    }
                }
            }
        }
    }
    
    return m_output;
}

const std::vector<float>& Upsample2D::backward(const std::vector<float>& gradOutput) {
    int gradInputSize = m_inH * m_inW * m_inC;
    
    // Resize only if needed (avoid reallocation)
    if (static_cast<int>(m_gradInput.size()) < gradInputSize) {
        m_gradInput.resize(gradInputSize);
    }
    std::fill(m_gradInput.begin(), m_gradInput.begin() + gradInputSize, 0.0f);
    
    for (int c = 0; c < m_inC; ++c) {
        for (int ih = 0; ih < m_inH; ++ih) {
            for (int iw = 0; iw < m_inW; ++iw) {
                float gradSum = 0.0f;
                
                // Sum from replicated positions
                for (int sh = 0; sh < m_scaleFactor; ++sh) {
                    for (int sw = 0; sw < m_scaleFactor; ++sw) {
                        int oh = ih * m_scaleFactor + sh;
                        int ow = iw * m_scaleFactor + sw;
                        int outIdx = (oh * m_outW + ow) * m_inC + c;
                        gradSum += gradOutput[outIdx];
                    }
                }
                
                int inIdx = (ih * m_inW + iw) * m_inC + c;
                m_gradInput[inIdx] = gradSum;
            }
        }
    }
    
    return m_gradInput;
}

