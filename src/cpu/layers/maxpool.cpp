#include "maxpool.h"
#include <cstddef>
#include <limits>

MaxPool2D::MaxPool2D(int poolSize, int stride)
    : m_poolSize(poolSize), m_stride(stride),
      m_inH(0), m_inW(0), m_inC(0), m_outH(0), m_outW(0) {
}

const std::vector<float>& MaxPool2D::forward(const std::vector<float>& input, int H, int W, int C) {
    m_inH = H;
    m_inW = W;
    m_inC = C;
    m_outH = (H - m_poolSize) / m_stride + 1;
    m_outW = (W - m_poolSize) / m_stride + 1;
    
    int outputSize = m_outH * m_outW * C;
    
    // Resize only if needed (avoid reallocation)
    if (static_cast<int>(m_output.size()) < outputSize) {
        m_output.resize(outputSize);
    }
    if (static_cast<int>(m_maxIndices.size()) < outputSize) {
        m_maxIndices.resize(outputSize);
    }
    
    for (int c = 0; c < C; ++c) {
        for (int oh = 0; oh < m_outH; ++oh) {
            for (int ow = 0; ow < m_outW; ++ow) {
                float maxVal = -std::numeric_limits<float>::infinity();
                int maxIdx = -1;
                
                for (int ph = 0; ph < m_poolSize; ++ph) {
                    for (int pw = 0; pw < m_poolSize; ++pw) {
                        int ih = oh * m_stride + ph;
                        int iw = ow * m_stride + pw;
                        int inIdx = (ih * m_inW + iw) * m_inC + c;
                        
                        if (input[inIdx] > maxVal) {
                            maxVal = input[inIdx];
                            maxIdx = inIdx;
                        }
                    }
                }
                
                int outIdx = (oh * m_outW + ow) * m_inC + c;
                m_output[outIdx] = maxVal;
                m_maxIndices[outIdx] = maxIdx;
            }
        }
    }
    
    return m_output;
}

const std::vector<float>& MaxPool2D::backward(const std::vector<float>& gradOutput) {
    int gradInputSize = m_inH * m_inW * m_inC;
    
    // Resize only if needed (avoid reallocation)
    if (static_cast<int>(m_gradInput.size()) < gradInputSize) {
        m_gradInput.resize(gradInputSize);
    }
    std::fill(m_gradInput.begin(), m_gradInput.begin() + gradInputSize, 0.0f);
    
    // Route gradient to max value positions
    int outputSize = m_outH * m_outW * m_inC;
    for (int i = 0; i < outputSize; ++i) {
        int maxIdx = m_maxIndices[i];
        if (maxIdx >= 0) {
            m_gradInput[maxIdx] += gradOutput[i];
        }
    }
    
    return m_gradInput;
}

