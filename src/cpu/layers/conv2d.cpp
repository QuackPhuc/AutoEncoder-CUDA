#include "conv2d.h"
#include <cstddef>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

Conv2D::Conv2D(int inChannels, int outChannels, 
               int kernelSize, int stride, int padding)
    : m_inChannels(inChannels), m_outChannels(outChannels),
      m_kernelSize(kernelSize), m_stride(stride), m_padding(padding),
      m_inH(0), m_inW(0), m_outH(0), m_outW(0) {
    
    int weightSize = m_kernelSize * m_kernelSize * m_inChannels * m_outChannels;
    m_weights.resize(weightSize);
    m_gradWeights.resize(weightSize, 0.0f);
    
    m_bias.resize(m_outChannels, 0.0f);
    m_gradBias.resize(m_outChannels, 0.0f);
    
    initializeWeights();
}

void Conv2D::initializeWeights() {
    // He initialization: std = sqrt(2 / fan_in)
    std::random_device rd;
    std::mt19937 gen(rd());
    float std = std::sqrt(2.0f / (m_kernelSize * m_kernelSize * m_inChannels));
    std::normal_distribution<float> dist(0.0f, std);
    
    for (auto& w : m_weights) {
        w = dist(gen);
    }
    std::fill(m_bias.begin(), m_bias.end(), 0.0f);
}

void Conv2D::ensureOutputSize(int size) {
    if (static_cast<int>(m_output.size()) < size) {
        m_output.resize(size);
    }
}

void Conv2D::ensureGradInputSize(int size) {
    if (static_cast<int>(m_gradInput.size()) < size) {
        m_gradInput.resize(size);
    }
}

const std::vector<float>& Conv2D::forward(const std::vector<float>& input, int H, int W) {
    m_inH = H;
    m_inW = W;
    m_outH = (H + 2 * m_padding - m_kernelSize) / m_stride + 1;
    m_outW = (W + 2 * m_padding - m_kernelSize) / m_stride + 1;
    
    m_cachedInput = input;
    
    // Use pre-allocated buffer (resize only if needed, no deallocation)
    int outputSize = m_outH * m_outW * m_outChannels;
    ensureOutputSize(outputSize);
    std::fill(m_output.begin(), m_output.begin() + outputSize, 0.0f);
    
    for (int oc = 0; oc < m_outChannels; ++oc) {
        for (int oh = 0; oh < m_outH; ++oh) {
            for (int ow = 0; ow < m_outW; ++ow) {
                float sum = m_bias[oc];
                
                for (int kh = 0; kh < m_kernelSize; ++kh) {
                    for (int kw = 0; kw < m_kernelSize; ++kw) {
                        for (int ic = 0; ic < m_inChannels; ++ic) {
                            int ih = oh * m_stride + kh - m_padding;
                            int iw = ow * m_stride + kw - m_padding;
                            
                            float inputVal = getPaddedValue(input, ih, iw, ic);
                            // Weight index: [kh][kw][ic][oc]
                            int wIdx = ((kh * m_kernelSize + kw) * m_inChannels + ic) * m_outChannels + oc;
                            sum += inputVal * m_weights[wIdx];
                        }
                    }
                }
                
                // Output index: [oh][ow][oc]
                int outIdx = (oh * m_outW + ow) * m_outChannels + oc;
                m_output[outIdx] = sum;
            }
        }
    }
    
    return m_output;
}

void Conv2D::zeroGradients() {
    std::fill(m_gradWeights.begin(), m_gradWeights.end(), 0.0f);
    std::fill(m_gradBias.begin(), m_gradBias.end(), 0.0f);
}

const std::vector<float>& Conv2D::backward(const std::vector<float>& gradOutput) {
    // NOTE: Gradients are ACCUMULATED, not cleared!
    // Call zeroGradients() at the start of each mini-batch
    
    int gradInputSize = m_inH * m_inW * m_inChannels;
    ensureGradInputSize(gradInputSize);
    std::fill(m_gradInput.begin(), m_gradInput.begin() + gradInputSize, 0.0f);
    
    for (int oc = 0; oc < m_outChannels; ++oc) {
        for (int oh = 0; oh < m_outH; ++oh) {
            for (int ow = 0; ow < m_outW; ++ow) {
                int outIdx = (oh * m_outW + ow) * m_outChannels + oc;
                float grad = gradOutput[outIdx];
                
                m_gradBias[oc] += grad;
                
                for (int kh = 0; kh < m_kernelSize; ++kh) {
                    for (int kw = 0; kw < m_kernelSize; ++kw) {
                        for (int ic = 0; ic < m_inChannels; ++ic) {
                            int ih = oh * m_stride + kh - m_padding;
                            int iw = ow * m_stride + kw - m_padding;
                            
                            float inputVal = getPaddedValue(m_cachedInput, ih, iw, ic);
                            int wIdx = ((kh * m_kernelSize + kw) * m_inChannels + ic) * m_outChannels + oc;
                            m_gradWeights[wIdx] += grad * inputVal;
                            
                            if (ih >= 0 && ih < m_inH && iw >= 0 && iw < m_inW) {
                                int inIdx = (ih * m_inW + iw) * m_inChannels + ic;
                                m_gradInput[inIdx] += grad * m_weights[wIdx];
                            }
                        }
                    }
                }
            }
        }
    }
    
    return m_gradInput;
}

void Conv2D::updateWeights(float learningRate) {
    for (size_t i = 0; i < m_weights.size(); ++i) {
        m_weights[i] -= learningRate * m_gradWeights[i];
    }
    for (size_t i = 0; i < m_bias.size(); ++i) {
        m_bias[i] -= learningRate * m_gradBias[i];
    }
}

float Conv2D::getPaddedValue(const std::vector<float>& input, int h, int w, int c) const {
    if (h < 0 || h >= m_inH || w < 0 || w >= m_inW) {
        return 0.0f;
    }
    int idx = (h * m_inW + w) * m_inChannels + c;
    return input[idx];
}
