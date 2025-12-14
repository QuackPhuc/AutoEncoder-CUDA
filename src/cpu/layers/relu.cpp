#include "relu.h"
#include <cstddef>

const std::vector<float>& ReLU::forward(const std::vector<float>& input) {
    size_t size = input.size();
    
    // Resize only if needed (avoid reallocation)
    if (m_output.size() < size) {
        m_output.resize(size);
    }
    if (m_mask.size() < size) {
        m_mask.resize(size);
    }
    
    for (size_t i = 0; i < size; ++i) {
        if (input[i] > 0.0f) {
            m_output[i] = input[i];
            m_mask[i] = 1.0f;
        } else {
            m_output[i] = 0.0f;
            m_mask[i] = 0.0f;
        }
    }
    
    return m_output;
}

const std::vector<float>& ReLU::backward(const std::vector<float>& gradOutput) {
    size_t size = gradOutput.size();
    
    // Resize only if needed (avoid reallocation)
    if (m_gradInput.size() < size) {
        m_gradInput.resize(size);
    }
    
    for (size_t i = 0; i < size; ++i) {
        m_gradInput[i] = gradOutput[i] * m_mask[i];
    }
    
    return m_gradInput;
}

