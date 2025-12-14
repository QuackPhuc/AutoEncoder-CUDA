#ifndef RELU_H
#define RELU_H

#include <vector>

// ReLU activation layer.
// Forward: max(0, x), Backward: grad * (x > 0 ? 1 : 0)
// Best practices: Pre-allocated buffers to avoid heap allocation in hot paths.
class ReLU {
public:
    ReLU() = default;
    
    const std::vector<float>& forward(const std::vector<float>& input);
    const std::vector<float>& backward(const std::vector<float>& gradOutput);

private:
    std::vector<float> m_mask;    // 1 if input > 0, else 0
    std::vector<float> m_output;  // Pre-allocated output buffer
    std::vector<float> m_gradInput;  // Pre-allocated gradient buffer
};

#endif // RELU_H
