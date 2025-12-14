#include "mse_loss.h"

float MSELoss::forward(const std::vector<float>& predictions,
                      const std::vector<float>& targets) {
    m_N = static_cast<int>(predictions.size());
    m_cachedDiff.resize(m_N);
    
    float loss = 0.0f;
    for (int i = 0; i < m_N; ++i) {
        float diff = predictions[i] - targets[i];
        m_cachedDiff[i] = diff;
        loss += diff * diff;
    }
    
    return loss / static_cast<float>(m_N);
}

std::vector<float> MSELoss::backward() {
    std::vector<float> gradient(m_N);
    
    for (int i = 0; i < m_N; ++i) {
        gradient[i] = 2.0f * m_cachedDiff[i] / static_cast<float>(m_N);
    }
    
    return gradient;
}
