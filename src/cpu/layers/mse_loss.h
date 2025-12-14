#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <vector>

// Mean Squared Error loss.
// Loss = (1/N) * sum((target - prediction)^2)
class MSELoss {
public:
    MSELoss() = default;
    
    // Returns scalar loss value
    float forward(const std::vector<float>& predictions,
                  const std::vector<float>& targets);
    
    // Gradient = 2 * (prediction - target) / N
    std::vector<float> backward();

private:
    std::vector<float> m_cachedDiff;  // prediction - target
    int m_N;
};

#endif // MSE_LOSS_H
