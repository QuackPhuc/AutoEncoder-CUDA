#ifndef GPU_INFERENCE_FEATURE_EXTRACTOR_H
#define GPU_INFERENCE_FEATURE_EXTRACTOR_H

#include <vector>
#include <string>

// Forward declaration
class GPUAutoencoder;

class FeatureExtractor {
public:
    // Constructor: loads trained encoder weights
    explicit FeatureExtractor(const std::string& encoderWeightsPath, int batchSize = 128);
    
    // Destructor: frees GPU memory
    ~FeatureExtractor();
    
    // Extract features for entire dataset with batching
    // Returns flattened features (numImages * 8192)
    std::vector<float> extract_all(
        const std::vector<float>& images,
        int numImages,
        int batchSize = -1);  // -1 = use constructor batch size
    
    // Get feature dimension (always 8192 for this architecture)
    int get_feature_dim() const { return 8 * 8 * 128; }

private:
    GPUAutoencoder* m_encoder;
    int m_maxBatchSize;
    
    // Disable copy
    FeatureExtractor(const FeatureExtractor&) = delete;
    FeatureExtractor& operator=(const FeatureExtractor&) = delete;
};

#endif // GPU_INFERENCE_FEATURE_EXTRACTOR_H
