#include "feature_extractor.h"
#include "gpu/model/autoencoder.h"
#include "gpu/core/cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

FeatureExtractor::FeatureExtractor(const std::string& encoderWeightsPath, int batchSize)
    : m_encoder(nullptr)
    , m_maxBatchSize(batchSize) {
    
    m_encoder = new GPUAutoencoder(m_maxBatchSize, 0.001f);
    m_encoder->loadModel(encoderWeightsPath);
}

FeatureExtractor::~FeatureExtractor() {
    delete m_encoder;
}

std::vector<float> FeatureExtractor::extract_all(
    const std::vector<float>& images,
    int numImages,
    int batchSize
) {
    if (batchSize <= 0) {
        batchSize = m_maxBatchSize;
    }
    
    if (batchSize > m_maxBatchSize) {
        throw std::invalid_argument(
            "Requested batch size exceeds maximum: " + 
            std::to_string(batchSize) + " > " + std::to_string(m_maxBatchSize));
    }
    
    const int imgSize = 32 * 32 * 3;
    const int featureDim = get_feature_dim();
    
    std::vector<float> allFeatures(numImages * featureDim);
    
    int numBatches = (numImages + batchSize - 1) / batchSize;
    
    std::cout << "Extracting " << numImages << " images (batch=" << batchSize << ")..." << std::flush;
    
    for (int b = 0; b < numBatches; ++b) {
        int currentBatchSize = std::min(batchSize, numImages - b * batchSize);
        int batchStartIdx = b * batchSize;
        
        // Use pointer-based API directly (zero-copy from source data)
        const float* batchStart = images.data() + batchStartIdx * imgSize;
        std::vector<float> batchFeatures = m_encoder->extractBatchFeatures(batchStart, currentBatchSize);
        
        // Copy results to output buffer
        std::memcpy(allFeatures.data() + batchStartIdx * featureDim,
                    batchFeatures.data(),
                    currentBatchSize * featureDim * sizeof(float));
    }
    
    std::cout << " done." << std::endl;
    
    return allFeatures;
}
