#include "cifar_loader.h"
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <random>

CIFAR10Dataset::CIFAR10Dataset(const std::string& dataDir) 
    : m_dataDir(dataDir) {
}

void CIFAR10Dataset::loadTrainData(int maxSamples) {
    m_trainImages.clear();
    m_trainLabels.clear();
    
    int samplesLoaded = 0;
    
    // Load 5 training batch files (10,000 images each)
    for (int i = 1; i <= 5; ++i) {
        if (maxSamples > 0 && samplesLoaded >= maxSamples) {
            break;
        }
        
        std::string filename = m_dataDir + "/data_batch_" + std::to_string(i) + ".bin";
        
        size_t beforeSize = m_trainLabels.size();
        loadBinaryFile(filename, m_trainImages, m_trainLabels);
        size_t loadedFromFile = m_trainLabels.size() - beforeSize;
        
        samplesLoaded += loadedFromFile;
        
        // Trim excess if loaded more than needed
        if (maxSamples > 0 && samplesLoaded > maxSamples) {
            int excess = samplesLoaded - maxSamples;
            m_trainImages.resize(m_trainImages.size() - (excess * IMAGE_SIZE));
            m_trainLabels.resize(m_trainLabels.size() - excess);
            samplesLoaded = maxSamples;
            break;
        }
    }
    
    // Initialize shuffle indices
    m_shuffleIndices.resize(getTrainSize());
    for (int i = 0; i < getTrainSize(); ++i) {
        m_shuffleIndices[i] = i;
    }
    
    std::cout << "Train: " << getTrainSize() << " images";
    if (maxSamples > 0) {
        std::cout << " (limited)";
    }
    std::cout << std::endl;
}

void CIFAR10Dataset::loadTestData() {
    m_testImages.clear();
    m_testLabels.clear();
    
    std::string filename = m_dataDir + "/test_batch.bin";
    loadBinaryFile(filename, m_testImages, m_testLabels);
    
    std::cout << "Test:  " << getTestSize() << " images" << std::endl;
}

void CIFAR10Dataset::loadBinaryFile(const std::string& filepath,
                                   std::vector<float>& images,
                                   std::vector<uint8_t>& labels) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // CIFAR-10 binary format:
    // Each record: 1 label byte + 3072 image bytes (1024 R + 1024 G + 1024 B)
    
    while (file.good()) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (file.gcount() == 0) break;
        
        std::vector<uint8_t> imageData(IMAGE_SIZE);
        file.read(reinterpret_cast<char*>(imageData.data()), IMAGE_SIZE);
        
        if (file.gcount() != IMAGE_SIZE) {
            throw std::runtime_error("Incomplete image data in file: " + filepath);
        }
        
        // Normalize [0, 255] -> [0, 1]
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            images.push_back(static_cast<float>(imageData[i]) / 255.0f);
        }
        
        labels.push_back(label);
    }
    
    file.close();
}

void CIFAR10Dataset::shuffleTrainData() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(m_shuffleIndices.begin(), m_shuffleIndices.end(), gen);
}

std::vector<float> CIFAR10Dataset::getBatch(int batchIdx, int batchSize) {
    std::vector<float> batch;
    batch.reserve(batchSize * IMAGE_SIZE);
    
    int startIdx = batchIdx * batchSize;
    int numImages = getTrainSize();
    
    for (int i = 0; i < batchSize; ++i) {
        int imgIdx = startIdx + i;
        if (imgIdx >= numImages) break;
        
        int shuffledIdx = m_shuffleIndices[imgIdx];
        int offset = shuffledIdx * IMAGE_SIZE;
        
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            batch.push_back(m_trainImages[offset + j]);
        }
    }
    
    return batch;
}
