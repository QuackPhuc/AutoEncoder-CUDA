#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H

#include <vector>
#include <string>
#include <cstdint>

// CIFAR-10 Dataset loader and manager
// Loads binary CIFAR-10 files and provides normalized float data
class CIFAR10Dataset {
public:
    CIFAR10Dataset(const std::string& dataDir);
    
    // 0 = load all
    void loadTrainData(int maxSamples = 0);

    void loadTestData();
    
    const std::vector<float>& getTrainImages() const { return m_trainImages; }
    
    const std::vector<uint8_t>& getTrainLabels() const { return m_trainLabels; }
    
    const std::vector<float>& getTestImages() const { return m_testImages; }
    
    const std::vector<uint8_t>& getTestLabels() const { return m_testLabels; }
    
    void shuffleTrainData();
    
    std::vector<float> getBatch(int batchIdx, int batchSize);
    
    int getTrainSize() const { return m_trainImages.size() / IMAGE_SIZE; }
    int getTestSize() const { return m_testImages.size() / IMAGE_SIZE; }

private:
    void loadBinaryFile(const std::string& filepath, 
                       std::vector<float>& images, 
                       std::vector<uint8_t>& labels);
    
    std::string m_dataDir;
    std::vector<float> m_trainImages;      // Normalized [0,1], shape: (N, 32*32*3)
    std::vector<uint8_t> m_trainLabels;    // Values [0-9]
    std::vector<float> m_testImages;       // Normalized [0,1], shape: (N, 32*32*3)
    std::vector<uint8_t> m_testLabels;     // Values [0-9]
    std::vector<int> m_shuffleIndices;     // For batch randomization, each image is 32*32*3 float.
    
    static constexpr int IMAGE_SIZE = 32 * 32 * 3;  // 3072 pixels
    static constexpr int RECORD_SIZE = 1 + IMAGE_SIZE;  // 1 label + 3072 pixels
};

#endif // CIFAR_LOADER_H
