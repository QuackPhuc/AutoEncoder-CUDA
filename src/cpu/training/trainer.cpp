#include "trainer.h"
#include <iostream>

Trainer::Trainer(Autoencoder& model, CIFAR10Dataset& dataset, float learningRate)
    : m_model(model), m_dataset(dataset), m_learningRate(learningRate),
      m_logger("output/training_log.txt") {
}

void Trainer::train(int epochs, int batchSize) {
    std::cout << "CPU Training: " << epochs << " epochs, batch=" << batchSize 
              << " (proper mini-batch SGD)\n";
    
    Timer totalTimer;
    totalTimer.start();
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        m_dataset.shuffleTrainData();
        
        m_timer.start();
        float epochLoss = trainEpoch(batchSize);
        double epochTime = m_timer.stop();
        
        m_logger.logEpoch(epoch, epochs, epochLoss, epochTime);
    }
    
    double totalTime = totalTimer.stop();
    std::cout << "Total: " << totalTime << "s (" << (totalTime / epochs) << "s/epoch)\n";
}

float Trainer::trainEpoch(int batchSize) {
    float totalLoss = 0.0f;
    int numBatches = m_dataset.getTrainSize() / batchSize;
    
    for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
        auto batchImages = m_dataset.getBatch(batchIdx, batchSize);
        
        float batchLoss = 0.0f;
        constexpr int IMAGE_SIZE = 32 * 32 * 3;  // CIFAR-10: 32x32 RGB = 3072 floats
        int actualBatchSize = static_cast<int>(batchImages.size()) / IMAGE_SIZE;
        
        // MINI-BATCH SGD: Zero gradients at the START of each batch
        m_model.zeroGradients();
        
        for (int i = 0; i < actualBatchSize; ++i) {
            std::vector<float> image(IMAGE_SIZE);
            std::copy(batchImages.begin() + i * IMAGE_SIZE,
                     batchImages.begin() + (i + 1) * IMAGE_SIZE,
                     image.begin());
            
            // Forward pass
            auto reconstructed = m_model.forward(image);
            float loss = m_lossFunction.forward(reconstructed, image);
            batchLoss += loss;
            
            // Backward pass - gradients are ACCUMULATED across the batch
            auto gradLoss = m_lossFunction.backward();
            m_model.backward(gradLoss);
        }
        
        // MINI-BATCH SGD: Update weights ONCE per batch (with averaged gradient)
        // Scale learning rate by batch size since gradients are summed, not averaged
        m_model.updateWeights(m_learningRate / static_cast<float>(actualBatchSize));
        
        totalLoss += batchLoss / static_cast<float>(actualBatchSize);
    }
    
    return totalLoss / static_cast<float>(numBatches);
}

