#ifndef TRAINER_H
#define TRAINER_H

#include "cpu/model/autoencoder.h"
#include "cpu/data/cifar_loader.h"
#include "cpu/layers/mse_loss.h"
#include "utils/logger.h"
#include "utils/timer.h"

// Training loop orchestrator for autoencoder
class Trainer {
public:
    Trainer(Autoencoder& model, CIFAR10Dataset& dataset, float learningRate);
    
    // Main training loop
    void train(int epochs, int batchSize);
    
    // Train a single epoch, returns average loss
    float trainEpoch(int batchSize);

private:
    Autoencoder& m_model;
    CIFAR10Dataset& m_dataset;
    float m_learningRate;
    MSELoss m_lossFunction;
    Logger m_logger;
    Timer m_timer;
};

#endif // TRAINER_H
