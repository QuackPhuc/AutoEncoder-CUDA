#include <iostream>
#include <chrono>
#include <cstring>
#include "cpu/data/cifar_loader.h"
#include "cpu/model/autoencoder.h"
#include "cpu/training/trainer.h"
#include "utils/timer.h"

void showUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --epochs <N>        Number of training epochs (default: 20)\n";
    std::cout << "  --samples <N>       Number of training samples (0=all, default: 0)\n";
    std::cout << "  --batch-size <N>    Batch size (default: 32)\n";
    std::cout << "  --save-weights <P>  Custom weights output path\n";
    std::cout << "  --help              Show this help message\n";
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    int epochs = 20;
    int numSamples = 0;
    int batchSize = 32;
    std::string saveWeightsPath = "";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
            if (epochs <= 0) {
                std::cerr << "Error: epochs must be a positive integer\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            numSamples = std::atoi(argv[++i]);
            if (numSamples < 0) {
                std::cerr << "Error: samples must be a non-negative integer\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batchSize = std::atoi(argv[++i]);
            if (batchSize <= 0) {
                std::cerr << "Error: batch-size must be a positive integer\n";
                return 1;
            }
        } else if (strcmp(argv[i], "--save-weights") == 0 && i + 1 < argc) {
            saveWeightsPath = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            showUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            showUsage(argv[0]);
            return 1;
        }
    }
    
    try {
        std::cout << "========================================\n";
        std::cout << " AutoEncoder CUDA - CPU Baseline\n";
        std::cout << "========================================\n\n";
        
        std::cout << "Configuration:\n";
        std::cout << "  Epochs:     " << epochs << "\n";
        std::cout << "  Samples:    " << (numSamples == 0 ? "all" : std::to_string(numSamples)) << "\n";
        std::cout << "  Batch Size: " << batchSize << "\n\n";
        
        std::cout << "Loading CIFAR-10 dataset...\n";
        CIFAR10Dataset dataset("./data");
        dataset.loadTrainData(numSamples);
        dataset.loadTestData();
        std::cout << "Dataset loaded.\n\n";
        
        std::cout << "Creating autoencoder model...\n";
        Autoencoder model;
        std::cout << "Model created.\n\n";
        
        std::cout << "Starting training...\n";
        Trainer trainer(model, dataset, 0.001f);
        
        Timer totalTimer;
        totalTimer.start();
        trainer.train(epochs, batchSize);
        double totalTime = totalTimer.stop();
        
        std::string modelPath = saveWeightsPath.empty() 
            ? "./checkpoints/cpu_baseline.weights" 
            : saveWeightsPath;
        model.saveModel(modelPath);
        
        std::cout << "\n========================================\n";
        std::cout << " Training Complete\n";
        std::cout << "========================================\n";
        std::cout << "Total time:     " << totalTime << " s\n";
        std::cout << "Time per epoch: " << (totalTime / epochs) << " s\n";
        std::cout << "Model saved:    " << modelPath << "\n";
        std::cout << "========================================\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
