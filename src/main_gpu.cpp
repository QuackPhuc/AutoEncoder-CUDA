#include <iostream>
#include <cstring>
#include "gpu/core/device_reset.h"
#include "gpu/model/autoencoder.h"
#include "cpu/data/cifar_loader.h"
#include "config/gpu_config.h"
#include "benchmarking/profiler.h"

void showUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --gpu-version <N>      GPU version (default: 1)\n";
    std::cout << "                         0 = CPU Baseline\n";
    std::cout << "                         1 = GPU Basic\n";
    std::cout << "                         2 = GPU Opt v1 (NCHW+2DGrid+WarpShuffle)\n";
    std::cout << "                         3 = GPU Opt v2 (im2col+GEMM)\n";
    std::cout << "  --epochs <N>           Number of training epochs (default: 20)\n";
    std::cout << "  --samples <N>          Number of training samples (0=all, default: 0)\n";
    std::cout << "  --batch-size <N>       Batch size (0=auto, default: 0)\n";
    std::cout << "  --save-weights <PATH>  Custom weights output path\n";
    std::cout << "  --help                 Show this help message\n";
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    resetGPUDevice();
    
    GPUConfig& config = GPUConfig::getInstance();
    config.parseCommandLine(argc, argv);
    
    int epochs = 20;
    int numSamples = 0;
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
        } else if (strcmp(argv[i], "--save-weights") == 0 && i + 1 < argc) {
            saveWeightsPath = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            showUsage(argv[0]);
            return 0;
        }
    }
    
    try {
        std::cout << "AutoEncoder CUDA | " << config.getVersionName() << "\n";
        std::cout << "Epochs: " << epochs << " | Batch: " << config.getBatchSize()
                  << " | Samples: " << (numSamples == 0 ? "all" : std::to_string(numSamples)) << "\n";
        
        CIFAR10Dataset dataset("./data");
        dataset.loadTrainData(numSamples);
        dataset.loadTestData();
        
        int batchSize = config.getBatchSize();
        GPUAutoencoder model(batchSize, 0.001f);
        
        GPUProfiler profiler;
        auto metrics = profiler.profileTraining(model, dataset, epochs);
        
        profiler.printMetrics(metrics, config.getVersionName());
        
        // Use custom path if provided, otherwise use version-based naming
        std::string modelPath = saveWeightsPath.empty() 
            ? "./checkpoints/" + config.getVersionFileName() + ".weights"
            : saveWeightsPath;
        model.saveModel(modelPath);
        
        std::cout << "\nModel saved: " << modelPath << "\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
