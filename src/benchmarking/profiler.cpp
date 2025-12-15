#include "profiler.h"
#include "gpu/model/autoencoder.h"
#include "cpu/data/cifar_loader.h"
#include "config/gpu_config.h"
#include "utils/timer.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

namespace {
constexpr double BYTES_TO_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
constexpr int IMAGE_SIZE = 32 * 32 * 3;  // CIFAR-10: 32x32 RGB
}

GPUProfiler::Metrics GPUProfiler::profileTraining(
    GPUAutoencoder& model,
    CIFAR10Dataset& dataset,
    int epochs)
{
    Metrics metrics;
    Timer totalTimer;
    totalTimer.start();

    int batchSize = GPUConfig::getInstance().getBatchSize();
    
    const auto& trainImages = dataset.getTrainImages();
    int numBatches = trainImages.size() / (batchSize * IMAGE_SIZE);

    std::cout << "\nTraining: " << epochs << " epochs, " 
              << numBatches << " batches/epoch\n" << std::flush;

    float finalLoss = 0.0f;
    Timer epochTimer;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        epochTimer.start();
        float epochLoss = 0.0f;
        
        for (int batch = 0; batch < numBatches; ++batch) {
            std::vector<float> batchData(
                trainImages.begin() + batch * batchSize * IMAGE_SIZE,
                trainImages.begin() + (batch + 1) * batchSize * IMAGE_SIZE
            );
            model.trainStep(batchData);
            epochLoss = model.getLoss();
        }
        
        double epochTime = epochTimer.stop();
        
        finalLoss = epochLoss;
        std::cout << "  Epoch " << std::setw(2) << (epoch + 1) << "/" << epochs 
                  << " | Loss: " << std::fixed << std::setprecision(6) << epochLoss
                  << " | " << std::setprecision(1) << epochTime << "s\n" << std::flush;
    }

    metrics.trainingTimeSec = totalTimer.stop();
    metrics.timePerEpochSec = metrics.trainingTimeSec / epochs;
    metrics.finalLoss = finalLoss;
    metrics.gpuMemoryUsedBytes = getGPUMemoryUsage();

    return metrics;
}

void GPUProfiler::printMetrics(
    const Metrics& metrics,
    const std::string& versionName) const
{
    std::cout << "\n========================================\n";
    std::cout << " Performance Metrics: " << versionName << "\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Training Time:     " << metrics.trainingTimeSec << " sec\n";
    std::cout << "Time per Epoch:    " << metrics.timePerEpochSec << " sec\n";
    std::cout << "Final Loss:        " << metrics.finalLoss << "\n";
    std::cout << std::setprecision(1);
    std::cout << "GPU Memory Used:   "
              << (metrics.gpuMemoryUsedBytes * BYTES_TO_GB) << " GB\n";

    if (metrics.kernelOccupancy > 0.0f) {
        std::cout << "Kernel Occupancy:  "
                  << (metrics.kernelOccupancy * 100.0f) << "%\n";
    }
    if (metrics.memoryBandwidthUtil > 0.0f) {
        std::cout << "Memory Bandwidth:  "
                  << (metrics.memoryBandwidthUtil * 100.0f) << "%\n";
    }

    std::cout << "========================================\n";
}

size_t GPUProfiler::getGPUMemoryUsage() const {
    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        std::cerr << "Warning: Could not query GPU memory: "
                  << cudaGetErrorString(err) << "\n";
        return 0;
    }
    return totalMem - freeMem;
}

