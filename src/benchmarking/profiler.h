#ifndef BENCHMARKING_PROFILER_H
#define BENCHMARKING_PROFILER_H

#include <cstddef>
#include <string>

class GPUAutoencoder;
class CIFAR10Dataset;

// GPU training performance profiler.
// Collects training time, loss, and memory usage metrics.
class GPUProfiler {
public:
    struct Metrics {
        double trainingTimeSec;
        double timePerEpochSec;
        float finalLoss;
        size_t gpuMemoryUsedBytes;
        float kernelOccupancy;       // From nvprof (optional)
        float memoryBandwidthUtil;   // From nvprof (optional)

        Metrics()
            : trainingTimeSec(0.0)
            , timePerEpochSec(0.0)
            , finalLoss(0.0f)
            , gpuMemoryUsedBytes(0)
            , kernelOccupancy(0.0f)
            , memoryBandwidthUtil(0.0f) {}
    };

    Metrics profileTraining(GPUAutoencoder& model,
                            CIFAR10Dataset& dataset,
                            int epochs);

    void printMetrics(const Metrics& metrics,
                      const std::string& versionName) const;

    // Appends metrics to CSV file
    void saveMetrics(const Metrics& metrics,
                     const std::string& versionName,
                     const std::string& filepath) const;

private:
    size_t getGPUMemoryUsage() const;
};

#endif // BENCHMARKING_PROFILER_H
