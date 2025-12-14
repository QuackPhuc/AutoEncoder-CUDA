#ifndef BENCHMARKING_COMPARATOR_H
#define BENCHMARKING_COMPARATOR_H

#include "profiler.h"
#include "config/gpu_config.h"
#include <map>
#include <string>

// Performance comparison tool across GPU versions.
// Collects results and generates comparison tables/CSV for reports.
class PerformanceComparator {
public:
    void addResult(GPUVersion version, const GPUProfiler::Metrics& metrics);

    // Prints table: Phase | Time | Speedup vs CPU | Incremental | Memory | Optimization
    void generateComparisonTable() const;

    // Outputs CSV: Version,TrainingTime,SpeedupVsCPU,IncrementalSpeedup,MemoryGB
    void generateSpeedupChart(const std::string& filepath) const;

    void printSummary() const;

private:
    std::map<GPUVersion, GPUProfiler::Metrics> m_results;

    std::string getVersionName(GPUVersion version) const;
    std::string getOptimizationName(GPUVersion version) const;
};

#endif // BENCHMARKING_COMPARATOR_H
