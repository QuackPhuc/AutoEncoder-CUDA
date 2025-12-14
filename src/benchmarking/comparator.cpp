#include "comparator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>

namespace {
constexpr double BYTES_TO_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
constexpr int TABLE_WIDTH = 118;

// Helper function to format memory size with proper precision
std::string formatMemoryGB(double memoryGB) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << memoryGB << " GB";
    return oss.str();
}

// Helper function to format speedup with proper precision
std::string formatSpeedup(double speedup) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << speedup << "x";
    return oss.str();
}
}

void PerformanceComparator::addResult(
    GPUVersion version,
    const GPUProfiler::Metrics& metrics)
{
    m_results[version] = metrics;
}

void PerformanceComparator::generateComparisonTable() const {
    if (m_results.empty()) {
        std::cout << "No results to compare.\n";
        return;
    }

    std::cout << "\n========================================\n";
    std::cout << " Performance Comparison Across Phases\n";
    std::cout << "========================================\n\n";

    std::cout << std::left
              << std::setw(20) << "Phase"
              << std::setw(15) << "Training Time"
              << std::setw(18) << "Speedup (vs CPU)"
              << std::setw(20) << "Incremental Speedup"
              << std::setw(15) << "Memory Usage"
              << std::setw(30) << "Key Optimization"
              << "\n";

    std::cout << std::string(TABLE_WIDTH, '-') << "\n";

    double cpuTime = 0.0;
    auto cpuIt = m_results.find(GPUVersion::CPU_BASELINE);
    if (cpuIt != m_results.end()) {
        cpuTime = cpuIt->second.trainingTimeSec;
    }

    double prevTime = cpuTime;

    for (const auto& [version, metrics] : m_results) {
        double speedupVsCPU = (cpuTime > 0.0) ? (cpuTime / metrics.trainingTimeSec) : 1.0;
        double incrementalSpeedup = (prevTime > 0.0) ? (prevTime / metrics.trainingTimeSec) : 1.0;

        std::cout << std::fixed << std::setprecision(1);
        std::cout << std::setw(20) << getVersionName(version);
        std::cout << std::setw(15) << (std::to_string(static_cast<int>(metrics.trainingTimeSec)) + "s");

        std::cout << std::setw(18);
        if (version == GPUVersion::CPU_BASELINE) {
            std::cout << "1.0x";
        } else {
            std::cout << formatSpeedup(speedupVsCPU);
        }

        std::cout << std::setw(20);
        if (version == GPUVersion::CPU_BASELINE) {
            std::cout << "-";
        } else {
            std::cout << formatSpeedup(incrementalSpeedup);
        }

        double memoryGB = metrics.gpuMemoryUsedBytes * BYTES_TO_GB;
        std::cout << std::setw(15) << formatMemoryGB(memoryGB);
        std::cout << std::setw(30) << getOptimizationName(version);
        std::cout << "\n";

        prevTime = metrics.trainingTimeSec;
    }

    std::cout << std::string(TABLE_WIDTH, '-') << "\n\n";
}

void PerformanceComparator::generateSpeedupChart(const std::string& filepath) const {
    std::ofstream outFile(filepath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open " << filepath << " for writing\n";
        return;
    }

    outFile << "Version,TrainingTime,SpeedupVsCPU,IncrementalSpeedup,MemoryGB\n";

    double cpuTime = 0.0;
    auto cpuIt = m_results.find(GPUVersion::CPU_BASELINE);
    if (cpuIt != m_results.end()) {
        cpuTime = cpuIt->second.trainingTimeSec;
    }

    double prevTime = cpuTime;

    for (const auto& [version, metrics] : m_results) {
        double speedupVsCPU = (cpuTime > 0.0) ? (cpuTime / metrics.trainingTimeSec) : 1.0;
        double incrementalSpeedup = (prevTime > 0.0) ? (prevTime / metrics.trainingTimeSec) : 1.0;
        double memoryGB = metrics.gpuMemoryUsedBytes * BYTES_TO_GB;

        outFile << getVersionName(version) << ","
                << metrics.trainingTimeSec << ","
                << speedupVsCPU << ","
                << incrementalSpeedup << ","
                << memoryGB << "\n";

        prevTime = metrics.trainingTimeSec;
    }

    std::cout << "Speedup chart data saved to: " << filepath << "\n";
}

void PerformanceComparator::printSummary() const {
    if (m_results.empty()) {
        std::cout << "No results available.\n";
        return;
    }

    std::cout << "\n========================================\n";
    std::cout << " Summary Statistics\n";
    std::cout << "========================================\n";

    GPUVersion fastestVersion = GPUVersion::CPU_BASELINE;
    double fastestTime = std::numeric_limits<double>::max();

    for (const auto& [version, metrics] : m_results) {
        if (metrics.trainingTimeSec < fastestTime) {
            fastestTime = metrics.trainingTimeSec;
            fastestVersion = version;
        }
    }

    std::cout << "Fastest Version:   " << getVersionName(fastestVersion) << "\n";
    std::cout << "Best Time:         " << static_cast<int>(fastestTime) << " seconds\n";

    auto cpuIt = m_results.find(GPUVersion::CPU_BASELINE);
    if (cpuIt != m_results.end()) {
        double totalSpeedup = cpuIt->second.trainingTimeSec / fastestTime;
        std::cout << "Total Speedup:     " << static_cast<int>(totalSpeedup) << "x vs CPU\n";
    }

    std::cout << "========================================\n\n";
}

std::string PerformanceComparator::getVersionName(GPUVersion version) const {
    switch (version) {
        case GPUVersion::CPU_BASELINE: return "CPU Baseline";
        case GPUVersion::GPU_BASIC:    return "GPU Basic";
        case GPUVersion::GPU_OPT_V1:   return "GPU Opt v1 (NCHW)";
        case GPUVersion::GPU_OPT_V2:   return "GPU Opt v2 (GEMM)";
        default:                       return "Unknown";
    }
}

std::string PerformanceComparator::getOptimizationName(GPUVersion version) const {
    switch (version) {
        case GPUVersion::CPU_BASELINE: return "-";
        case GPUVersion::GPU_BASIC:    return "Parallelization";
        case GPUVersion::GPU_OPT_V1:   return "NCHW Layout + 2D Grid";
        case GPUVersion::GPU_OPT_V2:   return "im2col + cuBLAS GEMM";
        default:                       return "Unknown";
    }
}
