#include "gpu_config.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

GPUConfig& GPUConfig::getInstance() {
    static GPUConfig instance;
    return instance;
}

void GPUConfig::parseCommandLine(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu-version") == 0 && i + 1 < argc) {
            try {
                int versionNum = std::stoi(argv[++i]);
                if (versionNum < 0 || versionNum > 4) {
                    std::cerr << "Error: gpu-version must be 0-4" << std::endl;
                    std::cerr << "  0 = CPU Baseline" << std::endl;
                    std::cerr << "  1 = GPU Basic (Phase 2)" << std::endl;
                    std::cerr << "  2 = GPU Opt v1 (NCHW)" << std::endl;
                    std::cerr << "  3 = GPU Opt v2 (im2col+GEMM)" << std::endl;
                    std::cerr << "  4 = GPU Opt v3 (im2col+GEMM+Streams)" << std::endl;
                    std::exit(1);
                }
                m_version = static_cast<GPUVersion>(versionNum);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid gpu-version value. Must be an integer 0-4." << std::endl;
                std::exit(1);
            }
        } else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            try {
                int batchSize = std::stoi(argv[++i]);
                if (batchSize > 0) {
                    m_batchSize = batchSize;
                } else {
                    std::cerr << "Warning: batch-size must be positive. Ignoring invalid value." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid batch-size value. Using default." << std::endl;
            }
        }
    }
}

std::string GPUConfig::getVersionName() const {
    switch (m_version) {
        case GPUVersion::CPU_BASELINE: return "CPU Baseline";
        case GPUVersion::GPU_BASIC:    return "GPU Basic (Phase 2)";
        case GPUVersion::GPU_OPT_V1:   return "GPU Opt v1 (NCHW)";
        case GPUVersion::GPU_OPT_V2:   return "GPU Opt v2 (im2col+GEMM)";
        case GPUVersion::GPU_OPT_V3:   return "GPU Opt v3 (Streams)";
        default:                       return "Unknown";
    }
}

std::string GPUConfig::getVersionFileName() const {
    switch (m_version) {
        case GPUVersion::CPU_BASELINE: return "cpu_baseline";
        case GPUVersion::GPU_BASIC:    return "gpu_basic";
        case GPUVersion::GPU_OPT_V1:   return "gpu_opt_v1";
        case GPUVersion::GPU_OPT_V2:   return "gpu_opt_v2";
        case GPUVersion::GPU_OPT_V3:   return "gpu_opt_v3";
        default:                       return "model";
    }
}

int GPUConfig::getBatchSize() const {
    if (m_batchSize > 0) {
        return m_batchSize;
    }
    
    switch (m_version) {
        case GPUVersion::CPU_BASELINE: return CPU_DEFAULT_BATCH;
        case GPUVersion::GPU_BASIC:    return GPU_DEFAULT_BATCH;
        case GPUVersion::GPU_OPT_V1:   return GPU_DEFAULT_BATCH;
        case GPUVersion::GPU_OPT_V2:   return GPU_DEFAULT_BATCH;
        case GPUVersion::GPU_OPT_V3:   return GPU_DEFAULT_BATCH;
        default:                       return CPU_DEFAULT_BATCH;
    }
}
