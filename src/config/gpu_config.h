#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <string>

// GPU implementation versions for performance comparison
enum class GPUVersion {
    CPU_BASELINE = 0, // CPU baseline
    GPU_BASIC = 1,    // GPU basic
    GPU_OPT_V1 = 2,   // NCHW layout + 2D grid indexing (formerly V3)
    GPU_OPT_V2 = 3    // im2col + cuBLAS GEMM (formerly V4)
};

// Default configuration constants
static constexpr int DEFAULT_BLOCK_SIZE = 256;
static constexpr int CPU_DEFAULT_BATCH = 32;
static constexpr int GPU_DEFAULT_BATCH = 64;

// Singleton configuration for GPU version selection
class GPUConfig {
public:
    static GPUConfig& getInstance();
    void parseCommandLine(int argc, char** argv);
    
    GPUVersion getVersion() const { return m_version; }
    std::string getVersionName() const;
    std::string getVersionFileName() const;
    
    int getBatchSize() const;
    void setBatchSize(int batchSize);
    void setVersion(GPUVersion version) { m_version = version; }
    
    // Returns CUDA block size (threads per block)
    int getBlockSize() const;
    
    // Feature flags based on version
    bool useSharedMemory() const;
    bool useKernelFusion() const;
    bool useStreams() const;
    
    std::string getOptimizationDesc() const;
    
private:
    GPUConfig() : m_version(GPUVersion::GPU_BASIC), m_batchSize(0) {}
    GPUConfig(const GPUConfig&) = delete;
    GPUConfig& operator=(const GPUConfig&) = delete;
    
    GPUVersion m_version;
    int m_batchSize;
};

#endif // GPU_CONFIG_H
