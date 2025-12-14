cmake_minimum_required(VERSION 3.18)
project(AutoEncoderCUDA LANGUAGES CXX CUDA)

# ==============================================================================
# Build Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architectures: Tesla T4 (75) is primary for Colab, include others for compatibility
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Find CUDA Toolkit for include directories (needed by ThunderSVM headers in .cpp files)
find_package(CUDAToolkit REQUIRED)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# Source Files
# ==============================================================================
file(GLOB_RECURSE CPU_DATA_SOURCES CONFIGURE_DEPENDS "src/cpu/data/*.cpp")
file(GLOB_RECURSE CPU_LAYERS_SOURCES CONFIGURE_DEPENDS "src/cpu/layers/*.cpp")
file(GLOB_RECURSE CPU_MODEL_SOURCES CONFIGURE_DEPENDS "src/cpu/model/*.cpp")
file(GLOB_RECURSE CPU_TRAINING_SOURCES CONFIGURE_DEPENDS "src/cpu/training/*.cpp")
file(GLOB_RECURSE UTIL_SOURCES CONFIGURE_DEPENDS "src/utils/*.cpp")
file(GLOB_RECURSE GPU_SOURCES CONFIGURE_DEPENDS "src/gpu/*.cu")
file(GLOB_RECURSE CONFIG_SOURCES CONFIGURE_DEPENDS "src/config/*.cpp")
file(GLOB_RECURSE BENCHMARK_SOURCES CONFIGURE_DEPENDS "src/benchmarking/*.cpp")
file(GLOB_RECURSE EVAL_SOURCES CONFIGURE_DEPENDS "src/cpu/evaluation/*.cpp")
file(GLOB_RECURSE GPU_SVM_SOURCES CONFIGURE_DEPENDS "src/gpu/svm/*.cpp")

# ==============================================================================
# ThunderSVM Integration
# ==============================================================================
set(THUNDERSVM_DIR "${PROJECT_SOURCE_DIR}/external/thundersvm")
if(NOT EXISTS "${THUNDERSVM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ThunderSVM not found. Run: git submodule update --init --recursive")
endif()

# ThunderSVM uses legacy cuda_add_library which ignores CMAKE_CUDA_ARCHITECTURES.
# We must set CUDA_NVCC_FLAGS explicitly for Tesla T4 (sm_75) compatibility.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=compute_75")

add_subdirectory(${THUNDERSVM_DIR})

# ==============================================================================
# Helper Function: Configure CUDA Target
# ==============================================================================
function(configure_cuda_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2>
            $<$<COMPILE_LANGUAGE:CUDA>:/O2>
        )
    else()
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        )
    endif()
endfunction()

# ==============================================================================
# CPU Executable
# ==============================================================================
add_executable(autoencoder_cpu
    src/main.cpp
    ${CPU_DATA_SOURCES}
    ${CPU_LAYERS_SOURCES}
    ${CPU_MODEL_SOURCES}
    ${CPU_TRAINING_SOURCES}
    ${UTIL_SOURCES}
)
target_include_directories(autoencoder_cpu PRIVATE ${PROJECT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(autoencoder_cpu PRIVATE /W4 /O2)
else()
    target_compile_options(autoencoder_cpu PRIVATE -Wall -Wextra -O3)
endif()

# ==============================================================================
# GPU Executable
# ==============================================================================
add_executable(autoencoder_gpu
    src/main_gpu.cpp
    ${CPU_DATA_SOURCES}
    ${UTIL_SOURCES}
    ${GPU_SOURCES}
    ${CONFIG_SOURCES}
    ${BENCHMARK_SOURCES}
)
target_link_libraries(autoencoder_gpu PRIVATE CUDA::cublas CUDA::cudart)
configure_cuda_target(autoencoder_gpu)

# ==============================================================================
# Inference Executable (ThunderSVM)
# ==============================================================================
add_executable(autoencoder_inference
    src/main_inference.cpp
    ${CPU_DATA_SOURCES}
    ${UTIL_SOURCES}
    ${EVAL_SOURCES}
    ${GPU_SVM_SOURCES}
    ${GPU_SOURCES}
    ${CONFIG_SOURCES}
)
target_include_directories(autoencoder_inference PRIVATE 
    ${THUNDERSVM_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/external/thundersvm
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(autoencoder_inference PRIVATE thundersvm CUDA::cublas CUDA::cudart)
configure_cuda_target(autoencoder_inference)

# ==============================================================================
# Output Directories
# ==============================================================================
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/checkpoints)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

# ==============================================================================
# Build Summary
# ==============================================================================
message(STATUS "========================================")
message(STATUS "AutoEncoderCUDA Build Configuration")
message(STATUS "========================================")
message(STATUS "Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:     ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "========================================")
