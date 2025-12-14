#ifndef CIFAR10_CONSTANTS_H
#define CIFAR10_CONSTANTS_H

#include <string>
#include <vector>

namespace evaluation {

// Single source of truth for CIFAR-10 class names
// Used by metrics.cpp, visualizer.cpp, and any other evaluation code
inline const std::vector<std::string> CIFAR10_CLASS_NAMES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// Number of classes in CIFAR-10
constexpr int CIFAR10_NUM_CLASSES = 10;

// Image dimensions
constexpr int CIFAR10_IMAGE_WIDTH = 32;
constexpr int CIFAR10_IMAGE_HEIGHT = 32;
constexpr int CIFAR10_IMAGE_CHANNELS = 3;
constexpr int CIFAR10_IMAGE_SIZE = CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT * CIFAR10_IMAGE_CHANNELS;

} // namespace evaluation

#endif // CIFAR10_CONSTANTS_H
