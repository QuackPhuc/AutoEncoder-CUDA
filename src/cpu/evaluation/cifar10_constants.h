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

} // namespace evaluation

#endif // CIFAR10_CONSTANTS_H
