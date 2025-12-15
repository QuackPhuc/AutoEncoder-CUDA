#ifndef METRICS_H
#define METRICS_H

#include <vector>
#include <string>
#include <cstdint>

namespace evaluation {

// Structure holding classification metrics
struct ClassificationMetrics {
    float overall_accuracy;                        // Overall accuracy (%)
    std::vector<float> per_class_accuracy;         // Per-class accuracy (%)
    std::vector<std::vector<int>> confusion_matrix; // Confusion matrix (NxN)
    std::vector<int> class_counts;                 // Samples per class
    
    void print_summary() const;
};

// Metrics calculation utility class
class MetricsCalculator {
public:
    // Calculate classification metrics from predictions and ground truth
    static ClassificationMetrics calculate(
        const std::vector<int>& predictions,
        const std::vector<uint8_t>& ground_truth,
        int num_classes = 10
    );

private:
    // Build confusion matrix[true_class][predicted_class]
    static std::vector<std::vector<int>> build_confusion_matrix(
        const std::vector<int>& predictions,
        const std::vector<uint8_t>& ground_truth,
        int num_classes
    );
};

} // namespace evaluation

#endif // METRICS_H
