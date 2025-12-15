#include "metrics.h"
#include "cifar10_constants.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace evaluation {

void ClassificationMetrics::print_summary() const {
    std::cout << "\n========== Classification Metrics ==========\n";
    std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(2)
              << overall_accuracy << "%\n\n";
    
    std::cout << "Per-Class Accuracy:\n";
    std::cout << std::setw(15) << "Class" 
              << std::setw(12) << "Accuracy" 
              << std::setw(10) << "Count\n";
    std::cout << std::string(37, '-') << "\n";
    
    for (size_t i = 0; i < per_class_accuracy.size(); ++i) {
        const std::string& class_name = (i < CIFAR10_CLASS_NAMES.size()) 
            ? CIFAR10_CLASS_NAMES[i] : "Unknown";
        std::cout << std::setw(15) << class_name
                  << std::setw(11) << std::fixed << std::setprecision(2)
                  << per_class_accuracy[i] << "%"
                  << std::setw(10) << class_counts[i] << "\n";
    }
    std::cout << "============================================\n\n";
}

ClassificationMetrics MetricsCalculator::calculate(
    const std::vector<int>& predictions,
    const std::vector<uint8_t>& ground_truth,
    int num_classes) {
    
    if (predictions.size() != ground_truth.size()) {
        throw std::invalid_argument(
            "Predictions and ground truth must have same size");
    }
    
    ClassificationMetrics metrics;
    
    metrics.confusion_matrix = build_confusion_matrix(
        predictions, ground_truth, num_classes);
    
    // Overall accuracy
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == static_cast<int>(ground_truth[i])) {
            correct++;
        }
    }
    metrics.overall_accuracy = 
        static_cast<float>(correct) / predictions.size() * 100.0f;
    
    // Per-class accuracy
    metrics.per_class_accuracy.resize(num_classes, 0.0f);
    metrics.class_counts.resize(num_classes, 0);
    
    for (int c = 0; c < num_classes; ++c) {
        int class_correct = metrics.confusion_matrix[c][c];
        int class_total = 0;
        for (int j = 0; j < num_classes; ++j) {
            class_total += metrics.confusion_matrix[c][j];
        }
        
        metrics.class_counts[c] = class_total;
        if (class_total > 0) {
            metrics.per_class_accuracy[c] = 
                static_cast<float>(class_correct) / class_total * 100.0f;
        }
    }
    
    return metrics;
}

std::vector<std::vector<int>> MetricsCalculator::build_confusion_matrix(
    const std::vector<int>& predictions,
    const std::vector<uint8_t>& ground_truth,
    int num_classes) {
    
    std::vector<std::vector<int>> matrix(
        num_classes, std::vector<int>(num_classes, 0));
    
    // matrix[true_class][predicted_class]
    for (size_t i = 0; i < predictions.size(); ++i) {
        int true_class = static_cast<int>(ground_truth[i]);
        int pred_class = predictions[i];
        
        if (true_class >= 0 && true_class < num_classes &&
            pred_class >= 0 && pred_class < num_classes) {
            matrix[true_class][pred_class]++;
        }
    }
    
    return matrix;
}

} // namespace evaluation
