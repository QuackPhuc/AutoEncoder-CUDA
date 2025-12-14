#include "visualizer.h"
#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace evaluation {

void ResultsVisualizer::print_confusion_matrix(
    const std::vector<std::vector<int>>& matrix,
    const std::vector<std::string>& class_names) {
    
    int num_classes = static_cast<int>(matrix.size());
    
    std::cout << "\n========== Confusion Matrix ==========\n";
    std::cout << "Format: Rows=Actual, Columns=Predicted\n\n";
    
    // Header row
    std::cout << std::setw(12) << " ";
    for (int i = 0; i < num_classes; ++i) {
        std::string label = class_names[i].substr(0, 8);
        std::cout << std::setw(9) << label;
    }
    std::cout << "\n";
    std::cout << std::string(12 + 9 * num_classes, '-') << "\n";
    
    // Matrix rows
    for (int i = 0; i < num_classes; ++i) {
        std::string label = class_names[i].substr(0, 10);
        std::cout << std::setw(12) << label;
        
        for (int j = 0; j < num_classes; ++j) {
            std::cout << std::setw(9) << matrix[i][j];
        }
        std::cout << "\n";
    }
    
    std::cout << "======================================\n\n";
}

void ResultsVisualizer::save_confusion_matrix_csv(
    const std::vector<std::vector<int>>& matrix,
    const std::string& filepath) {
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    int num_classes = static_cast<int>(matrix.size());
    
    // Header row
    file << "Actual\\Predicted";
    for (int i = 0; i < num_classes; ++i) {
        file << "," << CIFAR10_CLASS_NAMES[i];
    }
    file << "\n";
    
    // Matrix rows
    for (int i = 0; i < num_classes; ++i) {
        file << CIFAR10_CLASS_NAMES[i];
        for (int j = 0; j < num_classes; ++j) {
            file << "," << matrix[i][j];
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "Confusion matrix saved to: " << filepath << "\n";
}

void ResultsVisualizer::generate_report(
    const ClassificationMetrics& metrics,
    const std::vector<std::string>& class_names,
    const std::string& output_path) {
    
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + output_path);
    }
    
    file << "=================================================\n";
    file << "      CIFAR-10 Classification Report\n";
    file << "=================================================\n\n";
    
    file << "OVERALL ACCURACY: " << std::fixed << std::setprecision(2)
         << metrics.overall_accuracy << "%\n\n";
    
    file << "PER-CLASS PERFORMANCE:\n";
    file << std::setw(15) << "Class" 
         << std::setw(12) << "Accuracy" 
         << std::setw(10) << "Samples\n";
    file << std::string(37, '-') << "\n";
    
    for (size_t i = 0; i < class_names.size(); ++i) {
        file << std::setw(15) << class_names[i]
             << std::setw(11) << std::fixed << std::setprecision(2)
             << metrics.per_class_accuracy[i] << "%"
             << std::setw(10) << metrics.class_counts[i] << "\n";
    }
    
    file << "\n";
    
    // Confusion matrix
    file << "CONFUSION MATRIX (Rows=Actual, Columns=Predicted):\n";
    file << std::setw(12) << " ";
    for (size_t i = 0; i < class_names.size(); ++i) {
        file << std::setw(9) << class_names[i].substr(0, 7);
    }
    file << "\n";
    
    for (size_t i = 0; i < class_names.size(); ++i) {
        file << std::setw(12) << class_names[i].substr(0, 10);
        for (size_t j = 0; j < class_names.size(); ++j) {
            file << std::setw(9) << metrics.confusion_matrix[i][j];
        }
        file << "\n";
    }
    
    file << "\n=================================================\n";
    file.close();
    std::cout << "Report saved to: " << output_path << "\n";
}

} // namespace evaluation
