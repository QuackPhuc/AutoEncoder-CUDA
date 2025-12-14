#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "metrics.h"
#include "cifar10_constants.h"
#include <vector>
#include <string>

namespace evaluation {

// Visualization and reporting utility class
class ResultsVisualizer {
public:
    // Print confusion matrix to console (Rows=Actual, Columns=Predicted)
    static void print_confusion_matrix(
        const std::vector<std::vector<int>>& matrix,
        const std::vector<std::string>& class_names
    );
    
    // Save confusion matrix as CSV file
    static void save_confusion_matrix_csv(
        const std::vector<std::vector<int>>& matrix,
        const std::string& filepath
    );
    
    // Generate comprehensive text report
    static void generate_report(
        const ClassificationMetrics& metrics,
        const std::vector<std::string>& class_names,
        const std::string& output_path
    );
};

} // namespace evaluation

#endif // VISUALIZER_H
