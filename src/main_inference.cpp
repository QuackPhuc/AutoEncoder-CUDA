#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include "gpu/core/device_reset.h"
#include "cpu/data/cifar_loader.h"
#include "gpu/svm/svm.h"
#include "cpu/evaluation/metrics.h"
#include "cpu/evaluation/visualizer.h"
#include "gpu/inference/feature_extractor.h"

void showHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "OPTIONS:\n"
              << "  --encoder-weights PATH  Encoder weights path (default: ./checkpoints/encoder.weights)\n"
              << "  --svm-model PATH        SVM model path (default: ./checkpoints/svm.bin)\n"
              << "  --batch-size N          Batch size for feature extraction (default: 128)\n"
              << "  --train-svm             Train SVM and save model (default behavior)\n"
              << "  --evaluate-only         Skip SVM training, load existing model and evaluate\n"
              << "  --help                  Show this help message\n";
}

int main(int argc, char** argv) {
    resetGPUDevice();
    
    try {
        // Default paths
        std::string data_dir = "./data";
        std::string encoder_weights = "./checkpoints/encoder.weights";
        std::string svm_model_path = "./checkpoints/svm.bin";
        std::string report_path = "./results/classification_report.txt";
        std::string confusion_csv = "./results/confusion_matrix.csv";
        int batch_size = 128;
        bool train_svm = true;  // Default: train SVM
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--encoder-weights") == 0 && i + 1 < argc) {
                encoder_weights = argv[++i];
            } else if (strcmp(argv[i], "--svm-model") == 0 && i + 1 < argc) {
                svm_model_path = argv[++i];
            } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
                batch_size = std::atoi(argv[++i]);
                if (batch_size <= 0) {
                    std::cerr << "Error: batch-size must be positive\n";
                    return 1;
                }
            } else if (strcmp(argv[i], "--train-svm") == 0) {
                train_svm = true;
            } else if (strcmp(argv[i], "--evaluate-only") == 0) {
                train_svm = false;
            } else if (strcmp(argv[i], "--help") == 0) {
                showHelp(argv[0]);
                return 0;
            }
        }
        
        // Check encoder weights exist
        {
            std::ifstream f(encoder_weights);
            if (!f.good()) {
                std::cerr << "Error: Encoder weights not found: " << encoder_weights << "\n";
                std::cerr << "Run training first: ./run.sh train-autoencoder\n";
                return 1;
            }
        }
        
        // Check SVM model if evaluate-only
        if (!train_svm) {
            std::ifstream f(svm_model_path);
            if (!f.good()) {
                std::cerr << "Error: SVM model not found: " << svm_model_path << "\n";
                std::cerr << "Run SVM training first: ./run.sh train-svm\n";
                return 1;
            }
        }
        
        std::cout << "=== Inference Pipeline ===\n";
        std::cout << "Encoder: " << encoder_weights << "\n";
        std::cout << "SVM:     " << svm_model_path << (train_svm ? " (will train)" : " (pre-trained)") << "\n\n";
        
        // Load dataset
        std::cout << "Loading CIFAR-10...\n";
        CIFAR10Dataset dataset(data_dir);
        dataset.loadTrainData();
        dataset.loadTestData();
        
        auto train_images = dataset.getTrainImages();
        auto train_labels = dataset.getTrainLabels();
        auto test_images = dataset.getTestImages();
        auto test_labels = dataset.getTestLabels();
        
        // Extract features using GPU encoder
        std::cout << "Extracting features...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        FeatureExtractor extractor(encoder_weights, batch_size);
        
        std::vector<float> train_features, test_features;
        
        if (train_svm) {
            train_features = extractor.extract_all(train_images, 50000, batch_size);
        }
        test_features = extractor.extract_all(test_images, 10000, batch_size);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double extraction_time = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Feature extraction: " << std::fixed << std::setprecision(1) << extraction_time << "s\n\n";
        
        // SVM
        gpu_svm::ThunderSVMTrainer svm_trainer(10.0, -1.0);
        double svm_train_time = 0;
        
        if (train_svm) {
            std::cout << "Training SVM (ThunderSVM GPU)...\n";
            auto svm_start = std::chrono::high_resolution_clock::now();
            
            svm_trainer.train(train_features, train_labels, 50000, extractor.get_feature_dim());
            svm_trainer.save_model(svm_model_path);
            
            // Free train_features memory after training
            train_features.clear();
            train_features.shrink_to_fit();
            
            auto svm_end = std::chrono::high_resolution_clock::now();
            svm_train_time = std::chrono::duration<double>(svm_end - svm_start).count();
            std::cout << "SVM training: " << svm_train_time << "s\n";
            std::cout << "Model saved: " << svm_model_path << "\n\n";
        } else {
            std::cout << "Loading SVM model: " << svm_model_path << "\n";
            svm_trainer.load_model(svm_model_path);
        }
        
        // Predict and evaluate
        std::cout << "Evaluating on test set...\n";
        auto predictions = svm_trainer.predict_batch(test_features, 10000, extractor.get_feature_dim());
        
        auto metrics = evaluation::MetricsCalculator::calculate(predictions, test_labels, 10);
        
        metrics.print_summary();
        evaluation::ResultsVisualizer::print_confusion_matrix(
            metrics.confusion_matrix,
            evaluation::CIFAR10_CLASS_NAMES);
        
        // Save results
        evaluation::ResultsVisualizer::generate_report(
            metrics, 
            evaluation::CIFAR10_CLASS_NAMES,
            report_path);
        
        evaluation::ResultsVisualizer::save_confusion_matrix_csv(
            metrics.confusion_matrix,
            confusion_csv);
        
        // Summary
        std::cout << "\n=== Results ===\n";
        std::cout << "Accuracy:    " << std::fixed << std::setprecision(2)
                  << metrics.overall_accuracy << "%\n";
        std::cout << "Extraction:  " << std::setprecision(1) << extraction_time << "s\n";
        if (train_svm) {
            std::cout << "SVM Train:   " << svm_train_time << "s\n";
        }
        std::cout << "Report:      " << report_path << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
}
