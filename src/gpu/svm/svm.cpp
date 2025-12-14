#include "svm.h"
#include <thundersvm/model/svc.h>
#include <thundersvm/dataset.h>
#include <thundersvm/svmparam.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>

namespace gpu_svm {

// Memory limit for ThunderSVM working set (1GB)
static constexpr size_t MAX_SVM_MEMORY_MB = 1024;

ThunderSVMTrainer::ThunderSVMTrainer(double C, double gamma)
    : model_(nullptr), C_(C), gamma_(gamma) {
}

ThunderSVMTrainer::~ThunderSVMTrainer() = default;

void ThunderSVMTrainer::train(
    const std::vector<float>& train_features,
    const std::vector<uint8_t>& train_labels,
    int num_samples,
    int feature_dim) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert labels to float vector for ThunderSVM
    std::vector<float> labels_float(train_labels.begin(), train_labels.end());
    
    // Create dataset using dense format
    DataSet dataset;
    dataset.load_from_dense(
        num_samples,
        feature_dim,
        const_cast<float*>(train_features.data()),
        labels_float.data());
    
    // Set up SVM parameters
    SvmParam param;
    param.svm_type = SvmParam::C_SVC;
    param.kernel_type = SvmParam::RBF;
    param.C = static_cast<float>(C_);
    
    // Auto gamma: 1/num_features
    if (gamma_ < 0) {
        param.gamma = 1.0f / feature_dim;
    } else {
        param.gamma = static_cast<float>(gamma_);
    }
    
    param.max_mem_size = MAX_SVM_MEMORY_MB << 20;
    
    // Create and train model
    model_ = std::make_unique<SVC>();
    model_->train(dataset, param);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double training_time = std::chrono::duration<double>(
        end_time - start_time).count();
    
    std::cout << "ThunderSVM: " << model_->total_sv() << " SVs, " 
              << training_time << "s\n";
}

void ThunderSVMTrainer::save_model(const std::string& filepath) const {
    if (!model_) {
        throw std::runtime_error("Cannot save model: no model trained");
    }
    model_->save_to_file(filepath);
}

void ThunderSVMTrainer::load_model(const std::string& filepath) {
    if (!std::ifstream(filepath).good()) {
        throw std::runtime_error("Model file not found: " + filepath);
    }
    model_ = std::make_unique<SVC>();
    model_->load_from_file(filepath);
}

std::vector<int> ThunderSVMTrainer::predict_batch(
    const std::vector<float>& features,
    int num_samples,
    int feature_dim) const {
    
    if (!model_) {
        throw std::runtime_error("Cannot predict: no model loaded");
    }
    
    // Convert dense features to sparse node2d format
    DataSet::node2d instances(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        instances[i].reserve(feature_dim);
        const float* sample = features.data() + i * feature_dim;
        
        for (int j = 0; j < feature_dim; ++j) {
            if (sample[j] != 0.0f) {  // Only non-zero values
                instances[i].emplace_back(j + 1, sample[j]);  // 1-based index (ThunderSVM convention)
            }
        }
    }
    
    // Predict using ThunderSVM (uses GPU internally)
    std::vector<double> predictions = model_->predict(instances, num_samples);
    
    // Convert float predictions to int labels
    std::vector<int> labels(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        labels[i] = static_cast<int>(predictions[i]);
    }
    
    return labels;
}

} // namespace gpu_svm
