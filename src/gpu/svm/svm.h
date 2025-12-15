#ifndef GPU_SVM_H
#define GPU_SVM_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

// Forward declarations to avoid exposing ThunderSVM headers
class SVC;
class DataSet;
struct SvmParam;

namespace gpu_svm {

class ThunderSVMTrainer {
public:
    explicit ThunderSVMTrainer(double C = 10.0, double gamma = -1.0);
    ~ThunderSVMTrainer();


    void train(
        const std::vector<float>& train_features,
        const std::vector<uint8_t>& train_labels,
        int num_samples,
        int feature_dim);

    void save_model(const std::string& filepath) const;
    void load_model(const std::string& filepath);
    
    std::vector<int> predict_batch(
        const std::vector<float>& features,
        int num_samples,
        int feature_dim) const;

private:
    std::unique_ptr<SVC> model_;
    double C_;
    double gamma_;

    ThunderSVMTrainer(const ThunderSVMTrainer&) = delete;
    ThunderSVMTrainer& operator=(const ThunderSVMTrainer&) = delete;
};

} // namespace gpu_svm

#endif // GPU_SVM_H
