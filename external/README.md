# External Dependencies

## ThunderSVM (GPU-accelerated SVM)

This project uses **ThunderSVM** for GPU-accelerated SVM training and prediction.

### Setup

```bash
# Clone with submodules (recommended)
git clone --recursive https://github.com/QuackPhuc/AutoEncoder-CUDA.git

# Or update existing clone
git submodule update --init --recursive
```

### Directory Structure

```
external/
└── thundersvm/          # ThunderSVM library (git submodule)
    ├── include/         # Header files
    ├── src/             # Source files
    └── CMakeLists.txt
```

### Build Integration

ThunderSVM is automatically built and linked via CMake. No manual setup required.

### Usage

The SVM wrapper is in `src/gpu/svm/`:

```cpp
#include "gpu/svm/svm.h"

gpu_svm::ThunderSVMTrainer trainer(C, gamma);
trainer.train(features, labels, num_samples, feature_dim);
trainer.save_model("./checkpoints/svm.bin");
trainer.load_model("./checkpoints/svm.bin");
auto predictions = trainer.predict_batch(features, num_samples, feature_dim);
```

### Troubleshooting

**Submodule not found:**

```bash
git submodule update --init --recursive
./build.sh --clean
```

**Option 2: Add Submodule Manually (if submodule fails)**

```bash
cd external
git submodule add https://github.com/Xtra-Computing/thundersvm.git
```

### Links

- ThunderSVM: <https://github.com/Xtra-Computing/thundersvm>
- Documentation: <https://thundersvm.readthedocs.io/>
