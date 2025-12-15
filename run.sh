#!/bin/bash
# AutoEncoder CUDA - Run Script

# set -e

BUILD_DIR="build"
MODE="pipeline"
DEVICE="gpu"
VERSION="v2"
EPOCHS=20
SAMPLES=0
BATCH_SIZE=0
ENCODER_WEIGHTS=""
SVM_MODEL=""

show_help() {
    cat << 'EOF'
Usage: ./run.sh [COMMAND] [OPTIONS]

COMMANDS:
    train-autoencoder   Train autoencoder only, save encoder weights
    train-svm           Train SVM using pre-trained encoder weights
    evaluate            Evaluate using pre-trained encoder + SVM weights
    pipeline            Full: train-autoencoder -> train-svm (default)

OPTIONS:
    --device cpu|gpu        Device to use (default: gpu)
    --version v             GPU version: naive | v1 | v2 | v3 (default: v2)
    --epochs N              Training epochs (default: 20)
    --samples N             Limit samples, 0=all (default: 0)
    --batch-size N          Batch size, 0=auto (default: 0)
    --encoder-weights PATH  Input encoder weights (default: ./checkpoints/encoder.weights)
                            Used by: train-svm, evaluate
                            Ignored by: pipeline (uses its own trained weights)
    --svm-model PATH        Input SVM model (default: ./checkpoints/svm.bin)
                            Used by: evaluate
                            Ignored by: pipeline (uses its own trained weights)

EXAMPLES:
    ./run.sh                                    # Full pipeline (uses own timestamped weights)
    ./run.sh train-autoencoder --epochs 20      # Train autoencoder -> timestamped output
    ./run.sh train-svm                          # Train SVM using default encoder.weights
    ./run.sh train-svm --encoder-weights ./checkpoints/encoder_custom.weights
    ./run.sh evaluate                           # Evaluate with default weights
    ./run.sh evaluate --encoder-weights ./checkpoints/encoder_custom.weights
EOF
}

# Parse command (first argument)
case ${1:-} in
    train-autoencoder) MODE="train-autoencoder"; shift ;;
    train-svm) MODE="train-svm"; shift ;;
    evaluate) MODE="evaluate"; shift ;;
    pipeline) MODE="pipeline"; shift ;;
    --*) ;;  # No command, keep default (pipeline), process as option
    -h|--help) show_help; exit 0 ;;
    "") ;;   # No arguments, use defaults
    *) echo "Unknown command: $1"; show_help; exit 1 ;;
esac

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --encoder-weights) ENCODER_WEIGHTS="$2"; shift 2 ;;
        --svm-model) SVM_MODEL="$2"; shift 2 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate device
[[ "$DEVICE" == "cpu" || "$DEVICE" == "gpu" ]] || {
    echo "Invalid device: $DEVICE (use: cpu | gpu)"
    exit 1
}

# Generate timestamp for training output
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default paths logic per mode:
# - train-autoencoder: output encoder with timestamp
# - train-svm: input encoder from default/config, output SVM with timestamp  
# - evaluate: input both from default/config
# - pipeline: ALWAYS use own timestamped weights (ignore user config)

DEFAULT_ENCODER="./checkpoints/encoder.weights"
DEFAULT_SVM="./checkpoints/svm.bin"

case "$MODE" in
    train-autoencoder)
        # Output: timestamped encoder weights
        ENCODER_OUTPUT="./checkpoints/encoder_${TIMESTAMP}.weights"
        ;;
    train-svm)
        # Input: encoder weights (default or config)
        # Output: timestamped SVM model
        [[ -z "$ENCODER_WEIGHTS" ]] && ENCODER_WEIGHTS="$DEFAULT_ENCODER"
        SVM_OUTPUT="./checkpoints/svm_${TIMESTAMP}.bin"
        ;;
    evaluate)
        # Input: both from default or config
        [[ -z "$ENCODER_WEIGHTS" ]] && ENCODER_WEIGHTS="$DEFAULT_ENCODER"
        [[ -z "$SVM_MODEL" ]] && SVM_MODEL="$DEFAULT_SVM"
        ;;
    pipeline)
        # Force own timestamped weights (ignore any user-provided paths)
        if [[ -n "$ENCODER_WEIGHTS" || -n "$SVM_MODEL" ]]; then
            echo "Warning: --encoder-weights and --svm-model are ignored in pipeline mode."
            echo "         Pipeline always uses its own trained weights."
        fi
        ENCODER_WEIGHTS="./checkpoints/encoder_${TIMESTAMP}.weights"
        SVM_MODEL="./checkpoints/svm_${TIMESTAMP}.bin"
        ;;
esac

# Select executables
TRAIN_EXE=""
INFER_EXE="$BUILD_DIR/bin/autoencoder_inference"

case "$DEVICE" in
    cpu) TRAIN_EXE="$BUILD_DIR/bin/autoencoder_cpu" ;;
    gpu) TRAIN_EXE="$BUILD_DIR/bin/autoencoder_gpu" ;;
esac

# Build training arguments
TRAIN_ARGS="--epochs $EPOCHS"
[[ $SAMPLES -gt 0 ]] && TRAIN_ARGS="$TRAIN_ARGS --samples $SAMPLES"
[[ $BATCH_SIZE -gt 0 ]] && TRAIN_ARGS="$TRAIN_ARGS --batch-size $BATCH_SIZE"

# GPU version: naive=1, v1=2, v2=3, v3=4
if [[ "$DEVICE" == "gpu" ]]; then
    case $VERSION in
        naive) TRAIN_ARGS="$TRAIN_ARGS --gpu-version 1" ;;
        v1)    TRAIN_ARGS="$TRAIN_ARGS --gpu-version 2" ;;
        v2)    TRAIN_ARGS="$TRAIN_ARGS --gpu-version 3" ;;
        v3)    TRAIN_ARGS="$TRAIN_ARGS --gpu-version 4" ;;
        *) echo "Invalid version: $VERSION"; exit 1 ;;
    esac
fi

# Build inference arguments with explicit encoder and svm paths
build_infer_args() {
    local encoder_path="$1"
    local svm_path="$2"
    INFER_ARGS="--encoder-weights $encoder_path --svm-model $svm_path"
    [[ $BATCH_SIZE -gt 0 ]] && INFER_ARGS="$INFER_ARGS --batch-size $BATCH_SIZE"
    
    # Pass GPU version to inference for fast feature extraction
    case $VERSION in
        naive) INFER_ARGS="$INFER_ARGS --gpu-version 1" ;;
        v1)    INFER_ARGS="$INFER_ARGS --gpu-version 2" ;;
        v2)    INFER_ARGS="$INFER_ARGS --gpu-version 3" ;;
        v3)    INFER_ARGS="$INFER_ARGS --gpu-version 4" ;;
    esac
}

# Display mode info
echo "[$MODE] device=$DEVICE | epochs=$EPOCHS | version=$VERSION"

check_train_exe() {
    if [[ -f "${TRAIN_EXE}.exe" ]]; then
        TRAIN_EXE="${TRAIN_EXE}.exe"
    elif [[ ! -f "$TRAIN_EXE" ]]; then
        echo "Error: $TRAIN_EXE not found. Run ./build.sh first."
        exit 1
    fi
}

check_infer_exe() {
    if [[ -f "${INFER_EXE}.exe" ]]; then
        INFER_EXE="${INFER_EXE}.exe"
    elif [[ ! -f "$INFER_EXE" ]]; then
        echo "Error: $INFER_EXE not found. Run ./build.sh first."
        exit 1
    fi
}

case "$MODE" in
    train-autoencoder)
        check_train_exe
        echo "  output: $ENCODER_OUTPUT"
        echo ""
        echo "=== Training Autoencoder ==="
        $TRAIN_EXE $TRAIN_ARGS --save-weights "$ENCODER_OUTPUT"
        ;;
        
    train-svm)
        check_infer_exe
        [[ -f "$ENCODER_WEIGHTS" ]] || {
            echo "Error: Encoder weights not found: $ENCODER_WEIGHTS"
            echo "Run './run.sh train-autoencoder' first, or specify --encoder-weights."
            exit 1
        }
        echo "  input encoder: $ENCODER_WEIGHTS"
        echo "  output svm:    $SVM_OUTPUT"
        build_infer_args "$ENCODER_WEIGHTS" "$SVM_OUTPUT"
        echo ""
        echo "=== Training SVM ==="
        $INFER_EXE $INFER_ARGS --train-svm
        ;;
        
    evaluate)
        check_infer_exe
        [[ -f "$ENCODER_WEIGHTS" ]] || {
            echo "Error: Encoder weights not found: $ENCODER_WEIGHTS"
            exit 1
        }
        [[ -f "$SVM_MODEL" ]] || {
            echo "Error: SVM model not found: $SVM_MODEL"
            exit 1
        }
        echo "  encoder: $ENCODER_WEIGHTS"
        echo "  svm:     $SVM_MODEL"
        build_infer_args "$ENCODER_WEIGHTS" "$SVM_MODEL"
        echo ""
        echo "=== Evaluating ==="
        $INFER_EXE $INFER_ARGS --evaluate-only
        ;;
        
    pipeline)
        check_train_exe
        check_infer_exe
        echo "  output encoder: $ENCODER_WEIGHTS"
        echo "  output svm:     $SVM_MODEL"
        
        echo ""
        echo "=== Step 1: Training Autoencoder ==="
        $TRAIN_EXE $TRAIN_ARGS --save-weights "$ENCODER_WEIGHTS" || {
            echo "Error: Autoencoder training failed with exit code $?"
            exit 1
        }
        echo "[Step 1 completed successfully]"
        
        build_infer_args "$ENCODER_WEIGHTS" "$SVM_MODEL"
        echo ""
        echo "=== Step 2: Training SVM ==="
        $INFER_EXE $INFER_ARGS --train-svm || {
            echo "Error: SVM training failed with exit code $?"
            exit 1
        }
        echo "[Step 2 completed successfully]"
        ;;
esac
