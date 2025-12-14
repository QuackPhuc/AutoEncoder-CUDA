#!/bin/bash
# Download and extract CIFAR-10 dataset

set -e

DATA_DIR="./data"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE="$DATA_DIR/cifar-10-binary.tar.gz"

REQUIRED_FILES=(data_batch_{1..5}.bin test_batch.bin)

# Check if already exists
check_dataset() {
    for f in "${REQUIRED_FILES[@]}"; do
        [[ ! -f "$DATA_DIR/$f" ]] && return 1
    done
    return 0
}

echo "[download] CIFAR-10 Dataset"

if check_dataset; then
    echo "[OK] Dataset already exists"
    exit 0
fi

mkdir -p "$DATA_DIR"

# Download
echo "  Downloading (162 MB)..."
if command -v curl &>/dev/null; then
    curl -sL "$URL" -o "$ARCHIVE"
elif command -v wget &>/dev/null; then
    wget -q "$URL" -O "$ARCHIVE"
else
    echo "[ERROR] curl or wget required"
    exit 1
fi

# Extract
echo "  Extracting..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
find "$DATA_DIR" -name "*.bin" -exec mv {} "$DATA_DIR/" \; 2>/dev/null || true

# Cleanup
rm -rf "$ARCHIVE" "$DATA_DIR/cifar-10-batches-bin" 2>/dev/null || true

# Verify
if check_dataset; then
    echo "[OK] Dataset ready"
else
    echo "[ERROR] Extraction failed"
    exit 1
fi
