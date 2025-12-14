#!/bin/bash
# Download pretrained weights from Google Drive
# Usage:
#   ./scripts/download_weights.sh          # Download all weights
#   ./scripts/download_weights.sh encoder  # Download encoder only
#   ./scripts/download_weights.sh svm      # Download SVM only

set -e

ENCODER_FILE_ID="1nfCUnKa6TmBBbhAiud0oItnoWFgiZt3v"
SVM_FILE_ID="1ZyAv2R0fYPRKBuQuxsQY0WSjYfhTDSEb"

CHECKPOINT_DIR="./checkpoints"
ENCODER_FILE="$CHECKPOINT_DIR/encoder.weights"
SVM_FILE="$CHECKPOINT_DIR/svm.bin"

ENCODER_EXPECTED_SIZE=0
SVM_EXPECTED_SIZE=0
print_header() {
    echo ""
    echo "========================================"
    echo " AutoEncoder CUDA - Download Weights"
    echo "========================================"
}

download_gdrive() {
    local file_id="$1"
    local output_file="$2"
    local file_name=$(basename "$output_file")
    
    echo "  Downloading $file_name..."
    
    # Google Drive direct download URL
    local url="https://drive.google.com/uc?export=download&id=$file_id"
    
    if command -v curl &>/dev/null; then
        # First request to get confirmation token (for large files)
        local confirm=$(curl -sc /tmp/gdrive_cookie "$url" | \
            grep -o 'confirm=[^&]*' | cut -d= -f2)
        
        if [[ -n "$confirm" ]]; then
            # Large file: use confirmation token
            curl -Lb /tmp/gdrive_cookie \
                "https://drive.google.com/uc?export=download&confirm=$confirm&id=$file_id" \
                -o "$output_file" --progress-bar
        else
            # Small file: direct download
            curl -L "$url" -o "$output_file" --progress-bar
        fi
        rm -f /tmp/gdrive_cookie
        
    elif command -v wget &>/dev/null; then
        # wget method for large files
        wget --load-cookies /tmp/gdrive_cookie \
            "https://drive.google.com/uc?export=download&id=$file_id" \
            -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p' > /tmp/gdrive_confirm
        
        local confirm=$(cat /tmp/gdrive_confirm)
        if [[ -n "$confirm" ]]; then
            wget --load-cookies /tmp/gdrive_cookie -q --show-progress \
                "https://drive.google.com/uc?export=download&confirm=$confirm&id=$file_id" \
                -O "$output_file"
        else
            wget -q --show-progress "$url" -O "$output_file"
        fi
        rm -f /tmp/gdrive_cookie /tmp/gdrive_confirm
    else
        echo "[ERROR] curl or wget required"
        exit 1
    fi
}

# Verify file size (optional)
verify_file() {
    local file="$1"
    local expected_size="$2"
    local file_name=$(basename "$file")
    
    if [[ ! -f "$file" ]]; then
        echo "  [ERROR] $file_name not found after download"
        return 1
    fi
    
    local actual_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
    
    if [[ "$expected_size" -gt 0 && "$actual_size" -ne "$expected_size" ]]; then
        echo "  [WARNING] $file_name size mismatch: expected $expected_size, got $actual_size"
        return 1
    fi
    
    echo "  [OK] $file_name downloaded ($(numfmt --to=iec-i --suffix=B "$actual_size" 2>/dev/null || echo "${actual_size} bytes"))"
    return 0
}

# Download encoder weights
download_encoder() {
    if [[ "$ENCODER_FILE_ID" == "YOUR_ENCODER_FILE_ID_HERE" ]]; then
        echo "[ERROR] Encoder file ID not configured!"
        echo "        Edit scripts/download_weights.sh and set ENCODER_FILE_ID"
        return 1
    fi
    
    if [[ -f "$ENCODER_FILE" ]]; then
        echo "  [SKIP] $ENCODER_FILE already exists"
        echo "         Delete it first if you want to re-download"
        return 0
    fi
    
    download_gdrive "$ENCODER_FILE_ID" "$ENCODER_FILE"
    verify_file "$ENCODER_FILE" "$ENCODER_EXPECTED_SIZE"
}

# Download SVM model
download_svm() {
    if [[ "$SVM_FILE_ID" == "YOUR_SVM_FILE_ID_HERE" ]]; then
        echo "[ERROR] SVM file ID not configured!"
        echo "        Edit scripts/download_weights.sh and set SVM_FILE_ID"
        return 1
    fi
    
    if [[ -f "$SVM_FILE" ]]; then
        echo "  [SKIP] $SVM_FILE already exists"
        echo "         Delete it first if you want to re-download"
        return 0
    fi
    
    download_gdrive "$SVM_FILE_ID" "$SVM_FILE"
    verify_file "$SVM_FILE" "$SVM_EXPECTED_SIZE"
}

# =====================================================
# Main
# =====================================================
print_header

# Create checkpoints directory if not exists
mkdir -p "$CHECKPOINT_DIR"

# Parse argument
TARGET="${1:-all}"

case "$TARGET" in
    encoder)
        echo "[download] Encoder weights only"
        download_encoder
        ;;
    svm)
        echo "[download] SVM model only"
        download_svm
        ;;
    all|"")
        echo "[download] All pretrained weights"
        echo ""
        echo "--- Encoder Weights ---"
        download_encoder
        echo ""
        echo "--- SVM Model ---"
        download_svm
        ;;
    --help|-h)
        echo "Usage: $0 [encoder|svm|all]"
        echo ""
        echo "  encoder  Download encoder.weights only"
        echo "  svm      Download svm.bin only"
        echo "  all      Download all weights (default)"
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown target: $TARGET"
        echo "Usage: $0 [encoder|svm|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo " Download complete!"
echo "========================================"
