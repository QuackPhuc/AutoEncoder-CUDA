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

# Minimum file sizes for validation
ENCODER_MIN_SIZE=2000000       # ~2MB (encoder weights)
SVM_MIN_SIZE=1000000000        # ~1GB minimum (actual is ~3.2GB)

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
    
    # Method 1: Use gdown if available (best for large files)
    if command -v gdown &>/dev/null; then
        gdown --id "$file_id" -O "$output_file" --quiet
        return $?
    fi
    
    # Method 2: curl with proper cookie handling for large files
    if command -v curl &>/dev/null; then
        local url="https://drive.google.com/uc?export=download&id=$file_id"
        local cookie_file="/tmp/gdrive_cookie_$$"
        
        # First request - get cookies and check for confirmation
        curl -sc "$cookie_file" -L "$url" -o /tmp/gdrive_response_$$ 2>/dev/null
        
        # Check if we got HTML (confirmation page) or the actual file
        if grep -q "confirm=" /tmp/gdrive_response_$$  2>/dev/null; then
            # Extract confirmation token
            local confirm=$(grep -o 'confirm=[^&"]*' /tmp/gdrive_response_$$ | head -1 | cut -d= -f2)
            if [[ -n "$confirm" ]]; then
                echo "  Large file detected, using confirmation token..."
                curl -Lb "$cookie_file" \
                    "https://drive.google.com/uc?export=download&confirm=$confirm&id=$file_id" \
                    -o "$output_file" --progress-bar
            else
                # Try with uuid token extraction
                local uuid=$(grep -o 'uuid=[^&"]*' /tmp/gdrive_response_$$ | head -1 | cut -d= -f2)
                if [[ -n "$uuid" ]]; then
                    curl -Lb "$cookie_file" \
                        "https://drive.google.com/uc?export=download&uuid=$uuid&id=$file_id" \
                        -o "$output_file" --progress-bar
                else
                    echo "  [WARNING] Could not extract confirmation token"
                    mv /tmp/gdrive_response_$$ "$output_file"
                fi
            fi
        else
            # Small file - direct download succeeded
            mv /tmp/gdrive_response_$$ "$output_file"
        fi
        rm -f "$cookie_file" /tmp/gdrive_response_$$
        
    elif command -v wget &>/dev/null; then
        # wget method
        wget --quiet --save-cookies /tmp/gdrive_cookie_$$ --keep-session-cookies \
            "https://drive.google.com/uc?export=download&id=$file_id" -O /tmp/gdrive_response_$$
        
        if grep -q "confirm=" /tmp/gdrive_response_$$ 2>/dev/null; then
            local confirm=$(grep -o 'confirm=[^&"]*' /tmp/gdrive_response_$$ | head -1 | cut -d= -f2)
            wget --load-cookies /tmp/gdrive_cookie_$$ -q --show-progress \
                "https://drive.google.com/uc?export=download&confirm=$confirm&id=$file_id" \
                -O "$output_file"
        else
            mv /tmp/gdrive_response_$$ "$output_file"
        fi
        rm -f /tmp/gdrive_cookie_$$ /tmp/gdrive_response_$$
    else
        echo "[ERROR] curl or wget required"
        exit 1
    fi
}

# Verify file size and content
verify_file() {
    local file="$1"
    local min_size="$2"  # Minimum expected size in bytes
    local file_name=$(basename "$file")
    
    if [[ ! -f "$file" ]]; then
        echo "  [ERROR] $file_name not found after download"
        return 1
    fi
    
    local actual_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")
    
    # Check if file is HTML (error page from Google Drive)
    if head -c 100 "$file" 2>/dev/null | grep -qi "<!DOCTYPE\|<html"; then
        echo "  [ERROR] $file_name is an HTML page, not the actual file!"
        echo "          Google Drive may require manual download for large files."
        echo "          Direct link: https://drive.google.com/uc?id=$file_id"
        rm -f "$file"
        return 1
    fi
    
    # Check minimum size
    if [[ "$min_size" -gt 0 && "$actual_size" -lt "$min_size" ]]; then
        local min_human=$(numfmt --to=iec-i --suffix=B "$min_size" 2>/dev/null || echo "$min_size bytes")
        local actual_human=$(numfmt --to=iec-i --suffix=B "$actual_size" 2>/dev/null || echo "$actual_size bytes")
        echo "  [ERROR] $file_name too small: got $actual_human, expected at least $min_human"
        echo "          Download may have failed. Delete the file and try again."
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
    verify_file "$ENCODER_FILE" "$ENCODER_MIN_SIZE"
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
    verify_file "$SVM_FILE" "$SVM_MIN_SIZE"
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
