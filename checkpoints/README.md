# Checkpoints Directory

This directory stores trained model weights.

## Files

| File | Description |
|------|-------------|
| `encoder.weights` | Trained autoencoder encoder weights |
| `svm.bin` | Trained SVM classifier model |
| `encoder_*.weights` | Timestamped encoder weights from training runs |
| `svm_*.bin` | Timestamped SVM models from training runs |

## Download Pretrained Weights

To skip training and use pretrained weights:

```bash
./scripts/download_weights.sh
```

This downloads:

- `encoder.weights` - Autoencoder trained on CIFAR-10 (20 epochs)
- `svm.bin` - SVM classifier trained on extracted features

## Google Drive Links

If automatic download fails, manually download from:

- **Encoder weights**: [Google Drive](https://drive.google.com/file/d/1nfCUnKa6TmBBbhAiud0oItnoWFgiZt3v/view?usp=drive_link)
- **SVM model**: [Google Drive](https://drive.google.com/file/d/1ZyAv2R0fYPRKBuQuxsQY0WSjYfhTDSEb/view?usp=drive_link)

## Training Your Own

```bash
# Train autoencoder (saves timestamped weights)
./run.sh train-autoencoder --epochs 20

# Train SVM using saved encoder
./run.sh train-svm --encoder-weights ./checkpoints/encoder_TIMESTAMP.weights
```
