# GhanaSegNet

This repository implements GhanaSegNet, a hybrid deep learning model combining CNN and Transformer layers to segment Ghanaian food images using the FRANI dataset.

It includes baseline comparisons with UNet, DeepLabV3+, and SegFormer-B0, and uses boundary-aware loss and dice loss for improved segmentation.

## Setup
GitHub Codespaces enabled via `.devcontainer/` for instant cloud development.

## Folders
- `models/`: All 4 architectures
- `scripts/`: Training, evaluation, visualization
- `utils/`: Losses, metrics
- `notebooks/`: Experimentation
