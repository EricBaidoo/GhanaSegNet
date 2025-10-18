# GhanaSegNet Training Notebook - Documentation

## Overview

This notebook implements architecture-specific hyperparameter optimization (Approach 2) for training GhanaSegNet and baseline segmentation models on the Ghanaian food dataset.

## Notebook Structure

### Section 1: Environment Setup and Verification
- Verifies Python and PyTorch installation
- Checks CUDA availability and GPU specifications
- Confirms project directory structure

### Section 2: Dataset Verification and Configuration
- Locates dataset in standard paths
- Verifies train/validation split structure
- Reports dataset statistics

### Section 3: Model Architecture Verification
- Loads GhanaSegNet architecture
- Computes parameter counts
- Compares model sizes with baselines

### Section 4: Training Configuration
**Approach 2: Architecture-Specific Hyperparameters**

| Parameter | GhanaSegNet/SegFormer | UNet/DeepLabV3+ |
|-----------|----------------------|-----------------|
| Learning Rate | 5×10⁻⁵ | 1×10⁻⁴ |
| Warmup Epochs | 5 | 0 |
| Gradient Clipping | max_norm=1.0 | max_norm=5.0 |
| Training Duration | 60 epochs | 60 epochs |
| Batch Size | 8 | 8 |

**Justification:** Transformer-based architectures require lower learning rates and gradual warmup for stable convergence of self-attention mechanisms.

### Section 5A: Execute GhanaSegNet Training
- Trains GhanaSegNet with optimized hyperparameters
- Real-time progress monitoring
- Automatic checkpoint saving

### Section 5B: Comprehensive Benchmarking (Optional)
- Sequential training of all four models
- Architecture-specific optimization for each model
- Fair comparison framework

### Section 6: Results Analysis
- Loads training metrics from checkpoints
- Visualizes training curves (loss, mIoU, accuracy, learning rate)
- Statistical performance summary

### Section 7: Model Comparison
- Compares performance across all trained models
- Parameter efficiency analysis
- Visualization of performance vs. model size

### Section 8: Export Results
- Generates JSON summary for programmatic access
- Creates LaTeX table for thesis inclusion
- Produces methodology text for thesis

## Usage Instructions

### Quick Start (GhanaSegNet Only)
1. Execute cells 1-4 (Setup and Configuration)
2. Skip Section 5B
3. Execute Section 5A (GhanaSegNet training)
4. Execute Sections 6-8 (Analysis and export)

### Full Benchmarking
1. Execute cells 1-4
2. In Section 5B, set `RUN_ALL_MODELS = True`
3. Execute Section 5B (trains all models)
4. Execute Sections 6-8

## Expected Outputs

### Checkpoint Files
- `checkpoints/ghanasegnet/best_ghanasegnet.pth` - Best model weights
- `checkpoints/ghanasegnet/ghanasegnet_results.json` - Performance metrics
- `checkpoints/ghanasegnet/training_history.json` - Epoch-by-epoch metrics

### Visualizations
- `checkpoints/ghanasegnet/training_curves.png` - Training progress plots
- `checkpoints/model_comparison.png` - Multi-model comparison

### Thesis Documentation
- `checkpoints/thesis_results_summary.json` - Comprehensive results
- `checkpoints/results_table.tex` - LaTeX table
- `checkpoints/methodology_text.txt` - Methodology description

## Performance Targets

**Baseline:** DeepLabV3+ at 0.2544 mIoU (15 epochs, undertrained)

**Target:** GhanaSegNet at 0.30-0.32 mIoU (60 epochs, optimized)

**Expected Improvement:** +5-8% mIoU through:
- Architecture-specific learning rate (5×10⁻⁵)
- Warmup schedule (5 epochs)
- Extended training duration (60 vs. 15 epochs)
- Optimized gradient clipping

## Scientific Justification

This approach follows research best practices established in:
- **Xie et al. (2021)** - SegFormer: Used different learning rates for transformer vs. CNN baselines
- **Liu et al. (2021)** - Swin Transformer: Architecture-specific optimization
- **Chen et al. (2018)** - DeepLabV3+: Model-specific hyperparameter tuning

The use of architecture-specific hyperparameters is standard practice in top-tier computer vision venues (CVPR, ICCV, NeurIPS) and ensures each model achieves its maximum potential performance.

## References

1. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *NeurIPS*.

2. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. *ICCV*.

3. Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV*.

---

**Last Updated:** October 17, 2025
