# Chapter 4: Implementation and System Architecture

## 4.1 Introduction

This chapter presents the comprehensive implementation of GhanaSegNet, a novel hybrid CNN-Transformer architecture specifically engineered for semantic segmentation of traditional Ghanaian foods. The implementation represents a convergence of cutting-edge deep learning methodologies, culturally-aware architectural design, and practical deployment considerations, establishing a new paradigm for domain-specific computer vision applications.

The architectural implementation is grounded in theoretical foundations from both convolutional neural networks (Krizhevsky et al., 2012; He et al., 2016) and transformer attention mechanisms (Vaswani et al., 2017; Dosovitskiy et al., 2020), while introducing novel fusion strategies optimized for the unique challenges of food image segmentation. The system architecture emphasizes modularity, extensibility, and computational efficiency, ensuring both research reproducibility and practical deployment viability.

This chapter elucidates the technical decisions, algorithmic innovations, and engineering solutions that enable GhanaSegNet to achieve state-of-the-art performance while maintaining the computational efficiency required for mobile health applications in resource-constrained environments. The implementation framework serves as both a research contribution and a practical foundation for automated nutritional assessment systems in developing nations.

## 4.1 Initial Model Evaluation

Purpose: Report the first evaluation of the implemented GhanaSegNet model (baseline run), provide reproducible commands to reproduce the short validation run, and summarize the initial metrics used to guide subsequent hyperparameter decisions.

Setup and assumptions:
- Dataset: FRANI (939 train / 202 val) prepared as described in Chapter 3.
- Model: `models/ghanasegnet.py` configured for C=6 classes.
- Environment: CUDA-enabled GPU (recommended) or Colab T4 for rapid iteration.

Initial (baseline) run summary (epoch 1 reported values from run JSON):

| Metric | Value |
|--------|-------|
| Validation mIoU | 0.2437 (24.37%) |
| Validation pixel accuracy | 0.7832 (78.32%) |
| Training loss (epoch 1 end) | 2.3107 |
| Validation loss (epoch 1) | 2.4208 |

Reproducible quick-run (PowerShell):

```powershell
# run a short 5-10 epoch baseline to reproduce reported metrics (writes artifacts to checkpoints/ghanasegnet/)
python scripts\train_baselines.py --model ghanasegnet --epochs 5 --batch-size 8 --checkpoint-dir checkpoints/ghanasegnet --seed 789

# evaluate a saved checkpoint on validation split
python scripts\evaluate.py --model ghanasegnet --checkpoint checkpoints/ghanasegnet/best_model.pth --split val
```

Figure/Table placeholders:
- Table 4.1: Baseline run hyperparameters and checkpoint metadata (link to `checkpoints/ghanasegnet/run_*.json`).
- Figure 4.1: Training and validation curves (loss, mIoU) for the baseline run.

Notes:
- These initial metrics were used to tune loss weighting and schedule further ablation studies described in Chapter 5.
- The run JSON and per-epoch metrics are available in `checkpoints/ghanasegnet/` and should be attached to any future reproduction attempts.

## 4.2 Theoretical Foundations and Design Principles

### 4.2.1 Hybrid Architecture Theoretical Framework

The GhanaSegNet architecture is founded on the theoretical principle that food segmentation requires both local texture understanding (optimally handled by CNNs) and global contextual reasoning (effectively addressed by transformers). This dual requirement motivates a hybrid approach that leverages the complementary strengths of both architectural paradigms.

**Mathematical Formulation:**

Let $\mathcal{F}_{CNN}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{H' \times W' \times D}$ represent the CNN feature extraction function, and $\mathcal{F}_{Transformer}: \mathbb{R}^{H' \times W' \times D} \rightarrow \mathbb{R}^{H' \times W' \times D}$ represent the transformer attention mechanism. The hybrid architecture computes:

$$\mathcal{F}_{Hybrid}(X) = \mathcal{F}_{Decoder}(\mathcal{F}_{Transformer}(\mathcal{F}_{CNN}(X)))$$

where $X \in \mathbb{R}^{H \times W \times 3}$ is the input image, and $\mathcal{F}_{Decoder}$ represents the enhanced U-Net decoder with skip connections.

**Design Principles:**

1. **Efficiency-First Design:** Optimize for mobile deployment constraints while maintaining accuracy
2. **Cultural Sensitivity:** Incorporate domain-specific inductive biases for traditional food presentations
3. **Modular Composition:** Enable component-wise analysis and future extensibility
4. **Scalable Architecture:** Support various input resolutions and deployment scenarios

### 4.2.2 Loss Function Theoretical Foundation

The food-aware combined loss function addresses three fundamental challenges in food segmentation:

**1. Class Imbalance:** Addressed through Dice loss component
$$\mathcal{L}_{Dice} = 1 - \frac{2|Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}$$

**2. Boundary Precision:** Handled by boundary-aware loss
$$\mathcal{L}_{Boundary} = -\sum_{i \in \partial \Omega} y_i \log(\hat{y}_i)$$

**3. Hard Example Focus:** Managed through focal loss component
$$\mathcal{L}_{Focal} = -\alpha(1-\hat{y})^\gamma \log(\hat{y})$$

The combined formulation:
$$\mathcal{L}_{Combined} = \alpha \mathcal{L}_{Dice} + \beta \mathcal{L}_{Boundary} + \gamma \mathcal{L}_{Focal}$$

where $\alpha + \beta + \gamma = 1$ and weights are optimized for food segmentation characteristics.

## 4.3 Software Architecture and Development Environment

### 4.3.1 Enterprise-Grade Technology Stack

The GhanaSegNet implementation leverages a carefully curated technology ecosystem that balances cutting-edge capabilities with production stability. The architecture follows microservices principles, enabling independent scaling and maintenance of different system components.

**Core Deep Learning Infrastructure:**
- **PyTorch:** Primary framework chosen for dynamic computational graphs and superior debugging capabilities — use a version compatible with your CUDA/runtime; see `requirements.txt` for guidance.
- **TorchVision:** Provides optimized computer vision operations and pre-trained model zoo; choose the matching version for the installed PyTorch.
- **EfficientNet / timm:** Use `timm` or `efficientnet-pytorch` as listed in `requirements.txt` for backbone implementations.
- **Transformers (optional):** Hugging Face utilities for transformer components; only required if you use HF transformer modules.
- **CUDA / cuDNN:** GPU acceleration with automatic mixed precision training; ensure CUDA toolkit and drivers align with the installed PyTorch wheel.

**Advanced Data Processing Pipeline:**
- **Albumentations 1.3.0:** Industry-standard augmentation library with food-specific transformations
- **OpenCV 4.12.0:** High-performance computer vision operations with hardware acceleration
- **Pillow 11.3.0:** Cross-platform image processing with extensive format support
- **NumPy 2.2.6:** Vectorized operations with Intel MKL optimization
- **Pandas 2.0+:** Structured data manipulation for metadata and results analysis

**Scientific Computing and Analysis:**
- **SciPy 1.16.1:** Statistical analysis and signal processing algorithms
- **Scikit-learn 1.7.1:** Machine learning utilities and advanced metrics computation
- **Matplotlib 3.9.0:** Publication-quality visualization with LaTeX integration
- **Seaborn 0.13.0:** Statistical visualization with aesthetic defaults
- **Weights & Biases (WandB):** Experiment tracking and hyperparameter optimization

### 4.3.2 Advanced Project Architecture and Design Patterns

The implementation follows Domain-Driven Design (DDD) principles with clear separation of concerns and dependency injection patterns:
# Chapter 4: Implementation and System Architecture

## 4.1 Introduction

This chapter describes the concrete implementation of GhanaSegNet and the surrounding engineering that makes experiments reproducible, auditable, and deployable. It documents the software layout, model internals, training pipeline, evaluation procedures, and deployment pathways. The goal is to provide enough detail for a reader with access to the FRANI dataset and a CUDA-enabled GPU to reproduce the presented results and extend the work.

## 4.1 Initial Model Evaluation

Purpose: Report the first evaluation of the implemented GhanaSegNet model (baseline run), provide reproducible commands to reproduce the short validation run, and summarize the initial metrics used to guide subsequent hyperparameter decisions.

Setup and assumptions:
- Dataset: FRANI (939 train / 202 val) prepared as described in Chapter 3.
- Model: `models/ghanasegnet.py` configured for C=6 classes.
- Environment: CUDA-enabled GPU (recommended) or Colab T4 for rapid iteration.

Initial (baseline) run summary (epoch 1 reported values from run JSON):

| Metric | Value |
|--------|-------|
| Validation mIoU | 0.2437 (24.37%) |
| Validation pixel accuracy | 0.7832 (78.32%) |
| Training loss (epoch 1 end) | 2.3107 |
| Validation loss (epoch 1) | 2.4208 |

Reproducible quick-run (PowerShell):

```powershell
# run a short 5-10 epoch baseline to reproduce reported metrics (writes artifacts to checkpoints/ghanasegnet/)
python scripts\train_baselines.py --model ghanasegnet --epochs 5 --batch-size 8 --checkpoint-dir checkpoints/ghanasegnet --seed 789

# evaluate a saved checkpoint on validation split
python scripts\evaluate.py --model ghanasegnet --checkpoint checkpoints/ghanasegnet/best_model.pth --split val
```

Figure/Table placeholders:
- Table 4.1: Baseline run hyperparameters and checkpoint metadata (link to `checkpoints/ghanasegnet/run_*.json`).
- Figure 4.1: Training and validation curves (loss, mIoU) for the baseline run.

Notes:
- These initial metrics were used to tune loss weighting and schedule further ablation studies described in Chapter 5.
- The run JSON and per-epoch metrics are available in `checkpoints/ghanasegnet/` and should be attached to any future reproduction attempts.

## 4.2 Theoretical Foundations and Design Principles

The GhanaSegNet design follows several guiding principles: efficiency for mobile/edge deployment, cultural sensitivity to Ghanaian food presentation, and modularity for extensibility.

### 4.2.1 Hybrid architecture formalization

We model the segmentation pipeline as:

$$\mathbf{Y} = \mathcal{D}(\mathcal{T}(\mathcal{E}(\mathbf{X}))))$$

where:
- $\mathbf{X} \in \mathbb{R}^{B\times3\times H \times W}$ is the input batch,
- $\mathcal{E}$ is the CNN encoder (EfficientNet-derived),
- $\mathcal{T}$ is a lightweight transformer bottleneck (or cascade of bottlenecks),
- $\mathcal{D}$ is the decoder producing per-pixel logits $\mathbf{Y} \in \mathbb{R}^{B\times C \times H \times W}$.

The objective is to minimize a combined loss (Chapter 3):

$$\mathcal{L} = \alpha \mathcal{L}_{Dice} + \beta \mathcal{L}_{Boundary} + \gamma \mathcal{L}_{Focal}$$

with weights selected via cross-validation.

### 4.2.2 Design trade-offs

- Transformer capacity vs. latency: we limit transformer token counts via 1×1 conv reductions at the bottleneck to keep inference latency reasonable on GPU and mobile.
- Decoder complexity vs. boundary precision: learnable skip-adapters preserve high-resolution cues while keeping parameter counts moderate.

## 4.3 Software architecture and development environment

### 4.3.1 Repository layout

Key directories and their purpose:

- `models/` — model implementations (`ghanasegnet.py`, `unet.py`, `deeplabv3plus.py`, `segformer_b0.py`).
- `scripts/` — training and evaluation entry points (`train_baselines.py`, `evaluate.py`, `test.py`).
- `infrastructure/` — data loaders, augmentation pipelines, validators.
- `utils/` — `losses.py`, `metrics.py`, and helper utilities.
- `checkpoints/` — run outputs, checkpoints, JSON summaries.
- `notebooks/` — experimental notebooks for visualization and analysis.

### 4.3.2 Dependencies and environment

- Use `requirements.txt` for reproducible Python dependencies. Prefer creating an isolated venv or using the provided Dockerfile for production-like runs.
- Recommended Python runtime: 3.8–3.10. Use a PyTorch wheel that matches your CUDA driver (see README).

### 4.3.3 DevOps / CI suggestions

- Basic GitHub Actions workflow is included to run tests across Python/PyTorch matrix. Keep the test matrix minimal to reduce CI costs; run full experiments outside CI (Colab or dedicated GPU nodes).

## 4.4 Detailed model implementation

This subsection maps the algorithmic design to the concrete implementation in `models/ghanasegnet.py`. The aim is to make the reader understand how tensor shapes flow through the network and where to edit for further experiments.

### 4.4.1 Component overview

- Encoder ($\\mathcal{E}$): EfficientNet-derived backbone producing hierarchical features at multiple scales: $F_1, F_2, F_3, F_4$.
- Bottleneck / Transformers ($\\mathcal{T}$): 1×1 conv reduces channels to a compact token dimension; a small stack of transformer blocks (self-attention + MLP) injects global context.
- Decoder ($\\mathcal{D}$): Food-aware decoder that upsamples the bottleneck while fusing skip features via learned adapters and channel attention.
- Skip adapters: lightweight 1×1 convs that align channel dimensions for skip fusion.

### 4.4.2 Tensor shapes (working example, input 384×384)

| Stage | Tensor shape (B=1 example) | Notes |
|-------|-----------------------------|-------|
| Input | [1, 3, 384, 384] | Raw RGB image |
| Encoder output (richest) | [1, 1280, 12, 12] | after downsampling in EfficientNet |
| Feature processor | [1, 256, 12, 12] | 1×1 conv reductions |
| Transformer tokens | [1, 256, 144] | flattened tokens = 12×12 |
| Decoder output logits | [1, C, 384, 384] | upsampled to input size |

### 4.4.3 Forward pass (concise pseudo-code)

```python
def forward(self, x):
    # x: [B, 3, H, W]
    skips = self.encoder.extract_skips(x)  # list of multi-scale features
    bottleneck = self.feature_processor(skips[-1])  # [B, 256, h, w]
    tokens = bottleneck.flatten(2).permute(0, 2, 1)  # [B, N, D]
    tokens = self.transformer(tokens)
    bottleneck = tokens.permute(0, 2, 1).view_as(bottleneck)
    out = self.decoder(bottleneck, skips[:-1])
    return out  # [B, C, H, W]
```

(Refer to `models/ghanasegnet.py` for the full implementation.)

### 4.4.4 Parameter counts and performance

- Reported parameter count for the GhanaSegNet variant used in experiments: ~6.8M (check `scripts/compute_model_size.py` if present).
- FLOPs estimates are sensitive to input size; use `ptflops` or `fvcore` profiling utilities for precise measurements per variant.

## 4.5 Training pipeline and utilities

This section documents `scripts/train_baselines.py` and related utilities that implement data loading, training loop, logging, and checkpointing.

### 4.5.1 Data loading and augmentation

- Datasets are implemented under `infrastructure/data/datasets/ghana_food_dataset.py` and use Albumentations for online augmentation.
- Progressive resizing and augmentation schedules are implemented in the training script and the augmentation pipeline; verify the schedule via CLI flags.

### 4.5.2 Losses, optimizer and schedulers

- Loss composition is defined in `utils/losses.py` and constructed per-run in the training script.
- Optimizer: AdamW; Scheduler: cosine annealing or CosineAnnealingWarmRestarts for long runs.
- Gradient clipping and mixed-precision (AMP) are supported; enable AMP with the `--use-amp` flag.

### 4.5.3 Checkpointing and logging

- Checkpoints: periodic saving and best-model-by-mIoU saving into `checkpoints/ghanasegnet/`.
- Logging: per-epoch metrics are written to a `run_summary.json` and optionally to Weights & Biases if `--use-wandb` is enabled.

### 4.5.4 Example training command

```powershell
python scripts\\train_baselines.py --model ghanasegnet --epochs 100 --batch-size 8 --lr 1e-4 --checkpoint-dir checkpoints/ghanasegnet --use-amp --seed 789
```

## 4.6 Evaluation, inference and visualization

### 4.6.1 Evaluation protocol

- Convert logits to predicted masks via argmax before passing to IoU/Dice metric functions to avoid shape/type mismatches.
- Compute per-class IoU, mean IoU, Dice, pixel accuracy, and boundary F1.

### 4.6.2 Inference utilities

- `scripts/test.py` performs single-image inference and saves visualizations to `outputs/`.
- Post-processing: optional CRF or morphological smoothing can be applied to refine predictions; these steps are parameterized in `scripts/test.py`.

### 4.6.3 Visualization

- Notebooks under `notebooks/` contain code to plot training curves, per-class performance heatmaps, and qualitative comparisons across models.

## 4.7 Deployment and optimization

### 4.7.1 Model export and mobile inference

- Export formats: TorchScript and ONNX are supported for deployment. Optimize models using PyTorch Mobile tools or ONNX Runtime for mobile/edge.
- Quantization: post-training dynamic quantization or quantization-aware training are available in the `deployment/model_optimization/` utilities.

### 4.7.2 Edge considerations

- For mobile deployment prefer EfficientNet-lite backbones and smaller decoder variants; profile memory and latency on target devices.

## 4.8 Reproducibility, testing, and CI

### 4.8.1 Reproducibility

- Save run metadata (seed, hyperparameters) in `checkpoints/` alongside checkpoints.
- Use deterministic settings where possible and document the environment (Python, PyTorch, CUDA versions).

### 4.8.2 Tests and CI

- The `tests/` directory contains smoke tests: import tests, small training loop tests, and evaluation checks. Run them locally via `pytest tests/`.
- CI should run linting and these smoke tests; avoid running full training in CI.

## 4.9 Implementation limitations and engineering notes

- Import chain issues: large imports like `torchvision` may cause long import times; prefer isolated environments or container images to avoid accidental blocking.
- Determinism: AMP and certain CUDA ops may be nondeterministic; use `--benchmark-mode` and fixed seeds for better reproducibility but expect small numerical differences.
- Memory hotspots: transformer tokenization and high input resolutions require careful batch sizing; prefer progressive resizing when GPU memory is constrained.

## 4.10 Chapter summary

This chapter documented the end-to-end implementation of GhanaSegNet: the theoretical grounding, concrete model design, training pipeline, evaluation protocols, deployment options, and reproducibility practices. The design choices prioritize a balance between performance and deployability for nutrition-assessment applications. Chapter 5 presents experiments, ablation studies, and quantitative results that validate these choices.
            mlp_dim=512, 
            dropout=dropout,
            use_learnable_scaling=True
        )
        
        self.transformer2 = TransformerBlock(
            dim=256, 
            heads=8, 
            mlp_dim=512, 
            dropout=dropout,
            use_learnable_scaling=True
        )
        
        # Enhanced decoder with food-specific optimizations
        self.decoder = FoodAwareDecoder(
            input_channels=256,
            skip_channels=[112, 40, 24, 16],  # EfficientNet-B0 feature dimensions
            output_channels=num_classes,
            use_attention=True
        )
        
        # Skip connection adaptation layers (novel contribution)
        # Enables fusion of features at different semantic levels
        self.skip_adapters = nn.ModuleList([
            SkipConnectionAdapter(in_ch, 256) for in_ch in [112, 40, 24, 16]
        ])
        
        # Learnable fusion weights for multi-scale integration
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)
        
        # Initialize weights using theoretical best practices
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Advanced weight initialization strategy based on theoretical foundations:
        - Kaiming initialization for ReLU activations (He et al., 2015)
        - Xavier initialization for attention mechanisms (Glorot & Bengio, 2010)
        - Conservative initialization for learnable scaling parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, '_is_pretrained') and m._is_pretrained:
                    continue  # Skip pretrained weights
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass implementing the hybrid CNN-Transformer architecture
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        batch_size, _, input_h, input_w = x.shape
        
        # Multi-scale feature extraction with EfficientNet backbone
        skip_features = []
        features = x
        
        # Extract hierarchical features for skip connections
        features = self.encoder._conv_stem(features)
        features = self.encoder._bn0(features)
        features = self.encoder._swish(features)
        
        # Progressive feature extraction with skip collection
        for idx, block in enumerate(self.encoder._blocks):
            features = block(features)
            
            # Strategic skip feature collection based on semantic levels
            if idx in [2, 5, 10, 15]:  # Empirically optimized collection points
                skip_features.append(features)
        
        # Advanced bottleneck processing with global context modeling
        features = self.feature_processor(features)
        
        # Dual transformer processing for enhanced global understanding
        # First transformer: Capture primary global dependencies
        features_t1 = self.transformer1(features)
        
        # Second transformer: Refine global understanding and capture interactions
        features_t2 = self.transformer2(features_t1)
        
        # Food-aware decoding with adaptive skip connection integration
        output = self.decoder(
            bottleneck_features=features_t2,
            skip_features=skip_features,
            skip_adapters=self.skip_adapters,
            fusion_weights=self.fusion_weights,
            target_size=(input_h, input_w)
        )
        
        return output

class TransformerBlock(nn.Module):
    """
    Advanced transformer block with food-specific optimizations
    
    Theoretical Foundation:
    - Multi-head self-attention for global dependency modeling (Vaswani et al., 2017)
    - Learnable scaling parameters for stable training (innovative contribution)
    - Optimized MLP design for food texture understanding
    """
    
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.1, use_learnable_scaling=True):
        super(TransformerBlock, self).__init__()
        
        # Layer normalization for stable training
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention with optimized parameters
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced MLP with food-specific activation pattern
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # Gaussian Error Linear Unit for smoother gradients
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Learnable scaling parameters (novel contribution)
        if use_learnable_scaling:
            self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)  # Conservative initialization
            self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
    
    def forward(self, x):
        """
        Forward pass with learnable residual scaling
        
        Mathematical formulation:
        x' = x + γ₁ * MultiHeadAttention(LayerNorm(x))
        x'' = x' + γ₂ * MLP(LayerNorm(x'))
        """
        B, C, H, W = x.shape
        
        # Reshape for transformer processing: [B, C, H, W] -> [B, HW, C]
        x_reshaped = x.flatten(2).transpose(1, 2)
        
        # Self-attention with learnable scaling
        x_norm = self.norm1(x_reshaped)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + self.gamma1 * attn_output
        
        # MLP with learnable scaling
        x_reshaped = x_reshaped + self.gamma2 * self.mlp(self.norm2(x_reshaped))
        
        # Reshape back to feature map: [B, HW, C] -> [B, C, H, W]
        x_output = x_reshaped.transpose(1, 2).view(B, C, H, W)
        
        return x_output

class FoodAwareDecoder(nn.Module):
    """
    Advanced decoder with food-specific architectural optimizations
    
    Innovation:
    - Adaptive skip connection fusion with learnable weights
    - Food-aware channel attention for texture discrimination
    - Progressive feature refinement for boundary precision
    """
    
    def __init__(self, input_channels, skip_channels, output_channels, use_attention=True):
        super(FoodAwareDecoder, self).__init__()
        
        self.use_attention = use_attention
        
        # Progressive upsampling blocks
        self.up_blocks = nn.ModuleList([
            DecoderBlock(input_channels, 128, use_attention),
            DecoderBlock(128, 64, use_attention),
            DecoderBlock(64, 32, use_attention),
            DecoderBlock(32, 16, use_attention)
        ])
        
        # Final classification head with food-specific design
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, output_channels, 1)  # 1x1 conv for classification
        )
    
    def forward(self, bottleneck_features, skip_features, skip_adapters, fusion_weights, target_size):
        """
        Advanced decoding with adaptive skip connection fusion
        """
        x = bottleneck_features
        
        # Progressive upsampling with skip connection integration
        for i, (up_block, skip_adapter) in enumerate(zip(self.up_blocks, skip_adapters)):
            # Upsample current features
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Process and integrate skip connection if available
            if i < len(skip_features):
                skip_feat = skip_features[-(i+1)]  # Reverse order for decoder
                adapted_skip = skip_adapter(skip_feat)
                
                # Adaptive fusion with learnable weights
                weight = torch.sigmoid(fusion_weights[i])  # Ensure [0,1] range
                x = x + weight * adapted_skip
            
            # Apply decoder block
            x = up_block(x)
        
        # Final upsampling to target resolution
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Final classification
        output = self.final_conv(x)
        
        return output

class SkipConnectionAdapter(nn.Module):
    """
    Adaptive skip connection processing for multi-scale feature fusion
    """
    
    def __init__(self, in_channels, out_channels):
        super(SkipConnectionAdapter, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)
```

### 4.4.2 Advanced Loss Function Architecture

The food-aware combined loss function represents a significant theoretical and practical contribution to semantic segmentation for cultural food applications:

```python
class FoodAwareCombinedLoss(nn.Module):
    """
    Advanced multi-component loss function optimized for food segmentation
    
    Theoretical Foundation:
    - Addresses class imbalance through Dice coefficient optimization
    - Enhances boundary detection via morphological boundary loss
    - Focuses on hard examples through adaptive focal weighting
    - Incorporates cultural food presentation characteristics
    
    Mathematical Formulation:
    L_total = α * L_dice + β * L_boundary + γ * L_focal + δ * L_consistency
    
    where weights are optimized for food segmentation characteristics
    """
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, delta=0.0, num_classes=6):
        super(FoodAwareCombinedLoss, self).__init__()
        
        # Validate weight normalization
        total_weight = alpha + beta + gamma + delta
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        self.alpha = alpha    # Dice loss weight
        self.beta = beta      # Boundary loss weight  
        self.gamma = gamma    # Focal loss weight
        self.delta = delta    # Consistency loss weight
        self.num_classes = num_classes
        
        # Component loss functions
        self.dice_loss = GeneralizedDiceLoss(num_classes=num_classes)
        self.boundary_loss = MorphologicalBoundaryLoss()
        self.focal_loss = AdaptiveFocalLoss(alpha=0.25, gamma=2.0)
        
        if delta > 0:
            self.consistency_loss = ConsistencyRegularizationLoss()
    
    def forward(self, predictions, targets, auxiliary_predictions=None):
        """
        Compute combined loss with optional auxiliary supervision
        
        Args:
            predictions: Main model predictions [B, C, H, W]
            targets: Ground truth segmentation [B, H, W]
            auxiliary_predictions: Optional auxiliary predictions for consistency
            
        Returns:
            Combined loss value and component losses for analysis
        """
        device = predictions.device
        
        # Ensure targets are on the same device and proper format
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets = targets.to(device, dtype=torch.long)
        
        # Component loss computation
        dice_loss = self.dice_loss(predictions, targets)
        boundary_loss = self.boundary_loss(predictions, targets)
        focal_loss = self.focal_loss(predictions, targets)
        
        # Combined loss computation
        total_loss = (self.alpha * dice_loss + 
                     self.beta * boundary_loss + 
                     self.gamma * focal_loss)
        
        # Optional consistency regularization
        consistency_loss = torch.tensor(0.0, device=device)
        if self.delta > 0 and auxiliary_predictions is not None:
            consistency_loss = self.consistency_loss(predictions, auxiliary_predictions)
            total_loss += self.delta * consistency_loss
        
        # Return detailed loss information for analysis
        loss_components = {
            'total_loss': total_loss,
            'dice_loss': dice_loss,
            'boundary_loss': boundary_loss,
            'focal_loss': focal_loss,
            'consistency_loss': consistency_loss
        }
        
        return total_loss, loss_components

class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss optimized for food segmentation with class balancing
    
    Mathematical Foundation:
    GDL = 1 - (2 * Σᵢ wᵢ * |Pᵢ ∩ Tᵢ|) / (Σᵢ wᵢ * (|Pᵢ| + |Tᵢ|))
    
    where wᵢ = 1 / (Σⱼ Tᵢⱼ)² provides inverse frequency weighting
    """
    
    def __init__(self, num_classes, smooth=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute class weights based on inverse frequency
        class_weights = []
        for c in range(self.num_classes):
            class_sum = targets_one_hot[:, c].sum()
            if class_sum > 0:
                weight = 1.0 / (class_sum.pow(2) + self.smooth)
            else:
                weight = 0.0
            class_weights.append(weight)
        
        class_weights = torch.tensor(class_weights, device=predictions.device)
        
        # Compute generalized Dice loss
        intersection = (predictions * targets_one_hot).sum(dim=(0, 2, 3))
        union = predictions.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice_scores = (2 * intersection + self.smooth) / (union + self.smooth)
        weighted_dice = (class_weights * dice_scores).sum() / (class_weights.sum() + self.smooth)
        
        return 1 - weighted_dice

class MorphologicalBoundaryLoss(nn.Module):
    """
    Advanced boundary loss using morphological operations for precise edge detection
    
    Theoretical Foundation:
    - Utilizes morphological gradients to extract boundary information
    - Applies distance transform for boundary proximity weighting
    - Optimized for irregular food boundaries common in traditional presentations
    """
    
    def __init__(self, kernel_size=3):
        super(MorphologicalBoundaryLoss, self).__init__()
        self.kernel_size = kernel_size
        
        # Morphological kernels for boundary extraction
        kernel = torch.ones(kernel_size, kernel_size)
        self.register_buffer('morphological_kernel', kernel)
    
    def _extract_boundaries(self, masks):
        """Extract boundary maps using morphological operations"""
        boundaries = []
        
        for mask in masks:
            # Convert to binary masks for each class
            class_boundaries = []
            for c in range(mask.max().item() + 1):
                class_mask = (mask == c).float()
                
                # Morphological gradient for boundary extraction
                dilated = F.max_pool2d(class_mask.unsqueeze(0), 
                                     kernel_size=self.kernel_size, 
                                     stride=1, 
                                     padding=self.kernel_size//2)
                eroded = -F.max_pool2d(-class_mask.unsqueeze(0), 
                                     kernel_size=self.kernel_size, 
                                     stride=1, 
                                     padding=self.kernel_size//2)
                boundary = dilated - eroded
                class_boundaries.append(boundary.squeeze(0))
            
            boundaries.append(torch.stack(class_boundaries))
        
        return torch.stack(boundaries)
    
    def forward(self, predictions, targets):
        # Extract predicted boundaries
        pred_probs = F.softmax(predictions, dim=1)
        pred_classes = torch.argmax(pred_probs, dim=1)
        pred_boundaries = self._extract_boundaries(pred_classes)
        
        # Extract target boundaries
        target_boundaries = self._extract_boundaries(targets)
        
        # Compute boundary-aware loss
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_boundaries, target_boundaries.float(), reduction='mean'
        )
        
        return boundary_loss

class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss with dynamic gamma adjustment for food segmentation
    
    Innovation:
    - Adaptive gamma parameter based on class difficulty
    - Enhanced for handling class imbalance in food categories
    - Optimized for boundary regions and hard examples
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, adaptive_gamma=True):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.adaptive_gamma = adaptive_gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Adaptive gamma based on prediction confidence
        if self.adaptive_gamma:
            # Higher gamma for more confident incorrect predictions
            adaptive_gamma = self.gamma * (2 - pt)
        else:
            adaptive_gamma = self.gamma
        
        focal_loss = self.alpha * (1 - pt) ** adaptive_gamma * ce_loss
        
        return focal_loss.mean()
```

### 4.4.3 Advanced Training Pipeline with Theoretical Optimization

The training pipeline incorporates state-of-the-art optimization techniques grounded in deep learning theory and empirical food segmentation research:

```python
class AdvancedTrainingPipeline:
    """
    Comprehensive training pipeline with theoretical optimization foundations
    
    Key Innovations:
    - Adaptive learning rate scheduling based on loss landscape analysis
    - Progressive image augmentation aligned with curriculum learning
    - Advanced mixed precision training for computational efficiency
    - Real-time performance monitoring with early stopping mechanisms
    """
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Advanced optimizer with theoretical grounding
        self.optimizer = self._setup_optimizer()
        
        # Learning rate scheduler with adaptive mechanisms
        self.scheduler = self._setup_scheduler()
        
        # Loss function optimized for food segmentation
        self.criterion = FoodAwareCombinedLoss(
            alpha=config.get('dice_weight', 0.6),
            beta=config.get('boundary_weight', 0.3),
            gamma=config.get('focal_weight', 0.1),
            num_classes=config['num_classes']
        )
        
        # Mixed precision training for efficiency
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Performance tracking and early stopping
        self.best_val_iou = 0.0
        self.patience_counter = 0
        self.training_history = []
    
    def _setup_optimizer(self):
        """
        Advanced optimizer setup with theoretical foundations
        
        Uses AdamW with decoupled weight decay (Loshchilov & Hutter, 2017)
        Optimized hyperparameters for food segmentation tasks
        """
        # Differential learning rates for pretrained vs. new components
        backbone_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
        
        return AdamW([
            {'params': backbone_params, 'lr': self.config['lr'] * 0.1},  # Lower LR for pretrained
            {'params': new_params, 'lr': self.config['lr']}              # Higher LR for new components
        ], weight_decay=self.config.get('weight_decay', 1e-4))
    
    def _setup_scheduler(self):
        """
        Advanced learning rate scheduling with plateau detection
        
        Combines cosine annealing with warm restarts for optimal convergence
        """
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=self.config.get('scheduler_tmult', 2),
            eta_min=self.config.get('min_lr', 1e-6)
        )
    
    def train_epoch(self, epoch):
        """
        Advanced training epoch with theoretical optimization principles
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'iou': [], 'dice': [], 'boundary_f1': []}
        
        # Progressive augmentation intensity based on training progress
        augmentation_intensity = min(1.0, epoch / 20.0)  # Curriculum learning principle
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.cuda(), targets.cuda()
            
            # Mixed precision training for efficiency
            if self.scaler:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_components = self.criterion(predictions, targets)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping for stable training
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, loss_components = self.criterion(predictions, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Real-time metrics computation
            with torch.no_grad():
                batch_metrics = self._compute_metrics(predictions, targets)
                for key, value in batch_metrics.items():
                    epoch_metrics[key].append(value)
            
            epoch_losses.append(loss.item())
            
            # Dynamic progress bar updates
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_metrics["iou"]:.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Epoch-level statistics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate_epoch(self, epoch):
        """
        Comprehensive validation with multiple evaluation metrics
        """
        self.model.eval()
        val_losses = []
        val_metrics = {'iou': [], 'dice': [], 'boundary_f1': [], 'hausdorff': []}
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc=f'Validation Epoch {epoch}'):
                images, targets = images.cuda(), targets.cuda()
                
                predictions = self.model(images)
                loss, _ = self.criterion(predictions, targets)
                val_losses.append(loss.item())
                
                # Comprehensive metric computation
                batch_metrics = self._compute_comprehensive_metrics(predictions, targets)
                for key, value in batch_metrics.items():
                    val_metrics[key].append(value)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
        
        return avg_val_loss, avg_val_metrics
    
    def _compute_comprehensive_metrics(self, predictions, targets):
        """
        Comprehensive evaluation metrics for food segmentation assessment
        """
        pred_classes = torch.argmax(predictions, dim=1)
        
        # Standard segmentation metrics
        iou = self._compute_iou(pred_classes, targets)
        dice = self._compute_dice(pred_classes, targets)
        boundary_f1 = self._compute_boundary_f1(pred_classes, targets)
        
        # Advanced geometric metrics for food presentation analysis
        hausdorff = self._compute_hausdorff_distance(pred_classes, targets)
        
        return {
            'iou': iou,
            'dice': dice,
            'boundary_f1': boundary_f1,
            'hausdorff': hausdorff
        }
    
    def train(self):
        """
        Complete training loop with advanced optimization and monitoring
        """
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Model checkpointing based on validation IoU
            if val_metrics['iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['iou']
                best_model_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_iou': self.best_val_iou,
                    'config': self.config
                }, f'best_ghanasegnet_epoch_{epoch}_iou_{self.best_val_iou:.4f}.pth')
            else:
                self.patience_counter += 1
            
            # Training history logging
            epoch_history = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_history)
            
            # Early stopping check
            if self.patience_counter >= self.config.get('patience', 20):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Comprehensive logging
            self._log_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        
        # Load best model for final evaluation
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.training_history
```

## 4.5 Performance Optimization and Computational Efficiency

### 4.5.1 Memory Optimization Strategies

The GhanaSegNet implementation incorporates several memory optimization techniques essential for deployment in resource-constrained environments:

```python
class MemoryOptimizedGhanaSegNet(GhanaSegNet):
    """
    Memory-optimized version of GhanaSegNet for resource-constrained deployment
    
    Optimization Techniques:
    - Gradient checkpointing for memory-time trade-off
    - Dynamic input resolution based on available memory
    - Efficient attention computation with linear complexity
    - Optimized feature map management
    """
    
    def __init__(self, *args, memory_efficient=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_efficient = memory_efficient
        
        if memory_efficient:
            # Enable gradient checkpointing for transformer blocks
            self.transformer1 = checkpoint_wrapper(self.transformer1)
            self.transformer2 = checkpoint_wrapper(self.transformer2)
    
    def forward(self, x):
        """Memory-efficient forward pass with dynamic optimization"""
        if self.memory_efficient:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        """Implementation with memory optimization strategies"""
        # Dynamic input size adjustment based on available memory
        if self.training and self._should_reduce_resolution():
            x = F.interpolate(x, scale_factor=0.75, mode='bilinear', align_corners=False)
        
        return super().forward(x)
    
    def _should_reduce_resolution(self):
        """Dynamic memory monitoring for resolution adjustment"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return memory_used > 0.85  # Reduce resolution if >85% memory used
        return False

class EfficientAttentionMechanism(nn.Module):
    """
    Linear complexity attention mechanism for large image processing
    
    Theoretical Foundation:
    - Implements efficient attention approximation (Choromanski et al., 2020)
    - Reduces complexity from O(n²) to O(n) for sequence length n
    - Maintains attention quality while improving computational efficiency
    """
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """Linear complexity attention computation"""
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Efficient attention computation using FAVOR+ algorithm
        q = q * self.scale
        
        # Positive feature map for linear attention
        q_pos = F.relu(q) + 1e-6
        k_pos = F.relu(k) + 1e-6
        
        # Linear attention computation
        context = torch.einsum('bhnd,bhne->bhde', k_pos, v)
        attention_weights = torch.einsum('bhnd,bhde->bhne', q_pos, context)
        
        # Normalization
        normalizer = torch.einsum('bhnd,bhd->bhn', q_pos, k_pos.sum(dim=2))
        attention_weights = attention_weights / (normalizer.unsqueeze(-1) + 1e-8)
        
        out = rearrange(attention_weights, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

### 4.5.2 Mobile Deployment Optimization

For deployment on mobile devices and edge computing platforms, the architecture includes specialized optimization strategies:

```python
class MobileGhanaSegNet(nn.Module):
    """
    Mobile-optimized version of GhanaSegNet for edge deployment
    
    Optimization Features:
    - Depthwise separable convolutions for efficiency
    - Quantization-aware training support
    - Dynamic inference scaling
    - Hardware-specific optimizations
    """
    
    def __init__(self, num_classes=6, quantization_aware=True):
        super().__init__()
        
        # Mobile-optimized backbone
        self.backbone = efficientnet_b0(pretrained=True)
        
        # Lightweight transformer blocks
        self.transformer = MobileTransformerBlock(256, heads=4, mlp_dim=256)
        
        # Efficient decoder with depthwise separable convolutions
        self.decoder = MobileDecoder(256, num_classes)
        
        if quantization_aware:
            self._prepare_quantization()
    
    def _prepare_quantization(self):
        """Prepare model for quantization-aware training"""
        self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self, inplace=True)
    
    def forward(self, x):
        """Optimized forward pass for mobile deployment"""
        # Adaptive input processing based on device capabilities
        if self._is_mobile_device():
            x = self._mobile_preprocessing(x)
        
        features = self.backbone.features(x)
        features = self.transformer(features)
        output = self.decoder(features)
        
        return output
    
    def _mobile_preprocessing(self, x):
        """Mobile-specific preprocessing optimizations"""
        # Dynamic resolution adjustment
        _, _, h, w = x.shape
        if h > 512 or w > 512:
            scale_factor = 512 / max(h, w)
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear')
        
        return x
    
    def export_mobile_model(self, example_input):
        """Export optimized model for mobile deployment"""
        # Convert to TorchScript for mobile deployment
        self.eval()
        traced_model = torch.jit.trace(self, example_input)
        
        # Optimize for mobile
        optimized_model = optimize_for_mobile(traced_model)
        
        return optimized_model

class DepthwiseSeparableConv(nn.Module):
    """
    Efficient depthwise separable convolution implementation
    
    Theoretical Foundation:
    - Reduces parameters from k²·C_in·C_out to k²·C_in + C_in·C_out
    - Maintains representation capacity while improving efficiency
    - Essential for mobile deployment constraints
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.activation(x)
```

## 4.6 Comprehensive Testing and Validation Framework

### 4.6.1 Unit Testing Architecture

The implementation includes comprehensive testing frameworks ensuring reliability and maintainability:

```python
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

class TestGhanaSegNetArchitecture:
    """
    Comprehensive test suite for GhanaSegNet architecture validation
    
    Testing Philosophy:
    - Unit tests for individual components
    - Integration tests for end-to-end functionality
    - Performance benchmarks for optimization validation
    - Robustness tests for edge cases
    """
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing"""
        return GhanaSegNet(num_classes=6)
    
    @pytest.fixture
    def sample_input(self):
        """Generate sample input tensors"""
        return torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    def test_model_initialization(self, sample_model):
        """Test proper model initialization"""
        assert isinstance(sample_model, nn.Module)
        assert hasattr(sample_model, 'encoder')
        assert hasattr(sample_model, 'transformer1')
        assert hasattr(sample_model, 'transformer2')
        assert hasattr(sample_model, 'decoder')
    
    def test_forward_pass_shape(self, sample_model, sample_input):
        """Test output shape consistency"""
        with torch.no_grad():
            output = sample_model(sample_input)
        
        expected_shape = (2, 6, 224, 224)  # [batch, classes, height, width]
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_gradient_flow(self, sample_model, sample_input):
        """Test gradient computation and backpropagation"""
        sample_model.train()
        output = sample_model(sample_input)
        
        # Create dummy loss
        target = torch.randint(0, 6, (2, 224, 224))
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backpropagate
        loss.backward()
        
        # Check that gradients are computed
        for name, param in sample_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
    
    def test_memory_efficiency(self, sample_model, sample_input):
        """Test memory consumption within acceptable limits"""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        with torch.no_grad():
            output = sample_model(sample_input)
        
        peak_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = peak_memory - initial_memory
        
        # Assert memory usage is reasonable (adjust threshold as needed)
        max_expected_memory = 500 * 1024 * 1024  # 500 MB
        assert memory_usage < max_expected_memory, f"Memory usage {memory_usage} exceeds threshold"
    
    def test_loss_function_convergence(self):
        """Test loss function numerical stability and convergence"""
        loss_fn = FoodAwareCombinedLoss(num_classes=6)
        
        # Test with perfect predictions
        perfect_pred = torch.eye(6).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 32, 32) * 100
        perfect_target = torch.arange(6).unsqueeze(0).unsqueeze(-1).repeat(1, 32, 32) 
        
        loss, components = loss_fn(perfect_pred, perfect_target)
        
        # Loss should be close to zero for perfect predictions
        assert loss.item() < 0.1, f"Loss too high for perfect predictions: {loss.item()}"
    
    @pytest.mark.parametrize("input_size", [(224, 224), (384, 384), (512, 512)])
    def test_multi_resolution_support(self, sample_model, input_size):
        """Test model performance across different input resolutions"""
        h, w = input_size
        test_input = torch.randn(1, 3, h, w)
        
        with torch.no_grad():
            output = sample_model(test_input)
        
        expected_shape = (1, 6, h, w)
        assert output.shape == expected_shape, f"Failed for input size {input_size}"

class TestTrainingPipeline:
    """Test suite for training pipeline validation"""
    
    def test_training_step_execution(self):
        """Test individual training step execution"""
        # Mock components
        model = MagicMock()
        optimizer = MagicMock()
        criterion = MagicMock()
        
        # Mock data
        sample_batch = (torch.randn(2, 3, 224, 224), torch.randint(0, 6, (2, 224, 224)))
        
        # Simulate training step
        images, targets = sample_batch
        model.return_value = torch.randn(2, 6, 224, 224)
        criterion.return_value = (torch.tensor(0.5), {})
        
        # Verify training step execution
        predictions = model(images)
        loss, _ = criterion(predictions, targets)
        
        optimizer.zero_grad.assert_called_once()
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_validation_metrics_computation(self):
        """Test validation metrics computation accuracy"""
        # Create sample predictions and targets
        pred = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]])  # [1, 2, 2, 2]
        target = torch.tensor([[[1, 0], [0, 1]]])  # [1, 0, 0, 1]
        
        # Compute IoU manually for verification
        pred_classes = torch.argmax(pred, dim=1)
        
        # Verify IoU computation
        # Class 0: intersection=1, union=2 -> IoU=0.5
        # Class 1: intersection=1, union=2 -> IoU=0.5
        # Mean IoU should be 0.5
        
        iou = self._compute_iou_manual(pred_classes, target)
        expected_iou = 0.5
        
        assert abs(iou - expected_iou) < 1e-6, f"IoU mismatch: expected {expected_iou}, got {iou}"
    
    def _compute_iou_manual(self, pred, target):
        """Manual IoU computation for test verification"""
        num_classes = int(max(pred.max(), target.max())) + 1
        iou_scores = []
        
        for c in range(num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            
            if union > 0:
                iou_scores.append(intersection / union)
        
        return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

class PerformanceBenchmarkTests:
    """Performance benchmarking and optimization validation"""
    
    def test_inference_speed_benchmark(self):
        """Benchmark inference speed across different configurations"""
        model = GhanaSegNet(num_classes=6)
        model.eval()
        
        input_tensor = torch.randn(1, 3, 384, 384)
        
        # Warmup runs
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Benchmark runs
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / 100
        
        # Assert reasonable inference time (adjust threshold as needed)
        max_inference_time = 0.1  # 100ms threshold
        assert avg_inference_time < max_inference_time, \
            f"Inference too slow: {avg_inference_time:.4f}s > {max_inference_time}s"
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage for different batch sizes"""
        model = GhanaSegNet(num_classes=6)
        
        memory_usage = {}
        
        for batch_size in [1, 2, 4, 8]:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            input_tensor = torch.randn(batch_size, 3, 384, 384)
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage[batch_size] = peak_memory
        
        # Verify memory scales reasonably with batch size
        if len(memory_usage) > 1:
            memory_items = list(memory_usage.items())
            for i in range(1, len(memory_items)):
                batch_prev, mem_prev = memory_items[i-1]
                batch_curr, mem_curr = memory_items[i]
                
                # Memory should not increase more than linearly with batch size
                memory_ratio = mem_curr / mem_prev
                batch_ratio = batch_curr / batch_prev
                
                assert memory_ratio <= batch_ratio * 1.2, \
                    f"Memory usage not scaling linearly: {memory_ratio} > {batch_ratio * 1.2}"
```
```
```
├── scripts/                    # Training and evaluation scripts
│   ├── train_baselines.py     # Multi-model training pipeline
│   ├── evaluate.py            # Model evaluation framework
│   └── inference.py           # Real-time inference utilities
├── notebooks/                  # Research and analysis notebooks
│   ├── baseline_segmentation.ipynb
│   └── results_visualization.ipynb
└── configs/                    # Configuration files
    ├── training_config.yaml   # Training hyperparameters
    └── model_config.yaml      # Model architecture configurations
```

### 4.2.3 Development Environment Configuration

The development environment supports both local development and cloud-based training scenarios:

**Local Development Setup:**
- Python 3.13.3 with virtual environment isolation
- PyTorch with CPU/CUDA support for flexible development
- Jupyter notebooks for interactive research and visualization
- Git version control with comprehensive .gitignore configuration

**Cloud Training Environment:**
- Google Colab integration for GPU-accelerated training
- Automated dependency installation and environment setup
- Persistent storage integration for dataset and model checkpoints
- Scalable compute resource allocation based on training requirements

## 4.3 GhanaSegNet Architecture Implementation

### 4.3.1 Hybrid CNN-Transformer Design

The GhanaSegNet architecture represents a novel fusion of convolutional neural networks and transformer attention mechanisms, specifically optimized for food segmentation tasks. The implementation consists of three primary components:

**1. EfficientNet-B0 Encoder:**
```python
class GhanaSegNet(nn.Module):
    def __init__(self, num_classes=6, dropout=0.1):
        super(GhanaSegNet, self).__init__()
        
        # EfficientNet-B0 backbone (ImageNet pretrained)
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Enhanced channel reduction with better feature processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
```

The EfficientNet-B0 encoder provides an optimal balance between computational efficiency and feature extraction capability. The choice of EfficientNet-B0 is motivated by its superior parameter efficiency (5.3M parameters) compared to traditional ResNet architectures while maintaining competitive performance.

**2. Multi-Scale Transformer Integration:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable scaling parameters for better training dynamics
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)
```

The transformer blocks incorporate learnable scaling parameters (γ₁, γ₂) to improve training stability and convergence. This design addresses the challenge of integrating transformer attention with convolutional features at different scales.

**3. Enhanced U-Net Decoder with Skip Connections:**
```python
def forward(self, x):
    # Extract multi-scale features from EfficientNet encoder
    encoder_features = []
    x_enc = self.encoder._conv_stem(x)
    
    for idx, block in enumerate(self.encoder._blocks):
        x_enc = block(x_enc)
        if idx in [2, 5, 10, 15]:  # Strategic feature collection
            encoder_features.append(x_enc)
    
    # Enhanced bottleneck with global context
    features = self.conv1(x_enc)
    features = self.transformer1(features)
    features = self.transformer2(features)
    
    # Decoder with weighted skip connections
    d1 = self.up1(features)
    if len(encoder_features) >= 4:
        skip_feat = encoder_features[-2]
        d1 = d1 + 0.5 * self._adapt_skip_connection(skip_feat, d1)
```

### 4.3.2 Advanced Feature Fusion Mechanisms

The implementation incorporates sophisticated feature fusion strategies to effectively combine multi-scale CNN features with global transformer attention:

**Weighted Skip Connection Strategy:**
- Strategic feature extraction at encoder stages [2, 5, 10, 15]
- Adaptive channel alignment for skip connections
- Weighted fusion coefficients (0.5, 0.3) optimized through ablation studies

**Channel Attention Integration:**
```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # Attention mechanism for feature selection
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
```

### 4.3.3 Architectural Optimization Strategies

**Parameter Initialization:**
The implementation employs careful weight initialization strategies to ensure stable training convergence:

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d) and not hasattr(m, '_is_pretrained'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

**Memory Optimization:**
- Gradient checkpointing for reduced memory consumption
- Efficient tensor operations with in-place computations
- Dynamic input size handling for flexible deployment scenarios

## 4.4 Loss Function Design and Implementation

### 4.4.1 Food-Aware Combined Loss Function

The implementation introduces a novel food-aware loss function that addresses the specific challenges of food segmentation:

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Dice weight
        self.beta = beta    # Boundary weight
        self.gamma = 1.0 - alpha - beta  # Focal weight
        
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.focal = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        focal_loss = self.focal(inputs, targets.long())
        
        # Optimized weighting for Ghana food segmentation
        total_loss = (self.alpha * dice_loss + 
                     self.beta * boundary_loss + 
                     self.gamma * focal_loss)
        return total_loss
```

**Component Analysis:**

1. **Dice Loss (60% weight):** Addresses class imbalance and promotes overlap maximization
2. **Boundary Loss (30% weight):** Enhances edge detection for irregular food boundaries
3. **Focal Loss (10% weight):** Focuses training on hard examples and misclassified pixels

### 4.4.2 Boundary-Aware Loss Implementation

The boundary loss component specifically targets the challenge of irregular food boundaries common in traditional Ghanaian meal presentations:

```python
class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        
    def forward(self, pred, target):
        # Compute boundary maps using morphological operations
        target_boundary = self._compute_boundary(target)
        pred_boundary = self._compute_boundary(torch.argmax(pred, dim=1))
        
        # Calculate boundary-specific loss
        boundary_loss = F.binary_cross_entropy_with_logits(
            pred_boundary.float(), 
            target_boundary.float()
        )
        return boundary_loss
```

## 4.5 Training Pipeline Implementation

### 4.5.1 Multi-Model Training Framework

The implementation provides a comprehensive training framework supporting multiple baseline comparisons:

```python
def train_model(model_name, config):
    """Unified training function for all model architectures"""
    
    # Model initialization with architecture-specific optimizations
    model, criterion = get_model_and_criterion(model_name, config['num_classes'])
    
    # Optimizer configuration with model-specific learning rates
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Advanced learning rate scheduling
    if config['epochs'] <= 10:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[config['epochs']//3, 2*config['epochs']//3],
            gamma=0.1
        )
```

### 4.5.2 Data Loading and Augmentation Pipeline

**Custom Dataset Implementation:**
```python
class GhanaFoodDataset(Dataset):
    def __init__(self, split='train', target_size=(384, 384), augment=True):
        self.split = split
        self.target_size = target_size
        self.augment = augment and split == 'train'
        
        # Load file paths with validation
        self.image_paths, self.mask_paths = self._load_file_paths()
        
        # Initialize augmentation pipeline
        if self.augment:
            self.transform = self._create_augmentation_pipeline()
```

**Advanced Augmentation Strategy:**
```python
def _create_augmentation_pipeline(self):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
```

### 4.5.3 Training Optimization Strategies

**Gradient Clipping and Regularization:**
```python
# Training loop with advanced optimization
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item()
    
    scheduler.step()
```

**Model Checkpointing and Best Model Selection:**
```python
# Save best model based on validation IoU
if avg_iou > best_iou:
    best_iou = avg_iou
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou,
        'training_history': training_history
    }
    torch.save(checkpoint, f'checkpoints/{model_name}_best.pth')
```

## 4.6 Evaluation Framework Implementation

### 4.6.1 Comprehensive Metrics Suite

The implementation includes a comprehensive evaluation framework with multiple performance metrics:

```python
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def update(self, predictions, targets):
        # IoU calculation per class
        for class_id in range(self.num_classes):
            pred_mask = (predictions == class_id)
            true_mask = (targets == class_id)
            
            intersection = (pred_mask & true_mask).sum().item()
            union = (pred_mask | true_mask).sum().item()
            
            if union > 0:
                self.class_ious[class_id].append(intersection / union)
    
    def compute_metrics(self):
        return {
            'mean_iou': np.mean([np.mean(ious) for ious in self.class_ious if ious]),
            'class_ious': {f'class_{i}': np.mean(ious) if ious else 0.0 
                          for i, ious in enumerate(self.class_ious)},
            'pixel_accuracy': self.correct_pixels / self.total_pixels
        }
```

### 4.6.2 Statistical Analysis Framework

**Performance Comparison and Significance Testing:**
```python
def statistical_comparison(results_dict):
    """Compare model performances with statistical significance testing"""
    from scipy import stats
    
    models = list(results_dict.keys())
    comparisons = {}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Paired t-test for IoU scores
            t_stat, p_value = stats.ttest_rel(
                results_dict[model1]['iou_scores'],
                results_dict[model2]['iou_scores']
            )
            
            comparisons[f"{model1}_vs_{model2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return comparisons
```

## 4.7 Deployment and Optimization Considerations

### 4.7.1 Model Optimization for Deployment

**Quantization and Compression:**
```python
def optimize_for_deployment(model):
    """Apply deployment optimizations"""
    
    # Quantization-aware training
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare_qat(model)
    
    # Model compression through pruning
    parameters_to_prune = [
        (module, 'weight') for module in model.modules() 
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.utils.prune.L1Unstructured,
        amount=0.2,
    )
    
    return model_prepared
```

### 4.7.2 Real-Time Inference Pipeline

**Efficient Inference Implementation:**
```python
class GhanaSegNetInference:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self._load_optimized_model(model_path)
        self.transform = self._create_inference_transform()
    
    def predict(self, image):
        """Single image prediction with preprocessing"""
        preprocessed = self.transform(image=image)['image']
        
        with torch.no_grad():
            output = self.model(preprocessed.unsqueeze(0).to(self.device))
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return prediction
    
    def batch_predict(self, images, batch_size=8):
        """Efficient batch prediction"""
        predictions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_tensor = torch.stack([
                self.transform(image=img)['image'] for img in batch
            ]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_preds)
        
        return predictions
```

## 4.8 Implementation Challenges and Solutions

### 4.8.1 Memory Optimization Challenges

**Challenge:** Training large transformer-based models on limited GPU memory.

**Solution:** Implementation of gradient checkpointing and mixed-precision training:
```python
# Mixed precision training implementation
scaler = torch.cuda.amp.GradScaler()

for images, masks in train_loader:
    optimizer.zero_grad()
    
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, masks)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4.8.2 Training Stability Issues

**Challenge:** Unstable training convergence with hybrid CNN-Transformer architecture.

**Solution:** Learnable scaling parameters and careful initialization:
```python
# Learnable scaling for stable training
self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)  # Conservative initialization
self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)

# Application in forward pass
x = x + self.gamma1 * self.transformer_output
```

### 4.8.3 Cross-Platform Compatibility

**Challenge:** Ensuring consistent performance across different hardware configurations.

**Solution:** Dynamic device detection and adaptive batch sizing:
```python
def get_optimal_batch_size(model, device):
    """Dynamically determine optimal batch size"""
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        return min(16, max(2, gpu_memory // (1024**3)))  # Scale with GPU memory
    else:
        return 4  # Conservative batch size for CPU training
```

## 4.9 Code Quality and Testing Framework

### 4.9.1 Unit Testing Implementation

```python
import unittest
import torch

class TestGhanaSegNet(unittest.TestCase):
    def setUp(self):
        self.model = GhanaSegNet(num_classes=6)
        self.batch_size = 2
        self.input_tensor = torch.randn(self.batch_size, 3, 384, 384)
    
    def test_forward_pass_shape(self):
        """Test output shape consistency"""
        output = self.model(self.input_tensor)
        expected_shape = (self.batch_size, 6, 384, 384)
        self.assertEqual(output.shape, expected_shape)
    
    def test_gradient_flow(self):
        """Test gradient computation"""
        output = self.model(self.input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
```

### 4.9.2 Performance Profiling

```python
def profile_model_performance(model, input_size=(1, 3, 384, 384), iterations=100):
    """Profile model inference performance"""
    import time
    
    model.eval()
    dummy_input = torch.randn(input_size)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Timing
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    avg_inference_time = (time.time() - start_time) / iterations
    fps = 1.0 / avg_inference_time
    
    return {
        'avg_inference_time_ms': avg_inference_time * 1000,
        'fps': fps,
        'parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    }
```

## 4.10 Chapter Summary

This chapter has presented the comprehensive implementation details of the GhanaSegNet architecture and training framework. Key implementation contributions include:

1. **Modular Architecture Design:** Clean separation of concerns with reusable components
2. **Hybrid CNN-Transformer Implementation:** Novel fusion of EfficientNet and transformer attention
3. **Food-Aware Loss Function:** Specialized loss components for food segmentation challenges
4. **Comprehensive Training Pipeline:** Multi-model comparison framework with advanced optimizations
5. **Deployment-Ready Optimization:** Quantization, pruning, and efficient inference implementations
6. **Robust Testing Framework:** Unit tests, performance profiling, and cross-platform compatibility

The implementation demonstrates software engineering best practices while delivering cutting-edge deep learning capabilities specifically tailored for Ghanaian food segmentation. The modular design ensures maintainability and extensibility for future research directions, while the optimization strategies enable practical deployment in resource-constrained environments.

The next chapter will present the experimental results and performance analysis of this implementation, demonstrating the effectiveness of the proposed architectural innovations and training strategies.
