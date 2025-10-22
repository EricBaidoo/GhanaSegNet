# Chapter 3: Methodology

3.1 Overview

Purpose: This chapter formalizes the methodology behind GhanaSegNet and describes the design choices made to address the specific challenges of segmenting traditional Ghanaian foods. The goal is to present a reproducible pipeline (data → model → training → evaluation) and to provide a compact contract that specifies inputs, outputs, key assumptions, and failure modes.

Contract (inputs / outputs / success criteria):
- Inputs: RGB images X ∈ R^{H×W×3} and corresponding ground-truth masks Y ∈ {0,...,C-1}^{H×W} (C=6 in the FRANI experiments). Batches have shape [B, 3, H, W].
- Outputs: Per-pixel class logits L ∈ R^{B×C×H×W}, softmax probabilities P, and predicted masks Ŷ = argmax_c P. Checkpoints save model weights and a run JSON containing hyperparameters and scalar metrics.
- Success criteria: Primary metric is mean Intersection over Union (mIoU) on the FRANI validation split; reproducible runs should report mIoU ≈ 0.2437 (24.37%) for the baseline run described in this thesis and provide checkpoint/run JSON artifacts.
- Failure modes: mismatched mask encodings (palette vs integer class ids), shape mismatches between logits and labels during validation, and nondeterministic behavior from unspecified random seeds or mixed-precision nondeterminism.

High-level pipeline summary:
1. Data preparation and augmentation: normalize to ImageNet statistics, progressive resizing (256→320→384), and heavy augmentations implemented with Albumentations to simulate cultural presentation variability.
2. Multi-stage transfer learning: ImageNet backbone → food-domain finetuning (Food101/Nutrition5k) → African cuisine adaptation → FRANI specialization with staged unfreezing and differential learning rates.
3. Model: EfficientNet-derived encoder + lightweight transformer bottleneck(s) + food-aware decoder with adaptive skip-fusion and attention.
4. Training: Combined loss (Dice + Boundary + Focal), AdamW optimizer, mixed precision when available, and learning-rate scheduling with warm restarts or cosine annealing. Early stopping uses validation mIoU.
5. Evaluation: Save per-epoch checkpoints and run JSONs to `checkpoints/ghanasegnet/`; compute mIoU, Dice, pixel accuracy and boundary F1 on the validation set; retain per-class IoU for analysis.

Figure / table placeholders:
- Figure 3.1: End-to-end pipeline (data → pretraining → model → training → evaluation).
- Table 3.1: Input/output shapes and primary run hyperparameters (see `scripts/train_baselines.py`).

Edge cases and mitigations:
- Small-object sensitivity: use multiscale inputs and weighted losses (Generalized Dice) to upweight rare classes.
- Lighting / occlusion: include strong photometric and geometric augmentations.
- Label noise / palette mismatches: include a preprocessing validator in the data loader that checks mask value ranges and file correspondences.

GhanaSegNet integrates these components into a compact, deployable research pipeline designed for reproducibility and cultural sensitivity.


3.2 Multi-Stage Transfer Learning Framework

Purpose: Reduce domain gap progressively so the model learns generic visual patterns (edges, textures) before specializing on culturally specific food presentation styles.

Method summary:
- Stage 0 (ImageNet pretraining): Initialize encoder weights from ImageNet-pretrained EfficientNet variants (or `timm` backbones) to capture general-purpose features.
- Stage 1 (Food-domain finetuning): Finetune on large, diverse food datasets (Food101, Nutrition5k) to learn food-specific textures and color distributions.
- Stage 2 (Regional adaptation): Fine-tune on African cuisine proxies and synthetic examples that reflect Ghanaian serving styles (augmented lighting, plating variations, synthetic compositing) to reduce cultural domain shift.
- Stage 3 (FRANI specialization): Final fine-tuning on FRANI with staged unfreezing: initially train decoder/head, then progressively unfreeze earlier encoder blocks while reducing learning rates.

Practical notes and contract:
- Use lower learning rates for pretrained encoder layers (e.g., lr_encoder = 0.1 * lr_head).
- Apply staged unfreezing after N epochs of stable decoder training (empirically N ∈ [3, 10]).
- Save run metadata at every stage to enable later ablation studies.

3.3 Hybrid Architecture Design (CNN + Transformer)

Purpose: Combine local, translation-invariant feature extraction from CNNs with transformer's global attention to resolve ambiguous food-food boundaries and contextual occlusions.

Architecture components (concise):
- Encoder: EfficientNet-derived feature extractor producing multi-scale feature maps {F_1, F_2, F_3, F_4}.
- Transformer bottleneck(s): Lightweight transformer blocks applied to a reduced-channel bottleneck to add global context with low compute overhead.
- Decoder: Food-aware decoder with learnable skip adapters and channel attention modules; outputs per-pixel logits L ∈ R^{B×C×H×W}.

Implementation pointers:
- See `models/ghanasegnet.py` for the concrete implementation. The bottleneck uses 1×1 conv reduction to keep transformer token counts reasonable for typical input resolutions (256–384 px).
- Use `timm` or `efficientnet-pytorch` for backbones; if memory is constrained, prefer EfficientNet-lite variants.

3.4 Loss Function Design

Purpose: Improve boundary precision and handle class imbalance common in food segmentation.

Loss components and rationale:
- Dice / Generalized Dice: Overlap-based loss that compensates for class imbalance.
- Boundary loss: Penalizes mistakes near class boundaries (computed via morphological gradients or distance transforms).
- Focal term: Focuses gradients on hard examples (misclassified pixels) by down-weighting easy negatives.
- Optional consistency/contour terms: Encourage coherent predictions under geometric augmentations.

Combined loss used in reported runs:
L_combined = α L_dice + β L_boundary + γ L_focal (+ δ L_consistency)

Practical hyperparameters:
- Typical weights used: α=0.6, β=0.4, γ=0.0 for boundary-focused experiments; add γ>0 for heavy hard-example regimes.
- Implement boundary loss efficiently using a distance transform computed once per mask in preprocessing when possible.

3.5 Training Strategy

Purpose: Stabilize and accelerate convergence while preserving generalization.

Key strategies:
- Progressive multiresolution schedule: start training at 256×256, then finetune at 320×320 and 384×384 to improve boundary delineation.
- Optimizer: AdamW with weight decay; learning rate scheduler: cosine annealing or cosine annealing with warm restarts for long runs.
- Mixed-precision training (AMP): use PyTorch's native AMP for speed and memory savings; guard for nondeterministic ops when exact reproducibility is required.
- Early stopping and checkpointing: monitor validation mIoU; checkpoint best models and save run JSON with hyperparameters.

Implementation details and hyperparameters (reported runs):
- Batch size: 8 (adjust per GPU memory)
- Initial learning rate: 1e-4 (encoder lr reduced as noted)
- Weight decay: 1e-4
- Gradient clipping: max-norm 1.0
- Epochs: up to 100 with early stopping (patience=10)

3.6 Dataset and Preprocessing Strategy

Purpose: Prepare FRANI images for robust training while preserving label integrity.

Dataset summary:
- FRANI: 1,141 labeled images (939 train / 202 val); 6 semantic classes.

Preprocessing pipeline:
- Read images and masks, validate filename correspondences, and verify mask pixel ranges are within [0, C-1].
- Normalize images to ImageNet mean/std.
- Apply online augmentation (Albumentations): random crop, flip, rotate, color jitter, elastic transforms, perspective, Gaussian blur/noise, brightness/contrast adjustments.
- For boundary-aware loss: precompute boundary maps or distance transforms for masks during dataset preparation to avoid repeated CPU work in the training loop.

Notes on class balancing:
- Use inverse-frequency weighting in Generalized Dice or oversample rare-class images during training epochs.



3.7 Implementation and Reproducibility

This section provides concrete, actionable details for reproducing the experiments and for locating the code that implements the methodology described above. The intention is that a reader with access to the FRANI dataset and a CUDA-enabled GPU can run the reported experiments end-to-end.

3.7.1 Code organization and key files

- Model implementations: `models/ghanasegnet.py`, `models/unet.py`, `models/deeplabv3plus.py`, `models/segformer_b0.py`.
- Training and evaluation scripts: `scripts/train_baselines.py`, `scripts/evaluate.py`, `scripts/test.py`.
- Data & augmentations: `infrastructure/data/augmentation_pipeline.py`, `infrastructure/data/datasets/ghana_food_dataset.py`.
- Losses & metrics: `utils/losses.py`, `utils/metrics.py`.
- Notebooks: `Enhanced_GhanaSegNet_Training.ipynb`, `notebooks/results_visualization.ipynb` (exploratory analysis and plots).

3.7.2 Environment, containers and dependency management

- Use the repository `requirements.txt` to create a reproducible Python environment. For example (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- For robust runs, containerized execution is recommended; the repository includes a reference Dockerfile in the docs (see `Chapter_4_Implementation.md` for a multi-stage example). Use a PyTorch image that matches your CUDA driver version.
- Recommended hardware: a modern CUDA-enabled GPU (T4, V100, RTX family). When using mixed precision (AMP) be aware that some ops are nondeterministic; enable `--benchmark-mode` and set explicit seeds for as much determinism as possible.

3.7.3 Reproducible run example and expected artifacts

Run a short smoke training (PowerShell example):

```powershell
# Quick smoke run (10 epochs) — outputs saved to checkpoints/ghanasegnet/
python scripts\train_baselines.py --model ghanasegnet --epochs 10 --batch-size 8 --checkpoint-dir checkpoints/ghanasegnet --seed 789
```

What to expect after the run:
- `checkpoints/ghanasegnet/run_<timestamp>_config.json` — contains hyperparameters used for the run.
- `checkpoints/ghanasegnet/best_model.pth` — best checkpoint by validation mIoU.
- `checkpoints/ghanasegnet/metrics.csv` or `run_summary.json` — per-epoch metrics (train/val loss, mIoU, pixel accuracy).
- Example log fields in run JSON: `{ "seed": 789, "model": "ghanasegnet", "epochs": 10, "best_val_miou": 0.2437, "best_epoch": 1 }`.

3.7.4 Tests and quick verifications

Before long runs, verify environment and imports:

```powershell
python -m pytest tests/test_import.py -q
python -m pytest tests/test_training_env.py::test_small_epoch -q
python scripts/evaluate.py --model ghanasegnet --checkpoint checkpoints/ghanasegnet/best_model.pth --subset val
```

The repository includes simple tests (`test_import.py`, `test_training_env.py`, `test_evaluation.py`) that exercise basic importability and a mini training loop; use them as smoke checks in CI or locally.

3.8 Ethical considerations and cultural validation

Ethical considerations are central to GhanaSegNet's design and evaluation. This subsection records the assumptions, limitations, and governance choices made during dataset collection and algorithm development.

Data governance and consent:
- Use only images collected with appropriate consent or under licenses permitting research use. Exclude identifiable personal data and faces when consent cannot be documented.
- Maintain provenance metadata for each image (source, collection date, annotator, consent flag) in dataset manifests.

Bias mitigation and cultural sensitivity:
- Evaluate per-class and per-presentation-style performance (see Chapter 5) to detect systematic underperformance on particular food types or presentation styles.
- Include domain experts (nutritionists, local food experts) in labeling and validation loops to reduce annotation drift.
- Use augmentation and regional pretraining to reduce distribution shift, but avoid synthetic examples that distort cultural context or misrepresent traditional presentation.

Deployment cautions:
- Before clinical or nutritional deployment, perform local user studies and obtain ethical approval where required.
- Provide clear failure modes and human-in-the-loop recommendations: when confidence is low, flag images for manual review rather than returning an automated nutritional assessment.

3.9 Chapter summary

This chapter formalized the GhanaSegNet methodology. Key takeaways:
- A staged transfer learning pipeline was used to progressively adapt general-purpose features to the Ghanaian food domain.
- A hybrid CNN–Transformer architecture combined multi-scale local features with lightweight global attention to improve boundary delineation.
- A food-aware combined loss and progressive training schedule stabilized learning on the relatively small FRANI dataset.
- Practical reproducibility details (commands, expected artifacts, tests) and ethical considerations were provided to support transparent research and responsible deployment.

The following chapters describe the concrete implementation (Chapter 4), experiments and quantitative results (Chapter 5), and final conclusions and future work (Chapter 6).

