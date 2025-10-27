MIT License — see `LICENSE`

# GhanaSegNet

One-line: GhanaSegNet — a master's-level research codebase for semantic segmentation of Ghanaian food imagery.

## ABSTRACT

Food recognition and automated nutritional assessment remain complex challenges in sub-Saharan Africa, particularly in Ghana, where diverse traditional meals pose unique computational difficulties for artificial intelligence systems. The limited representation of African cuisines in existing food image datasets has created a significant gap in global food computing research. Current models are largely trained on Western-centric datasets such as Food101 and Recipe1M, resulting in algorithmic bias and poor performance when applied to culturally specific dishes. This thesis addresses this challenge through the development of GhanaSegNet, a hybrid convolutional–transformer-based semantic segmentation model specifically designed for Ghanaian food imagery. The study introduces a multi-stage transfer learning framework that progressively adapts pretrained models from general visual domains to culturally relevant food contexts. The research employs the FRANI dataset, comprising 1,141 annotated images of common Ghanaian dishes categorized into six semantic classes. A comprehensive data preprocessing and augmentation pipeline was developed to enhance model robustness, simulate real-world presentation styles, and mitigate class imbalance.

The GhanaSegNet architecture integrates the efficiency of convolutional encoders with the contextual reasoning power of transformer bottlenecks, enabling the model to capture both local and global visual dependencies. A composite loss function combining Dice and Boundary losses was employed to address class imbalance and improve segmentation precision, particularly along object boundaries. The model was trained using a structured, multi-resolution training protocol and evaluated on a held-out validation split using mean Intersection over Union (mIoU) as the primary performance metric. Experimental results demonstrated that GhanaSegNet achieved an average validation mIoU of 24.47%, performing competitively with the more complex DeepLabV3+ model (25.44%) while maintaining a significantly smaller parameter count. The model also exhibited superior boundary delineation and stable convergence across epochs, highlighting the effectiveness of its hybrid design and composite loss formulation.

The findings of this study provide evidence that culturally tailored computer vision models can achieve high segmentation accuracy and efficiency even within limited-resource settings. The multi-stage transfer learning approach and boundary-aware training framework demonstrate a practical pathway for developing inclusive and contextually relevant AI models in the African setting. The research contributes to addressing algorithmic bias in food computing and establishes a foundation for future work on mobile deployment and real-world nutritional assessment. Overall, GhanaSegNet presents a significant step toward the integration of culturally responsive artificial intelligence systems for health and nutrition applications in Ghana and beyond.

**Keywords:** semantic segmentation, GhanaSegNet, food recognition, transfer learning, artificial intelligence, computer vision

---

## Quick start (short)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Generate figures and tables from the available results (example):

```powershell
python .\scripts\generate_all_analysis_figures.py --results-dir results --out-dir figures --tables-dir tables
```

---

## Quick commands (inference / checkpoints / logs)

- Inference (single image):

```powershell
python predict.py --weights checkpoints\best.pth --input path\to\image.jpg --output out.png --overlay
```

- Inference (folder):

```powershell
python predict.py --weights checkpoints\best.pth --input data\FRANI\images\ --output outputs\predictions\
```

- Checkpoints & logs:

  - Checkpoints are saved to `checkpoints/` by default (use `--save-dir` to change).
  - Visualize training with TensorBoard: `tensorboard --logdir runs/`.

---

## Highlights / key results (from `results/`)

- deeplabv3plus — best validation mIoU = 0.25440692532084563
- ghanasegnet — best validation mIoU = 0.24474198825134152
- segformer — best validation mIoU = 0.24369344503983206
- unet — best validation mIoU = 0.24369284914571407
- GhanaSegNet parameter count: 6,754,261 total (trainable)

---

## Repository layout (important files)

- `models/` — model implementations (e.g., `ghanasegnet.py`, `unet.py`, `deeplabv3plus.py`, `segformer.py`).
- `scripts/` — training, evaluation, and utilities (e.g., `train_baselines.py`, `evaluate.py`, `smoke_test.py`, `generate_all_analysis_figures.py`).
- `results/` — per-model JSON result files and `val_iou_stats_summary.json` (used by analysis scripts).
- `notebooks/` — Colab/analysis notebooks for reproducing the paper figures and interactive exploration.
- `figures/`, `tables/` — generated outputs (plots, CSV/LaTeX) used in the thesis.
- `utils/` — loss functions and metric helpers.

---

## Design & reproducibility notes

- Experiments are reproducible provided the same dataset split and checkpoints. Results are serialized as JSON to capture training history, best metrics, and configuration (learning rate, weight decay, epochs, batch size, num_classes).
- The generator script (`scripts/generate_all_analysis_figures.py`) accepts a `--results-dir` argument so you can point it at local copies of result JSONs (useful if you mount Google Drive in Colab and copy the JSONs locally).
- A restoration provenance entry exists in `results/ghanasegnet_results.json` recording the Oct 11 selective restore commit id and timestamp for the baseline.

---

## Visualization guidance

- Figures are generated to be thesis-ready: labeled axes, numeric annotations, consistent palettes, and layout tuned for print. Use `--out-dir` and `--tables-dir` to control output locations. Increase DPI in `save_svg_and_png()` or use `savefig(..., dpi=600)` for high-resolution exports.

---

## Repro tips

- Run the smoke test before long runs to validate the environment:

```powershell
python .\scripts\smoke_test.py
```

- If you need the Oct 11 baseline exactly, consult the `restoration` block in `results/ghanasegnet_results.json` (it records the original commit used during the selective restore).

---

## Development & contribution

- Follow the repository coding style. Open a topic branch and a PR for proposed changes. Include unit tests for utilities and short notebooks for new analyses.

---

## Citation

If you use this work in research, please cite the repository and the associated thesis chapter.

---

## Contact

Repo owner: EricBaidoo — open an issue for reproducibility questions and include your reproduction recipe and exact command.

---

## Notes

- Results and large checkpoints are hosted on Google Drive and referenced by the Colab notebooks; they are not committed to this repository. Use the Colab notebooks to mount Drive and access artifacts.

---

For a full developer guide (environment, training, evaluation, tests and CI) request `docs/DEVELOPER_GUIDE.md` or I can generate it and add it to the repo.

**GHANASEGNET: ADVANCED DEEP LEARNING FOR GHANAIAN FOOD SEGMENTATION**
MIT License — see `LICENSE`

# GhanaSegNet

ABSTRACT

Food recognition and automated nutritional assessment remain complex challenges in sub-Saharan Africa, particularly in Ghana, where diverse traditional meals pose unique computational difficulties for artificial intelligence systems. The limited representation of African cuisines in existing food image datasets has created a significant gap in global food computing research. Current models are largely trained on Western-centric datasets such as Food101 and Recipe1M, resulting in algorithmic bias and poor performance when applied to culturally specific dishes. This thesis addresses this challenge through the development of GhanaSegNet, a hybrid convolutional–transformer-based semantic segmentation model specifically designed for Ghanaian food imagery. The study introduces a multi-stage transfer learning framework that progressively adapts pretrained models from general visual domains to culturally relevant food contexts. The research employs the FRANI dataset, comprising 1,141 annotated images of common Ghanaian dishes categorized into six semantic classes. A comprehensive data preprocessing and augmentation pipeline was developed to enhance model robustness, simulate real-world presentation styles, and mitigate class imbalance.

The GhanaSegNet architecture integrates the efficiency of convolutional encoders with the contextual reasoning power of transformer bottlenecks, enabling the model to capture both local and global visual dependencies. A composite loss function combining Dice and Boundary losses was employed to address class imbalance and improve segmentation precision, particularly along object boundaries. The model was trained using a structured, multi-resolution training protocol and evaluated on a held-out validation split using mean Intersection over Union (mIoU) as the primary performance metric. Experimental results demonstrated that GhanaSegNet achieved an average validation mIoU of 24.47%, performing competitively with the more complex DeepLabV3+ model (25.44%) while maintaining a significantly smaller parameter count. The model also exhibited superior boundary delineation and stable convergence across epochs, highlighting the effectiveness of its hybrid design and composite loss formulation.

The findings of this study provide evidence that culturally tailored computer vision models can achieve high segmentation accuracy and efficiency even within limited-resource settings. The multi-stage transfer learning approach and boundary-aware training framework demonstrate a practical pathway for developing inclusive and contextually relevant AI models in the African setting. The research contributes to addressing algorithmic bias in food computing and establishes a foundation for future work on mobile deployment and real-world nutritional assessment. Overall, GhanaSegNet presents a significant step toward the integration of culturally responsive artificial intelligence systems for health and nutrition applications in Ghana and beyond.
Keywords: semantic segmentation, GhanaSegNet, food recognition, transfer learning, artificial intelligence, computer vision

Quick start (short)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Generate figures and tables from the available results (example):

```powershell
python .\scripts\generate_all_analysis_figures.py --results-dir results --out-dir figures --tables-dir tables
```
**- Inference / Predict**
**  - python predict.py --weights checkpoints/best.pth --input path/to/image.jpg --output out.png --overlay**
**  - For folder inference: --input data/FRANI/images/ --output outputs/predictions/**

**- Checkpoints & logs**
**  - Checkpoints saved to checkpoints/ by default; specify --save-dir to change.**
**  - Visualize training with TensorBoard: tensorboard --logdir runs/**

**- Tips**
**  - Set CUDA devices: export CUDA_VISIBLE_DEVICES=0 (Linux/macOS) or set CUDA_VISIBLE_DEVICES=0 (Windows PowerShell).**
**  - Reduce batch size if you hit OOM.**
**  - Specify a custom weights path with --weights wherever required.**

**Use these commands from the repository root. Adjust config flags to match your hardware and dataset paths.**




**License**

MIT License

Copyright (c) 2025 GhanaSegNet contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
