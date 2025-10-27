
**GHANASEGNET: ADVANCED DEEP LEARNING FOR GHANAIAN FOOD SEGMENTATION**

Food recognition and automated nutritional assessment remain complex challenges in sub-Saharan Africa, particularly in Ghana, where diverse traditional meals pose unique computational difficulties for artificial intelligence systems. The limited representation of African cuisines in existing food image datasets has created a significant gap in global food computing research. Current models are largely trained on Western-centric datasets such as Food101 and Recipe1M, resulting in algorithmic bias and poor performance when applied to culturally specific dishes. This thesis addresses this challenge through the development of GhanaSegNet, a hybrid convolutional–transformer-based semantic segmentation model specifically designed for Ghanaian food imagery. The study introduces a multi-stage transfer learning framework that progressively adapts pretrained models from general visual domains to culturally relevant food contexts. The research employs the FRANI dataset, comprising 1,141 annotated images of common Ghanaian dishes categorized into six semantic classes. A comprehensive data preprocessing and augmentation pipeline was developed to enhance model robustness, simulate real-world presentation styles, and mitigate class imbalance.

The GhanaSegNet architecture integrates the efficiency of convolutional encoders with the contextual reasoning power of transformer bottlenecks, enabling the model to capture both local and global visual dependencies. A composite loss function combining Dice and Boundary losses was employed to address class imbalance and improve segmentation precision, particularly along object boundaries. The model was trained using a structured, multi-resolution training protocol and evaluated on a held-out validation split using mean Intersection over Union (mIoU) as the primary performance metric. Experimental results demonstrated that GhanaSegNet achieved an average validation mIoU of 24.47%, performing competitively with the more complex DeepLabV3+ model (25.44%) while maintaining a significantly smaller parameter count. The model also exhibited superior boundary delineation and stable convergence across epochs, highlighting the effectiveness of its hybrid design and composite loss formulation.

The findings of this study provide evidence that culturally tailored computer vision models can achieve high segmentation accuracy and efficiency even within limited-resource settings. The multi-stage transfer learning approach and boundary-aware training framework demonstrate a practical pathway for developing inclusive and contextually relevant AI models in the African setting. The research contributes to addressing algorithmic bias in food computing and establishes a foundation for future work on mobile deployment and real-world nutritional assessment. Overall, GhanaSegNet presents a significant step toward the integration of culturally responsive artificial intelligence systems for health and nutrition applications in Ghana and beyond.**
**Keywords: semantic segmentation, GhanaSegNet, food recognition, transfer learning, artificial intelligence, computer vision**

**Quick start (short)**
**- Prerequisites**
**  - Python 3.8–3.10, GPU with CUDA (optional but recommended).**
**  - From project root: python -m venv .venv && .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS).**
**  - pip install -r requirements.txt**

**- Prepare data**
**  - Place FRANI images and masks under data/FRANI/**
**    - data/FRANI/images/  (RGB images)**
**    - data/FRANI/masks/   (indexed segmentation masks, PNG)**
**  - Ensure a dataset config exists (example: data/frani.yaml) with paths and class list.**

**- Train (single-GPU)**
**  - python train.py --config configs/ghanasnet.yaml --data data/frani.yaml --epochs 50 --batch-size 8 --weights checkpoints/pretrained.pth**
**  - For quick smoke test: --epochs 1 --batch-size 2**

**- Train (multi-GPU)**
**  - torchrun --nproc_per_node=4 train.py --config configs/ghanasnet.yaml --data data/frani.yaml --epochs 50**

**- Evaluate**
**  - python evaluate.py --weights checkpoints/best.pth --data data/frani.yaml --split val**

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
