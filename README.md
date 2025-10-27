# GhanaSegNet: Advanced Deep Learning for Ghanaian Food Segmentation
# GhanaSegNet

Purpose
- Research code and artifacts for semantic segmentation of Ghanaian food images.

Repository facts
- Models: implementations in `models/` (ghanasegnet, unet, deeplabv3plus, segformer).
- Results: training histories and metrics in `results/` (e.g. `results/ghanasegnet_results.json`).
- Figures: publication-ready images are in `figures/`.
- Scripts: training and evaluation helpers in `scripts/`.
- Notebooks: analysis and Colab notebooks in `notebooks/` and root `.ipynb` files.
- Chapters: thesis chapters as `Chapter_*.md` at repo root.

Quick commands (facts only)
- Clone: `git clone https://github.com/EricBaidoo/GhanaSegNet.git`
- Install: `pip install -r requirements.txt`
- Smoke test (checks model import; requires PyTorch): `python scripts/smoke_test.py`
- Train baseline(s): `python scripts/train_baselines.py --model ghanasegnet --epochs 15`
- Evaluate a checkpoint: `python scripts/evaluate.py --model ghanasegnet --checkpoint <path>`

Notes
- `results/ghanasegnet_results.json` contains the recorded training history and parameter counts.
- A restoration of model and training files was performed from commit `6b921a9` (see `results/ghanasegnet_results.json` -> `restoration`).
- This README is concise and factual; use notebooks for detailed analysis and reproduction steps.

License
- MIT (see `LICENSE`).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EricBaidoo/GhanaSegNet/blob/main/GhanaSegNet_Colab.ipynb)



