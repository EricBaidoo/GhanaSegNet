# GhanaSegNet: Advanced Deep Learning for Ghanaian Food Segmentation
# GhanaSegNet

Concise project summary

- Purpose: code and artifacts developed for semantic segmentation of Ghanaian food images.
- Implementations: GhanaSegNet (baseline), Enhanced GhanaSegNet, and baseline models (U-Net, DeepLabV3+, SegFormer) in `models/`.
- Analysis: notebooks and scripts for evaluation and figure generation are provided in `notebooks/` and `scripts/`.
- Thesis: written chapters are at the repository root as `Chapter_*.md` and figures are in `figures/`.

Key results (recorded)

- GhanaSegNet (this repo): best validation mIoU = 0.244742 (final_epoch = 15).
- Model size: total parameters = 6,754,261 (trainable = 6,754,261).
- Full per-model metrics and training histories are available under `results/` as JSON files.

What was done

- Implemented and trained GhanaSegNet variants and standard baselines.
- Automated evaluation and plotting pipelines in notebooks for thesis figures and quantitative tables.
- Added scripts to generate qualitative grids (mask-only and synthetic prediction alternatives) for drafting without large checkpoints.
- Restored model and training code from an Oct 11 commit to reproduce the baseline setup.

Reproduction pointers (facts only)

- Dependencies: listed in `requirements.txt`.
- Train: `python scripts/train_baselines.py --model ghanasegnet --epochs 15` (training scripts in `scripts/`).
- Evaluate: `python scripts/evaluate.py --model ghanasegnet --checkpoint <path>`.

License

- MIT — see `LICENSE`.

For detailed instructions and analysis, open the notebooks in `notebooks/`.

License
- MIT (see `LICENSE`).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EricBaidoo/GhanaSegNet/blob/main/GhanaSegNet_Colab.ipynb)



