
# GhanaSegNet

Master-level README — concise, factual, and actionable.

TL;DR
- GhanaSegNet is a research codebase for semantic segmentation of Ghanaian food imagery. The repository contains model implementations (GhanaSegNet, Enhanced GhanaSegNet) and standard baselines (U-Net, DeepLabV3+, SegFormer), training/evaluation scripts, analysis notebooks, and utilities to reproduce results and generate thesis-ready figures and tables.

Highlights / key results
- Best validation mIoU (per available results):
	- deeplabv3plus: 0.25440692532084563
	- ghanasegnet: 0.24474198825134152
	- segformer: 0.24369344503983206
	- unet: 0.24369284914571407
- GhanaSegNet parameter count: 6,754,261 total (trainable)
- All per-model metrics and training histories are stored as JSON under `results/`.

Repository status and provenance
- The repository contains a restoration provenance entry recording a selective restore of model and training code from an Oct 11 commit to reproduce the baseline setup. See `results/ghanasegnet_results.json` -> `restoration` block for the source commit id and timestamp.

Quick start (reproduce locally)
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install runtime dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Run a smoke test (quick model import + parameter count):

```powershell
python .\scripts\smoke_test.py
```

4. Train a baseline model (example):

```powershell
python .\scripts\train_baselines.py --model ghanasegnet --epochs 15 --batch-size 8
```

5. Evaluate a trained checkpoint:

```powershell
python .\scripts\evaluate.py --model ghanasegnet --checkpoint path\to\checkpoint.pth
```

6. Generate thesis figures and tables (uses JSON results under `results/` or any directory you point to):

```powershell
python .\scripts\generate_all_analysis_figures.py --results-dir results --out-dir figures --tables-dir tables
```

Repository layout (important files)
- `models/` — model implementations: `ghanasegnet.py`, `unet.py`, `deeplabv3plus.py`, `segformer.py`.
- `scripts/` — training, evaluation, and utilities (e.g., `train_baselines.py`, `evaluate.py`, `smoke_test.py`, `generate_all_analysis_figures.py`).
- `results/` — per-model JSON result files and `val_iou_stats_summary.json` (used by analysis scripts).
- `notebooks/` — Colab/analysis notebooks for reproducing the paper figures and interactive exploration.
- `figures/`, `tables/` — generated outputs (plots, CSV/LaTeX) used in the thesis.
- `utils/` — loss functions and metric helpers.

Design & reproducibility notes
- Experiments are reproducible provided the same dataset split and checkpoints: results are serialized as JSON to capture training history, best metrics, and configuration (learning rate, weight decay, epochs, batch size, num_classes).
- The generator script (`scripts/generate_all_analysis_figures.py`) accepts a `--results-dir` argument so you can point it at local copies of result JSONs (useful if you mount Google Drive in Colab and copy the JSONs locally).

Visualization and figure guidance
- Figures are generated to be thesis-ready: labeled axes, numeric annotations, consistent palettes, and layout tuned for print. Use the generator's `--out-dir` and `--tables-dir` to control output locations. For higher-resolution exports, open the script and increase the DPI inside `save_svg_and_png()` or use the matplotlib `savefig(..., dpi=600)` option.

Repro tips
- If you need to reproduce exactly the Oct 11 baseline, consult the `restoration` block added to `results/ghanasegnet_results.json` (it records the original commit id used during restore).
- Use the provided smoke test before long runs to validate the environment (`python .\scripts\smoke_test.py`).

Development & contribution
- Follow the code style used in the repository. Proposed changes should come as topic branches and pull requests. Include unit tests for new utilities and small notebooks demonstrating new analysis.

Citation
- If you use this code in research, please cite the repository and any associated paper or thesis chapter where applicable.

License
- MIT — see `LICENSE` for full terms.

Contact
- Repo owner: EricBaidoo (see repository settings). For reproducibility questions open an issue with a short reproduction recipe and the exact command you ran.

Notes
- Results and large checkpoints are hosted on Google Drive and are referenced by the Colab notebooks; they are not all committed to this repository. Use the notebooks/Colab workflow to access Drive-mounted artifacts when needed.

-----
This README is intentionally concise and factual to support reproducible research and thesis preparation. If you want a condensed one-page abstract or a longer developer guide, tell me which and I will add it as `docs/DEVELOPER_GUIDE.md`.



