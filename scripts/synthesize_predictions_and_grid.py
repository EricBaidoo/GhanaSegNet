"""Synthesize simple predictions from masks/images and create a qualitative grid (input / ground-truth / synthetic-prediction).
Approach:
- If ground-truth masks exist, create synthetic predictions by eroding/dilating the mask or shifting classes.
- Otherwise, produce a color-quantized segmentation of the input image as a fake prediction.

Usage examples:
python scripts\synthesize_predictions_and_grid.py --images data/val/images --masks data/val/masks --out figures/qual_grid_synth.png --n 5
python scripts\synthesize_predictions_and_grid.py --images data/val/images --out figures/qual_grid_synth.png --n 5
"""
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True)
parser.add_argument('--masks', required=False)
parser.add_argument('--out', default='figures/qual_grid_synth.png')
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()

images_dir = Path(args.images)
mask_dir = Path(args.masks) if args.masks else None
out = Path(args.out)

imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])[:args.n]
if not imgs:
    print('No input images found in', images_dir)
    raise SystemExit(1)

rows = []
for img_p in imgs:
    stem = img_p.stem
    img = Image.open(img_p).convert('RGB')
    w = 256
    img_t = ImageOps.fit(img, (w, w))
    # ground truth
    if mask_dir:
        mask_p = mask_dir / (stem + '.png')
    else:
        mask_p = None
    if mask_p and mask_p.exists():
        gt = Image.open(mask_p).convert('L')
        gt_col = ImageOps.colorize(gt, black='black', white='green')
        # synth prediction: apply a small blur and shift
        pred = gt.filter(ImageFilter.GaussianBlur(radius=2))
        pred = pred.offset(2,2) if hasattr(pred, 'offset') else pred
        pred_col = ImageOps.colorize(pred.convert('L'), black='black', white='orange')
    else:
        gt_col = Image.new('RGB', (w,w), (240,240,240))
        # synth prediction: color-quantize the image to N colors
        arr = np.array(img_t)
        # simple kmeans-like quantization: reduce to 4 colors using reshape
        flat = arr.reshape(-1,3).astype(np.float32)
        # pick 4 centroids by sampling
        idx = np.random.choice(len(flat), size=4, replace=False)
        centroids = flat[idx]
        # assign pixels to nearest centroid
        dists = np.linalg.norm(flat[:,None,:]-centroids[None,:,:], axis=2)
        labels = dists.argmin(axis=1)
        quant = centroids[labels].reshape(arr.shape).astype(np.uint8)
        pred_col = Image.fromarray(quant)
    rows.append([img_t, gt_col, pred_col])

cols = 3
cell_w, cell_h = rows[0][0].size
grid = Image.new('RGB', (cell_w*cols, cell_h*len(rows)), (255,255,255))
for r, trip in enumerate(rows):
    for c, im in enumerate(trip):
        grid.paste(im, (c*cell_w, r*cell_h))

out.parent.mkdir(parents=True, exist_ok=True)
grid.save(out)
print('Saved synthetic qualitative grid to', out)
