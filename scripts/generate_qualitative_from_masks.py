"""Create qualitative grids from images and ground-truth masks only (no model required).
Usage: python scripts\generate_qualitative_from_masks.py --images <images_dir> --masks <masks_dir> --out figures\qual_grid_masks.png --n 5
"""
from pathlib import Path
from PIL import Image, ImageOps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images', required=True)
parser.add_argument('--masks', required=True)
parser.add_argument('--out', default='figures/qual_grid_masks.png')
parser.add_argument('--n', type=int, default=5)
args = parser.parse_args()

images_dir = Path(args.images)
masks_dir = Path(args.masks)
out = Path(args.out)

imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])[:args.n]
if not imgs:
    print('No input images found in', images_dir)
    raise SystemExit(1)

rows = []
for img_p in imgs:
    stem = img_p.stem
    mask_p = masks_dir / (stem + '.png')
    img = Image.open(img_p).convert('RGB')
    if mask_p.exists():
        mask = Image.open(mask_p).convert('L')
        # colorize mask for visibility
        mask_col = ImageOps.colorize(mask, black="black", white="red")
    else:
        mask_col = Image.new('RGB', img.size, (240,240,240))
    # Resize to same width
    w = 256
    img = ImageOps.fit(img, (w, w))
    mask_col = ImageOps.fit(mask_col, (w, w))
    rows.append([img, mask_col])

# Make grid: columns = input, gt
cols = 2
cell_w, cell_h = rows[0][0].size
grid = Image.new('RGB', (cell_w*cols, cell_h*len(rows)), (255,255,255))
for r, pair in enumerate(rows):
    for c, im in enumerate(pair):
        grid.paste(im, (c*cell_w, r*cell_h))

out.parent.mkdir(parents=True, exist_ok=True)
grid.save(out)
print('Saved qualitative grid to', out)
