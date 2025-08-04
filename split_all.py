import os
import shutil
import random
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

# Source directories
base_dir = Path("annotated_pete_updated")
images_dir = base_dir / "images"
masks_dir = base_dir / "masks"
color_masks_dir = base_dir / "color_masks"
watershed_masks_dir = base_dir / "watershed_masks"

# Output base
output_dir = Path("data")
output_dir.mkdir(parents=True, exist_ok=True)

# Collect all image stems
image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
# Extract stem names from image files for clarity
image_stems = [f.stem for f in image_files]
# Set a fixed random seed for reproducibility of dataset splits
random.seed(42)
random.shuffle(image_stems)
random.seed(42)
random.shuffle(image_stems)

n = len(image_stems)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

splits = {
    "train": image_stems[:train_end],
    "val": image_stems[train_end:val_end],
    "test": image_stems[val_end:]
}

# Suffixes for file naming conventions
IMAGE_SUFFIX = ".jpg"
GRAYSCALE_MASK_SUFFIX = "_mask.png"
COLOR_MASK_SUFFIX = "_color_mask.png"
WATERSHED_MASK_SUFFIX = "_watershed_mask.png"

# Validate suffixes exist in source directories
def validate_suffix(src_dir, suffix):
    files = list(src_dir.glob(f"*{suffix}"))
    if not files:
        logging.error(f"No files found in {src_dir} with suffix '{suffix}'")
        return False
    return True

# Subfolders to copy
subfolders = {
    "images": (images_dir, IMAGE_SUFFIX),
    "grayscale_masks": (masks_dir, GRAYSCALE_MASK_SUFFIX),
    "color_masks": (color_masks_dir, COLOR_MASK_SUFFIX),
    "watershed_masks": (watershed_masks_dir, WATERSHED_MASK_SUFFIX),
}
for sub_name, (src_dir, suffix) in subfolders.items():
    if not validate_suffix(src_dir, suffix):
        raise ValueError(f"Suffix '{suffix}' for '{sub_name}' is invalid or missing files.")
def copy_file(src_path, dest_path):
    if src_path.exists():
        shutil.copy(src_path, dest_path)
    else:
        logging.warning(f"Missing file: {src_path}")
for split, stems in splits.items():
    for sub_name, (src_dir, suffix) in subfolders.items():
        dest_dir = output_dir / split / sub_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for stem in stems:
            filename = stem + suffix
            src_path = src_dir / filename
            dest_path = dest_dir / filename
            tasks.append((src_path, dest_path))

        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: copy_file(*args), tasks)
            executor.map(lambda args: copy_file(*args), tasks)
