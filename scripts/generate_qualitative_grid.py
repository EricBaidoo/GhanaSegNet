#!/usr/bin/env python3
"""Generate a qualitative grid (input / ground-truth / prediction)

Usage (example):
  python scripts/generate_qualitative_grid.py \
      --checkpoint checkpoints/ghanasegnet/best_model.pth \
      --dataset-root data \
      --model ghanasegnet \
      --n 3

This script expects the dataset layout:
  <dataset-root>/val/images/*.jpg
  <dataset-root>/val/masks/*.png

It writes prediction images to a temporary folder and creates
figures/figure4_2_qualitative_grid.png using the existing
analysis/create_figures_pipeline.py helper.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
except Exception as e:
    print('Required packages missing. Please install torch, torchvision, pillow. Error:', e)
    sys.exit(1)


MODEL_MAP = {
    'ghanasegnet': ('models.ghanasegnet', 'GhanaSegNet'),
    'deeplabv3plus': ('models.deeplabv3plus', 'DeepLabV3Plus'),
    'unet': ('models.unet', 'UNet'),
    'segformer': ('models.segformer', 'SegFormer'),
}


def import_model_class(model_key):
    if model_key not in MODEL_MAP:
        raise ValueError(f'Unknown model: {model_key}. Supported: {list(MODEL_MAP.keys())}')
    mod_name, cls_name = MODEL_MAP[model_key]
    module = __import__(mod_name, fromlist=[cls_name])
    return getattr(module, cls_name)


def preprocess_image(img_path, img_size=None):
    img = Image.open(img_path).convert('RGB')
    if img_size:
        img = img.resize(img_size, resample=Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0), img.size  # return tensor, (W,H)


def save_mask(pred_tensor, out_path):
    # pred_tensor is [H,W] or numpy array of ints
    if not isinstance(pred_tensor, Image.Image):
        im = Image.fromarray(pred_tensor.astype('uint8'))
    else:
        im = pred_tensor
    im.save(out_path)


def run_inference_and_save_preds(model, device, image_paths, pred_out_dir, img_size=None):
    os.makedirs(pred_out_dir, exist_ok=True)
    pred_paths = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for p in image_paths:
            x, orig_size = preprocess_image(p, img_size)
            x = x.to(device)
            out = model(x)
            # model may return logits or (logits, aux)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            # out expected [B, C, H, W]
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).squeeze(0).cpu().numpy().astype('uint8')
            fname = Path(p).stem + '_pred.png'
            out_path = os.path.join(pred_out_dir, fname)
            save_mask(pred, out_path)
            pred_paths.append(out_path)
    return pred_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--dataset-root', required=True, help='Path to dataset root (expects val/images and val/masks)')
    parser.add_argument('--model', default='ghanasegnet', help='Model key: ghanasegnet, deeplabv3plus, unet, segformer')
    parser.add_argument('--n', type=int, default=3, help='Number of sample images to include')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-figure', default='figures/figure4_2_qualitative_grid.png')
    parser.add_argument('--img-size', type=int, nargs=2, help='Optional resize W H for model input/output')
    args = parser.parse_args()

    ds_root = Path(args.dataset_root)
    images_dir = ds_root / 'val' / 'images'
    masks_dir = ds_root / 'val' / 'masks'
    if not images_dir.exists() or not masks_dir.exists():
        print('Could not find val/images or val/masks under dataset root:', ds_root)
        sys.exit(1)

    # Find image files
    imgs = sorted([str(p) for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.png','.jpeg')])
    if not imgs:
        print('No images found in', images_dir)
        sys.exit(1)
    sample_imgs = imgs[:args.n]

    # Corresponding masks
    mask_paths = []
    for p in sample_imgs:
        stem = Path(p).stem
        # try png mask
        cand = masks_dir / (stem + '.png')
        if not cand.exists():
            cand = masks_dir / (stem + '.jpg')
        if not cand.exists():
            print('Warning: mask not found for', p)
            mask_paths.append('')
        else:
            mask_paths.append(str(cand))

    # Import model class and instantiate
    ModelClass = import_model_class(args.model)
    # try to instantiate - assume constructor accepts num_classes
    try:
        model = ModelClass(num_classes=6)
    except Exception:
        # fallback: no args
        model = ModelClass()

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print('Checkpoint not found:', ckpt_path)
        sys.exit(1)
    try:
        state = torch.load(str(ckpt_path), map_location=args.device)
        # support both state_dict and full checkpoint
        if 'state_dict' in state:
            sd = state['state_dict']
        else:
            sd = state
        # Some checkpoints use module prefix
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            # try removing 'module.' prefixes
            new_sd = {k.replace('module.',''):v for k,v in sd.items()}
            model.load_state_dict(new_sd)
    except Exception as e:
        print('Error loading checkpoint:', e)
        sys.exit(1)

    # Run inference and save prediction masks
    pred_out_dir = Path('figures') / 'temp_preds'
    if pred_out_dir.exists():
        shutil.rmtree(pred_out_dir)
    pred_out_dir.mkdir(parents=True, exist_ok=True)

    img_size = tuple(args.img_size) if args.img_size else None
    pred_paths = run_inference_and_save_preds(model, args.device, sample_imgs, str(pred_out_dir), img_size)

    # Create figures directory
    os.makedirs('figures', exist_ok=True)

    # Call the existing create_qualitative_grid helper
    try:
        sys.path.insert(0, os.getcwd())
        from analysis.create_figures_pipeline import create_qualitative_grid
        # convert lists
        image_paths = sample_imgs
        mask_paths_list = mask_paths
        pred_paths_list = pred_paths
        # fill missing masks with a blank placeholder if necessary
        for i, mp in enumerate(mask_paths_list):
            if not mp:
                # create a blank mask image matching first image size
                blank = Image.new('L', Image.open(image_paths[i]).size, color=0)
                blank_path = pred_out_dir / (Path(image_paths[i]).stem + '_mask_placeholder.png')
                blank.save(str(blank_path))
                mask_paths_list[i] = str(blank_path)

        # create the grid and save to the requested path
        create_qualitative_grid(image_paths, mask_paths_list, pred_paths_list)
        # rename output file to the desired out-figure if different
        default_out = Path('figures') / 'figure3_4_qualitative_grid.png'
        desired_out = Path(args.out_figure)
        if default_out.exists():
            default_out.replace(desired_out)
        print('Qualitative grid saved to', desired_out)

    except Exception as e:
        print('Failed to create qualitative grid using create_figures_pipeline:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
