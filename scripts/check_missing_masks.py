import os


images_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', 'images')
masks_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', 'masks')

images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
masks_set = set([f for f in os.listdir(masks_dir) if f.endswith('.png') or f.endswith('.jpg')])

def get_mask_name(image_name):
    base, ext = os.path.splitext(image_name)
    return f"{base}_mask.png"

missing_masks = []
for img in images:
    mask_name = get_mask_name(img)
    if mask_name not in masks_set:
        missing_masks.append(img)

if missing_masks:
    print('Images missing masks:')
    for img in missing_masks:
        print(img)
    print(f'Total missing: {len(missing_masks)}')
else:
    print('All images have corresponding masks.')
