import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GhanaFoodDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        """
        root_dir: path to split (e.g., data/train, data/val, data/test)
        Assumes images are in 'images/' and masks are in 'masks/' within root_dir.
        """
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.jpg', '.png'))
        ])
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask with class IDs

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
