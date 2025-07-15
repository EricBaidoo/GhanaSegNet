# Dataset loader for FRANI dataset

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GhanaFoodDataset(Dataset):
    def __init__(self, root='data/frani/', split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.image_dir = os.path.join(root, split, 'images')
        self.mask_dir = os.path.join(root, split, 'masks')

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale for labels

        if self.transform:
            image = self.transform(image)

        mask = np.array(mask, dtype=np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
