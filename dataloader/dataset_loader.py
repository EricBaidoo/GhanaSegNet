"""
GhanaFoodDataset Loader
Loads Ghanaian food segmentation dataset for training

Expected directory structure:
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/

Author: EricBaidoo
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF


class GhanaFoodDataset(Dataset):
    """
    Dataset class for Ghanaian food segmentation
    
    Args:
        split (str): 'train' or 'val'
        target_size (tuple): Target image size (height, width)
        data_root (str): Root directory containing the dataset
        transform (callable, optional): Additional transformations
        num_classes (int): Number of segmentation classes (default: 6)
    """
    
    def __init__(self, split='train', target_size=(384, 384), data_root='data', 
                 transform=None, num_classes=6):
        """
        Initialize the dataset
        """
        self.split = split
        self.target_size = target_size
        self.data_root = data_root
        self.transform = transform
        self.num_classes = num_classes
        
        # Construct paths
        self.images_dir = os.path.join(data_root, split, 'images')
        self.masks_dir = os.path.join(data_root, split, 'masks')
        
        # Verify directories exist
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                f"Please ensure dataset is located at: {data_root}\n"
                f"Expected structure: {data_root}/{split}/images/"
            )
        
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(
                f"Masks directory not found: {self.masks_dir}\n"
                f"Please ensure dataset is located at: {data_root}\n"
                f"Expected structure: {data_root}/{split}/masks/"
            )
        
        # Get list of image files
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.PNG'))
        ])
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found in {self.images_dir}\n"
                f"Please check that images are present in the directory."
            )
        
        # Define basic transforms
        self.resize_transform = transforms.Resize(target_size, interpolation=Image.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset
        
        Returns:
            image (torch.Tensor): Preprocessed image tensor [3, H, W]
            mask (torch.Tensor): Segmentation mask tensor [H, W]
        """
        # Get image filename
        img_name = self.image_files[idx]
        
        # Construct full paths
        img_path = os.path.join(self.images_dir, img_name)
        
        # Try to find corresponding mask (handle different extensions)
        mask_name = None
        base_name = os.path.splitext(img_name)[0]
        
        for ext in ['.png', '.PNG', '.jpg', '.jpeg', '.JPG', '.JPEG']:
            potential_mask = base_name + ext
            mask_path = os.path.join(self.masks_dir, potential_mask)
            if os.path.exists(mask_path):
                mask_name = potential_mask
                break
        
        if mask_name is None:
            raise FileNotFoundError(
                f"Mask not found for image: {img_name}\n"
                f"Searched in: {self.masks_dir}\n"
                f"Tried extensions: .png, .PNG, .jpg, .jpeg"
            )
        
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image and mask
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Grayscale for masks
        except Exception as e:
            raise IOError(f"Error loading image/mask pair:\n{img_path}\n{mask_path}\n{str(e)}")
        
        # Resize image and mask
        image = self.resize_transform(image)
        mask = TF.resize(mask, self.target_size, interpolation=Image.NEAREST)
        
        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default transforms
            image = self.to_tensor(image)
            image = self.normalize(image)
        
        # Convert mask to tensor
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        
        # Ensure mask values are within valid range [0, num_classes-1]
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        
        return image, mask
    
    def get_class_distribution(self):
        """
        Compute class distribution across the dataset
        Useful for class balancing
        
        Returns:
            dict: Class counts {class_id: count}
        """
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        print(f"Computing class distribution for {self.split} set...")
        for idx in range(len(self)):
            _, mask = self[idx]
            for class_id in range(self.num_classes):
                class_counts[class_id] += (mask == class_id).sum().item()
        
        return class_counts


def verify_dataset(data_root='data'):
    """
    Verify dataset structure and print statistics
    
    Args:
        data_root (str): Root directory containing the dataset
    """
    print("DATASET VERIFICATION")
    print("=" * 60)
    print(f"Dataset root: {data_root}")
    print("")
    
    for split in ['train', 'val']:
        print(f"{split.upper()} SET:")
        try:
            dataset = GhanaFoodDataset(split=split, data_root=data_root)
            print(f"  Number of samples: {len(dataset)}")
            print(f"  Images directory: {dataset.images_dir}")
            print(f"  Masks directory: {dataset.masks_dir}")
            
            # Load one sample to verify
            img, mask = dataset[0]
            print(f"  Image shape: {img.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Unique mask values: {torch.unique(mask).tolist()}")
            print("")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print("")
    
    print("Verification complete.")


if __name__ == "__main__":
    # Test the dataset loader
    import sys
    
    data_root = sys.argv[1] if len(sys.argv) > 1 else 'data'
    verify_dataset(data_root)
