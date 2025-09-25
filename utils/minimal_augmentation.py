"""
Minimal but Effective Data Augmentation for GhanaSegNet 30%+ Target
Strategic augmentation focusing on highest-impact, lowest-overhead techniques

Author: EricBaidoo
"""

import torch
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageEnhance

class MinimalEffectiveAugmentation:
    """
    Minimal but highly effective augmentation targeting specific food segmentation challenges
    """
    def __init__(self, p=0.5):
        self.p = p
        
        # Core augmentations with proven impact
        self.geometric_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Food presentation symmetry
            transforms.RandomRotation(degrees=10),    # Slight rotation for robustness
        ])
        
        # Food-specific color augmentations
        self.color_variations = [
            self.adjust_lighting,
            self.adjust_saturation,
            self.adjust_warmth
        ]
    
    def adjust_lighting(self, image):
        """Simulate different lighting conditions"""
        if random.random() < self.p:
            # Realistic lighting range for food photos
            factor = random.uniform(0.8, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        return image
    
    def adjust_saturation(self, image):
        """Enhance food colors for better segmentation"""
        if random.random() < self.p:
            # Food colors should remain appetizing
            factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        return image
    
    def adjust_warmth(self, image):
        """Simulate different camera white balance"""
        if random.random() < self.p:
            factor = random.uniform(0.9, 1.1)
            image_np = np.array(image).astype(np.float32)
            
            # Subtle warmth adjustment
            image_np[:, :, 0] *= factor      # Red
            image_np[:, :, 2] *= (2.0 - factor)  # Blue
            
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        return image
    
    def __call__(self, image):
        """Apply minimal effective augmentation"""
        # 1. Geometric (always beneficial for food segmentation)
        image = self.geometric_transforms(image)
        
        # 2. One random color augmentation (avoid over-augmentation)
        if random.random() < 0.6:  # 60% chance of color augmentation
            color_aug = random.choice(self.color_variations)
            image = color_aug(image)
        
        return image

class StandardNormalization:
    """
    Standard normalization for consistency
    """
    def __init__(self):
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image):
        return self.normalize(image)

class SmartGhanaFoodDataset:
    """
    Smart dataset with minimal but effective augmentation
    """
    def __init__(self, split, target_size=(384, 384), use_augmentation=True):
        from data.dataset_loader import GhanaFoodDataset
        self.base_dataset = GhanaFoodDataset(split, target_size)
        self.use_augmentation = use_augmentation and split == 'train'
        
        # Minimal effective augmentation
        if self.use_augmentation:
            self.augmentation = MinimalEffectiveAugmentation(p=0.5)
        
        self.normalization = StandardNormalization()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        if self.use_augmentation:
            # Apply minimal augmentation
            image = self.augmentation(image)
        
        # Standard normalization
        image_tensor = self.normalization(image)
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        
        return image_tensor, mask_tensor