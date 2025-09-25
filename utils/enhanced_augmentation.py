"""
Enhanced Data Augmentation Pipeline for 30%+ mIoU Performance
Specialized for Ghanaian Food Segmentation

Key Augmentations:
- MixUp and CutMix for better generalization
- Food-specific color augmentations
- Advanced geometric transformations
- Test-time augmentation
- Progressive augmentation scheduling

Author: EricBaidoo
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class FoodSpecificAugmentation:
    """
    Food-specific augmentations for Ghanaian dishes
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def enhance_saturation(self, image, factor_range=(0.7, 1.5)):
        """Enhance food colors - important for food segmentation"""
        if random.random() < self.p:
            factor = random.uniform(*factor_range)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        return image
    
    def adjust_warmth(self, image, factor_range=(0.8, 1.3)):
        """Adjust color temperature - simulate different lighting"""
        if random.random() < self.p:
            factor = random.uniform(*factor_range)
            image = np.array(image).astype(np.float32)
            
            # Warm/cool adjustment
            image[:, :, 0] *= factor        # Red channel
            image[:, :, 2] *= (2.0 - factor)  # Blue channel
            
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        return image
    
    def simulate_lighting(self, image, brightness_range=(0.7, 1.4)):
        """Simulate different lighting conditions"""
        if random.random() < self.p:
            factor = random.uniform(*brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        return image
    
    def add_shadow(self, image, mask):
        """Add realistic shadows to food images"""
        if random.random() < self.p:
            image_np = np.array(image)
            
            # Create shadow mask
            shadow_strength = random.uniform(0.3, 0.7)
            shadow_kernel = np.ones((20, 20), np.uint8)
            shadow_mask = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, shadow_kernel)
            shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
            
            # Apply shadow
            shadow_factor = 1.0 - shadow_strength * (shadow_mask / 255.0)
            for c in range(3):
                image_np[:, :, c] = image_np[:, :, c] * shadow_factor
            
            image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
        return image

class MixUpAugmentation:
    """
    MixUp augmentation for better generalization
    """
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, images, targets):
        if random.random() > self.p:
            return images, targets
            
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix targets (for segmentation, we use spatial mixing)
        mixed_targets = (lam * targets.unsqueeze(1).float() + 
                        (1 - lam) * targets[indices].unsqueeze(1).float()).squeeze(1).long()
        
        return mixed_images, mixed_targets

class CutMixAugmentation:
    """
    CutMix augmentation for segmentation
    """
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p
    
    def __call__(self, images, targets):
        if random.random() > self.p:
            return images, targets
            
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Get random crop box
        H, W = images.shape[2], images.shape[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_targets = targets.clone()
        
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        mixed_targets[:, bby1:bby2, bbx1:bbx2] = targets[indices, bby1:bby2, bbx1:bbx2]
        
        return mixed_images, mixed_targets

class TestTimeAugmentation:
    """
    Test-time augmentation for improved inference
    """
    def __init__(self, scales=[0.8, 1.0, 1.2], flips=True):
        self.scales = scales
        self.flips = flips
    
    def augment_batch(self, image):
        """Apply multiple augmentations to create batch"""
        augmented_images = []
        transformations = []
        
        for scale in self.scales:
            for flip in ([False, True] if self.flips else [False]):
                img = image.clone()
                
                # Scale
                if scale != 1.0:
                    h, w = img.shape[2], img.shape[3]
                    new_h, new_w = int(h * scale), int(w * scale)
                    img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # Flip
                if flip:
                    img = torch.flip(img, dims=[3])
                
                augmented_images.append(img)
                transformations.append({'scale': scale, 'flip': flip})
        
        return torch.cat(augmented_images, dim=0), transformations
    
    def merge_predictions(self, predictions, transformations, original_size):
        """Merge predictions from augmented images"""
        merged_pred = torch.zeros((1, predictions.shape[1], *original_size), 
                                 device=predictions.device)
        
        for i, (pred, transform) in enumerate(zip(predictions, transformations)):
            pred = pred.unsqueeze(0)
            
            # Reverse flip
            if transform['flip']:
                pred = torch.flip(pred, dims=[3])
            
            # Reverse scale
            if transform['scale'] != 1.0:
                pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            
            merged_pred += pred
        
        return merged_pred / len(transformations)

class AdvancedGhanaFoodDataset:
    """
    Enhanced dataset with advanced augmentations for 30%+ performance
    """
    def __init__(self, split, target_size=(512, 512), enhanced_augment=True):
        from data.dataset_loader import GhanaFoodDataset
        self.base_dataset = GhanaFoodDataset(split, target_size)
        self.enhanced_augment = enhanced_augment and split == 'train'
        self.target_size = target_size
        
        # Food-specific augmentations
        self.food_aug = FoodSpecificAugmentation(p=0.6)
        
        # Advanced augmentations for training
        if self.enhanced_augment:
            self.augmentations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.4,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            ])
        else:
            self.augmentations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        if self.enhanced_augment:
            # Apply food-specific augmentations
            image = self.food_aug.enhance_saturation(image)
            image = self.food_aug.adjust_warmth(image)
            image = self.food_aug.simulate_lighting(image)
            image = self.food_aug.add_shadow(image, mask)
            
            # Convert to tensor for geometric augmentations
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = torch.from_numpy(np.array(mask)).long()
            
            # Apply synchronized augmentations
            seed = random.randint(0, 2**32)
            
            # Augment image
            random.seed(seed)
            torch.manual_seed(seed)
            image_tensor = self.augmentations(image_tensor)
            
            # Augment mask with same transformations
            random.seed(seed)
            torch.manual_seed(seed)
            mask_pil = transforms.ToPILImage()(mask_tensor.unsqueeze(0).float())
            mask_tensor = transforms.ToTensor()(
                self.augmentations(mask_pil)
            ).squeeze(0).long()
            
            # Normalize image
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image_tensor)
            
            return image_tensor, mask_tensor
        else:
            # Standard preprocessing for validation
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image_tensor)
            mask_tensor = torch.from_numpy(np.array(mask)).long()
            
            return image_tensor, mask_tensor

class ProgressiveAugmentationScheduler:
    """
    Progressive augmentation intensity scheduling
    """
    def __init__(self, max_epochs=50):
        self.max_epochs = max_epochs
        self.current_epoch = 0
    
    def step(self, epoch):
        self.current_epoch = epoch
    
    def get_augmentation_strength(self):
        """Get current augmentation strength (0.0 to 1.0)"""
        # Start with strong augmentation, gradually reduce
        progress = self.current_epoch / self.max_epochs
        return max(0.3, 1.0 - progress * 0.7)  # 1.0 -> 0.3
    
    def should_use_mixup(self):
        """Whether to use MixUp in current epoch"""
        return self.current_epoch < self.max_epochs * 0.7  # First 70% of training
    
    def should_use_cutmix(self):
        """Whether to use CutMix in current epoch"""
        return self.current_epoch < self.max_epochs * 0.8  # First 80% of training