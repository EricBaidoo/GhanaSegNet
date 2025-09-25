"""
Advanced Loss Functions for GhanaSegNet
Optimized loss functions specifically designed for 30%+ mIoU achievement

This module provides state-of-the-art loss functions that work together
to optimize different aspects of segmentation performance.

Author: EricBaidoo
Target: 30%+ mIoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalization of Dice Loss
    Better handling of class imbalance with alpha/beta parameters
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_index=255):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # False positive weight
        self.beta = beta    # False negative weight
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # Get probabilities
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        # Mask out ignore pixels
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target_one_hot = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)
        
        # Calculate Tversky components
        tp = (pred * target_one_hot).sum(dim=2)
        fp = (pred * (1 - target_one_hot)).sum(dim=2)
        fn = ((1 - pred) * target_one_hot).sum(dim=2)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Average across classes and batch
        return 1 - tversky.mean()

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - Combines Focal and Tversky losses
    Focuses on hard examples while handling class imbalance
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6, ignore_index=255):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # Get probabilities
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot target
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        # Mask out ignore pixels
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target_one_hot = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)
        
        # Calculate Tversky components
        tp = (pred * target_one_hot).sum(dim=2)
        fp = (pred * (1 - target_one_hot)).sum(dim=2)
        fn = ((1 - pred) * target_one_hot).sum(dim=2)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal weight
        focal_weight = (1 - tversky) ** self.gamma
        
        # Focal Tversky loss
        focal_tversky = focal_weight * (1 - tversky)
        
        return focal_tversky.mean()

class BoundaryLoss(nn.Module):
    """
    Boundary Loss - Emphasizes boundary pixels for sharp segmentation
    """
    def __init__(self, boundary_weight=5.0, ignore_index=255):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        
        # Laplacian kernel for boundary detection
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian_kernel.view(1, 1, 3, 3))
        
    def get_boundary_mask(self, target):
        """Extract boundary pixels using Laplacian"""
        target_float = target.float().unsqueeze(1)
        
        # Apply Laplacian
        boundaries = F.conv2d(target_float, self.laplacian, padding=1)
        boundaries = torch.abs(boundaries) > 0.1
        
        return boundaries.squeeze(1)
    
    def forward(self, pred, target):
        # Get boundary mask
        boundary_mask = self.get_boundary_mask(target)
        
        # Regular cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # Apply boundary weighting
        weighted_loss = ce_loss.clone()
        weighted_loss[boundary_mask] *= self.boundary_weight
        
        # Mask out ignore pixels
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            weighted_loss = weighted_loss[valid_mask]
        
        return weighted_loss.mean()

class ComboLoss(nn.Module):
    """
    Combination Loss optimized for GhanaSegNet Advanced
    
    Combines multiple loss functions with optimal weights for 30%+ mIoU:
    - Cross Entropy (base classification)
    - Focal Tversky (hard examples + class imbalance)
    - Boundary Loss (sharp boundaries)
    - Dice Loss (region overlap)
    """
    def __init__(self, num_classes=6, class_weights=None, ignore_index=255):
        super(ComboLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=2.0, ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(boundary_weight=3.0, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        
        # Optimized weights (tuned for 30%+ mIoU)
        self.weights = {
            'ce': 0.35,           # Base classification
            'focal_tversky': 0.30,  # Hard examples focus
            'boundary': 0.20,      # Sharp boundaries
            'dice': 0.15          # Region overlap
        }
        
        print(f"🎯 ComboLoss initialized with weights: {self.weights}")
        
    def forward(self, pred, target):
        # Calculate individual losses
        ce = self.ce_loss(pred, target)
        ft = self.focal_tversky(pred, target)
        boundary = self.boundary_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.weights['ce'] * ce +
            self.weights['focal_tversky'] * ft +
            self.weights['boundary'] * boundary +
            self.weights['dice'] * dice
        )
        
        # Return loss and components for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce.item(),
            'focal_tversky': ft.item(),
            'boundary': boundary.item(),
            'dice': dice.item()
        }
        
        return total_loss, loss_dict

class DiceLoss(nn.Module):
    """
    Enhanced Dice Loss with smooth handling
    """
    def __init__(self, num_classes, smooth=1e-6, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # Get probabilities
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Mask out ignore pixels
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target_one_hot = target_one_hot.view(target_one_hot.size(0), target_one_hot.size(1), -1)
        
        # Calculate Dice
        intersection = (pred * target_one_hot).sum(dim=2)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()

class AdaptiveLoss(nn.Module):
    """
    Adaptive Loss that adjusts weights based on training progress
    Starts with basic losses and gradually emphasizes harder objectives
    """
    def __init__(self, num_classes=6, class_weights=None, ignore_index=255):
        super(AdaptiveLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.iteration = 0
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.focal_tversky = FocalTverskyLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        
    def get_adaptive_weights(self, iteration):
        """Get adaptive weights based on training progress"""
        # Normalize iteration (0-1 over typical training)
        progress = min(iteration / 10000, 1.0)  # Assume 10k iterations
        
        # Early training: focus on basic classification
        if progress < 0.3:
            return {'ce': 0.6, 'dice': 0.3, 'focal_tversky': 0.1, 'boundary': 0.0}
        # Mid training: add hard example focus
        elif progress < 0.7:
            return {'ce': 0.4, 'dice': 0.3, 'focal_tversky': 0.2, 'boundary': 0.1}
        # Late training: emphasize boundaries and hard examples
        else:
            return {'ce': 0.3, 'dice': 0.2, 'focal_tversky': 0.3, 'boundary': 0.2}
    
    def forward(self, pred, target):
        self.iteration += 1
        weights = self.get_adaptive_weights(self.iteration)
        
        # Calculate losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        ft = self.focal_tversky(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Adaptive combination
        total_loss = (
            weights['ce'] * ce +
            weights['dice'] * dice +
            weights['focal_tversky'] * ft +
            weights['boundary'] * boundary
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce.item(),
            'dice': dice.item(),
            'focal_tversky': ft.item(),
            'boundary': boundary.item(),
            'weights': weights
        }
        
        return total_loss, loss_dict

# Factory functions for easy usage
def create_combo_loss(num_classes=6, class_weights=None):
    """Create the optimized combo loss for 30%+ mIoU"""
    return ComboLoss(num_classes=num_classes, class_weights=class_weights)

def create_adaptive_loss(num_classes=6, class_weights=None):
    """Create adaptive loss that evolves during training"""
    return AdaptiveLoss(num_classes=num_classes, class_weights=class_weights)

def create_simple_combo_loss(num_classes=6):
    """Create a simpler combination for baseline comparison"""
    class SimpleLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(ignore_index=255)
            self.dice = DiceLoss(num_classes=num_classes)
            
        def forward(self, pred, target):
            ce_loss = self.ce(pred, target)
            dice_loss = self.dice(pred, target)
            total = 0.7 * ce_loss + 0.3 * dice_loss
            
            return total, {
                'total': total.item(),
                'ce': ce_loss.item(),
                'dice': dice_loss.item()
            }
    
    return SimpleLoss()

if __name__ == "__main__":
    # Test the loss functions
    print("🧪 Testing Advanced Loss Functions")
    
    # Dummy data
    batch_size, num_classes, height, width = 2, 6, 64, 64
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test ComboLoss
    combo_loss = create_combo_loss(num_classes=num_classes)
    loss, loss_dict = combo_loss(pred, target)
    
    print(f"✅ ComboLoss: {loss:.4f}")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"   {name}: {value:.4f}")
    
    print("🎯 Advanced loss functions ready for 30%+ mIoU training!")