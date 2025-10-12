"""
Multi-Scale Test-Time Augmentation for Enhanced GhanaSegNet
Implements ensemble inference for maximum performance boost
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class MultiScaleTTA:
    """
    Multi-scale Test-Time Augmentation for Enhanced GhanaSegNet
    Combines predictions from multiple scales and transformations
    """
    def __init__(self, model, scales=[0.8, 1.0, 1.2, 1.4], flip=True):
        self.model = model
        self.scales = scales
        self.flip = flip
        
        # Store original model training state
        self.original_training_state = model.training
        
    def __enter__(self):
        self.model.eval()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.original_training_state)
    
    def predict_single_scale(self, image, scale=1.0, flip=False):
        """Predict at single scale with optional flipping"""
        B, C, H, W = image.shape
        
        # Scale image
        if scale != 1.0:
            new_h, new_w = int(H * scale), int(W * scale)
            # Ensure dimensions are divisible by 32 for proper feature map sizes
            new_h = ((new_h + 31) // 32) * 32
            new_w = ((new_w + 31) // 32) * 32
            scaled_image = F.interpolate(
                image, size=(new_h, new_w), 
                mode='bilinear', align_corners=False
            )
        else:
            scaled_image = image
            new_h, new_w = H, W
        
        # Apply horizontal flip if requested
        if flip:
            scaled_image = torch.flip(scaled_image, dims=[3])
        
        # Get prediction
        with torch.no_grad():
            pred = self.model(scaled_image)
            if isinstance(pred, tuple):
                pred = pred[0]  # Take main output if auxiliary outputs exist
        
        # Flip prediction back if image was flipped
        if flip:
            pred = torch.flip(pred, dims=[3])
        
        # Resize prediction back to original size
        if scale != 1.0:
            pred = F.interpolate(
                pred, size=(H, W), 
                mode='bilinear', align_corners=False
            )
        
        return pred
    
    def predict_with_tta(self, image):
        """
        Predict with full test-time augmentation ensemble
        
        Args:
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Ensemble prediction [B, num_classes, H, W]
        """
        predictions = []
        
        # Multi-scale predictions
        for scale in self.scales:
            # Normal prediction
            pred = self.predict_single_scale(image, scale=scale, flip=False)
            predictions.append(pred)
            
            # Flipped prediction (if enabled)
            if self.flip:
                pred_flip = self.predict_single_scale(image, scale=scale, flip=True)
                predictions.append(pred_flip)
        
        # Ensemble averaging
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred
    
    def predict_with_confidence(self, image, return_uncertainty=False):
        """
        Predict with confidence estimation
        
        Args:
            image: Input image tensor
            return_uncertainty: Whether to return prediction uncertainty
            
        Returns:
            prediction: Ensemble prediction
            confidence: Prediction confidence (if return_uncertainty=True)
        """
        predictions = []
        
        # Collect all predictions
        for scale in self.scales:
            pred = self.predict_single_scale(image, scale=scale, flip=False)
            predictions.append(F.softmax(pred, dim=1))
            
            if self.flip:
                pred_flip = self.predict_single_scale(image, scale=scale, flip=True)
                predictions.append(F.softmax(pred_flip, dim=1))
        
        # Stack predictions
        all_preds = torch.stack(predictions, dim=0)  # [num_predictions, B, C, H, W]
        
        # Ensemble mean
        mean_pred = all_preds.mean(dim=0)
        
        if return_uncertainty:
            # Compute prediction variance as uncertainty measure
            variance = all_preds.var(dim=0).mean(dim=1, keepdim=True)  # [B, 1, H, W]
            # Convert variance to confidence (higher variance = lower confidence)
            confidence = 1.0 / (1.0 + variance)
            return mean_pred, confidence
        
        return mean_pred


class AdvancedTTAStrategy:
    """
    Advanced TTA strategy with adaptive scaling and rotation
    """
    def __init__(self, model, adaptive_scales=True, use_rotation=False):
        self.model = model
        self.adaptive_scales = adaptive_scales
        self.use_rotation = use_rotation
        
        # Base scales
        self.base_scales = [0.8, 0.9, 1.0, 1.1, 1.25]
        
        # Rotation angles (if enabled)
        self.rotation_angles = [0, 90, 180, 270] if use_rotation else [0]
    
    def get_adaptive_scales(self, image_size):
        """Get adaptive scales based on image size"""
        H, W = image_size
        
        # Adjust scales based on image resolution
        if H < 256 or W < 256:
            # For small images, use more aggressive upscaling
            return [0.9, 1.0, 1.2, 1.4, 1.6]
        elif H > 512 or W > 512:
            # For large images, use more conservative scaling
            return [0.7, 0.85, 1.0, 1.15]
        else:
            # Standard scaling for medium images
            return self.base_scales
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        if angle == 0:
            return image
        elif angle == 90:
            return torch.rot90(image, k=1, dims=[2, 3])
        elif angle == 180:
            return torch.rot90(image, k=2, dims=[2, 3])
        elif angle == 270:
            return torch.rot90(image, k=3, dims=[2, 3])
        else:
            # For arbitrary angles, would need torchvision.transforms.functional.rotate
            return image
    
    def predict_with_advanced_tta(self, image):
        """Advanced TTA with adaptive scaling and optional rotation"""
        predictions = []
        
        # Get adaptive scales
        if self.adaptive_scales:
            scales = self.get_adaptive_scales(image.shape[2:])
        else:
            scales = self.base_scales
        
        # Multi-scale and multi-rotation predictions
        for scale in scales:
            for angle in self.rotation_angles:
                # Apply rotation
                rotated_image = self.rotate_image(image, angle)
                
                # Scale image
                if scale != 1.0:
                    H, W = rotated_image.shape[2:]
                    new_h, new_w = int(H * scale), int(W * scale)
                    new_h = ((new_h + 31) // 32) * 32
                    new_w = ((new_w + 31) // 32) * 32
                    scaled_image = F.interpolate(
                        rotated_image, size=(new_h, new_w),
                        mode='bilinear', align_corners=False
                    )
                else:
                    scaled_image = rotated_image
                
                # Get prediction
                with torch.no_grad():
                    pred = self.model(scaled_image)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                
                # Resize back to original size
                if scale != 1.0:
                    pred = F.interpolate(
                        pred, size=rotated_image.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                
                # Rotate prediction back
                if angle != 0:
                    if angle == 90:
                        pred = torch.rot90(pred, k=-1, dims=[2, 3])
                    elif angle == 180:
                        pred = torch.rot90(pred, k=-2, dims=[2, 3])
                    elif angle == 270:
                        pred = torch.rot90(pred, k=-3, dims=[2, 3])
                
                # Final resize to match input
                if pred.shape[2:] != image.shape[2:]:
                    pred = F.interpolate(
                        pred, size=image.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                
                predictions.append(F.softmax(pred, dim=1))
        
        # Ensemble averaging
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        return ensemble_pred


def create_tta_predictor(model, strategy='standard'):
    """
    Create TTA predictor for Enhanced GhanaSegNet
    
    Args:
        model: Trained Enhanced GhanaSegNet model
        strategy: 'standard', 'advanced', or 'fast'
    
    Returns:
        TTA predictor instance
    """
    if strategy == 'standard':
        return MultiScaleTTA(
            model=model,
            scales=[0.9, 1.0, 1.1, 1.2],
            flip=True
        )
    elif strategy == 'advanced':
        return AdvancedTTAStrategy(
            model=model,
            adaptive_scales=True,
            use_rotation=False  # Rotation can be expensive
        )
    elif strategy == 'fast':
        return MultiScaleTTA(
            model=model,
            scales=[1.0, 1.1],
            flip=True
        )
    else:
        raise ValueError(f"Unknown TTA strategy: {strategy}")


# Usage example:
"""
# After training your model
model = EnhancedGhanaSegNet(num_classes=6)
model.load_state_dict(torch.load('best_model.pth'))

# Create TTA predictor
tta_predictor = create_tta_predictor(model, strategy='standard')

# Use TTA for inference
with tta_predictor:
    ensemble_pred = tta_predictor.predict_with_tta(test_image)
    
    # Or with confidence estimation
    pred, confidence = tta_predictor.predict_with_confidence(test_image, return_uncertainty=True)
"""