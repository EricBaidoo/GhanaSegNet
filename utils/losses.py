import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        # Robustly ensure targets shape is [batch, H, W]
        if targets.dim() > 3:
            # Always squeeze all singleton dimensions
            targets = targets.squeeze()
        if targets.dim() == 2:
            # If shape is [H, W], add batch dimension
            targets = targets.unsqueeze(0)
        if targets.dim() != 3:
            raise ValueError(f"Unexpected mask shape after squeeze: {targets.shape}")
        # Now targets should be [batch, H, W]
        targets = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.sobel = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # Register sobel kernel as a buffer so it moves with the model
        sobel_kernel = torch.tensor([[[[-1, -2, -1],
                                       [0,  0,  0],
                                       [1,  2,  1]]]], dtype=torch.float32)
        self.register_buffer('sobel_kernel', sobel_kernel)
        self.sobel.weight.requires_grad = False

    def get_edges(self, mask):
        # Ensure sobel weights are on the same device as input
        self.sobel.weight.data = self.sobel_kernel.to(mask.device)
        edge = torch.abs(self.sobel(mask.unsqueeze(1)))
        return (edge > 0.1).float()

    def forward(self, inputs, targets):
        # Use softmax probabilities instead of argmax for differentiability
        inputs_prob = F.softmax(inputs, dim=1)
        # Get the most probable class as a soft approximation
        inputs_soft = torch.sum(inputs_prob * torch.arange(inputs.size(1), device=inputs.device).view(1, -1, 1, 1).float(), dim=1)
        
        inputs_edge = self.get_edges(inputs_soft)
        targets_edge = self.get_edges(targets.float())

        return F.binary_cross_entropy(inputs_edge, targets_edge)


class CombinedLoss(nn.Module):
    """
    Enhanced Combined Loss for 30% mIoU target with multi-scale supervision
    """
    def __init__(self, alpha=0.7, aux_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.aux_weight = aux_weight  # Weight for auxiliary losses
        
        # Enhanced loss components
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Class-balanced weights for food segmentation (background, banku, rice, fufu, kenkey, other)
        class_weights = torch.tensor([0.5, 2.0, 1.5, 2.5, 1.8, 1.2])
        self.balanced_ce = nn.CrossEntropyLoss(weight=class_weights)
        
        # Focal loss for hard examples
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def focal_loss(self, inputs, targets):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets, aux_outputs=None):
        # Handle different input formats
        if isinstance(inputs, tuple):
            inputs, aux_outputs = inputs[0], inputs[1] if len(inputs) > 1 else None
        
        # Main loss computation - enhanced for 30% mIoU target
        dice_loss = self.dice(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        
        # Move class weights to same device as inputs
        if hasattr(self.balanced_ce, 'weight') and self.balanced_ce.weight is not None:
            self.balanced_ce.weight = self.balanced_ce.weight.to(inputs.device)
        balanced_ce_loss = self.balanced_ce(inputs, targets)
        
        # Enhanced main loss combining multiple components for better segmentation
        main_loss = (self.alpha * (dice_loss + focal_loss * 0.5 + balanced_ce_loss * 0.3) + 
                    (1 - self.alpha) * boundary_loss + 
                    ce_loss * 0.1)  # Small CE loss for stability
        
        # Multi-scale auxiliary supervision for 30% mIoU target
        aux_loss = 0.0
        if aux_outputs is not None and len(aux_outputs) > 0:
            for aux_out in aux_outputs:
                aux_dice = self.dice(aux_out, targets)
                aux_ce = self.ce_loss(aux_out, targets)
                aux_focal = self.focal_loss(aux_out, targets)
                aux_loss += (aux_dice + aux_ce * 0.5 + aux_focal * 0.3)
            
            aux_loss = aux_loss / len(aux_outputs)  # Average auxiliary losses
            total_loss = main_loss + self.aux_weight * aux_loss
        else:
            total_loss = main_loss
        
        return total_loss
