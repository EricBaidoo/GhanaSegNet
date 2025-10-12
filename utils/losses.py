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


class AdvancedBoundaryLoss(nn.Module):
    """
    Advanced boundary-aware loss for Enhanced GhanaSegNet
    Optimized for 30% mIoU achievement with food segmentation boundaries
    """
    def __init__(self, boundary_weight=2.0, distance_weight=1.0):
        super(AdvancedBoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.distance_weight = distance_weight
        
        # Multi-scale edge detection kernels
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.laplacian = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Register edge detection kernels
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]]]], dtype=torch.float32)
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1],
                                        [0,  0,  0],
                                        [1,  2,  1]]]], dtype=torch.float32)
        laplacian_kernel = torch.tensor([[[[0, -1, 0],
                                          [-1, 4, -1],
                                          [0, -1, 0]]]], dtype=torch.float32)
        
        self.register_buffer('sobel_x_kernel', sobel_x_kernel)
        self.register_buffer('sobel_y_kernel', sobel_y_kernel)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
        # Disable gradient computation for edge kernels
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        self.laplacian.weight.requires_grad = False

    def get_multi_scale_edges(self, mask):
        """Extract multi-scale boundary information"""
        # Ensure kernels are on the same device as input
        self.sobel_x.weight.data = self.sobel_x_kernel.to(mask.device)
        self.sobel_y.weight.data = self.sobel_y_kernel.to(mask.device)
        self.laplacian.weight.data = self.laplacian_kernel.to(mask.device)
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4 and mask.size(1) > 1:
            # Convert multi-class to single channel by taking argmax
            mask = mask.argmax(dim=1, keepdim=True).float()
        
        # Compute gradients
        grad_x = self.sobel_x(mask)
        grad_y = self.sobel_y(mask)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Laplacian for fine boundaries
        laplacian = torch.abs(self.laplacian(mask))
        
        # Combine multi-scale edge information
        edges = torch.max(grad_magnitude, laplacian)
        return (edges > 0.1).float()

    def compute_distance_transform(self, edges):
        """Compute approximate distance transform for boundary weighting"""
        # Simple distance approximation using max pooling
        distances = edges.clone()
        for _ in range(3):  # 3 iterations for reasonable distance approximation
            distances = F.max_pool2d(distances, 3, stride=1, padding=1)
        return distances

    def forward(self, inputs, targets):
        """
        Advanced boundary loss computation with multi-scale edge detection
        """
        # Use softmax probabilities for differentiable edge detection
        inputs_prob = F.softmax(inputs, dim=1)
        inputs_soft = torch.sum(inputs_prob * torch.arange(inputs.size(1), device=inputs.device).view(1, -1, 1, 1).float(), dim=1)
        
        # Extract multi-scale boundary information
        inputs_edges = self.get_multi_scale_edges(inputs_soft)
        targets_edges = self.get_multi_scale_edges(targets.float())
        
        # Ensure same spatial dimensions
        if inputs_edges.shape != targets_edges.shape:
            targets_edges = F.interpolate(targets_edges, size=inputs_edges.shape[-2:], mode='nearest')
        
        # Compute distance-weighted boundary loss
        target_distances = self.compute_distance_transform(targets_edges)
        
        # Ensure boundary weights match edge dimensions
        if target_distances.shape != inputs_edges.shape:
            target_distances = F.interpolate(target_distances, size=inputs_edges.shape[-2:], mode='nearest')
        
        boundary_weights = 1.0 + self.distance_weight * target_distances
        
        # Weighted binary cross-entropy on boundaries
        boundary_loss = F.binary_cross_entropy(inputs_edges, targets_edges, reduction='none')
        weighted_boundary_loss = (boundary_loss * boundary_weights).mean()
        
        # Additional edge consistency loss for smooth boundaries
        edge_consistency = F.mse_loss(inputs_edges, targets_edges)
        
        return self.boundary_weight * weighted_boundary_loss + 0.1 * edge_consistency


class AdvancedFocalLoss(nn.Module):
    """
    Advanced Focal Loss with class balancing for food segmentation
    Optimized for 30% mIoU achievement
    """
    def __init__(self, alpha=None, gamma=2.0, num_classes=6):
        super(AdvancedFocalLoss, self).__init__()
        if alpha is None:
            # Food segmentation class weights (background, banku, rice, fufu, kenkey, other)
            alpha = torch.FloatTensor([0.5, 2.0, 2.0, 2.5, 2.0, 1.5])
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Dynamic alpha balancing
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.view(-1)).view(targets.shape)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    SUPER-ENHANCED Combined Loss for 30% mIoU target with multi-scale supervision
    """
    def __init__(self, alpha=0.6, aux_weight=0.4, adaptive_weights=True):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.aux_weight = aux_weight  # Weight for auxiliary losses
        self.adaptive_weights = adaptive_weights
        
        # Enhanced loss components
        self.dice = DiceLoss()
        self.boundary = AdvancedBoundaryLoss()
        self.focal_loss = AdvancedFocalLoss()
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
