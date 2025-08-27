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
        self.sobel.weight.data = torch.tensor([[[[-1, -2, -1],
                                                 [0,  0,  0],
                                                 [1,  2,  1]]]], dtype=torch.float32)
        self.sobel.weight.requires_grad = False

    def get_edges(self, mask):
        edge = torch.abs(self.sobel(mask.unsqueeze(1)))
        return (edge > 0.1).float()

    def forward(self, inputs, targets):
        inputs = torch.argmax(inputs, dim=1)
        inputs_edge = self.get_edges(inputs.float())
        targets_edge = self.get_edges(targets.float())

        return F.binary_cross_entropy(inputs_edge, targets_edge)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        return self.alpha * dice_loss + (1 - self.alpha) * boundary_loss
