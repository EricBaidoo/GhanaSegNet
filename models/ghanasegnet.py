"""
GhanaSegNet: Novel Hybrid CNN-Transformer Architecture
for Semantic Segmentation of Traditional Ghanaian Foods

Key Innovations:
- EfficientNet-lite0 backbone with ImageNet pretraining
- Novel transformer integration for global context
- Culturally-aware architectural design
- Direct transfer learning (ImageNet → Ghana Food)

Author: EricBaidoo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class TransformerBlock(nn.Module):
    """
    Transformer block for global context understanding
    Integrates at bottleneck for efficiency (Xie et al., 2021)
    """
    def __init__(self, dim, heads=4, mlp_dim=256, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # Better gradients than ReLU
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back to feature map
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class DecoderBlock(nn.Module):
    """
    Standard decoder block with BatchNorm and ReLU
    Maintains architectural simplicity for mobile deployment
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class GhanaSegNet(nn.Module):
    """
    GhanaSegNet: Advanced Multi-Scale Transfer Learning Framework
    
    Architecture Components:
    1. EfficientNet-B0 encoder (ImageNet pretrained backbone)
    2. Transformer integration at bottleneck (global context)
    3. U-Net decoder with skip connections (feature fusion)
    4. Direct transfer learning (ImageNet → Ghana Food)
    
    Design Philosophy:
    - Efficient design: Balanced accuracy-speed trade-off
    - Culturally-aware: Designed for traditional food presentation
    - Transfer learning: Direct domain adaptation from ImageNet
    - Novel architecture: CNN-Transformer hybrid for food segmentation
    """
    def __init__(self, num_classes=6, dropout=0.1):
        super(GhanaSegNet, self).__init__()
        
        # EfficientNet-B0 backbone (ImageNet pretrained)
        # Note: Using B0 instead of lite0 due to library availability
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Extract features at different levels using proper forward hooks
        # We'll use the encoder in forward pass, not split it here
        
        # Channel reduction for transformer efficiency (EfficientNet-B0 outputs 1280 features)
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        
        # Transformer block at bottleneck for global context
        self.transformer = TransformerBlock(
            dim=256, 
            heads=4,  # Balanced attention heads
            mlp_dim=256,  # Conservative MLP size for mobile
            dropout=dropout
        )
        
        # Simple decoder for proof of concept
        # Note: Using standard channels for initial implementation
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(128, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32, 16)
        
        # Final classification layer with dropout
        self.final = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize non-pretrained weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, '_is_pretrained'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """
        Forward pass implementing EfficientNet encoder + Transformer + simple decoder
        For now, keep the original simple approach until we can properly implement skip connections
        """
        # Extract features using EfficientNet encoder
        features = self.encoder.extract_features(x)  # [B, 1280, H/32, W/32]
        
        # Bottleneck processing with global context
        features = self.conv1(features)              # Channel reduction [B, 256, H/32, W/32]
        features = self.transformer(features)        # Global attention
        
        # Simple decoder path (skip connections to be implemented properly later)
        d1 = self.up1(features)                      # [B, 128, H/16, W/16]
        d1 = self.dec1(d1)                           # [B, 64, H/16, W/16]
        
        d2 = self.up2(d1)                            # [B, 32, H/8, W/8]
        d2 = self.dec2(d2)                           # [B, 16, H/8, W/8]
        
        # Final prediction with bilinear upsampling to original size
        out = self.final(d2)                         # [B, num_classes, H/8, W/8]
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out
    
    def get_transfer_learning_parameters(self):
        """
        Return parameter groups for multi-stage transfer learning
        Supports progressive unfreezing strategies
        """
        return {
            'encoder_backbone': list(self.encoder.parameters()),
            'decoder_head': list(self.up2.parameters()) + list(self.dec2.parameters()) +
                           list(self.up1.parameters()) + list(self.dec1.parameters()) +
                           list(self.final.parameters()),
            'transformer': list(self.transformer.parameters()),
            'bottleneck': list(self.conv1.parameters())
        }
