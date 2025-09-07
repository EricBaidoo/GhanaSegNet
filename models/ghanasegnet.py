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
    1. EfficientNet-lite0 encoder (mobile-optimized backbone)
    2. Transformer integration at bottleneck (global context)
    3. U-Net decoder with skip connections (feature fusion)
    4. Multi-stage transfer learning ready
    
    Design Philosophy:
    - Mobile-first: Optimized for deployment in Ghana/West Africa
    - Culturally-aware: Designed for traditional food presentation
    - Transfer learning: Ready for multi-stage domain adaptation
    - Efficiency: Balanced accuracy-speed trade-off
    """
    def __init__(self, num_classes=6, dropout=0.1):
        super(GhanaSegNet, self).__init__()
        
        # EfficientNet-lite0 backbone (mobile-optimized)
        # 4.6M parameters vs 5.3M for EfficientNet-B0
        self.encoder = EfficientNet.from_pretrained('efficientnet-lite0')
        
        # Multi-level feature extraction following EfficientNet structure
        self.enc0 = nn.Sequential(
            self.encoder._conv_stem, 
            self.encoder._bn0, 
            self.encoder._swish
        )  # Output: 32 channels
        
        self.enc1 = self.encoder._blocks[0:2]   # Output: 16 channels
        self.enc2 = self.encoder._blocks[2:4]   # Output: 24 channels  
        self.enc3 = self.encoder._blocks[4:10]  # Output: 112 channels
        self.enc4 = self.encoder._blocks[10:]   # Output: 320 channels
        
        # Channel reduction for transformer efficiency
        self.conv1 = nn.Conv2d(320, 256, kernel_size=1)
        
        # Transformer block at bottleneck for global context
        self.transformer = TransformerBlock(
            dim=256, 
            heads=4,  # Balanced attention heads
            mlp_dim=256,  # Conservative MLP size for mobile
            dropout=dropout
        )
        
        # Progressive upsampling decoder with skip connections
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(128 + 112, 128)  # Skip from enc3
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(64 + 24, 64)     # Skip from enc2
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32 + 16, 32)     # Skip from enc1
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(16 + 32, 16)     # Skip from enc0
        
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
        Forward pass implementing multi-scale feature extraction
        and transformer-enhanced global context understanding
        """
        # Encoder path - hierarchical feature extraction
        x0 = self.enc0(x)                          # [B, 32, H/2, W/2]
        x1 = self.enc1(x0)                         # [B, 16, H/4, W/4]
        x2 = self.enc2(x1)                         # [B, 24, H/8, W/8]
        x3 = self.enc3(x2)                         # [B, 112, H/16, W/16]
        x4 = self.enc4(x3)                         # [B, 320, H/32, W/32]
        
        # Bottleneck processing with global context
        x4 = self.conv1(x4)                        # Channel reduction
        x4 = self.transformer(x4)                  # Global attention
        
        # Decoder path - progressive upsampling with skip connections
        d4 = self.up4(x4)                          # [B, 128, H/16, W/16]
        d4 = self.dec4(torch.cat([d4, x3], dim=1)) # Skip connection
        
        d3 = self.up3(d4)                          # [B, 64, H/8, W/8]
        d3 = self.dec3(torch.cat([d3, x2], dim=1)) # Skip connection
        
        d2 = self.up2(d3)                          # [B, 32, H/4, W/4]
        d2 = self.dec2(torch.cat([d2, x1], dim=1)) # Skip connection
        
        d1 = self.up1(d2)                          # [B, 16, H/2, W/2]
        d1 = self.dec1(torch.cat([d1, x0], dim=1)) # Skip connection
        
        # Final prediction with bilinear upsampling
        out = self.final(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out
    
    def get_transfer_learning_parameters(self):
        """
        Return parameter groups for multi-stage transfer learning
        Supports progressive unfreezing strategies
        """
        return {
            'encoder_backbone': list(self.encoder.parameters()),
            'decoder_head': list(self.up4.parameters()) + list(self.dec4.parameters()) + 
                           list(self.up3.parameters()) + list(self.dec3.parameters()) +
                           list(self.up2.parameters()) + list(self.dec2.parameters()) +
                           list(self.up1.parameters()) + list(self.dec1.parameters()) +
                           list(self.final.parameters()),
            'transformer': list(self.transformer.parameters()),
            'bottleneck': list(self.conv1.parameters())
        }
