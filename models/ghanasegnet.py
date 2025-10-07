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

class SpatialAttention(nn.Module):
    """Lightweight spatial attention module"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class DecoderBlock(nn.Module):
    """
    Enhanced decoder block with skip connections and spatial attention
    """
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super(DecoderBlock, self).__init__()
        total_in_channels = in_channels + skip_channels
        
        self.conv1 = nn.Conv2d(total_in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Add spatial attention
        self.attention = SpatialAttention(out_channels) if use_attention else None
        
        # Add residual connection if dimensions match
        self.residual = nn.Conv2d(total_in_channels, out_channels, 1) if total_in_channels != out_channels else None

    def forward(self, x, skip=None):
        identity = x
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        # Store for residual
        if self.residual is not None:
            identity = self.residual(x)
        
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        if self.attention is not None:
            out = self.attention(out)
        
        # Add residual connection
        if self.residual is not None:
            out = out + identity
        
        return F.relu(out)

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
        
        # Channel reduction for transformer efficiency (EfficientNet-B0 outputs 1280 features)
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        
        # Transformer block at bottleneck for global context
        self.transformer = TransformerBlock(
            dim=256, 
            heads=4,  # Balanced attention heads
            mlp_dim=256,  # Conservative MLP size for mobile
            dropout=dropout
        )
        
        # Skip connection channel adapters for EfficientNet-B0 features
        # Based on actual EfficientNet-B0 structure: [24@H/4, 112@H/16, 320@H/32]
        self.skip_conv1 = nn.Conv2d(320, 64, kernel_size=1)   # From block 15: 320 -> 64 channels
        self.skip_conv2 = nn.Conv2d(112, 32, kernel_size=1)   # From block 10: 112 -> 32 channels
        self.skip_conv3 = nn.Conv2d(24, 16, kernel_size=1)    # From block 2: 24 -> 16 channels
        
        # Enhanced decoder with skip connections, attention, and residuals
        # Decoder path 1: 256 -> 128 + skip from block 15 (64 channels) = 192 total input
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=128, use_attention=True)
        
        # Decoder path 2: 128 -> 64 + skip from block 10 (32 channels) = 96 total input  
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(in_channels=64, skip_channels=32, out_channels=64, use_attention=True)
        
        # Decoder path 3: 64 -> 32 + skip from block 2 (16 channels) = 48 total input
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4)  # 4x upsampling H/16->H/4  
        self.dec3 = DecoderBlock(in_channels=32, skip_channels=16, out_channels=32, use_attention=True)
        

        
        # Final classification layer with dropout
        self.final = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, kernel_size=1)
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
        Forward pass with proper skip connections from EfficientNet encoder
        Extracts multi-scale features for U-Net style decoder
        """
        # Extract intermediate features from EfficientNet encoder
        skip_features = []
        
        # Start with proper EfficientNet preprocessing
        x_enc = self.encoder._swish(self.encoder._bn0(self.encoder._conv_stem(x)))
        
        # Forward through encoder blocks while collecting skip connections
        for i, block in enumerate(self.encoder._blocks):
            x_enc = block(x_enc)
            # Collect features after specific blocks for skip connections
            if i == 2:    # After block 2: H/4, W/4, 24 channels
                skip_features.append(x_enc)
            elif i == 10:  # After block 10: H/16, W/16, 112 channels
                skip_features.append(x_enc)
            elif i == 15:  # After block 15: H/32, W/32, 320 channels  
                skip_features.append(x_enc)
        
        # Final encoder features (bottleneck)
        features = self.encoder._conv_head(x_enc)    # [B, 1280, H/32, W/32]
        features = self.encoder._bn1(features)
        features = self.encoder._swish(features)
        
        # Bottleneck processing with global context
        features = self.conv1(features)              # Channel reduction [B, 256, H/32, W/32]
        features = self.transformer(features)        # Global attention
        
        # Enhanced decoder with skip connections, attention, and residuals
        # Stage 1: Upsample and fuse with skip from block 15 (320 channels)
        d1 = self.up1(features)                      # [B, 128, H/16, W/16]
        skip1 = self.skip_conv1(skip_features[2])    # Adapt channels [B, 64, H/32, W/32]
        d1 = self.dec1(d1, skip1)                    # Fuse and decode with attention [B, 128, H/16, W/16]
        
        # Stage 2: Upsample and fuse with skip from block 10 (112 channels)
        d2 = self.up2(d1)                            # [B, 64, H/8, W/8]
        skip2 = self.skip_conv2(skip_features[1])    # Adapt channels [B, 32, H/16, W/16]
        d2 = self.dec2(d2, skip2)                    # Fuse and decode with attention [B, 64, H/8, W/8]
        
        # Stage 3: Upsample and fuse with skip from block 2 (24 channels)
        d3 = self.up3(d2)                            # [B, 32, H/4, W/4]
        skip3 = self.skip_conv3(skip_features[0])    # Adapt channels [B, 16, H/4, W/4]
        d3 = self.dec3(d3, skip3)                    # Fuse and decode with attention [B, 32, H/4, W/4]
        
        # Final classification at H/4 resolution, then upsample to input size
        out = self.final(d3)                         # [B, num_classes, H/4, W/4]
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
