"""
GhanaSegNet v2.0 - Advanced Model for Ghana Food Segmentation
Incorporates domain-specific innovations for African food segmentation
Author: EricBaidoo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class CulturalAttention(nn.Module):
    """
    Cultural Context Attention Module
    Learns cultural patterns specific to Ghanaian food presentation
    """
    def __init__(self, in_channels, reduction=16):
        super(CulturalAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-scale feature fusion for handling diverse food textures
    Captures both fine details (rice grains) and coarse patterns (stew)
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.scales = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // 4, 1),  # 1x1
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),  # 3x3
            nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),  # 5x5
            nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, out_channels // 4, 1)
            )  # pooling
        ])
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        features = [scale(x) for scale in self.scales]
        fused = torch.cat(features, dim=1)
        return self.fusion(fused)

class ContextualTransformerBlock(nn.Module):
    """
    Enhanced transformer with positional encoding for spatial context
    Better for understanding spatial relationships in food arrangements
    """
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.1):
        super(ContextualTransformerBlock, self).__init__()
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for spatial awareness
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # Add positional encoding
        seq_len = x.shape[1]
        if seq_len <= 1024:
            x = x + self.pos_embed[:, :seq_len, :]
        
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        # Reshape back
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class EnhancedDecoderBlock(nn.Module):
    """
    Enhanced decoder with cultural attention and residual connections
    """
    def __init__(self, in_channels, out_channels):
        super(EnhancedDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cultural_attn = CulturalAttention(out_channels)
        
        # Skip connection if dimensions match
        self.skip = nn.Identity() if in_channels == out_channels else \
                   nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cultural_attn(out)
        
        out += identity
        return F.relu(out)

class GhanaSegNetV2(nn.Module):
    """
    Advanced GhanaSegNet v2.0 with cultural context understanding
    
    Key innovations:
    1. Cultural attention mechanisms
    2. Multi-scale feature fusion
    3. Enhanced positional transformer
    4. Residual decoder blocks
    5. Ghana-specific architectural choices
    """
    def __init__(self, num_classes=6, dropout=0.1):
        super(GhanaSegNetV2, self).__init__()
        
        # Efficient backbone for mobile deployment
        self.encoder = EfficientNet.from_pretrained('efficientnet-b2')  # Slightly larger for better features
        
        # Multi-level feature extraction
        self.enc0 = nn.Sequential(
            self.encoder._conv_stem, 
            self.encoder._bn0, 
            self.encoder._swish
        )  # 32 channels
        self.enc1 = self.encoder._blocks[0:2]   # 16 channels
        self.enc2 = self.encoder._blocks[2:4]   # 24 channels  
        self.enc3 = self.encoder._blocks[4:10]  # 48 channels
        self.enc4 = self.encoder._blocks[10:]   # 120 channels
        
        # Enhanced feature processing
        self.multi_scale_fusion = MultiScaleFeatureFusion(1408, 512)  # EfficientNet-B2 final: 1408
        self.conv_reduce = nn.Conv2d(512, 256, kernel_size=1)
        
        # Cultural context transformer
        self.transformer = ContextualTransformerBlock(
            dim=256, 
            heads=8, 
            mlp_dim=512,
            dropout=dropout
        )
        
        # Enhanced decoder with cultural attention
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = EnhancedDecoderBlock(128 + 120, 128)  # Skip from enc4
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = EnhancedDecoderBlock(64 + 48, 64)     # Skip from enc3
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = EnhancedDecoderBlock(32 + 24, 32)     # Skip from enc2
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = EnhancedDecoderBlock(16 + 16, 16)     # Skip from enc1
        
        # Final classification with dropout for regularization
        self.final_conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Encoder path with skip connections
        x0 = self.enc0(x)                          # [B, 32, H/2, W/2]
        x1 = self.enc1(x0)                         # [B, 16, H/4, W/4]
        x2 = self.enc2(x1)                         # [B, 24, H/8, W/8]
        x3 = self.enc3(x2)                         # [B, 48, H/16, W/16]
        x4 = self.enc4(x3)                         # [B, 120, H/32, W/32]
        
        # Enhanced feature processing
        x4_enhanced = self.multi_scale_fusion(x4)
        x4_reduced = self.conv_reduce(x4_enhanced)
        x4_context = self.transformer(x4_reduced)
        
        # Decoder path with enhanced blocks
        d4 = self.up4(x4_context)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))
        
        # Final prediction
        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out

# Create alias for backward compatibility
GhanaSegNet = GhanaSegNetV2
