"""
GhanaSegNet: Enhanced Hybrid CNN-Transformer Architecture
for Semantic Segmentation of Traditional Ghanaian Foods

Key Innovations:
- EfficientNet-B0 backbone with ImageNet pretraining
- Enhanced transformer with 8 attention heads (vs 4)
- ASPP module for multi-scale feature extraction
- Enhanced spatial attention with channel attention
- Improved decoder with progressive feature fusion
- Optimized for fair benchmarking against DeepLabV3+

Architectural Enhancements for Benchmarking:
- Multi-scale context via ASPP (like DeepLabV3+)
- Deeper transformer with better gradient flow
- Enhanced attention mechanisms
- Progressive decoder improvements

Author: EricBaidoo
Date: Enhanced October 11, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class TransformerBlock(nn.Module):
    """
    Enhanced Transformer block for global context understanding
    Improvements: 8 heads (vs 4), deeper MLP, better normalization, gradient scaling
    """
    def __init__(self, dim, heads=8, mlp_dim=512, dropout=0.15):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads,  # Enhanced from 4 to 8 heads
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        # Enhanced MLP with deeper architecture
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, dim),
            nn.Dropout(dropout)
        )
        
        # Learnable scaling factors for better gradient flow
        self.scale_attn = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_mlp = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # Enhanced self-attention with scaling
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.scale_attn * attn_out
        
        # Enhanced MLP with scaling
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.scale_mlp * mlp_out
        
        # Reshape back to feature map
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class SpatialAttention(nn.Module):
    """Enhanced spatial attention with channel attention integration"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        # Spatial attention branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Enhanced channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        sa = self.spatial_conv(x_ca)
        x_sa = x_ca * sa
        
        # Feature fusion
        output = self.fusion(x_sa)
        return output

class ASPPModule(nn.Module):
    """
    Enhanced Atrous Spatial Pyramid Pooling for multi-scale feature extraction
    Similar to DeepLabV3+ but integrated into GhanaSegNet architecture
    """
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPPModule, self).__init__()
        
        # Multiple dilation rates for multi-scale context
        self.aspp_blocks = nn.ModuleList()
        dilations = [1] + rates  # Include 1x1 conv
        
        for dilation in dilations:
            if dilation == 1:
                # 1x1 convolution
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Dilated convolution
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.aspp_blocks.append(block)
        
        # Global average pooling branch
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels * (len(dilations) + 1)  # +1 for global pooling
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply ASPP blocks
        aspp_features = []
        for aspp_block in self.aspp_blocks:
            aspp_features.append(aspp_block(x))
        
        # Global average pooling
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        aspp_features.append(global_feat)
        
        # Concatenate and project
        concat_feat = torch.cat(aspp_features, dim=1)
        output = self.project(concat_feat)
        
        return output

class EnhancedDecoderBlock(nn.Module):
    """
    Enhanced decoder block with improved feature fusion and attention
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(EnhancedDecoderBlock, self).__init__()
        
        # Skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Main path processing
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced attention
        self.attention = SpatialAttention(out_channels)
    
    def forward(self, x, skip):
        # Upsample main path to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.main_conv(x)
        
        # Process skip connection
        skip = self.skip_conv(skip)
        
        # Fuse features
        fused = torch.cat([x, skip], dim=1)
        fused = self.fusion(fused)
        
        # Apply enhanced attention
        output = self.attention(fused)
        
        return output

class GhanaSegNet(nn.Module):
    """
    GhanaSegNet Enhanced: Advanced Multi-Scale Transfer Learning Framework
    
    Enhanced Architecture Components:
    1. EfficientNet-B0 encoder with ImageNet pretraining
    2. ASPP module for multi-scale feature extraction (like DeepLabV3+)
    3. Enhanced transformer with 8 attention heads (vs 4 original)
    4. Enhanced spatial+channel attention mechanisms
    5. Improved decoder with progressive feature fusion
    6. Optimized for fair benchmarking performance
    
    Key Enhancements for Benchmarking:
    - Multi-scale context via ASPP competing with DeepLabV3+
    - Deeper transformer with better gradient flow and scaling
    - Enhanced attention combining spatial and channel mechanisms
    - Progressive decoder with better skip connection fusion
    - Optimized regularization and normalization
    """
    def __init__(self, num_classes=6, dropout=0.15):
        super(GhanaSegNet, self).__init__()
        
        # EfficientNet-B0 backbone (ImageNet pretrained)
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Channel reduction for transformer efficiency
        self.conv_reduce = nn.Conv2d(1280, 256, kernel_size=1)
        
        # Enhanced ASPP module for multi-scale features (DeepLabV3+ style)
        self.aspp = ASPPModule(256, 256)
        
        # Enhanced transformer block with 8 attention heads
        self.transformer = TransformerBlock(
            dim=256, 
            heads=8,  # Enhanced from 4 to 8 heads for better performance
            mlp_dim=512,  # Deeper MLP for better feature learning
            dropout=dropout
        )
        
        # Enhanced decoder blocks with correct EfficientNet-B0 channels
        # EfficientNet-B0 feature channels: 16, 24, 40, 112, 320
        self.dec4 = EnhancedDecoderBlock(256, 320, 256)  # Bottleneck -> Dec4 (block 15: 320 channels)
        self.dec3 = EnhancedDecoderBlock(256, 112, 128)  # Dec4 -> Dec3 (block 10: 112 channels) 
        self.dec2 = EnhancedDecoderBlock(128, 40, 64)    # Dec3 -> Dec2 (block 4: 40 channels)
        self.dec1 = EnhancedDecoderBlock(64, 24, 32)     # Dec2 -> Dec1 (block 2: 24 channels)
        
        # Final classification head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Initialize enhanced components
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for enhanced components"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Enhanced forward pass with ASPP, improved transformer, and progressive decoder
        """
        # Store input size for final upsampling
        input_size = x.shape[2:]
        
        # Extract multi-scale features from EfficientNet encoder
        features = []
        x_enc = self.encoder._conv_stem(x)
        x_enc = self.encoder._bn0(x_enc)
        x_enc = self.encoder._swish(x_enc)
        features.append(x_enc)  # 16 channels
        
        # Forward through encoder blocks and extract skip features
        for idx, block in enumerate(self.encoder._blocks):
            x_enc = block(x_enc)
            if idx in [2, 4, 10, 15]:  # Extract at key stages for skip connections
                features.append(x_enc)
        
        # Final encoder processing
        x_enc = self.encoder._conv_head(x_enc)
        x_enc = self.encoder._bn1(x_enc)
        x_enc = self.encoder._swish(x_enc)
        
        # Bottleneck processing with enhancements
        x_bottleneck = self.conv_reduce(x_enc)      # Channel reduction to 256
        x_bottleneck = self.aspp(x_bottleneck)      # Multi-scale features (ASPP)
        x_bottleneck = self.transformer(x_bottleneck)  # Enhanced transformer (8 heads)
        
        # Enhanced progressive decoder
        x = self.dec4(x_bottleneck, features[4])    # features[4]: 112 channels
        x = self.dec3(x, features[3])               # features[3]: 40 channels
        x = self.dec2(x, features[2])               # features[2]: 24 channels
        x = self.dec1(x, features[1])               # features[1]: 16 channels
        
        # Final classification
        x = self.final_conv(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
    
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
