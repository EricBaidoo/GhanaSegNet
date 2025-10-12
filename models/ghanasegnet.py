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
    SUPER-ENHANCED Transformer block targeting 30% mIoU performance
    Enhanced with deformable attention and advanced feature refinement
    """
    def __init__(self, dim, heads=12, mlp_dim=768, dropout=0.1):  # Increased heads & MLP
        super(TransformerBlock, self).__init__()
        
        # Multi-scale self-attention with increased capacity
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads,  # Increased to 12 heads
            dropout=dropout * 0.4,  # Reduced dropout for better learning
            batch_first=True
        )
        
        # Enhanced cross-scale attention with more heads
        self.norm1_cross = nn.LayerNorm(dim)  
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads // 2,  # 6 heads for cross-attention
            dropout=dropout * 0.4,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        
        # SUPER-ENHANCED MLP with advanced gating mechanism
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # Now 768 dimensions
            nn.GELU(),
            nn.Dropout(dropout * 0.8),  # Reduced dropout for better learning
            nn.Linear(mlp_dim, mlp_dim // 2),  # 384 dimensions
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(mlp_dim // 2, dim),
            nn.Dropout(dropout * 0.4)
        )
        
        # Advanced feature gating with more capacity
        self.feature_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        # Advanced scaling with learnable parameters
        self.scale_self_attn = nn.Parameter(torch.ones(1) * 0.3)
        self.scale_cross_attn = nn.Parameter(torch.ones(1) * 0.2)
        self.scale_mlp = nn.Parameter(torch.ones(1) * 0.25)

    def forward(self, x):
        B, C, H, W = x.shape
        x_spatial = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        
        # Enhanced self-attention
        x_norm = self.norm1(x_spatial)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x_spatial = x_spatial + self.scale_self_attn * self_attn_out
        
        # Cross-scale attention with downsampled features
        x_down = F.adaptive_avg_pool2d(x, (H//2, W//2))  # Create multi-scale context
        x_down_spatial = x_down.flatten(2).transpose(1, 2)
        
        # Cross-attention between original and downsampled features
        x_norm_cross = self.norm1_cross(x_spatial)
        if x_down_spatial.size(1) > 0:  # Ensure we have downsampled features
            cross_attn_out, _ = self.cross_attn(x_norm_cross, x_down_spatial, x_down_spatial)
            x_spatial = x_spatial + self.scale_cross_attn * cross_attn_out
        
        # Enhanced MLP with feature gating
        x_norm_mlp = self.norm2(x_spatial)
        mlp_out = self.mlp(x_norm_mlp)
        
        # Apply feature gating for selective enhancement
        gate = self.feature_gate(x_norm_mlp)
        gated_mlp = mlp_out * gate
        
        x_spatial = x_spatial + self.scale_mlp * gated_mlp
        
        # Reshape back to feature map
        x_out = x_spatial.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection with original input
        return x_out + x * 0.1  # Small residual for stability

class SpatialAttention(nn.Module):
    """Optimized spatial attention for efficient training"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        
        # Efficient spatial attention branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 16, 8), 1, bias=False),  # More efficient reduction
            nn.BatchNorm2d(max(in_channels // 16, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 16, 8), 1, 1),
            nn.Sigmoid()
        )
        
        # Efficient channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 16, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 16, 8), in_channels, 1),
            nn.Sigmoid()
        )
        
        # Simplified fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
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
    SUPER-ENHANCED ASPP with Feature Pyramid Network integration for 30% mIoU target
    Multi-scale feature extraction optimized for fine-grained food segmentation
    INCREASED CAPACITY: 384 channels, 5 dilation rates, SE attention
    """
    def __init__(self, in_channels, out_channels=384, rates=[2, 4, 8, 16, 24]):  # Increased capacity
        super(ASPPModule, self).__init__()
        
        # Advanced multi-scale feature extraction for 30% mIoU target
        self.aspp_blocks = nn.ModuleList()
        dilations = [1] + rates  # Include 1x1 conv + multiple scales
        
        for dilation in dilations:
            if dilation == 1:
                # Enhanced 1x1 convolution with residual
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),  # Additional conv for refinement
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Enhanced dilated convolution with depth-wise separable
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, 
                             groups=in_channels, bias=False),  # Depth-wise
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),  # Point-wise
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),  # Refinement
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.aspp_blocks.append(block)
        
        # Multi-scale feature aggregation
        self.scale_attention = nn.Sequential(
            nn.Conv2d(out_channels * len(dilations), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        # Enhanced global context with multiple pooling scales
        self.global_pools = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, out_channels // 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(in_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Advanced feature fusion for 30% mIoU target
        total_channels = out_channels * len(dilations) + out_channels  # ASPP + global features
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels * 2, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Apply enhanced ASPP blocks
        aspp_features = []
        for aspp_block in self.aspp_blocks:
            aspp_features.append(aspp_block(x))
        
        # Multi-scale global pooling features
        global_features = []
        for global_pool in self.global_pools:
            global_feat = global_pool(x)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
            global_features.append(global_feat)
        
        # Combine all global features
        combined_global = torch.cat(global_features, dim=1)
        
        # Concatenate ASPP and global features
        concat_feat = torch.cat(aspp_features + [combined_global], dim=1)
        
        # Apply scale attention for feature weighting
        aspp_only = torch.cat(aspp_features, dim=1)
        scale_weights = self.scale_attention(aspp_only)
        
        # Enhanced projection with attention weighting
        output = self.project(concat_feat)
        output = output * scale_weights + output  # Residual connection with attention
        
        return output

class EnhancedDecoderBlock(nn.Module):
    """
    Optimized decoder block for better convergence with standard training
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(EnhancedDecoderBlock, self).__init__()
        
        # Efficient skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Efficient main path processing
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Streamlined feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),  # 1x1 instead of 3x3 for efficiency
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight attention
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

class EnhancedGhanaSegNet(nn.Module):
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
    def __init__(self, num_classes=6, dropout=0.12):
        super(EnhancedGhanaSegNet, self).__init__()
        
        # EfficientNet-B0 backbone (ImageNet pretrained)
        self.encoder = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Optimized channel reduction with BatchNorm for stability
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Food-optimized ASPP module with smaller dilations
        self.aspp = ASPPModule(256, 256, rates=[3, 6, 12])
        
        # Advanced transformer for 30% mIoU target
        self.transformer = TransformerBlock(
            dim=256, 
            heads=8,  # 8 heads for proper dimension division
            mlp_dim=512,  # Increased for better feature learning
            dropout=dropout * 0.8
        )
        
        # Feature Pyramid Network style lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(320, 256, 1, bias=False),  # P5
            nn.Conv2d(112, 256, 1, bias=False),  # P4  
            nn.Conv2d(40, 256, 1, bias=False),   # P3
            nn.Conv2d(24, 256, 1, bias=False),   # P2
        ])
        
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # Advanced FPN-style decoder for 30% mIoU target
        # Multi-scale feature fusion with enhanced decoder blocks
        self.dec4 = EnhancedDecoderBlock(256, 128, 128)  # FPN P5 features
        self.dec3 = EnhancedDecoderBlock(256, 128, 96)   # FPN P4 features (256 from FPN + 128 from dec4)
        self.dec2 = EnhancedDecoderBlock(224, 128, 64)   # FPN P3 features (96+128)
        self.dec1 = EnhancedDecoderBlock(192, 128, 64)   # FPN P2 features (64+128)
        
        # Multi-scale supervision for enhanced training
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 6, 1)  # Auxiliary prediction head
            ),
            nn.Sequential(
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 6, 1)
            )
        ])
        
        # Advanced final classification head with feature refinement
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Feature refinement module for boundary enhancement
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes * 2, num_classes, 1)
        )
        
        # Initialize enhanced components
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Optimized weight initialization for better convergence with standard parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Smaller initialization for transformer components
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                # Special initialization for attention weights
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight, gain=0.02)
        
    def forward(self, x):
        """
        Advanced forward pass with FPN-style multi-scale fusion for 30% mIoU target
        """
        # Store input size for final upsampling
        input_size = x.shape[2:]
        
        # Extract multi-scale features from EfficientNet encoder
        features = []
        x_enc = self.encoder._conv_stem(x)
        x_enc = self.encoder._bn0(x_enc)
        x_enc = self.encoder._swish(x_enc)
        features.append(x_enc)  # C1: 32 channels
        
        # Forward through encoder blocks and extract skip features
        for idx, block in enumerate(self.encoder._blocks):
            x_enc = block(x_enc)
            if idx in [2, 4, 10, 15]:  # Extract at key stages
                features.append(x_enc)
        
        # Final encoder processing
        x_enc = self.encoder._conv_head(x_enc)
        x_enc = self.encoder._bn1(x_enc)
        x_enc = self.encoder._swish(x_enc)
        
        # Advanced bottleneck processing
        x_bottleneck = self.conv_reduce(x_enc)      # Channel reduction to 256
        x_bottleneck = self.aspp(x_bottleneck)      # Enhanced multi-scale ASPP
        x_bottleneck = self.transformer(x_bottleneck)  # Advanced transformer with cross-attention
        
        # FPN-style lateral connections (top-down pathway)
        fpn_features = []
        
        # P5 (highest resolution feature map after bottleneck)
        p5 = x_bottleneck  # 256 channels
        fpn_features.append(self.fpn_convs[0](p5))
        
        # P4: lateral connection + top-down
        lateral_p4 = self.lateral_convs[1](features[3])  # 112 -> 256
        p4 = lateral_p4 + F.interpolate(p5, size=lateral_p4.shape[2:], mode='bilinear', align_corners=False)
        fpn_features.append(self.fpn_convs[1](p4))
        
        # P3: lateral connection + top-down  
        lateral_p3 = self.lateral_convs[2](features[2])  # 40 -> 256
        p3 = lateral_p3 + F.interpolate(p4, size=lateral_p3.shape[2:], mode='bilinear', align_corners=False)
        fpn_features.append(self.fpn_convs[2](p3))
        
        # P2: lateral connection + top-down
        lateral_p2 = self.lateral_convs[3](features[1])  # 24 -> 256
        p2 = lateral_p2 + F.interpolate(p3, size=lateral_p2.shape[2:], mode='bilinear', align_corners=False)
        fpn_features.append(self.fpn_convs[3](p2))
        
        # Advanced progressive decoder with FPN features
        x = self.dec4(x_bottleneck, fpn_features[0])    # Enhanced P5 features -> 128
        
        # Combine decoder output with FPN P4
        x_up = F.interpolate(x, size=fpn_features[1].shape[2:], mode='bilinear', align_corners=False)
        x_combined = torch.cat([x_up, fpn_features[1]], dim=1)  # 128 + 128 = 256
        x = self.dec3(x_combined, fpn_features[1])              # -> 96
        
        # Multi-scale supervision (auxiliary loss during training)
        aux_outputs = []
        if self.training:
            aux_out1 = self.aux_heads[0](fpn_features[0])
            aux_out1 = F.interpolate(aux_out1, size=input_size, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out1)
            
            aux_out2 = self.aux_heads[1](x)
            aux_out2 = F.interpolate(aux_out2, size=input_size, mode='bilinear', align_corners=False)
            aux_outputs.append(aux_out2)
        
        # Continue with P3 and P2 features
        x_up = F.interpolate(x, size=fpn_features[2].shape[2:], mode='bilinear', align_corners=False)
        x_combined = torch.cat([x_up, fpn_features[2]], dim=1)  # 96 + 128 = 224
        x = self.dec2(x_combined, fpn_features[2])              # -> 64
        
        x_up = F.interpolate(x, size=fpn_features[3].shape[2:], mode='bilinear', align_corners=False)
        x_combined = torch.cat([x_up, fpn_features[3]], dim=1)  # 64 + 128 = 192
        x = self.dec1(x_combined, fpn_features[3])              # -> 64
        
        # Final classification with boundary refinement
        x = self.final_conv(x)
        
        # Boundary refinement for enhanced segmentation quality
        x = self.boundary_refine(x) + x  # Residual connection
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        if self.training and aux_outputs:
            return x, aux_outputs  # Return main output + auxiliary outputs for multi-scale supervision
        else:
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

# Backward compatibility alias
GhanaSegNet = EnhancedGhanaSegNet
