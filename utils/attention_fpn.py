"""
Enhanced Feature Pyramid Network improvements for GhanaSegNet
Implements attention-based feature fusion for better performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFPN(nn.Module):
    """
    Attention-based Feature Pyramid Network for Enhanced GhanaSegNet
    Improves feature fusion with cross-scale attention mechanisms
    """
    def __init__(self, in_channels_list, out_channels=256):
        super(AttentionFPN, self).__init__()
        
        # Lateral convolutions for each scale
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])
        
        # Cross-scale attention modules
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)
            for _ in range(len(in_channels_list))
        ])
        
        # Feature refinement after attention
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list))
        ])
        
        # Global context module
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * len(in_channels_list), 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales [P2, P3, P4, P5]
        Returns:
            Enhanced multi-scale features
        """
        # Apply lateral convolutions
        lateral_features = []
        for i, feature in enumerate(features):
            lateral = self.lateral_convs[i](feature)
            lateral_features.append(lateral)
        
        # Top-down pathway with attention
        refined_features = []
        prev_feature = None
        
        for i in range(len(lateral_features) - 1, -1, -1):
            current = lateral_features[i]
            
            if prev_feature is not None:
                # Upsample previous feature to match current size
                prev_upsampled = F.interpolate(
                    prev_feature, size=current.shape[2:], 
                    mode='bilinear', align_corners=False
                )
                
                # Apply cross-scale attention
                B, C, H, W = current.shape
                current_flat = current.flatten(2).transpose(1, 2)  # (B, HW, C)
                prev_flat = prev_upsampled.flatten(2).transpose(1, 2)
                
                attended, _ = self.cross_attentions[i](current_flat, prev_flat, prev_flat)
                attended = attended.transpose(1, 2).view(B, C, H, W)
                
                # Combine with residual connection
                current = current + attended + prev_upsampled
            
            # Apply refinement
            refined = self.refinement_convs[i](current)
            refined_features.insert(0, refined)
            prev_feature = refined
        
        # Apply global context attention
        concat_features = torch.cat([
            F.adaptive_avg_pool2d(f, 1) for f in refined_features
        ], dim=1)
        
        global_weights = self.global_context(concat_features)
        global_weights = global_weights.view(
            global_weights.size(0), len(refined_features), -1, 1, 1
        )
        
        # Apply global attention weights
        for i, feature in enumerate(refined_features):
            refined_features[i] = feature * global_weights[:, i]
        
        return refined_features


class EnhancedCrossScaleFusion(nn.Module):
    """
    Cross-scale feature fusion with learnable attention weights
    """
    def __init__(self, channels):
        super(EnhancedCrossScaleFusion, self).__init__()
        
        self.scale_attention = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4, 1),
            nn.Softmax(dim=1)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, features):
        """Fuse multi-scale features with attention"""
        # Resize all features to the same size (largest)
        target_size = features[0].shape[2:]
        resized_features = []
        
        for feature in features:
            if feature.shape[2:] != target_size:
                resized = F.interpolate(
                    feature, size=target_size, 
                    mode='bilinear', align_corners=False
                )
            else:
                resized = feature
            resized_features.append(resized)
        
        # Concatenate all features
        concat_features = torch.cat(resized_features, dim=1)
        
        # Compute attention weights
        attention_weights = self.scale_attention(concat_features)
        
        # Apply attention weights to individual features
        weighted_features = []
        for i, feature in enumerate(resized_features):
            weighted = feature * attention_weights[:, i:i+1]
            weighted_features.append(weighted)
        
        # Final fusion
        final_concat = torch.cat(weighted_features, dim=1)
        output = self.feature_fusion(final_concat)
        
        return output