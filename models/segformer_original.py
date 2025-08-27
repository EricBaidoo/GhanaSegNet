"""
Original SegFormer-B0 Implementation (Xie et al., 2021)
Clean implementation without pre-trained weights for fair baseline comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerOriginal(nn.Module):
    """
    Original SegFormer-B0 with random initialization
    No pre-trained weights for fair baseline comparison
    """
    def __init__(self, num_classes=6):
        super(SegFormerOriginal, self).__init__()
        
        # SegFormer-B0 configuration (from original paper)
        config = SegformerConfig(
            num_channels=3,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],  # B0 depths
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],  # B0 hidden sizes
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],  # B0 attention heads
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            initializer_range=0.02,
            drop_path_rate=0.1,
            layer_norm_eps=1e-6,
            decoder_hidden_size=256,
            num_labels=num_classes,
            semantic_loss_ignore_index=255,
        )
        
        # Initialize with random weights (no pre-training)
        self.model = SegformerForSemanticSegmentation(config)
        
        # Initialize weights randomly
        self.model.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights according to original paper"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # SegFormer expects pixel_values input
        outputs = self.model(pixel_values=x)
        return outputs.logits

# Alias for consistency
SegFormer = SegFormerOriginal
