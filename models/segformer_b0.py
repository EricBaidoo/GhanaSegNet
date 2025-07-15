import torch
import torch.nn as nn
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

class SegFormerB0(nn.Module):
    def __init__(self, num_classes=6):
        super(SegFormerB0, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # allows replacing final head
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits
