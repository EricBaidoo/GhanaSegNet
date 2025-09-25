"""
Quick Parameter Count Check for Model Comparison
"""

import torch
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

def count_parameters(model):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())

def quick_model_check():
    """Quick check of available models and their parameter counts"""
    
    try:
        from unet import UNet
        unet = UNet(n_channels=3, n_classes=6)
        unet_params = count_parameters(unet)
        print(f"UNet: {unet_params:,} parameters")
    except Exception as e:
        print(f"UNet error: {e}")
    
    try:
        from deeplabv3plus import DeepLabV3Plus
        deeplabv3 = DeepLabV3Plus(num_classes=6, backbone='resnet50')
        deeplabv3_params = count_parameters(deeplabv3)
        print(f"DeepLabV3+: {deeplabv3_params:,} parameters")
    except Exception as e:
        print(f"DeepLabV3+ error: {e}")
    
    try:
        from ghanasegnet import GhanaSegNet
        ghanasegnet = GhanaSegNet(num_classes=6)
        ghanasegnet_params = count_parameters(ghanasegnet)
        print(f"GhanaSegNet (Original): {ghanasegnet_params:,} parameters")
    except Exception as e:
        print(f"GhanaSegNet error: {e}")
    
    try:
        from ghanasegnet_v2 import GhanaSegNetV2
        ghanasegnet_v2 = GhanaSegNetV2(num_classes=6)
        v2_params = count_parameters(ghanasegnet_v2)
        print(f"GhanaSegNet v2: {v2_params:,} parameters")
        
        # Detailed breakdown for v2
        print(f"\nGhanaSegNet v2 breakdown:")
        for name, module in ghanasegnet_v2.named_children():
            module_params = count_parameters(module)
            print(f"  {name}: {module_params:,} parameters")
        
    except Exception as e:
        print(f"GhanaSegNet v2 error: {e}")

if __name__ == "__main__":
    quick_model_check()