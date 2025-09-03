#!/usr/bin/env python3
"""
Pre-training validation script for GhanaSegNet baseline models
Tests all components before starting actual training
"""

import sys
import os
sys.path.append('.')

def test_environment():
    """Test the training environment"""
    print("🔍 TESTING TRAINING ENVIRONMENT")
    print("=" * 50)
    
    # Test 1: Basic imports
    try:
        import torch
        import torch.nn as nn
        import torchvision
        import numpy as np
        from PIL import Image
        print("✅ Basic dependencies: PyTorch, NumPy, PIL")
    except Exception as e:
        print(f"❌ Basic dependencies failed: {e}")
        return False
    
    # Test 2: Device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    # Test 3: Model imports
    try:
        from models.unet import UNetOriginal
        from models.deeplabv3plus import DeepLabV3Plus
        print("✅ UNet and DeepLabV3+ imported")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    # Test 4: Model creation
    try:
        unet = UNetOriginal(n_channels=3, n_classes=6)
        deeplabv3 = DeepLabV3Plus(num_classes=6)
        print("✅ Models created successfully")
        
        # Count parameters
        unet_params = sum(p.numel() for p in unet.parameters())
        deeplab_params = sum(p.numel() for p in deeplabv3.parameters())
        print(f"📊 UNet parameters: {unet_params:,}")
        print(f"📊 DeepLabV3+ parameters: {deeplab_params:,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 5: Dataset structure
    try:
        data_dirs = ['data/train', 'data/val', 'data/test']
        for dir_path in data_dirs:
            if os.path.exists(dir_path):
                subdirs = os.listdir(dir_path)
                print(f"✅ {dir_path}: {subdirs}")
            else:
                print(f"❌ Missing: {dir_path}")
                return False
    except Exception as e:
        print(f"❌ Dataset check failed: {e}")
        return False
    
    # Test 6: Utilities
    try:
        from utils.losses import CombinedLoss
        from utils.metrics import compute_iou, compute_pixel_accuracy
        criterion = CombinedLoss(alpha=0.8)
        print("✅ Loss and metrics imported")
    except Exception as e:
        print(f"❌ Utilities import failed: {e}")
        return False
    
    # Test 7: Forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            unet_output = unet(dummy_input)
            deeplab_output = deeplabv3(dummy_input)
            print(f"✅ Forward pass: UNet {dummy_input.shape} -> {unet_output.shape}")
            print(f"✅ Forward pass: DeepLab {dummy_input.shape} -> {deeplab_output.shape}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED - READY FOR TRAINING!")
    return True

def show_training_commands():
    """Show available training commands"""
    print("\n🚀 TRAINING COMMANDS:")
    print("=" * 50)
    print("Train individual models:")
    print("  python3 scripts/train_baselines.py --model unet --epochs 30")
    print("  python3 scripts/train_baselines.py --model deeplabv3plus --epochs 30")
    print("  python3 scripts/train_baselines.py --model segformer --epochs 30")
    print("\nTrain all models:")
    print("  python3 scripts/train_baselines.py --model all --epochs 30")
    print("\nCustom configuration:")
    print("  python3 scripts/train_baselines.py --model unet --epochs 50 --batch-size 4 --lr 1e-4")

if __name__ == "__main__":
    if test_environment():
        show_training_commands()
    else:
        print("❌ Environment test failed. Please fix issues before training.")
        sys.exit(1)
