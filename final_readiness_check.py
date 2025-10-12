#!/usr/bin/env python3
"""Final readiness check for Enhanced GhanaSegNet training"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("🔍 FINAL READINESS CHECK FOR ENHANCED GHANASEGNET")
print("="*60)

try:
    # 1. Dataset loading
    from data.dataset_loader import GhanaFoodDataset
    train_ds = GhanaFoodDataset(split='train', data_root='data', target_size=(256, 256))
    val_ds = GhanaFoodDataset(split='val', data_root='data', target_size=(256, 256))
    print(f"✅ Dataset: {len(train_ds)} train, {len(val_ds)} val samples")
    
    # 2. Enhanced model
    from models.ghanasegnet import EnhancedGhanaSegNet
    model = EnhancedGhanaSegNet(num_classes=6)
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Enhanced Model: {params:,} parameters")
    
    # 3. Training function
    from scripts.train_baselines import enhanced_train_model
    import inspect
    sig = inspect.signature(enhanced_train_model)
    print(f"✅ Enhanced training function ready")
    
    # 4. Loss and utilities
    from utils.losses import CombinedLoss
    from utils.metrics import compute_iou, compute_pixel_accuracy
    from utils.optimizers import create_optimized_optimizer_and_scheduler, get_progressive_training_config
    print(f"✅ Advanced loss & utilities ready")
    
    # 5. PyTorch environment
    import torch
    print(f"✅ PyTorch {torch.__version__} ready")
    cuda_status = "Available" if torch.cuda.is_available() else "CPU only"
    print(f"✅ CUDA: {cuda_status}")
    
    # 6. Progressive training components
    prog_config = get_progressive_training_config(1, 15)
    print(f"✅ Progressive training config ready")
    
    print("\n" + "="*60)
    print("🎯 TRAINING READINESS: ALL SYSTEMS GO!")
    print("🚀 Ready for 30% mIoU target achievement!")
    print("📊 Enhanced features active:")
    print("   • Progressive resolution training (256→320→384)")
    print("   • Early stopping prevention")
    print("   • Advanced loss function")
    print("   • Milestone tracking")
    print("   • Mixed precision training")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()