#!/usr/bin/env python3
"""Test the dataset loading fix"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from data.dataset_loader import GhanaFoodDataset
    
    print("✅ Dataset class imported")
    
    # Test the constructor calls that were failing
    dataset_path = "data"
    
    # Test the fixed calls
    print("🧪 Testing fixed dataset creation...")
    
    train_dataset = GhanaFoodDataset(split='train', data_root=dataset_path, target_size=(256, 256))
    val_dataset = GhanaFoodDataset(split='val', data_root=dataset_path, target_size=(256, 256))
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")
    
    print("🎉 Dataset loading fix successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()