#!/usr/bin/env python3

print("Testing imports...")

try:
    import sys
    import os
    sys.path.insert(0, '/workspaces/GhanaSegNet')
    
    print("1. Basic imports...")
    import torch
    from PIL import Image
    import numpy as np
    print("✅ Basic imports successful")
    
    print("2. Dataset loader import...")
    from data.dataset_loader import GhanaFoodDataset
    print("✅ GhanaFoodDataset imported successfully")
    
    print("3. Creating dataset instance...")
    dataset = GhanaFoodDataset('train')
    print(f"✅ Dataset created with {len(dataset)} samples")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
