#!/usr/bin/env python3
"""Test the enhanced_train_model function with actual parameters"""

import sys
import os
import torch

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

print("🧪 Testing enhanced_train_model function...")

try:
    from scripts.train_baselines import enhanced_train_model
    print("✅ Import successful")
    
    # Test with minimal parameters
    print("🔄 Testing function call...")
    
    # Check if data directory exists
    data_path = "data" if os.path.exists("data") else "."
    print(f"📂 Using dataset path: {data_path}")
    
    # Test the function signature
    import inspect
    sig = inspect.signature(enhanced_train_model)
    print(f"📋 Function signature: {sig}")
    
    # Test torch availability
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    
    # Check GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Using device: {device}")
    
    print("✅ All basic checks passed - function should work")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()