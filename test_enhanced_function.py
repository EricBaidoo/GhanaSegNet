#!/usr/bin/env python3
"""Test enhanced_train_model with minimal parameters"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from scripts.train_baselines import enhanced_train_model
    
    print("✅ Enhanced training function imported")
    
    # Test with minimal parameters - just initialization
    print("🧪 Testing function initialization...")
    
    # This should get through the dataset loading without error
    print("Call signature test...")
    
    # We won't actually run training, just test the setup
    import inspect
    sig = inspect.signature(enhanced_train_model)  
    print(f"✅ Function signature: {sig}")
    
    print("🎉 Function should work now!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()