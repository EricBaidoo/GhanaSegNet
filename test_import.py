#!/usr/bin/env python3
"""Test import of enhanced_train_model function"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from scripts.train_baselines import enhanced_train_model
    print("✅ enhanced_train_model imported successfully")
    print(f"Function type: {type(enhanced_train_model)}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Checking available modules...")
    try:
        import scripts.train_baselines as tb
        print(f"Available functions: {[attr for attr in dir(tb) if not attr.startswith('_')]}")
    except Exception as e2:
        print(f"Cannot import train_baselines: {e2}")
except Exception as e:
    print(f"❌ Other error: {e}")

print(f"Current directory: {os.getcwd()}")
print(f"Python path includes:")
for p in sys.path:
    print(f"  {p}")