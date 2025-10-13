#!/usr/bin/env python3
"""
Simple notebook test to identify execution issues
"""

print("🧪 NOTEBOOK EXECUTION DIAGNOSTIC")
print("="*50)

# Test 1: Basic Python execution
try:
    import sys
    import os
    print("✅ Basic Python imports work")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")

# Test 2: Jupyter/Colab detection
try:
    if 'google.colab' in sys.modules:
        print("✅ Running in Google Colab")
        environment = "colab"
    elif 'ipykernel' in sys.modules:
        print("✅ Running in Jupyter/IPython")
        environment = "jupyter"
    else:
        print("⚠️  Running in regular Python (not notebook)")
        environment = "python"
except Exception as e:
    print(f"❌ Environment detection failed: {e}")
    environment = "unknown"

# Test 3: PyTorch availability
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} available")
    print(f"✅ CUDA: {'Available' if torch.cuda.is_available() else 'CPU only'}")
except ImportError:
    print("❌ PyTorch not available - need to install")
except Exception as e:
    print(f"❌ PyTorch error: {e}")

# Test 4: Required packages
required_packages = [
    ('efficientnet_pytorch', 'EfficientNet'),
    ('tqdm', 'Progress bars'),
    ('PIL', 'Image processing'),
]

for package, description in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} ({description}) available")
    except ImportError:
        print(f"❌ {package} ({description}) missing - need to install")
    except Exception as e:
        print(f"⚠️  {package} issue: {e}")

# Test 5: Working directory and paths
try:
    cwd = os.getcwd()
    print(f"✅ Working directory: {cwd}")
    
    # Check if we're in the right place
    if 'GhanaSegNet' in cwd or os.path.exists('models'):
        print("✅ In GhanaSegNet directory")
    else:
        print("⚠️  May not be in correct directory")
        
    # Check key files exist
    key_files = [
        'models/ghanasegnet.py',
        'scripts/train_baselines.py',
        'data/dataset_loader.py'
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            
except Exception as e:
    print(f"❌ Directory check failed: {e}")

print("\n📋 DIAGNOSIS SUMMARY:")
print(f"Environment: {environment}")

if environment == "python":
    print("\n💡 SOLUTION: This notebook needs to run in:")
    print("   • Google Colab (recommended)")
    print("   • Jupyter Notebook")
    print("   • VS Code with Jupyter extension")
    print("\n🚀 TO RUN IN COLAB:")
    print("   1. Upload notebook to Google Drive")
    print("   2. Open with Google Colab")
    print("   3. Run cells sequentially")
elif environment == "colab":
    print("\n✅ Colab environment detected - notebook should work")
elif environment == "jupyter":
    print("\n✅ Jupyter environment detected - notebook should work")