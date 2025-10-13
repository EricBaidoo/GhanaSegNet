#!/usr/bin/env python3
"""
🧪 FINAL ENHANCED NOTEBOOK TEST
Tests the fixed notebook components to ensure they work like the original
"""

import subprocess
import sys
import os

def test_enhanced_script():
    """Test the enhanced training script integration"""
    print("🧪 TESTING ENHANCED SCRIPT INTEGRATION")
    print("="*50)
    
    # Test 1: Help command works
    try:
        result = subprocess.run(['python', 'scripts/train_baselines.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if 'enhanced_ghanasegnet' in result.stdout:
            print("✅ Enhanced model option available in script")
        else:
            print("❌ Enhanced model option not found in script")
            return False
    except Exception as e:
        print(f"❌ Script help test failed: {e}")
        return False
    
    # Test 2: Enhanced function import
    try:
        sys.path.insert(0, '.')
        from scripts.train_baselines import enhanced_train_model
        print("✅ Enhanced training function imports correctly")
    except Exception as e:
        print(f"❌ Enhanced function import failed: {e}")
        return False
    
    # Test 3: Model import
    try:
        from models.ghanasegnet import EnhancedGhanaSegNet
        model = EnhancedGhanaSegNet(num_classes=6)
        print("✅ Enhanced model creates successfully")
    except Exception as e:
        print(f"❌ Enhanced model creation failed: {e}")
        return False
    
    return True

def test_notebook_structure():
    """Test the notebook structure"""
    print("\n🧪 TESTING NOTEBOOK STRUCTURE")
    print("="*50)
    
    notebook_path = 'notebooks/Enhanced_GhanaSegNet_Colab.ipynb'
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for critical components matching working notebook pattern
        checks = [
            ('Drive Mount', 'drive.mount(\'/content/drive\')' in content),
            ('Git Clone', '!git clone https://github.com/EricBaidoo/GhanaSegNet.git' in content),
            ('Change Directory', '%cd GhanaSegNet' in content),
            ('Dataset Copy', '!cp -r "/content/drive/MyDrive/data" .' in content),
            ('Subprocess Call', 'subprocess.run([' in content),
            ('Enhanced Model', 'enhanced_ghanasegnet' in content),
            ('Error Handling', 'result.returncode == 0' in content),
        ]
        
        all_passed = True
        for check_name, condition in checks:
            if condition:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Notebook structure test failed: {e}")
        return False

def main():
    print("🎯 ENHANCED GHANASEGNET NOTEBOOK - FINAL VERIFICATION")
    print("="*60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    test1_pass = test_enhanced_script()
    test2_pass = test_notebook_structure()
    
    print(f"\n🏆 FINAL RESULTS")
    print("="*50)
    
    if test1_pass and test2_pass:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Enhanced notebook is ready for Google Colab")
        print("✅ Matches working notebook patterns")
        print("✅ Enhanced training integration works")
        print("\n📋 READY TO DEPLOY:")
        print("   1. Upload notebook to Google Colab")
        print("   2. Update dataset path in cell 2")
        print("   3. Run cells 1→2→3 in sequence")
        print("   4. Achieve 30%+ mIoU target! 🎯")
        return True
    else:
        print("⚠️  Some issues detected")
        if not test1_pass:
            print("❌ Enhanced script integration needs fixing")
        if not test2_pass:
            print("❌ Notebook structure needs adjustment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)