#!/usr/bin/env python3
"""
GhanaSegNet v2 Import and Basic Functionality Test
Tests model instantiation, forward pass, and parameter counting
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.ghanasegnet_v2 import GhanaSegNetV2
from utils.losses import CulturalLoss

def test_v2_model():
    """Test GhanaSegNet v2 model instantiation and basic operations"""
    
    print("🇬🇭 GhanaSegNet v2 - Import and Functionality Test")
    print("=" * 55)
    
    try:
        # Test model instantiation
        print("1. Testing model instantiation...")
        model = GhanaSegNetV2(num_classes=6)
        print("   ✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 384, 384)  # Batch=1, RGB, 384x384
            print(f"   📥 Input shape: {dummy_input.shape}")
            
            output = model(dummy_input)
            print(f"   📤 Output shape: {output.shape}")
            print(f"   ✅ Forward pass successful")
            
            # Check output dimensions
            expected_shape = (1, 6, 384, 384)  # Batch, Classes, Height, Width
            if output.shape == expected_shape:
                print(f"   ✅ Output shape correct: {output.shape}")
            else:
                print(f"   ❌ Output shape incorrect: got {output.shape}, expected {expected_shape}")
        
        # Test loss function
        print("\n3. Testing CulturalLoss...")
        loss_fn = CulturalLoss()
        dummy_target = torch.randint(0, 6, (1, 384, 384))  # Random segmentation mask
        
        model.train()
        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        print(f"   📊 Loss value: {loss.item():.4f}")
        print(f"   ✅ Loss computation successful")
        
        # Test backward pass
        print("\n4. Testing backward pass...")
        loss.backward()
        print("   ✅ Backward pass successful")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ GhanaSegNet v2 is ready for training!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cultural_components():
    """Test specific cultural intelligence components"""
    
    print(f"\n🎭 Cultural Intelligence Components Test")
    print("=" * 45)
    
    try:
        model = GhanaSegNetV2(num_classes=6)
        
        # Check if cultural attention modules exist
        has_cultural_attention = False
        has_transformer = False
        has_multiscale_fusion = False
        
        for name, module in model.named_modules():
            if 'cultural' in name.lower():
                has_cultural_attention = True
                print(f"   🎯 Found cultural component: {name}")
            if 'transformer' in name.lower():
                has_transformer = True
                print(f"   🔄 Found transformer component: {name}")
            if 'fusion' in name.lower():
                has_multiscale_fusion = True
                print(f"   🔗 Found fusion component: {name}")
        
        print(f"\n📊 Cultural Intelligence Assessment:")
        print(f"   Cultural Attention: {'✅' if has_cultural_attention else '❌'}")
        print(f"   Transformer Blocks: {'✅' if has_transformer else '❌'}")
        print(f"   Multi-scale Fusion: {'✅' if has_multiscale_fusion else '❌'}")
        
        if has_cultural_attention and has_transformer and has_multiscale_fusion:
            print(f"   🏆 Full cultural intelligence capability confirmed!")
            return True
        else:
            print(f"   ⚠️ Some cultural components may be missing")
            return False
            
    except Exception as e:
        print(f"❌ Cultural components test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 GhanaSegNet v2 Comprehensive Testing")
    print("=" * 60)
    
    # Run basic tests
    basic_test_passed = test_v2_model()
    
    # Run cultural components test
    cultural_test_passed = test_cultural_components()
    
    print(f"\n📋 TEST SUMMARY")
    print("=" * 25)
    print(f"Basic Functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"Cultural Components: {'✅ PASS' if cultural_test_passed else '❌ FAIL'}")
    
    if basic_test_passed and cultural_test_passed:
        print(f"\n🎉 GhanaSegNet v2 is fully ready for training!")
        print(f"🚀 Next steps:")
        print(f"   1. Run: python scripts/train_v2.py --mode test")
        print(f"   2. Or use: test_v2.bat (Windows)")
        print(f"   3. For full training: train_v2.bat")
    else:
        print(f"\n⚠️ Some tests failed. Please check the errors above.")
        
    print(f"\n{'=' * 60}")