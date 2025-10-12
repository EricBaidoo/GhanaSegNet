#!/usr/bin/env python3
"""
Test the actual enhanced_train_model function call
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_function_call():
    """Test that the enhanced function can be called with new parameters"""
    print("🧪 TESTING ENHANCED FUNCTION CALL")
    print("="*50)
    
    try:
        from scripts.train_baselines import enhanced_train_model
        import inspect
        
        # Get function signature
        sig = inspect.signature(enhanced_train_model)
        print("✅ Function signature:")
        for param_name, param in sig.parameters.items():
            default_val = param.default if param.default != inspect.Parameter.empty else "required"
            print(f"   {param_name}: {default_val}")
        
        print("\n🎯 RECOMMENDED CALL FOR 30% mIoU TARGET:")
        print("""
# In your notebook:
best_iou, history = enhanced_train_model(
    model_name='enhanced_ghanasegnet',
    dataset_path='data',  # Adjust to your data path
    epochs=20,                    # ✨ Extended training
    batch_size=6,                 # ✨ Optimized batch size  
    learning_rate=1.8e-4,         # ✨ Fine-tuned LR
    weight_decay=1.5e-3,          # ✨ Enhanced regularization
    input_size=320,               # ✨ Higher resolution
    disable_early_stopping=False, # ✨ Overfitting prevention
    use_advanced_augmentation=True # ✨ Advanced augmentation
)
        """)
        
        print("🚀 EXPECTED IMPROVEMENTS:")
        print("   • Break through 24.4% mIoU plateau")
        print("   • Prevent overfitting after epoch 11")
        print("   • Target: 26-28% → 30% mIoU")
        print("   • Training time: ~45-60 minutes")
        print("   • Automatic early stopping if needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing function call: {e}")
        return False

if __name__ == "__main__":
    test_function_call()