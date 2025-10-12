"""
Quick test of Enhanced GhanaSegNet training setup
Verify all components work before Colab execution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_training_setup():
    """Test that enhanced training components are ready"""
    print("🧪 TESTING ENHANCED TRAINING SETUP")
    print("="*50)
    
    try:
        # Test model import
        from models.ghanasegnet import EnhancedGhanaSegNet
        model = EnhancedGhanaSegNet(num_classes=6)
        print("✅ Enhanced GhanaSegNet import successful")
        
        # Test optimizer import
        from utils.optimizers import create_optimized_optimizer_and_scheduler
        config = {'learning_rate': 2.5e-4, 'weight_decay': 1.2e-3, 'epochs': 15}
        optimizer, scheduler = create_optimized_optimizer_and_scheduler(model, config)
        print("✅ Optimized optimizer and scheduler created")
        
        # Test loss function
        from utils.losses import CombinedLoss
        criterion = CombinedLoss(alpha=0.6, aux_weight=0.4, adaptive_weights=True)
        print("✅ Advanced combined loss function ready")
        
        # Test enhanced training function import
        from scripts.train_baselines import enhanced_train_model
        print("✅ Enhanced training function imported successfully")
        
        print("\n🎉 ALL COMPONENTS READY FOR COLAB EXECUTION!")
        print("🚀 Proceed to Colab for the ambitious 15-epoch training!")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_enhanced_training_setup()
    if success:
        print(f"\n🎯 READY TO ACHIEVE 30% mIoU!")
        print(f"📋 Next steps:")
        print(f"   1. Upload project to Google Colab")
        print(f"   2. Mount Google Drive with dataset")
        print(f"   3. Run the enhanced training cell")
        print(f"   4. Monitor progress toward 30% mIoU target!")
    else:
        print(f"\n🛠️ Fix issues before Colab execution")