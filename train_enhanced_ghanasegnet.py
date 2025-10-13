#!/usr/bin/env python3
"""
Enhanced GhanaSegNet Training Script (Python Version)
Converted from Jupyter notebook for direct execution
"""

import sys
import os
import torch

print("🚀 Enhanced GhanaSegNet Training - Python Version")
print("="*60)

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def main():
    """Main training execution"""
    
    # Step 1: Verify environment
    print("📂 Working directory:", os.getcwd())
    
    # Step 2: Import training function
    try:
        from scripts.train_baselines import enhanced_train_model
        print("✅ Training function imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Step 3: Set paths (local version)
    dataset_path = "data"
    
    # Step 4: Verify dataset
    if os.path.exists(dataset_path):
        train_imgs = os.path.join(dataset_path, 'train', 'images')
        if os.path.exists(train_imgs):
            train_count = len(os.listdir(train_imgs))
            print(f"✅ Dataset verified: {train_count} training images")
        else:
            print(f"⚠️  Dataset structure may differ - proceeding anyway")
    else:
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print("\n🔄 Progressive Training Schedule:")
    print("   • Epochs 1-5:   256×256px (batch=8) - Foundation")
    print("   • Epochs 6-11:  320×320px (batch=6) - Enhancement") 
    print("   • Epochs 12-15: 384×384px (batch=4) - Maximum detail")
    print("   • Early stopping: Prevents overfitting after epoch 11")
    print("   • Target: 30% mIoU (up from 24.4% baseline)")

    # Step 5: Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔥 Using device: {device}")
    if device == 'cuda':
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🎬 Training starting...")

    # Step 6: Launch enhanced training
    try:
        training_results = enhanced_train_model(
            model_name='enhanced_ghanasegnet',
            dataset_path=dataset_path,
            epochs=15,                           # Progressive schedule
            batch_size=6,                        # Auto-adjusts: 8→6→4
            learning_rate=1.8e-4,               # Optimized
            weight_decay=1.5e-3,                # Enhanced regularization
            input_size=320,                     # Progressive: 256→320→384
            disable_early_stopping=False,       # Prevent overfitting
            use_advanced_augmentation=True,     # Better generalization
            device=device
        )

        # Step 7: Display results
        best_iou = training_results['best_val_iou']
        achieved_milestones = training_results['achieved_milestones']
        target_achieved = training_results['target_achieved']

        print("\n" + "="*60)
        print("🏆 TRAINING COMPLETE!")
        print(f"📊 Best mIoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
        print(f"🎯 Target: 30.00%")
        print(f"📈 Improvement: {(best_iou*100 - 24.4):+.2f} points from baseline")

        if target_achieved:
            print("🎉 🏆 TARGET ACHIEVED! 30%+ mIoU reached!")
            print(f"🎯 Milestones achieved: {achieved_milestones}")
        elif best_iou >= 0.27:
            print("🎉 EXCELLENT! Very close to target!")
            print(f"🎯 Milestones achieved: {achieved_milestones}")
        else:
            print("📊 Good progress - try TTA next for additional boost")
            if achieved_milestones:
                print(f"🎯 Milestones achieved: {achieved_milestones}")
        
        print(f"\n💾 Results saved to: checkpoints/enhanced_ghanasegnet/")
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()