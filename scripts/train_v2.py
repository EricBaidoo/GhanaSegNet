#!/usr/bin/env python3
"""
GhanaSegNet v2 Training Command
Simple wrapper for training the cultural intelligence model
Author: EricBaidoo
"""

import os
import sys
import subprocess
from datetime import datetime

def train_ghanasegnet_v2(epochs=80, batch_size=8, lr=1e-4):
    """
    Train GhanaSegNet v2 with optimized parameters for cultural intelligence
    
    Args:
        epochs (int): Number of training epochs (default: 80)
        batch_size (int): Batch size (default: 8)
        lr (float): Learning rate (default: 1e-4)
    """
    
    print("🇬🇭 GhanaSegNet v2.0 - Cultural Intelligence Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  • Model: GhanaSegNet v2 (Cultural Intelligence)")
    print(f"  • Epochs: {epochs}")
    print(f"  • Batch Size: {batch_size}")
    print(f"  • Learning Rate: {lr}")
    print(f"  • Loss Function: CulturalLoss (Dice + Focal + Tversky + Boundary)")
    print(f"  • Expected mIoU: 30-32% with superior cultural understanding")
    print("=" * 60)
    
    # Build the training command
    train_script = os.path.join(os.path.dirname(__file__), 'train_baselines.py')
    cmd = [
        sys.executable,
        train_script,
        '--model', 'ghanasegnet_v2',
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--lr', str(lr),
        '--num-classes', '6'
    ]
    
    print(f"🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 60)
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("📊 Check 'checkpoints/ghanasegnet_v2/' for saved models")
        print("📈 Training history saved in JSON format")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        print("🔍 Check the error messages above for troubleshooting")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def quick_test():
    """Quick test training with minimal epochs"""
    print("🧪 Quick Test Mode - Training for 3 epochs")
    return train_ghanasegnet_v2(epochs=3, batch_size=4, lr=1e-4)

def full_training():
    """Full training with optimized parameters"""
    print("🎯 Full Training Mode - Optimized for best results")
    return train_ghanasegnet_v2(epochs=80, batch_size=8, lr=1e-4)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GhanaSegNet v2 with cultural intelligence')
    parser.add_argument('--mode', type=str, choices=['test', 'full'], default='full',
                       help='Training mode: test (3 epochs) or full (80 epochs)')
    parser.add_argument('--epochs', type=int, help='Number of epochs (overrides mode)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.epochs:
        # Custom epoch count
        success = train_ghanasegnet_v2(args.epochs, args.batch_size, args.lr)
    elif args.mode == 'test':
        # Quick test
        success = quick_test()
    else:
        # Full training
        success = full_training()
    
    if success:
        print("\n🎉 GhanaSegNet v2 training completed!")
        print("🔍 Next steps:")
        print("  1. Check training results in checkpoints/")
        print("  2. Run evaluation on test set")
        print("  3. Compare with baseline models")
        sys.exit(0)
    else:
        print("\n💥 Training failed!")
        print("🔧 Troubleshooting tips:")
        print("  1. Check dataset is properly loaded")
        print("  2. Verify CUDA/GPU availability")
        print("  3. Ensure sufficient memory")
        sys.exit(1)