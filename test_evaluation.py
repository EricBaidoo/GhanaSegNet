#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Starting UNet evaluation test...")

try:
    import torch
    print("✓ PyTorch imported successfully")
    
    # Test device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # Test UNet import
    from models.unet import UNet
    print("✓ UNet model imported successfully")
    
    # Create a simple UNet model
    model = UNet(num_classes=6)
    print("✓ UNet model created successfully")
    
    # Test if we can load a checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✓ Model checkpoint loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load checkpoint: {e}")
    else:
        print("⚠ No checkpoint found, using random weights")
    
    # Test with dummy data
    print("Testing with dummy data...")
    model.eval()
    
    # Create dummy input (batch_size=2, channels=3, height=256, width=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✓ Model forward pass successful. Output shape: {output.shape}")
        
        # Test prediction
        predictions = torch.argmax(output, dim=1)
        print(f"✓ Predictions shape: {predictions.shape}")
        
        # Simple accuracy calculation with dummy ground truth
        dummy_gt = torch.randint(0, 6, (2, 256, 256))
        correct_pixels = (predictions == dummy_gt).float().mean()
        print(f"✓ Dummy pixel accuracy: {correct_pixels:.4f}")
    
    print("\n🎉 UNet evaluation test completed successfully!")
    print("The model is ready for evaluation with real data.")
    
except Exception as e:
    print(f"❌ Error during evaluation test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
