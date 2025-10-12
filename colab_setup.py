"""
Colab Setup Cell for Enhanced GhanaSegNet Training
Copy this code to the beginning of your Colab notebook
"""

# ===============================
# COLAB SETUP CELL - RUN FIRST
# ===============================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Google Drive to local Colab storage
print("📂 Copying dataset from Google Drive...")
import os
import shutil

# Copy data from your Google Drive
source_path = "/content/drive/MyDrive/data"
target_path = "/content/data"

if os.path.exists(source_path):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)  # Remove existing data
    shutil.copytree(source_path, target_path)
    print(f"✅ Dataset copied from {source_path} to {target_path}")
else:
    print(f"❌ Source path not found: {source_path}")
    print("💡 Please check your Google Drive path")

# Verify dataset structure
print("\n📊 Verifying dataset structure...")
if os.path.exists('/content/data'):
    print("Dataset structure:")
    os.system("find /content/data -type f -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' | head -10")
    
    # Count files
    train_images = "/content/data/train/images"
    train_masks = "/content/data/train/masks"
    val_images = "/content/data/val/images"
    val_masks = "/content/data/val/masks"
    
    if os.path.exists(train_images):
        train_img_count = len([f for f in os.listdir(train_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"📈 Train images: {train_img_count}")
    
    if os.path.exists(train_masks):
        train_mask_count = len([f for f in os.listdir(train_masks) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"🎭 Train masks: {train_mask_count}")
    
    if os.path.exists(val_images):
        val_img_count = len([f for f in os.listdir(val_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"📊 Val images: {val_img_count}")
    else:
        print("⚠️  No validation images found")
    
    if os.path.exists(val_masks):
        val_mask_count = len([f for f in os.listdir(val_masks) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"🎭 Val masks: {val_mask_count}")
    else:
        print("⚠️  No validation masks found")

print("\n🎯 Dataset setup complete! Ready for enhanced training.")

# ===============================
# INSTALL REQUIRED PACKAGES
# ===============================

print("\n📦 Installing required packages...")

# Install efficientnet_pytorch if not already installed
try:
    import efficientnet_pytorch
    print("✅ efficientnet_pytorch already installed")
except ImportError:
    print("📦 Installing efficientnet_pytorch...")
    !pip install efficientnet_pytorch

# Install other required packages
required_packages = [
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'tqdm',
    'numpy',
    'opencv-python',
    'matplotlib',
    'Pillow',
    'scikit-learn'
]

for package in required_packages:
    try:
        # Simple check - try to import the main module
        module_name = package.split('>=')[0].split('==')[0]
        if module_name == 'opencv-python':
            module_name = 'cv2'
        elif module_name == 'Pillow':
            module_name = 'PIL'
        elif module_name == 'scikit-learn':
            module_name = 'sklearn'
        
        __import__(module_name)
        print(f"✅ {package} available")
    except ImportError:
        print(f"📦 Installing {package}...")
        !pip install {package}

print("\n🚀 All packages installed! Environment ready for training.")

# ===============================
# TRAINING COMMAND TEMPLATE
# ===============================

print("\n📝 READY TO TRAIN WITH ENHANCED FEATURES:")
print("""
# Run this in your next cell:
best_iou, history = enhanced_train_model(
    model_name='enhanced_ghanasegnet',
    dataset_path='/content/data',  # Colab data path
    epochs=15,                     # Progressive: 5+6+4 epochs
    batch_size=6,                  # Auto-adjusts: 8→6→4
    learning_rate=1.8e-4,          # Optimized LR
    weight_decay=1.5e-3,           # Enhanced regularization
    input_size=320,                # Progressive: 256→320→384
    disable_early_stopping=False,  # Prevent overfitting
    use_advanced_augmentation=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"🎯 Final Results:")
print(f"Best mIoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
print(f"Target: 30.00%")
if best_iou >= 0.30:
    print("🏆 TARGET ACHIEVED!")
""")

print("\n" + "="*60)
print("🎉 COLAB SETUP COMPLETE - READY FOR 30% mIoU TARGET!")
print("="*60)