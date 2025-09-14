"""
GhanaSegNet Complete Training - Colab Ready Code
Copy each section to separate Colab cells
"""

# ===== CELL 1: Setup and Dependencies =====
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python pillow tqdm matplotlib seaborn

import torch
import os
print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
print(f"🎮 GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
print(f"📊 PyTorch version: {torch.__version__}")

# ===== CELL 2: Navigate to Project (after uploading) =====
# Upload your GhanaSegNet folder to Colab first!
os.chdir('/content/GhanaSegNet')
print(f"📁 Current directory: {os.getcwd()}")
print(f"📄 Files: {os.listdir('.')[:10]}")  # Show first 10 files

# ===== CELL 3: Train GhanaSegNet (Your Model) =====
!python scripts/train_baselines.py --model ghanasegnet --epochs 15

# ===== CELL 4: Train SegFormer (Best Transformer Baseline) =====
!python scripts/train_baselines.py --model segformer --epochs 15

# ===== CELL 5: Train DeepLabV3+ (CNN State-of-the-art) =====
!python scripts/train_baselines.py --model deeplabv3plus --epochs 15

# ===== CELL 6: Train UNet (Medical Baseline) =====
!python scripts/train_baselines.py --model unet --epochs 15

# ===== CELL 7: Results Analysis =====
import json
import matplotlib.pyplot as plt
import numpy as np

# Load all results
models = ['ghanasegnet', 'segformer', 'deeplabv3plus', 'unet']
results = {}
print("📊 Loading results...")

for model in models:
    try:
        with open(f'checkpoints/{model}_results.json', 'r') as f:
            results[model] = json.load(f)
        print(f"✅ {model.upper()}: {results[model]['best_iou']*100:.2f}% mIoU")
    except FileNotFoundError:
        print(f"❌ {model.upper()}: Results not found")

# Create comprehensive comparison plot
if len(results) > 0:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model names and colors
    model_names = [m.upper() for m in results.keys()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(results)]
    
    # 1. Performance comparison
    ious = [results[m]['best_iou'] * 100 for m in results.keys()]
    bars1 = ax1.bar(model_names, ious, color=colors)
    ax1.set_ylabel('mIoU (%)', fontsize=12)
    ax1.set_title('🏆 Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(ious) * 1.2)
    for i, v in enumerate(ious):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Parameter efficiency
    params = [results[m]['total_parameters'] / 1e6 for m in results.keys()]
    bars2 = ax2.bar(model_names, params, color=colors)
    ax2.set_ylabel('Parameters (Millions)', fontsize=12)
    ax2.set_title('⚡ Model Efficiency Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(params):
        ax2.text(i, v + 0.5, f'{v:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training epochs (final)
    epochs = [results[m]['final_epoch'] for m in results.keys()]
    bars3 = ax3.bar(model_names, epochs, color=colors)
    ax3.set_ylabel('Training Epochs', fontsize=12)
    ax3.set_title('🛑 Early Stopping Analysis', fontsize=14, fontweight='bold')
    for i, v in enumerate(epochs):
        ax3.text(i, v + 0.2, f'{v}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficiency scatter plot
    ax4.scatter(params, ious, c=colors, s=200, alpha=0.7)
    for i, model in enumerate(model_names):
        ax4.annotate(model, (params[i], ious[i]), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    ax4.set_xlabel('Parameters (Millions)', fontsize=12)
    ax4.set_ylabel('mIoU (%)', fontsize=12)
    ax4.set_title('📈 Efficiency vs Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("🏆 GHANASEGNET FAIR COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'mIoU (%)':<10} {'Parameters':<12} {'Epochs':<8} {'Status'}")
    print("-"*80)
    
    best_iou = max(ious)
    for i, model in enumerate(results.keys()):
        status = "🥇 WINNER" if ious[i] == best_iou else ""
        print(f"{model.upper():<15} {ious[i]:<10.1f} {params[i]:<12.1f}M {epochs[i]:<8} {status}")
    
    print("-"*80)
    improvement = ((best_iou - max([iou for j, iou in enumerate(ious) if j != ious.index(best_iou)])) / 
                   max([iou for j, iou in enumerate(ious) if j != ious.index(best_iou)])) * 100
    print(f"🚀 GhanaSegNet improvement over best baseline: {improvement:.1f}%")
    print(f"⚡ Most efficient model: {model_names[params.index(min(params))]} ({min(params):.1f}M params)")
    print("="*80)

# ===== CELL 8: Download Results =====
# Create downloadable results package
import zipfile

def create_results_package():
    with zipfile.ZipFile('ghanasegnet_results.zip', 'w') as zipf:
        # Add model checkpoints
        for model in ['ghanasegnet', 'segformer', 'deeplabv3plus', 'unet']:
            try:
                zipf.write(f'checkpoints/best_{model}.pth')
                zipf.write(f'checkpoints/{model}_results.json')
                print(f"✅ Added {model} results to package")
            except FileNotFoundError:
                print(f"⚠️ {model} results not found")
        
        # Add updated training log
        try:
            zipf.write('Training_Results_Log.md')
            print("✅ Added training log")
        except FileNotFoundError:
            print("⚠️ Training log not found")
    
    print("📦 Results package created: ghanasegnet_results.zip")
    print("💾 Download this file to save your training results!")

create_results_package()

# Download command for Colab
from google.colab import files
files.download('ghanasegnet_results.zip')

print("🎉 Training completed! Check your downloads for the results package.")