"""
GhanaSegNet Complete Training - Colab Ready Code
Copy each section to separate Colab cells
"""

# ===== CELL 1: Setup and Dependencies =====
# Install all required dependencies
# Before running this script, ensure you have installed all dependencies:
# Run these commands in your terminal or Colab cell:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install opencv-python pillow tqdm matplotlib seaborn
# pip install efficientnet-pytorch
# pip install segmentation-models-pytorch

import torch
import os
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
print(f"PyTorch version: {torch.__version__}")

# Verify EfficientNet installation
try:
    from efficientnet_pytorch import EfficientNet
    print("EfficientNet-PyTorch installed successfully")
except ImportError:
    print("EfficientNet-PyTorch not found. Please install it using 'pip install efficientnet-pytorch' before running this script.")
    raise

# ===== CELL 2: Navigate to Project (after uploading) =====
# Upload your GhanaSegNet folder to Colab first!
os.chdir('/content/GhanaSegNet')
print(f"Current directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')[:10]}")  # Show first 10 files

# Verify all required modules can be imported
print("\nVerifying project imports...")
try:
    from models.ghanasegnet import GhanaSegNet
    print("GhanaSegNet model imported successfully")
except ImportError as e:
    print(f"GhanaSegNet import failed: {e}")

try:
    from models.unet import UNet
    from models.deeplabv3plus import DeepLabV3Plus
    from models.segformer import SegFormerB0
    print("All baseline models imported successfully")
except ImportError as e:
    print(f"Baseline model import failed: {e}")

# ===== CELL 3: Train GhanaSegNet (Your Model) =====
import subprocess

print("Starting GhanaSegNet training...")
print("Your novel hybrid CNN-Transformer architecture")
print("Expected performance: >24% mIoU (based on previous results)")
subprocess.run(['python', 'scripts/train_baselines.py', '--model', 'ghanasegnet', '--epochs', '15'])

# ===== CELL 4: Train SegFormer (Best Transformer Baseline) =====
print("Starting SegFormer training...")
print("Pure Transformer architecture baseline")
print("Expected performance: 18-23% mIoU")
subprocess.run(['python', 'scripts/train_baselines.py', '--model', 'segformer', '--epochs', '15'])

# ===== CELL 5: Train DeepLabV3+ (CNN State-of-the-art) =====
print("Starting DeepLabV3+ training...")
print("ResNet-50 backbone with atrous convolutions")
print("Expected performance: 15-21% mIoU")
subprocess.run(['python', 'scripts/train_baselines.py', '--model', 'deeplabv3plus', '--epochs', '15'])

# ===== CELL 6: Train UNet (Medical Baseline) =====
print("Starting UNet training...")
print("Medical imaging architecture baseline")
print("Expected performance: 12-18% mIoU")
subprocess.run(['python', 'scripts/train_baselines.py', '--model', 'unet', '--epochs', '15'])

# ===== CELL 7: Results Analysis =====
import json
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Please install it using 'pip install matplotlib' before running this script.")
    raise
import numpy as np

# Load all results
models = ['ghanasegnet', 'segformer', 'deeplabv3plus', 'unet']
results = {}
print("Loading results...")

for model in models:
    try:
        with open(f'checkpoints/{model}/{model}_results.json', 'r') as f:
            results[model] = json.load(f)
        print(f"{model.upper()}: {results[model]['best_iou']*100:.2f}% mIoU")
    except FileNotFoundError:
        print(f"{model.upper()}: Results not found")

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
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(ious) * 1.2)
    for i, v in enumerate(ious):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Parameter efficiency
    params = [results[m]['total_parameters'] / 1e6 for m in results.keys()]
    bars2 = ax2.bar(model_names, params, color=colors)
    ax2.set_ylabel('Parameters (Millions)', fontsize=12)
    ax2.set_title('Model Efficiency Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(params):
        ax2.text(i, v + 0.5, f'{v:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. Training epochs (final)
    epochs = [results[m]['final_epoch'] for m in results.keys()]
    bars3 = ax3.bar(model_names, epochs, color=colors)
    ax3.set_ylabel('Training Epochs', fontsize=12)
    ax3.set_title('Early Stopping Analysis', fontsize=14, fontweight='bold')
    for i, v in enumerate(epochs):
        ax3.text(i, v + 0.2, f'{v}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficiency scatter plot
    ax4.scatter(params, ious, c=colors, s=200, alpha=0.7)
    for i, model in enumerate(model_names):
        ax4.annotate(model, (params[i], ious[i]), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    ax4.set_xlabel('Parameters (Millions)', fontsize=12)
    ax4.set_ylabel('mIoU (%)', fontsize=12)
    ax4.set_title('Efficiency vs Performance', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("GHANASEGNET FAIR COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'mIoU (%)':<10} {'Parameters':<12} {'Epochs':<8} {'Status'}")
    print("-"*80)
    
    best_iou = max(ious)
    for i, model in enumerate(results.keys()):
        status = "WINNER" if ious[i] == best_iou else ""
        print(f"{model.upper():<15} {ious[i]:<10.1f} {params[i]:<12.1f}M {epochs[i]:<8} {status}")
    
    print("-"*80)
    if len(ious) > 1:
        sorted_ious = sorted(ious, reverse=True)
        improvement = ((sorted_ious[0] - sorted_ious[1]) / sorted_ious[1]) * 100
        print(f"Best model improvement over 2nd best: {improvement:.1f}%")
    print(f"Most efficient model: {model_names[params.index(min(params))]} ({min(params):.1f}M params)")
    print("="*80)

# ===== CELL 8: Download Results =====
# Create downloadable results package
import zipfile

def create_results_package():
    with zipfile.ZipFile('ghanasegnet_results.zip', 'w') as zipf:
        # Add model checkpoints
        for model in ['ghanasegnet', 'segformer', 'deeplabv3plus', 'unet']:
            try:
                zipf.write(f'checkpoints/{model}/best_{model}.pth')
                zipf.write(f'checkpoints/{model}/{model}_results.json')
                print(f"Added {model} results to package")
            except FileNotFoundError:
                print(f"{model} results not found")
        
        # Add training summary
        try:
            zipf.write('checkpoints/training_summary.json')
            print("Added training summary")
        except FileNotFoundError:
            print("Training summary not found")
            
        # Add updated training log
        try:
            zipf.write('Training_Results_Log.md')
            print("Added training log")
        except FileNotFoundError:
            print("Training log not found")
    
    print("Results package created: ghanasegnet_results.zip")
    print("Download this file to save your training results!")
try:
    from google.colab import files
    files.download('ghanasegnet_results.zip')
except ImportError:
    print("google.colab not found. If running outside Colab, please manually download 'ghanasegnet_results.zip'.")

# Download command for Colab
from google.colab import files
files.download('ghanasegnet_results.zip')

print("Training completed! Check your downloads for the results package.")
print(f"Total models trained: {len(results)}")
if results:
    best_model = max(results.keys(), key=lambda k: results[k]['best_iou'])
    print(f"Best performing model: {best_model.upper()} ({results[best_model]['best_iou']*100:.2f}% mIoU)")