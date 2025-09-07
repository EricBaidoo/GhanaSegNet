#!/usr/bin/env python3
"""
Multi-Model Training Script for GhanaSegNet Research Project
Trains baseline models: UNet, DeepLabV3+, SegFormer-B0
Author: EricBaidoo
"""

import sys
import os
import argparse
import json
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import original baseline models (unmodified for fair comparison)
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerB0
from models.ghanasegnet import GhanaSegNet

# Import utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset_loader import GhanaFoodDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy

def get_model_and_criterion(model_name, num_classes=6):
    """Initialize model and its ORIGINAL loss function for fair comparison"""
    models = {
        'unet': lambda: UNet(n_channels=3, n_classes=num_classes),
        'deeplabv3plus': lambda: DeepLabV3Plus(num_classes=num_classes),
        'segformer': lambda: SegFormerB0(num_classes=num_classes),
        'ghanasegnet': lambda: GhanaSegNet(num_classes=num_classes)
    }
    
    # Original loss functions from papers
    original_losses = {
        'unet': nn.CrossEntropyLoss(),  # Ronneberger et al., 2015
        'deeplabv3plus': nn.CrossEntropyLoss(),  # Chen et al., 2018  
        'segformer': nn.CrossEntropyLoss(),  # Xie et al., 2021
        'ghanasegnet': nn.CrossEntropyLoss()  # Novel hybrid architecture
    }
    
    paper_refs = {
        'unet': 'Ronneberger et al., 2015', 
        'deeplabv3plus': 'Chen et al., 2018', 
        'segformer': 'Xie et al., 2021'
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    
    model = models[model_name.lower()]()
    criterion = original_losses[model_name.lower()]
    
    print(f"📋 Using ORIGINAL loss for {model_name}: CrossEntropyLoss")
    print(f"📚 Paper reference: {paper_refs[model_name]}")
    
    return model, criterion

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples

def validate_epoch(model, val_loader, criterion, device, num_classes=6):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Compute metrics
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes)
            acc = compute_pixel_accuracy(preds, masks)
            
            total_loss += loss.item() * images.size(0)
            total_iou += iou * images.size(0)
            total_acc += acc * images.size(0)
            total_samples += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'acc': f'{acc:.4f}'
            })
    
    return {
        'loss': total_loss / total_samples,
        'iou': total_iou / total_samples,
        'accuracy': total_acc / total_samples
    }

def train_model(model_name, config):
    """Main training function"""
    print(f"🚀 Starting training for {model_name.upper()}")
    print(f"📋 Config: {json.dumps(config, indent=2)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Set original input sizes for each model
    model_input_sizes = {
        'unet': (572, 572),           # Original UNet paper
        'deeplabv3plus': (513, 513),  # Original DeepLabV3+ paper
        'segformer': (512, 512)       # SegFormer paper
    }
    input_size = model_input_sizes.get(model_name.lower(), (256, 256))

    print("📁 Loading datasets...")
    train_dataset = GhanaFoodDataset('train', target_size=input_size)
    val_dataset = GhanaFoodDataset('val', target_size=input_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True  # Drop incomplete batches to avoid BatchNorm errors
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        drop_last=True  # Drop incomplete batches to avoid BatchNorm errors
    )

    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model with ORIGINAL loss function
    print(f"🤖 Initializing {model_name} model...")
    model, criterion = get_model_and_criterion(model_name, config['num_classes'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🔓 Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer (using ORIGINAL loss function)
    # criterion already initialized with original loss
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_iou = 0.0
    train_history = []
    
    print(f"🎯 Starting training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, config['num_classes'])
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_iou': val_metrics['iou'],
            'val_accuracy': val_metrics['accuracy'],
            'lr': scheduler.get_last_lr()[0]
        }
        train_history.append(epoch_data)
        
        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"📊 Val Loss: {val_metrics['loss']:.4f}")
        print(f"📊 Val IoU: {val_metrics['iou']:.4f}")
        print(f"📊 Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"📊 Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, f"{checkpoint_dir}/best_model.pth")
            print(f"💾 New best model saved! IoU: {best_iou:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'config': config
        }, f"{checkpoint_dir}/latest_model.pth")
    
    # Save training history
    with open(f"{checkpoint_dir}/training_history.json", 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"🎉 Training completed!")
    print(f"🏆 Best IoU: {best_iou:.4f}")
    print(f"💾 Models saved in: {checkpoint_dir}")
    
    return best_iou, train_history

def main():
    parser = argparse.ArgumentParser(description='Train baseline models for GhanaSegNet research')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of classes')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'num_classes': args.num_classes,
        'timestamp': datetime.now().isoformat(),
        'note': 'Using ORIGINAL loss functions for fair baseline comparison'
    }
    
    # Train models
    if args.model == 'all':
        models_to_train = ['unet', 'deeplabv3plus', 'segformer']
        results = {}
        
        for model_name in models_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()}")
            print(f"{'='*60}")
            
            try:
                best_iou, history = train_model(model_name, config)
                results[model_name] = {
                    'best_iou': best_iou,
                    'status': 'completed'
                }
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                results[model_name] = {
                    'best_iou': 0.0,
                    'status': f'failed: {e}'
                }
        
        # Save overall results
        with open('checkpoints/training_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("🏁 TRAINING SUMMARY")
        print(f"{'='*60}")
        for model, result in results.items():
            print(f"{model.upper()}: IoU={result['best_iou']:.4f} ({result['status']})")
        
    else:
        train_model(args.model, config)

if __name__ == "__main__":
    main()
