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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

class EarlyStopping:
    """Early stopping to prevent overfitting - APPLIED TO ALL MODELS FAIRLY"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

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
        'ghanasegnet': CombinedLoss(alpha=0.6)  # Optimized: 60% Dice + 40% Boundary for better performance
    }
    
    paper_refs = {
        'unet': 'Ronneberger et al., 2015', 
        'deeplabv3plus': 'Chen et al., 2018', 
        'segformer': 'Xie et al., 2021',
        'ghanasegnet': 'Baidoo, E. (Novel Architecture)'
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    
    model = models[model_name.lower()]()
    criterion = original_losses[model_name.lower()]
    
    print(f"📋 Using ORIGINAL loss for {model_name}: {type(criterion).__name__}")
    if model_name == 'ghanasegnet':
        print("🍽️  Food-aware loss: Dice + Focal + Boundary losses combined")
    print(f"📚 Paper reference: {paper_refs[model_name]}")
    
    return model, criterion

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_name):
    """Train for one epoch with enhanced monitoring"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training [{model_name}]")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # FAIR ENHANCEMENT: Gradient clipping applied to ALL models
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples

def validate_epoch(model, val_loader, criterion, device, epoch, model_name, num_classes=6):
    """Validate for one epoch with enhanced monitoring"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation [{model_name}]")
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
        'segformer': (512, 512),      # SegFormer paper
        'ghanasegnet': (384, 384)     # Optimized for EfficientNet-B0 backbone + transformers
    }
    input_size = model_input_sizes.get(model_name.lower(), (256, 256))

    print("📁 Loading datasets...")
    train_dataset = GhanaFoodDataset('train', target_size=input_size)
    val_dataset = GhanaFoodDataset('val', target_size=input_size)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,  # Set to 0 for Windows compatibility
        drop_last=True  # Drop incomplete batches to avoid BatchNorm errors
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows compatibility
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
    # Enhanced scheduler with ReduceLROnPlateau for better overfitting prevention
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Monitor IoU (higher is better)
        factor=0.5, 
        patience=3,
        min_lr=1e-6
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Create checkpoint directory
    checkpoint_dir = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop with enhanced early stopping
    best_iou = 0.0
    train_history = []
    
    print(f"🎯 Starting enhanced training for {config['epochs']} epochs (with early stopping)...")
    
    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']} - {model_name.upper()}")
        print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, model_name)
        
        # Validate  
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch+1, model_name, config['num_classes'])
        
        # Update scheduler with validation IoU
        scheduler.step(val_metrics['iou'])
        
        # Extract metrics for cleaner logging
        val_loss = val_metrics['loss']
        val_iou = val_metrics['iou'] 
        val_accuracy = val_metrics['accuracy']
        
        # Log metrics
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_accuracy': val_accuracy,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_data)
        
        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}")
        print(f"📊 Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"📊 Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save best model and check early stopping
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config,
                'model_name': model_name
            }, f"{checkpoint_dir}/best_{model_name}.pth")
            print(f"💾 New best model saved! IoU: {best_iou:.4f}")
        
        # Early stopping check
        if early_stopping(val_iou, model):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"🏆 Best IoU achieved: {best_iou:.4f}")
            break
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'config': config,
            'model_name': model_name
        }, f"{checkpoint_dir}/latest_{model_name}.pth")
    
    # Save training history and final results
    final_results = {
        'model_name': model_name,
        'best_iou': float(best_iou),
        'final_epoch': epoch + 1,
        'training_history': train_history,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(f"{checkpoint_dir}/{model_name}_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"🎉 Training completed for {model_name.upper()}!")
    print(f"🏆 Best validation IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"💾 Results saved to: {checkpoint_dir}")
    
    return {
        'best_iou': best_iou,
        'final_epoch': epoch + 1,
        'model_name': model_name,
        'total_params': total_params
    }
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
        models_to_train = ['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet']
        results = {}
        
        print(f"\n🚀 STARTING COMPREHENSIVE MODEL COMPARISON")
        print(f"📋 Models to train: {', '.join([m.upper() for m in models_to_train])}")
        print(f"⚙️  Config: {args.epochs} epochs, batch size {args.batch_size}")
        print(f"{'='*80}")
        
        for i, model_name in enumerate(models_to_train, 1):
            print(f"\n{'='*80}")
            print(f"🔄 TRAINING MODEL {i}/{len(models_to_train)}: {model_name.upper()}")
            print(f"{'='*80}")
            
            try:
                best_iou, history = train_model(model_name, config)
                results[model_name] = {
                    'best_iou': best_iou,
                    'status': 'completed',
                    'model_type': 'baseline' if model_name != 'ghanasegnet' else 'novel'
                }
                print(f"✅ {model_name.upper()} completed - Best IoU: {best_iou:.4f}")
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                results[model_name] = {
                    'best_iou': 0.0,
                    'status': f'failed: {e}',
                    'model_type': 'baseline' if model_name != 'ghanasegnet' else 'novel'
                }
        
        # Save overall results
        import os
        os.makedirs('checkpoints', exist_ok=True)
        with open('checkpoints/training_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display final comparison
        print(f"\n{'='*80}")
        print("� FINAL MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'IoU':<10} {'Type':<10} {'Status'}")
        print(f"{'-'*50}")
        
        # Sort by IoU for ranking
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_iou'], reverse=True)
        
        for rank, (model, result) in enumerate(sorted_results, 1):
            status_icon = "✅" if result['status'] == 'completed' else "❌"
            model_type = result['model_type'].capitalize()
            print(f"#{rank} {model.upper():<12} {result['best_iou']:.4f}     {model_type:<10} {status_icon}")
        
        # Highlight GhanaSegNet performance
        if 'ghanasegnet' in results:
            ghanasegnet_result = results['ghanasegnet']
            baseline_ious = [r['best_iou'] for m, r in results.items() if m != 'ghanasegnet' and r['status'] == 'completed']
            if baseline_ious and ghanasegnet_result['status'] == 'completed':
                best_baseline = max(baseline_ious)
                improvement = ghanasegnet_result['best_iou'] - best_baseline
                print(f"\n🎯 NOVEL MODEL ANALYSIS:")
                print(f"   GhanaSegNet IoU: {ghanasegnet_result['best_iou']:.4f}")
                print(f"   Best Baseline IoU: {best_baseline:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement/best_baseline*100:+.1f}%)")
        
        print(f"\n📊 Results saved to: checkpoints/training_summary.json")
        
    else:
        train_model(args.model, config)

if __name__ == "__main__":
    main()
