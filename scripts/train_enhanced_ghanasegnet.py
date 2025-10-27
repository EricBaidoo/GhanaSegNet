#!/usr/bin/env python3
"""
Enhanced GhanaSegNet Optimized Training Script
Specifically tuned for the enhanced architecture (11.1M parameters)

Key Optimizations:
- Lower learning rate for larger model (5e-5 vs 1e-4)
- Progressive training with encoder freezing
- Enhanced regularization strategy
- Longer warmup for transformer components
- Adaptive dropout scheduling

Author: EricBaidoo
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import random
import numpy as np

# Import enhanced model and utilities
from models.ghanasegnet import GhanaSegNet
from data.dataset_loader import GhanaFoodDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy

class EnhancedEarlyStopping:
    """Enhanced early stopping for larger models with more patience"""
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
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

class AdaptiveDropout:
    """Adaptive dropout scheduling for enhanced architecture"""
    def __init__(self, model, initial_dropout=0.25, final_dropout=0.15, total_epochs=80):
        self.model = model
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.total_epochs = total_epochs
        
    def update_dropout(self, epoch):
        # Progressive dropout reduction
        progress = min(epoch / self.total_epochs, 1.0)
        current_dropout = self.initial_dropout - progress * (self.initial_dropout - self.final_dropout)
        
        # Update dropout in all relevant modules
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
        
        return current_dropout

def freeze_encoder(model, freeze=True):
    """Freeze/unfreeze encoder for progressive training"""
    for param in model.encoder.parameters():
        param.requires_grad = not freeze
    
    print(f"{'Freezing' if freeze else 'Unfreezing'} encoder parameters")

def get_parameter_groups(model):
    """Get parameter groups for differential learning rates"""
    encoder_params = list(model.encoder.parameters())
    aspp_params = list(model.aspp.parameters())
    transformer_params = list(model.transformer.parameters())
    decoder_params = (list(model.dec4.parameters()) + list(model.dec3.parameters()) + 
                     list(model.dec2.parameters()) + list(model.dec1.parameters()) + 
                     list(model.final_conv.parameters()))
    
    return {
        'encoder': encoder_params,
        'aspp': aspp_params,
        'transformer': transformer_params,
        'decoder': decoder_params
    }

def create_optimized_optimizer(model, base_lr=5e-5):
    """Create optimizer with differential learning rates for different components"""
    param_groups = get_parameter_groups(model)
    
    optimizer_params = [
        {'params': param_groups['encoder'], 'lr': base_lr * 0.1, 'weight_decay': 1e-4},
        {'params': param_groups['aspp'], 'lr': base_lr * 0.5, 'weight_decay': 5e-5},
        {'params': param_groups['transformer'], 'lr': base_lr * 0.3, 'weight_decay': 1e-5},
        {'params': param_groups['decoder'], 'lr': base_lr, 'weight_decay': 1e-4}
    ]
    
    return optim.AdamW(optimizer_params)

def train_enhanced_ghanasegnet(args):
    """Enhanced training function optimized for the enhanced architecture"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Enhanced GhanaSegNet on: {device}")
    
    # Set seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Initialize enhanced model
    model = GhanaSegNet(num_classes=6, dropout=args.initial_dropout)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️ Enhanced GhanaSegNet Architecture:")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Trainable Parameters: {trainable_params:,}")
    print(f"   • Enhanced Features: 8-head transformer, ASPP, enhanced attention")
    
    # Load datasets
    print("\n📊 Loading Datasets...")
    train_dataset = GhanaFoodDataset(split='train')
    val_dataset = GhanaFoodDataset(split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(val_dataset)}")
    
    # Initialize enhanced loss and optimizer
    criterion = CombinedLoss(alpha=0.6)  # Food-aware loss
    optimizer = create_optimized_optimizer(model, args.lr)
    
    # Enhanced learning rate scheduling
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    early_stopping = EnhancedEarlyStopping(patience=args.patience, min_delta=0.001)
    adaptive_dropout = AdaptiveDropout(model, args.initial_dropout, args.final_dropout, args.epochs)
    
    print(f"\n⚙️ Enhanced Training Configuration:")
    print(f"   • Base Learning Rate: {args.lr}")
    print(f"   • Differential LR: Encoder(0.1x), ASPP(0.5x), Transformer(0.3x), Decoder(1.0x)")
    print(f"   • Progressive Training: Freeze encoder first {args.freeze_epochs} epochs")
    print(f"   • Adaptive Dropout: {args.initial_dropout} → {args.final_dropout}")
    print(f"   • Enhanced Early Stopping: Patience={args.patience}")
    
    # Training history
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    best_epoch = 0
    
    # Phase 1: Frozen encoder training (stabilize decoder and new components)
    print(f"\n🔒 Phase 1: Frozen Encoder Training ({args.freeze_epochs} epochs)")
    freeze_encoder(model, freeze=True)
    
    for epoch in range(args.freeze_epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # Update adaptive dropout
        current_dropout = adaptive_dropout.update_dropout(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.freeze_epochs} [FROZEN]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dropout': f'{current_dropout:.3f}'})
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                # Compute IoU
                iou = compute_iou(outputs, masks)
                total_iou += iou
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
        
        print(f'Epoch {epoch+1}/{args.freeze_epochs} [FROZEN]: '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Val IoU: {avg_val_iou:.4f}, Best IoU: {best_iou:.4f}')
        
        scheduler.step()
    
    # Phase 2: Full model training (unfreeze encoder)
    print(f"\n🔓 Phase 2: Full Model Training ({args.epochs - args.freeze_epochs} epochs)")
    freeze_encoder(model, freeze=False)
    
    # Adjust learning rate for unfrozen training
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5  # Reduce LR when unfreezing
    
    for epoch in range(args.freeze_epochs, args.epochs):
        model.train()
        epoch_train_loss = 0.0
        
        # Update adaptive dropout
        current_dropout = adaptive_dropout.update_dropout(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [FULL]')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Dropout': f'{current_dropout:.3f}'})
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        total_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                iou = compute_iou(outputs, masks)
                total_iou += iou
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ious.append(avg_val_iou)
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            best_epoch = epoch + 1
        
        print(f'Epoch {epoch+1}/{args.epochs} [FULL]: '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Val IoU: {avg_val_iou:.4f}, Best IoU: {best_iou:.4f}')
        
        # Enhanced early stopping check
        if early_stopping(avg_val_iou, model):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
        scheduler.step()
    
    # Save results
    results = {
        'model_name': 'enhanced_ghanasegnet',
        'experiment_id': f'enhanced_ghanasegnet_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'timestamp': datetime.now().isoformat(),
        'training_config': {
            'epochs': args.epochs,
            'freeze_epochs': args.freeze_epochs,
            'batch_size': args.batch_size,
            'base_learning_rate': args.lr,
            'initial_dropout': args.initial_dropout,
            'final_dropout': args.final_dropout,
            'optimizer': 'AdamW_differential',
            'loss_function': 'CombinedLoss',
            'scheduler': 'CosineAnnealingWarmRestarts'
        },
        'performance_metrics': {
            'final_epoch': len(train_losses),
            'best_epoch': best_epoch,
            'best_iou': float(best_iou),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1]),
            'final_val_iou': float(val_ious[-1])
        },
        'training_history': {
            'train_losses': [float(x) for x in train_losses],
            'val_losses': [float(x) for x in val_losses],
            'val_ious': [float(x) for x in val_ious]
        },
        'model_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'enhancements': [
            'Enhanced Transformer (8 attention heads)',
            'ASPP Multi-scale feature extraction',
            'Enhanced spatial+channel attention',
            'Progressive 4-stage decoder',
            'Learnable gradient scaling',
            'Progressive training strategy',
            'Adaptive dropout scheduling',
            'Differential learning rates'
        ],
        'status': 'completed'
    }
    
    # Save checkpoint and results
    checkpoint_dir = f'checkpoints/enhanced_ghanasegnet'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_iou': best_iou,
        'epoch': best_epoch,
        'results': results
    }, f'{checkpoint_dir}/best_enhanced_ghanasegnet.pth')
    
    # Save results JSON
    with open(f'{checkpoint_dir}/enhanced_ghanasegnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🏆 Enhanced GhanaSegNet Training Complete!")
    print(f"   • Best mIoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"   • Best Epoch: {best_epoch}")
    print(f"   • Total Parameters: {total_params:,}")
    print(f"   • Model saved: {checkpoint_dir}/best_enhanced_ghanasegnet.pth")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced GhanaSegNet Optimized Training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs')
    parser.add_argument('--freeze-epochs', type=int, default=8, help='Epochs to freeze encoder')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Base learning rate')
    parser.add_argument('--initial-dropout', type=float, default=0.25, help='Initial dropout rate')
    parser.add_argument('--final-dropout', type=float, default=0.15, help='Final dropout rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced GhanaSegNet Optimized Training")
    print("=" * 50)
    print(f"Target: Improve from 24.47% → 28-30%+ mIoU")
    print(f"Strategy: Progressive training + enhanced regularization")
    
    results = train_enhanced_ghanasegnet(args)
    
    # Performance analysis
    baseline_iou = 0.2447
    enhanced_iou = results['performance_metrics']['best_iou']
    improvement = (enhanced_iou - baseline_iou) * 100
    
    print(f"\n📊 ENHANCEMENT IMPACT ANALYSIS:")
    print("=" * 40)
    print(f"🔄 Baseline (Oct 7): {baseline_iou*100:.2f}% mIoU")
    print(f"🚀 Enhanced Result: {enhanced_iou*100:.2f}% mIoU")
    print(f"➕ Improvement: {improvement:+.2f}% absolute")
    print(f"📈 Relative Gain: {(improvement/baseline_iou/100)*100:+.1f}%")
    
    if enhanced_iou > 0.28:
        print("🎯 SUCCESS: Target achieved! (28%+ mIoU)")
    elif enhanced_iou > baseline_iou + 0.02:
        print("✅ GOOD: Significant improvement achieved")
    else:
        print("⚠️  NEEDS TUNING: Consider longer training or further optimization")

if __name__ == '__main__':
    main()