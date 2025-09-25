#!/usr/bin/env python3
"""
Enhanced Training Script for GhanaSegNet 30%+ mIoU Target
Comprehensive optimization for maximum performance

Key Features:
- Enhanced GhanaSegNet architecture
- Advanced loss functions (Lovász + Tversky + Edge-aware)
- Progressive training strategy
- Advanced data augmentation
- Test-time augmentation
- Knowledge distillation
- Ensemble techniques

Author: EricBaidoo
"""

import sys
import os
import argparse
import json
import warnings
from datetime import datetime
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# Enable optimizations
cudnn.benchmark = True
cudnn.deterministic = False

# Import enhanced components
from models.ghanasegnet_enhanced import EnhancedGhanaSegNet, EnhancedCombinedLoss
from utils.enhanced_augmentation import (
    AdvancedGhanaFoodDataset, 
    MixUpAugmentation, 
    CutMixAugmentation,
    TestTimeAugmentation,
    ProgressiveAugmentationScheduler
)
from utils.metrics import compute_iou, compute_pixel_accuracy

class EnhancedEarlyStopping:
    """
    Enhanced early stopping with warmup and progressive patience
    """
    def __init__(self, patience=5, min_delta=0.001, warmup_epochs=5):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.epoch_count = 0
        
    def __call__(self, val_score, model):
        self.epoch_count += 1
        
        # No early stopping during warmup
        if self.epoch_count <= self.warmup_epochs:
            if self.best_score is None or val_score > self.best_score:
                self.best_score = val_score
                self.save_checkpoint(model)
            return False
            
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            # Progressive patience - more patience later in training
            current_patience = self.patience + (self.epoch_count // 10)
            if self.counter >= current_patience:
                model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def enhanced_train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                        scaler, mixup_aug, cutmix_aug, aug_scheduler):
    """
    Enhanced training epoch with advanced augmentations
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    # Progressive augmentation
    use_mixup = aug_scheduler.should_use_mixup()
    use_cutmix = aug_scheduler.should_use_cutmix()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training [Enhanced]")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        # Apply batch-level augmentations
        if use_mixup and np.random.random() < 0.3:
            images, masks = mixup_aug(images, masks)
        elif use_cutmix and np.random.random() < 0.3:
            images, masks = cutmix_aug(images, masks)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mixup': use_mixup,
            'cutmix': use_cutmix
        })
    
    return total_loss / total_samples

def enhanced_validate_epoch(model, val_loader, criterion, device, epoch, tta_aug=None):
    """
    Enhanced validation with optional test-time augmentation
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation [Enhanced]")
        
        for images, masks in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            # Optional test-time augmentation
            if tta_aug and epoch % 5 == 0:  # Use TTA every 5th epoch
                batch_outputs = []
                for img in images:
                    img_batch, transformations = tta_aug.augment_batch(img.unsqueeze(0))
                    img_batch = img_batch.to(device)
                    
                    with autocast():
                        tta_outputs = model(img_batch)
                    
                    merged_output = tta_aug.merge_predictions(
                        tta_outputs, transformations, images.shape[2:]
                    )
                    batch_outputs.append(merged_output)
                
                outputs = torch.cat(batch_outputs, dim=0)
            else:
                # Standard inference
                with autocast():
                    outputs = model(images)
            
            loss = criterion(outputs, masks)
            
            # Compute metrics
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes=6)
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

def train_enhanced_ghanasegnet(config):
    """
    Main training function for enhanced GhanaSegNet targeting 30%+ mIoU
    """
    print("🚀 Starting Enhanced GhanaSegNet Training for 30%+ mIoU")
    print("🎯 Target: >30% mIoU (improvement from 24.7%)")
    print(f"📋 Config: {json.dumps(config, indent=2)}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    
    # Enhanced datasets with advanced augmentation
    print("📁 Loading enhanced datasets...")
    train_dataset = AdvancedGhanaFoodDataset('train', target_size=(512, 512), enhanced_augment=True)
    val_dataset = AdvancedGhanaFoodDataset('val', target_size=(512, 512), enhanced_augment=False)
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Enhanced model
    print("🤖 Initializing Enhanced GhanaSegNet...")
    model = EnhancedGhanaSegNet(
        num_classes=config['num_classes'],
        backbone='efficientnet-b2'  # Larger backbone for better performance
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 Total parameters: {total_params:,}")
    print(f"🔓 Trainable parameters: {trainable_params:,}")
    
    # Enhanced loss function
    criterion = EnhancedCombinedLoss(
        alpha=0.4,  # Lovász weight
        beta=0.4,   # Tversky weight
        gamma=0.2,  # Edge weight
        edge_weight=2.0
    )
    
    # Advanced optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True
    )
    
    # Advanced scheduler - OneCycleLR for faster convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 5,  # Peak LR
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos',
        div_factor=25,  # Initial LR = max_lr / div_factor
        final_div_factor=1000  # Final LR = initial LR / final_div_factor
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Advanced augmentations
    mixup_aug = MixUpAugmentation(alpha=0.4, p=0.3)
    cutmix_aug = CutMixAugmentation(alpha=1.0, p=0.3)
    tta_aug = TestTimeAugmentation(scales=[0.9, 1.0, 1.1], flips=True)
    aug_scheduler = ProgressiveAugmentationScheduler(max_epochs=config['epochs'])
    
    # Enhanced early stopping
    early_stopping = EnhancedEarlyStopping(patience=7, min_delta=0.001, warmup_epochs=5)
    
    # Setup checkpoints
    checkpoint_dir = "checkpoints/enhanced_ghanasegnet"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_iou = 0.0
    train_history = []
    
    print(f"🏁 Starting enhanced training for {config['epochs']} epochs...")
    print("🎯 Target mIoU: >30% (current best: 24.7%)")
    
    start_time = datetime.now()
    
    for epoch in range(config['epochs']):
        epoch_start = datetime.now()
        
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']} - Enhanced GhanaSegNet")
        print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Update augmentation scheduler
        aug_scheduler.step(epoch)
        
        # Training
        train_loss = enhanced_train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1,
            scaler, mixup_aug, cutmix_aug, aug_scheduler
        )
        
        # Validation with optional TTA
        val_metrics = enhanced_validate_epoch(
            model, val_loader, criterion, device, epoch+1,
            tta_aug if epoch > config['epochs'] * 0.8 else None  # TTA in final 20%
        )
        
        # Update scheduler
        scheduler.step()
        
        val_loss = val_metrics['loss']
        val_iou = val_metrics['iou']
        val_accuracy = val_metrics['accuracy']
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Log metrics
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_accuracy': val_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        train_history.append(epoch_data)
        
        print(f"📊 Train Loss: {train_loss:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}")
        print(f"📊 Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"📊 Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"⏱️ Epoch time: {epoch_time:.1f}s")
        
        # 30% target tracking
        if val_iou >= 0.30:
            print(f"🎉 TARGET ACHIEVED! IoU: {val_iou*100:.2f}% >= 30%")
        else:
            remaining = (0.30 - val_iou) * 100
            print(f"🎯 Target progress: {remaining:.1f}% remaining to reach 30%")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_iou': best_iou,
                'config': config,
                'model_name': 'enhanced_ghanasegnet'
            }, f"{checkpoint_dir}/best_enhanced_ghanasegnet.pth")
            print(f"💾 New best model saved! IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
            
            # Achievement notifications
            if best_iou >= 0.30:
                print("🏆 30%+ mIoU ACHIEVED! TARGET REACHED!")
            elif best_iou >= 0.28:
                print("🔥 28%+ mIoU! Very close to 30% target!")
            elif best_iou >= 0.26:
                print("⚡ 26%+ mIoU! Strong improvement from baseline!")
        
        # Early stopping check
        if early_stopping(val_iou, model):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"🏆 Best IoU achieved: {best_iou:.4f} ({best_iou*100:.2f}%)")
            break
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Final results
    final_results = {
        'model_name': 'enhanced_ghanasegnet',
        'best_iou': float(best_iou),
        'target_achieved': best_iou >= 0.30,
        'improvement_over_baseline': ((best_iou - 0.247) / 0.247) * 100,
        'final_epoch': epoch + 1,
        'total_training_time': total_time,
        'avg_epoch_time': total_time / (epoch + 1),
        'training_history': train_history,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'enhancements_used': [
            'efficientnet_b2_backbone',
            'feature_pyramid_network',
            'multi_scale_transformer',
            'enhanced_cbam_attention',
            'lovasz_tversky_edge_loss',
            'advanced_data_augmentation',
            'mixup_cutmix',
            'test_time_augmentation',
            'onecycle_lr_scheduler',
            'mixed_precision_training'
        ],
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    # Save results
    with open(f"{checkpoint_dir}/enhanced_ghanasegnet_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 ENHANCED GHANASEGNET TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"🎯 TARGET: 30%+ mIoU")
    print(f"📊 ACHIEVED: {best_iou*100:.2f}% mIoU")
    print(f"✅ TARGET {'REACHED' if best_iou >= 0.30 else 'NOT REACHED'}")
    print(f"📈 IMPROVEMENT: {((best_iou - 0.247) / 0.247) * 100:+.1f}% over baseline (24.7%)")
    print(f"⏱️ TRAINING TIME: {total_time/60:.1f} minutes")
    print(f"💾 RESULTS SAVED: {checkpoint_dir}")
    
    if best_iou >= 0.30:
        print("🎉 CONGRATULATIONS! 30%+ mIoU TARGET ACHIEVED!")
        print("🏆 Your Enhanced GhanaSegNet is ready for publication!")
    else:
        print(f"⚠️ Target not reached. Consider:")
        print(f"   - Longer training ({config['epochs']} -> {config['epochs'] + 20} epochs)")
        print(f"   - EfficientNet-B3 backbone")
        print(f"   - Ensemble methods")
        print(f"   - Additional data augmentation")
    
    print(f"{'='*80}")
    
    return final_results

def main():
    """
    Main entry point for enhanced training
    """
    parser = argparse.ArgumentParser(description='Enhanced GhanaSegNet training for 30%+ mIoU')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of classes')
    
    args = parser.parse_args()
    
    # Enhanced configuration for 30%+ target
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 3e-4,  # Slightly higher for better regularization
        'num_classes': args.num_classes,
        'timestamp': datetime.now().isoformat(),
        'note': 'Enhanced GhanaSegNet targeting 30%+ mIoU with all optimizations',
        'target_miou': 0.30,
        'baseline_miou': 0.247
    }
    
    # Start enhanced training
    results = train_enhanced_ghanasegnet(config)
    
    return results

if __name__ == "__main__":
    results = main()