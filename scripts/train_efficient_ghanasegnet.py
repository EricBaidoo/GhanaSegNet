#!/usr/bin/env python3
"""
Efficient GhanaSegNet Training for 30%+ mIoU
Architecture-first approach with minimal but effective augmentation

Focus: Maximum gains from architecture and loss improvements
Augmentation: Minimal but strategic for 30%+ target

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
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# Enable optimizations
cudnn.benchmark = True

# Import components
from models.ghanasegnet_enhanced import EnhancedGhanaSegNet, EnhancedCombinedLoss
from utils.minimal_augmentation import SmartGhanaFoodDataset
from utils.metrics import compute_iou, compute_pixel_accuracy

class EfficientEarlyStopping:
    """
    Efficient early stopping for focused training
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
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
                model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def efficient_train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    """
    Efficient training epoch focused on convergence speed
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training [Efficient]")
    
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision for efficiency
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total_samples

def efficient_validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Efficient validation epoch
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation [Efficient]")
        
        for images, masks in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
            
            loss = criterion(outputs, masks)
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

def train_efficient_ghanasegnet(config):
    """
    Efficient training function focusing on architecture gains
    """
    print("🚀 Efficient GhanaSegNet Training for 30%+ mIoU")
    print("🎯 Strategy: Architecture-first, minimal augmentation")
    print(f"📋 Config: {json.dumps(config, indent=2)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Smart datasets with minimal effective augmentation
    print("📁 Loading smart datasets...")
    train_dataset = SmartGhanaFoodDataset('train', target_size=(384, 384), use_augmentation=True)
    val_dataset = SmartGhanaFoodDataset('val', target_size=(384, 384), use_augmentation=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Enhanced model (main source of gains)
    print("🤖 Initializing Enhanced GhanaSegNet...")
    model = EnhancedGhanaSegNet(
        num_classes=config['num_classes'],
        backbone='efficientnet-b2'
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Total parameters: {total_params:,}")
    
    # Enhanced loss (second source of gains)
    criterion = EnhancedCombinedLoss(
        alpha=0.4,  # Lovász
        beta=0.4,   # Tversky  
        gamma=0.2,  # Edge
        edge_weight=1.5
    )
    
    # Efficient optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR for faster convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 3,
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    scaler = GradScaler()
    early_stopping = EfficientEarlyStopping(patience=7, min_delta=0.001)
    
    # Setup checkpoints
    checkpoint_dir = "checkpoints/efficient_ghanasegnet"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_iou = 0.0
    train_history = []
    
    print(f"🏁 Starting efficient training for {config['epochs']} epochs...")
    start_time = datetime.now()
    
    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']} - Efficient GhanaSegNet")
        print(f"📈 Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Training
        train_loss = efficient_train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, scaler)
        
        # Validation
        val_metrics = efficient_validate_epoch(model, val_loader, criterion, device, epoch+1)
        
        scheduler.step()
        
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
        
        # Progress tracking
        if val_iou >= 0.30:
            print(f"🎉 TARGET ACHIEVED! IoU: {val_iou*100:.2f}% >= 30%")
        else:
            progress = (val_iou - 0.247) / (0.30 - 0.247) * 100
            print(f"🎯 Progress to 30%: {progress:.1f}% complete")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, f"{checkpoint_dir}/best_efficient_ghanasegnet.pth")
            print(f"💾 New best model saved! IoU: {best_iou:.4f}")
        
        # Early stopping
        if early_stopping(val_iou, model):
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Results
    results = {
        'model_name': 'efficient_ghanasegnet',
        'best_iou': float(best_iou),
        'target_achieved': best_iou >= 0.30,
        'improvement': ((best_iou - 0.247) / 0.247) * 100,
        'training_time': total_time,
        'strategy': 'architecture_first_minimal_augmentation',
        'training_history': train_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{checkpoint_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("🏆 EFFICIENT TRAINING COMPLETED")
    print(f"📊 Best IoU: {best_iou*100:.2f}%")
    print(f"🎯 Target {'ACHIEVED' if best_iou >= 0.30 else 'PROGRESS'}")
    print(f"⏱️ Training time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Efficient GhanaSegNet training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 3e-4,
        'num_classes': 6,
        'strategy': 'efficient_architecture_first'
    }
    
    return train_efficient_ghanasegnet(config)

if __name__ == "__main__":
    main()