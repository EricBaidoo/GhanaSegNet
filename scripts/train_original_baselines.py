#!/usr/bin/env python3
"""
Original Baseline Training Script for GhanaSegNet Research
Trains pure, unmodified baseline models for fair comparison:
- UNet (Ronneberger et al., 2015) - Original without BatchNorm
- DeepLabV3+ (Chen et al., 2018) - Standard implementation
- SegFormer-B0 (Xie et al., 2021) - Random init, no pre-training

Author: EricBaidoo
Date: August 2025
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
import numpy as np

# Import ORIGINAL baseline models
from models.unet import UNetOriginal
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerOriginal

# Import utilities
from data.dataset_loader import GhanaFoodDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy, compute_f1_per_class

def get_original_model(model_name, num_classes=6):
    """
    Initialize ORIGINAL baseline models (no modifications)
    """
    models = {
        'unet': lambda: UNetOriginal(n_channels=3, n_classes=num_classes, bilinear=False),
        'deeplabv3plus': lambda: DeepLabV3Plus(num_classes=num_classes),
        'segformer': lambda: SegFormerOriginal(num_classes=num_classes)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    
    print(f"🔧 Initializing ORIGINAL {model_name.upper()} (no modifications)")
    return models[model_name.lower()]()

def train_epoch(model, train_loader, criterion, optimizer, device, model_name):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Training {model_name.upper()}")
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images, masks = images.to(device), masks.to(device)
            # Debug: print shapes before forward and loss
            print(f"[DEBUG] Batch {batch_idx} - images shape: {images.shape}, masks shape: {masks.shape}")
            outputs = model(images)
            print(f"[DEBUG] Batch {batch_idx} - outputs shape: {outputs.shape}")
            # Assert mask shape is [batch_size, H, W]
            assert masks.dim() == 3, f"Expected masks shape [B, H, W], got {masks.shape}"
            # Assert output shape matches expected for segmentation
            assert outputs.shape[0] == masks.shape[0], f"Batch size mismatch: outputs {outputs.shape}, masks {masks.shape}"
            optimizer.zero_grad()
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        except Exception as e:
            print(f" Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / total_samples if total_samples > 0 else float('inf')

def validate_epoch(model, val_loader, criterion, device, num_classes, model_name):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating {model_name.upper()}")
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images, masks = images.to(device), masks.to(device)
                # Debug: print shapes before forward and loss
                print(f"[DEBUG] [VAL] Batch {batch_idx} - images shape: {images.shape}, masks shape: {masks.shape}")
                outputs = model(images)
                print(f"[DEBUG] [VAL] Batch {batch_idx} - outputs shape: {outputs.shape}")
                assert masks.dim() == 3, f"Expected masks shape [B, H, W], got {masks.shape}"
                assert outputs.shape[0] == masks.shape[0], f"Batch size mismatch: outputs {outputs.shape}, masks {masks.shape}"
                loss = criterion(outputs, masks)
                preds = torch.argmax(outputs, dim=1)
                iou = compute_iou(preds, masks, num_classes)
                acc = compute_pixel_accuracy(preds, masks)
                total_loss += loss.item() * images.size(0)
                total_iou += iou * images.size(0)
                total_acc += acc * images.size(0)
                total_samples += images.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(masks.cpu())
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'acc': f'{acc:.4f}'
                })
            except Exception as e:
                print(f" Error in validation batch {batch_idx}: {e}")
                continue
    
    # Compute F1 scores
    if all_preds and all_labels:
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        f1_scores = compute_f1_per_class(preds_cat, labels_cat, num_classes)
    else:
        f1_scores = np.zeros(num_classes)
    
    return {
        'loss': total_loss / total_samples if total_samples > 0 else float('inf'),
        'iou': total_iou / total_samples if total_samples > 0 else 0.0,
        'accuracy': total_acc / total_samples if total_samples > 0 else 0.0,
        'f1_scores': f1_scores,
        'mean_f1': np.mean(f1_scores)
    }

def train_baseline_model(model_name, config):
    """Main training function for baseline models"""
    print(f"\n{'='*60}")
    print(f" TRAINING ORIGINAL {model_name.upper()} BASELINE")
    print(f" Ensuring NO modifications for fair comparison")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create datasets
    print("Loading FRANI dataset...")
    train_dataset = GhanaFoodDataset('train')
    val_dataset = GhanaFoodDataset('val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f" Train samples: {len(train_dataset)}")
    print(f" Val samples: {len(val_dataset)}")
    
    # Initialize ORIGINAL model
    print(f" Initializing ORIGINAL {model_name} model...")
    model = get_original_model(model_name, config['num_classes'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Total parameters: {total_params:,}")
    print(f" Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=config['loss_alpha'])
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
    checkpoint_dir = f"checkpoints/baselines/{model_name}_original"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model configuration
    model_config = {
        'model_name': model_name,
        'model_type': 'original_baseline',
        'modifications': 'none',
        'paper_reference': {
            'unet': 'Ronneberger et al., 2015',
            'deeplabv3plus': 'Chen et al., 2018', 
            'segformer': 'Xie et al., 2021'
        }.get(model_name, 'unknown'),
        'training_config': config
    }
    
    with open(f"{checkpoint_dir}/model_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Training loop
    best_iou = 0.0
    best_f1 = 0.0
    train_history = []
    
    print(f" Starting training for {config['epochs']} epochs...")
    
    for epoch in range(config['epochs']):
        print(f"\n Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, model_name)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, 
                                   config['num_classes'], model_name)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_iou': val_metrics['iou'],
            'val_accuracy': val_metrics['accuracy'],
            'val_mean_f1': val_metrics['mean_f1'],
            'val_f1_scores': val_metrics['f1_scores'].tolist(),
            'lr': scheduler.get_last_lr()[0]
        }
        train_history.append(epoch_data)
        
        print(f" Train Loss: {train_loss:.4f}")
        print(f" Val Loss: {val_metrics['loss']:.4f}")
        print(f" Val IoU: {val_metrics['iou']:.4f}")
        print(f" Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f" Val Mean F1: {val_metrics['mean_f1']:.4f}")
        print(f" Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model (based on IoU)
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_f1 = val_metrics['mean_f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'best_f1': best_f1,
                'config': config,
                'model_config': model_config
            }, f"{checkpoint_dir}/best_model.pth")
            print(f" New best model saved! IoU: {best_iou:.4f}, F1: {best_f1:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_iou': best_iou,
            'best_f1': best_f1,
            'config': config,
            'model_config': model_config
        }, f"{checkpoint_dir}/latest_model.pth")
        
        # Save training history every 10 epochs
        if (epoch + 1) % 10 == 0:
            with open(f"{checkpoint_dir}/training_history.json", 'w') as f:
                json.dump(train_history, f, indent=2)
    
    # Save final training history
    with open(f"{checkpoint_dir}/training_history.json", 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"\n Training completed for {model_name.upper()}!")
    print(f" Best IoU: {best_iou:.4f}")
    print(f" Best F1: {best_f1:.4f}")
    print(f" Models saved in: {checkpoint_dir}")
    
    return {
        'model_name': model_name,
        'best_iou': best_iou,
        'best_f1': best_f1,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'status': 'completed',
        'checkpoint_dir': checkpoint_dir
    }

def main():
    parser = argparse.ArgumentParser(description='Train ORIGINAL baseline models for GhanaSegNet research')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['unet', 'deeplabv3plus', 'segformer', 'all'],
                       help='Original baseline model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
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
        'loss_alpha': 0.8,  # For CombinedLoss
        'timestamp': datetime.now().isoformat(),
        'note': 'Original baseline models - no modifications for fair comparison'
    }
    
    # Train models
    if args.model == 'all':
        models_to_train = ['unet', 'deeplabv3plus', 'segformer']
        results = {}
        
        print(f"\n{'='*80}")
        print(f" TRAINING ALL ORIGINAL BASELINE MODELS")
        print(f" For GhanaSegNet Research - Fair Comparison")
        print(f"{'='*80}")
        
        for model_name in models_to_train:
            try:
                result = train_baseline_model(model_name, config)
                results[model_name] = result
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                results[model_name] = {
                    'model_name': model_name,
                    'best_iou': 0.0,
                    'best_f1': 0.0,
                    'status': f'failed: {str(e)}'
                }
        
        # Save overall results
        os.makedirs('checkpoints/baselines', exist_ok=True)
        with open('checkpoints/baselines/training_summary.json', 'w') as f:
            json.dump({
                'training_date': datetime.now().isoformat(),
                'config': config,
                'results': results,
                'note': 'Original baseline models trained for GhanaSegNet comparison'
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print("BASELINE TRAINING SUMMARY")
        print(f"{'='*80}")
        for model, result in results.items():
            status = result['status']
            if status == 'completed':
                print(f" {model.upper():12} | IoU: {result['best_iou']:.4f} | F1: {result['best_f1']:.4f} | Params: {result.get('total_params', 0):,}")
            else:
                print(f" {model.upper():12} | {status}")
        print(f"{'='*80}")
        
    else:
        result = train_baseline_model(args.model, config)
        print(f"\n Training completed: {result}")

if __name__ == "__main__":
    main()
