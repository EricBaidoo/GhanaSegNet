#!/usr/bin/env python3
"""
Multi-Model Training Script for GhanaSegNet Research Project
Trains baseline models: UNet, DeepLabV3+, SegFormer-B0, GhanaSegNet
Author: EricBaidoo
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime

# Add parent directory to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import numpy as np

# Import baseline models
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerB0
from models.ghanasegnet import GhanaSegNet

# Import utilities
from data.dataset_loader import GhanaFoodDataset
from utils.losses import CombinedLoss
from utils.metrics import compute_iou, compute_pixel_accuracy
from utils.optimizers import create_optimized_optimizer_and_scheduler, get_progressive_training_config

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    Monitors validation score and restores best weights if needed.
    """
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
    """
    Initialize model and its original loss function for fair comparison.
    """
    models = {
        'unet': lambda: UNet(n_channels=3, n_classes=num_classes),
        'deeplabv3plus': lambda: DeepLabV3Plus(num_classes=num_classes),
        'segformer': lambda: SegFormerB0(num_classes=num_classes),
        'ghanasegnet': lambda: GhanaSegNet(num_classes=num_classes)
    }
    original_losses = {
        'unet': nn.CrossEntropyLoss(),
        'deeplabv3plus': nn.CrossEntropyLoss(),
        'segformer': nn.CrossEntropyLoss(),
        'ghanasegnet': CombinedLoss(alpha=0.7, aux_weight=0.4)  # Enhanced for 30% mIoU target
    }
    paper_refs = {
        'unet': 'Ronneberger et al., 2015', 
        'deeplabv3plus': 'Chen et al., 2018', 
        'segformer': 'Xie et al., 2021',
        'ghanasegnet': 'Baidoo, E. (Enhanced Architecture - ASPP + 8-Head Transformer)'
    }
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    model = models[model_name.lower()]()
    criterion = original_losses[model_name.lower()]
    print(f"Using ORIGINAL loss for {model_name}: {type(criterion).__name__}")
    if model_name == 'ghanasegnet':
        print("Enhanced 30% mIoU loss: Multi-scale supervision + Dice + Focal + Boundary + Class-balanced")
    print(f"Paper reference: {paper_refs[model_name]}")
    return model, criterion

def set_seed(seed, benchmark_mode=True):
    """
    Set random seed for reproducible benchmarking
    
    For research benchmarking, we need:
    1. Reproducible results within the same model
    2. Fair comparison across different models 
    3. Different random initialization per model (avoid identical weights)
    
    Args:
        seed: Random seed to use
        benchmark_mode: If True, enables deterministic operations for reproducible benchmarking
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if benchmark_mode:
        # For benchmarking: ensure deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"BENCHMARK MODE: Deterministic operations enabled (seed={seed})")
        print("   Results will be reproducible across runs")
        print("   Training may be slower due to deterministic operations")
    else:
        # For fast training: allow non-deterministic but faster operations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"FAST MODE: Non-deterministic operations enabled (seed={seed})")
        print("   Training will be faster")
        print("   Results may vary slightly between runs")

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, model_name):
    """
    Train for one epoch. Applies gradient clipping for stability.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training [{model_name}]")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle auxiliary outputs for enhanced GhanaSegNet
        if model_name == 'ghanasegnet' and isinstance(outputs, tuple):
            main_outputs, aux_outputs = outputs
            loss = criterion(main_outputs, masks, aux_outputs)
        else:
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take main output if tuple returned
            loss = criterion(outputs, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / total_samples

def validate_epoch(model, val_loader, criterion, device, epoch, model_name, num_classes=6):
    """
    Validate for one epoch. Computes loss, IoU, and pixel accuracy.
    """
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
            
            # Handle auxiliary outputs for enhanced GhanaSegNet validation
            if model_name == 'ghanasegnet' and isinstance(outputs, tuple):
                main_outputs, aux_outputs = outputs
                loss = criterion(main_outputs, masks, aux_outputs)
                preds = torch.argmax(main_outputs, dim=1)
            else:
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take main output if tuple returned
                loss = criterion(outputs, masks)
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
    """
    Main training function for a single model.
    Handles data loading, training loop, validation, checkpointing, and logging.
    """
    print(f"Starting training for {model_name.upper()}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # BENCHMARKING SETUP: Model-specific seeds for fair comparison
    # Each model gets a different seed to ensure:
    # 1. Different random weight initialization 
    # 2. Different data shuffling patterns
    # 3. Reproducible results within each model
    # 4. Fair comparison across models
    model_seeds = {
        'unet': 42,           # Standard baseline seed
        'deeplabv3plus': 123, # Different seed for fair comparison  
        'segformer': 456,     # Transformer-based model
        'ghanasegnet': 789    # Our novel enhanced architecture
    }
    
    # Use custom seed if provided, otherwise use model-specific seed
    seed = config.get('custom_seed') or model_seeds.get(model_name.lower(), 42)
    benchmark_mode = config.get('benchmark_mode', True)
    set_seed(seed, benchmark_mode)
    
    if config.get('custom_seed'):
        print(f"CUSTOM SEED: Using seed {seed} for {model_name.upper()}")
    else:
        print(f"BENCHMARKING: Using seed {seed} for {model_name.upper()}")
        print("   - Ensures reproducible results for this model")
        print("   - Different from other models for fair comparison")
        if benchmark_mode:
            print("   - Deterministic training for research validity")
        else:
            print("   - Fast mode enabled (non-deterministic operations)")
    
    # Handle device selection
    if config.get('device') == 'auto' or config.get('device') is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"Using device: {device}")

    # Set input sizes based on original papers
    model_input_sizes = {
        'unet': (572, 572),
        'deeplabv3plus': (513, 513),
        'segformer': (512, 512),
        'ghanasegnet': (384, 384)
    }
    input_size = model_input_sizes.get(model_name.lower(), (256, 256))

    print("Loading datasets...")
    data_root = config.get('dataset_path', 'data')
    print(f"Using dataset path: {data_root}")
    
    # Try new dataset loader with data_root parameter, fallback to old version
    try:
        train_dataset = GhanaFoodDataset('train', target_size=input_size, data_root=data_root)
        val_dataset = GhanaFoodDataset('val', target_size=input_size, data_root=data_root)
        print(f"✅ Using enhanced dataset loader with custom data path")
    except TypeError:
        # Fallback for older dataset loader without data_root parameter
        print(f"⚠️  Using fallback dataset loader (expects data in './data' folder)")
        train_dataset = GhanaFoodDataset('train', target_size=input_size)
        val_dataset = GhanaFoodDataset('val', target_size=input_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0,
        drop_last=True
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize model and criterion
    print(f"Initializing {model_name} model...")
    model, criterion = get_model_and_criterion(model_name, config['num_classes'])
    model = model.to(device)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and scheduler (Enhanced for better convergence)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),  # Slightly adjusted for transformer components
        eps=1e-8
    )
    
    # Enhanced learning rate scheduling for better convergence
    if config['epochs'] <= 20:  # For short training runs
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.7,  # Less aggressive for short training
            patience=5,  # More patience for short runs
            min_lr=1e-6,
            threshold=0.001
        )
    else:  # For longer training runs
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=0.5, 
            patience=3,
            min_lr=1e-6,
            threshold=0.0001
        )
    # Adaptive early stopping based on training length
    if config['epochs'] <= 20:  # For short training runs
        early_stopping = EarlyStopping(patience=10, min_delta=0.0005)  # More lenient for quick training
    else:  # For longer training runs
        early_stopping = EarlyStopping(patience=20, min_delta=0.0001)

    # Checkpoint directory
    checkpoint_dir = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_iou = 0.0
    train_history = []

    print(f"Starting enhanced training for {config['epochs']} epochs (with early stopping)...")
    
    # Warmup for transformer-based models (GhanaSegNet benefits from this)
    warmup_epochs = min(3, config['epochs'] // 4) if model_name == 'ghanasegnet' else 0
    base_lr = config['learning_rate']
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']} - {model_name.upper()}")
        
        # Apply warmup for transformer-based models
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warmup LR: {warmup_lr:.2e} (epoch {epoch+1}/{warmup_epochs})")
        else:
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1, model_name)
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch+1, model_name, config['num_classes'])
        
        # Only step scheduler after warmup period
        if warmup_epochs == 0 or epoch >= warmup_epochs:
            scheduler.step(val_metrics['iou'])

        val_loss = val_metrics['loss']
        val_iou = val_metrics['iou']
        val_accuracy = val_metrics['accuracy']

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_accuracy': val_accuracy,
            'lr': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_data)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)")
        print(f"Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        # Save best model
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
            print(f"New best model saved! IoU: {best_iou:.4f}")

        # Early stopping
        if early_stopping(val_iou, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best IoU achieved: {best_iou:.4f}")
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

    # Save training history and results
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

    print(f"Training completed for {model_name.upper()}!")
    print(f"Best validation IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"Results saved to: {checkpoint_dir}")

    return {
        'best_iou': best_iou,
        'final_epoch': epoch + 1,
        'model_name': model_name,
        'total_params': total_params
    }

def main():
    """
    Argument parsing and main entry point.
    Supports training a single model or all models for comparison.
    """
    parser = argparse.ArgumentParser(description='Train baseline models for GhanaSegNet research')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--dataset-path', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Override default model-specific seeds (for debugging only)')
    parser.add_argument('--benchmark-mode', action='store_true', default=True,
                       help='Enable deterministic operations for benchmarking (default: True)')
    parser.add_argument('--fast-mode', action='store_true', 
                       help='Disable deterministic ops for faster training (not recommended for benchmarking)')
    args = parser.parse_args()

    # Handle benchmarking mode
    benchmark_mode = args.benchmark_mode and not args.fast_mode
    
    config = {
        'epochs': args.epochs,
        'batch_size': getattr(args, 'batch_size', 8),
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'num_classes': args.num_classes,
        'custom_seed': args.seed,
        'benchmark_mode': benchmark_mode,
        'dataset_path': getattr(args, 'dataset_path', 'data'),
        'device': getattr(args, 'device', 'auto'),
        'timestamp': datetime.now().isoformat(),
        'note': 'Using ORIGINAL loss functions for fair baseline comparison'
    }
    
    # Print benchmarking info
    if benchmark_mode:
        print("BENCHMARK MODE ENABLED")
        print("   Deterministic operations for reproducible results")
        print("   Model-specific seeds for fair comparison")
        print("   Training may be slower but results are reproducible")
    else:
        print("FAST MODE ENABLED") 
        print("   Non-deterministic operations for faster training")
        print("   Results may vary slightly between runs")
    print()

    if args.model == 'all':
        models_to_train = ['unet', 'deeplabv3plus', 'segformer', 'ghanasegnet']
        results = {}
        print(f"\nSTARTING COMPREHENSIVE MODEL COMPARISON")
        print(f"Models to train: {', '.join([m.upper() for m in models_to_train])}")
        print(f"{'='*80}")
        for i, model_name in enumerate(models_to_train, 1):
            print(f"\n{'='*80}")
            print(f"TRAINING MODEL {i}/{len(models_to_train)}: {model_name.upper()}")
            print(f"{'='*80}")
            try:
                result = train_model(model_name, config)
                results[model_name] = {
                    'best_iou': result['best_iou'],
                    'status': 'completed',
                    'model_type': 'baseline' if model_name != 'ghanasegnet' else 'novel'
                }
                print(f"{model_name.upper()} completed - Best IoU: {result['best_iou']:.4f}")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'best_iou': 0.0,
                    'status': f'failed: {e}',
                    'model_type': 'baseline' if model_name != 'ghanasegnet' else 'novel'
                }
        os.makedirs('checkpoints', exist_ok=True)
        with open('checkpoints/training_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n{'='*80}")
        print("FINAL MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<15} {'IoU':<10} {'Type':<10} {'Status'}")
        print(f"{'-'*50}")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['best_iou'], reverse=True)
        for rank, (model, result) in enumerate(sorted_results, 1):
            status_icon = "OK" if result['status'] == 'completed' else "FAIL"
            model_type = result['model_type'].capitalize()
            print(f"#{rank} {model.upper():<12} {result['best_iou']:.4f}     {model_type:<10} {status_icon}")
        if 'ghanasegnet' in results:
            ghanasegnet_result = results['ghanasegnet']
            baseline_ious = [r['best_iou'] for m, r in results.items() if m != 'ghanasegnet' and r['status'] == 'completed']
            if baseline_ious and ghanasegnet_result['status'] == 'completed':
                best_baseline = max(baseline_ious)
                improvement = ghanasegnet_result['best_iou'] - best_baseline
                print(f"\nNOVEL MODEL ANALYSIS:")
                print(f"   GhanaSegNet IoU: {ghanasegnet_result['best_iou']:.4f}")
                print(f"   Best Baseline IoU: {best_baseline:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement/best_baseline*100:+.1f}%)")
        print(f"\nResults saved to: checkpoints/training_summary.json")
    else:
        train_model(args.model, config)

def enhanced_train_model(model_name='enhanced_ghanasegnet', epochs=15, batch_size=8, 
                        learning_rate=2.5e-4, weight_decay=1.2e-3, num_classes=6,
                        dataset_path='data', device='cuda', disable_early_stopping=True,
                        use_cosine_schedule=True, use_progressive_training=True,
                        mixed_precision=True, benchmark_mode=True, custom_seed=789):
    """
    Enhanced training function optimized for 30% mIoU target in 15 epochs
    """
    print("🚀 ENHANCED GHANASEGNET - AMBITIOUS 15-EPOCH TRAINING")
    print("="*60)
    print(f"🎯 TARGET: 30% mIoU | REALISTIC: 27-28% mIoU")
    print(f"🔧 ALL OPTIMIZATIONS ACTIVE")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(custom_seed)
    
    # Initialize enhanced model
    from models.ghanasegnet import EnhancedGhanaSegNet
    model = EnhancedGhanaSegNet(num_classes=num_classes).to(device)
    
    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Enhanced Model: {total_params:,} parameters")
    
    # Create optimized optimizer and scheduler
    config = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epochs': epochs
    }
    
    if use_cosine_schedule:
        optimizer, scheduler = create_optimized_optimizer_and_scheduler(model, config)
        print(f"✅ Cosine annealing scheduler with warmup")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Enhanced loss function
    criterion = CombinedLoss(alpha=0.6, aux_weight=0.4, adaptive_weights=True).to(device)
    print(f"✅ Advanced boundary-aware loss function")
    
    # Load dataset
    try:
        train_dataset = GhanaFoodDataset(dataset_path, split='train', data_root=dataset_path)
        val_dataset = GhanaFoodDataset(dataset_path, split='val', data_root=dataset_path)
    except:
        # Fallback for different dataset structure
        train_dataset = GhanaFoodDataset('data', split='train')
        val_dataset = GhanaFoodDataset('data', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Training tracking
    best_val_iou = 0.0
    training_history = []
    milestone_alerts = [25.0, 27.0, 28.0, 29.0, 30.0]  # mIoU milestones
    achieved_milestones = set()
    
    # Mixed precision training
    if mixed_precision and device == 'cuda':
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print(f"✅ Mixed precision training enabled")
    else:
        scaler = None
    
    print(f"\n🚀 STARTING AMBITIOUS 15-EPOCH TRAINING...")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Get progressive training configuration
        if use_progressive_training:
            prog_config = get_progressive_training_config(epoch, epochs)
            print(f"\n📊 EPOCH {epoch}/{epochs} - Progressive Config:")
            print(f"   Mixup: {prog_config['mixup_alpha']:.1f}, Augmentation: {prog_config['augmentation_strength']:.1f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            if scaler:  # Mixed precision
                with autocast():
                    outputs = model(images)
                    if isinstance(outputs, tuple):  # Handle auxiliary outputs
                        main_output, aux_outputs = outputs
                        loss = criterion(main_output, masks, aux_outputs)
                    else:
                        loss = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Regular training
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs
                    loss = criterion(main_output, masks, aux_outputs)
                else:
                    loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_samples += images.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        val_samples = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                
                if scaler:
                    with autocast():
                        outputs = model(images)
                        if isinstance(outputs, tuple):
                            main_output = outputs[0]
                        else:
                            main_output = outputs
                        loss = criterion(main_output, masks)
                else:
                    outputs = model(images)
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]
                    else:
                        main_output = outputs
                    loss = criterion(main_output, masks)
                
                val_loss += loss.item()
                
                # Compute metrics
                iou = compute_iou(main_output, masks)
                accuracy = compute_pixel_accuracy(main_output, masks)
                
                total_iou += iou
                total_accuracy += accuracy
                val_samples += images.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{iou:.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        avg_val_accuracy = total_accuracy / len(val_loader)
        
        # Learning rate scheduling
        if use_cosine_schedule:
            scheduler.step()
        else:
            scheduler.step(avg_val_iou)
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Check for new best
        is_best = avg_val_iou > best_val_iou
        if is_best:
            best_val_iou = avg_val_iou
            # Save best model
            os.makedirs(f'checkpoints/{model_name}', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'config': config
            }, f'checkpoints/{model_name}/best_model.pth')
        
        # Check milestones
        current_miou_percent = avg_val_iou * 100
        for milestone in milestone_alerts:
            if current_miou_percent >= milestone and milestone not in achieved_milestones:
                achieved_milestones.add(milestone)
                print(f"\n🎉 MILESTONE ACHIEVED: {milestone:.1f}% mIoU!")
                if milestone >= 30.0:
                    print(f"🏆 TARGET REACHED! 30% mIoU ACHIEVED AT EPOCH {epoch}!")
        
        # Record epoch results
        epoch_data = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_iou': avg_val_iou,
            'val_accuracy': avg_val_accuracy,
            'learning_rate': current_lr,
            'is_best': is_best,
            'time': epoch_time
        }
        training_history.append(epoch_data)
        
        # Progress report
        print(f"\n📊 EPOCH {epoch}/{epochs} RESULTS:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Val IoU: {avg_val_iou:.4f} ({current_miou_percent:.2f}%)")
        print(f"   Val Accuracy: {avg_val_accuracy:.4f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        print(f"   Best IoU: {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")
        print(f"   Epoch Time: {epoch_time:.1f}s")
        
        if is_best:
            print(f"   🎯 NEW BEST PERFORMANCE!")
        
        # Progress toward 30% target
        progress_to_target = (current_miou_percent - 24.4) / (30.0 - 24.4) * 100
        print(f"   📈 Progress to 30% target: {progress_to_target:.1f}%")
    
    # Final results
    print(f"\n" + "="*60)
    print(f"🏁 ENHANCED GHANASEGNET 15-EPOCH TRAINING COMPLETE!")
    print(f"="*60)
    print(f"🎯 FINAL RESULTS:")
    print(f"   Best mIoU: {best_val_iou:.4f} ({best_val_iou*100:.2f}%)")
    print(f"   Target: 30.00%")
    print(f"   Gap: {30.0 - best_val_iou*100:+.2f} percentage points")
    
    if best_val_iou >= 0.30:
        print(f"🏆 TARGET ACHIEVED! 30%+ mIoU reached!")
    elif best_val_iou >= 0.28:
        print(f"🎉 EXCELLENT! Within 2% of target!")
    elif best_val_iou >= 0.27:
        print(f"✅ GREAT! Solid improvement achieved!")
    else:
        print(f"📊 Results within expected range.")
    
    print(f"   Achieved milestones: {sorted(achieved_milestones)}")
    
    # Save training history
    with open(f'checkpoints/{model_name}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return {
        'best_val_iou': best_val_iou,
        'final_val_iou': training_history[-1]['val_iou'],
        'training_history': training_history,
        'achieved_milestones': list(achieved_milestones),
        'target_achieved': best_val_iou >= 0.30
    }

if __name__ == "__main__":
    main()
