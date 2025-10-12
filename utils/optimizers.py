"""
Advanced Learning Rate Scheduler for Enhanced GhanaSegNet
Designed for 30% mIoU achievement within 15 epochs

Features:
- Cosine annealing with warm restarts
- Linear warmup for stable initial training
- Adaptive learning rate based on validation performance
- Optimized for food segmentation task
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """
    Cosine annealing with warm restarts and initial warmup
    Optimized for Enhanced GhanaSegNet 30% mIoU target
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=2, last_epoch=-1, verbose=False):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.T_cur = 0
        self.T_i = T_0
        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Adjust for warmup period
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            
            if adjusted_epoch == 0:
                self.T_cur = 0
            elif adjusted_epoch % self.T_i == 0:
                self.T_cur = 0
                self.T_i *= self.T_mult
            else:
                self.T_cur += 1
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                   for base_lr in self.base_lrs]

def create_optimized_optimizer_and_scheduler(model, config):
    """
    Create optimized optimizer and scheduler for 30% mIoU target
    """
    # Enhanced AdamW optimizer with carefully tuned parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 2e-4),      # Increased from 1e-4
        weight_decay=config.get('weight_decay', 1e-3),  # Increased regularization
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True  # More stable convergence
    )
    
    # Advanced cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=1,  # Keep same cycle length
        eta_min=1e-6,  # Minimum learning rate
        warmup_epochs=2,  # 2 epochs warmup
        verbose=True
    )
    
    return optimizer, scheduler

class AdaptiveLRScheduler:
    """
    Adaptive learning rate scheduler based on validation performance
    Reduces LR when validation IoU plateaus
    """
    def __init__(self, optimizer, patience=3, factor=0.5, min_lr=1e-6):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_val_iou = 0
        self.patience_counter = 0
        
    def step(self, val_iou):
        if val_iou > self.best_val_iou:
            self.best_val_iou = val_iou
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self._reduce_lr()
            self.patience_counter = 0
            
    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")

def get_progressive_training_config(epoch, total_epochs=15):
    """
    Progressive training configuration for different training phases
    Optimized for 30% mIoU within 15 epochs
    """
    progress = epoch / total_epochs
    
    if progress < 0.3:  # Early phase (epochs 1-4)
        return {
            'mixup_alpha': 0.2,
            'cutmix_prob': 0.0,
            'augmentation_strength': 0.3,
            'loss_weights': {'dice': 0.5, 'focal': 0.3, 'boundary': 0.2}
        }
    elif progress < 0.7:  # Middle phase (epochs 5-10)
        return {
            'mixup_alpha': 0.4,
            'cutmix_prob': 0.3,
            'augmentation_strength': 0.5,
            'loss_weights': {'dice': 0.4, 'focal': 0.4, 'boundary': 0.2}
        }
    else:  # Late phase (epochs 11-15)
        return {
            'mixup_alpha': 0.1,
            'cutmix_prob': 0.1,
            'augmentation_strength': 0.2,
            'loss_weights': {'dice': 0.3, 'focal': 0.4, 'boundary': 0.3}
        }