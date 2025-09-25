"""
Training Script for GhanaSegNet Advanced
Optimized training pipeline targeting 30%+ mIoU performance

This script provides a clean, efficient training approach focusing on:
- Advanced model architecture
- Optimized loss functions
- Smart augmentation
- Efficient training loop

Author: EricBaidoo
Target: 30%+ mIoU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import time
from datetime import datetime
import json

# Import our advanced model
import sys
sys.path.append('/workspace')
from models.ghanasegnet_advanced import create_ghanasegnet_advanced
from utils.losses import DiceLoss, FocalLoss
from utils.metrics import IoUScore, PixelAccuracy

class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for semantic segmentation
    Optimizes for IoU directly
    """
    def __init__(self, ignore_index=255):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def lovasz_grad(self, gt_sorted):
        """Compute gradient of the Lovász extension w.r.t sorted errors"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if p == 0:
            return torch.zeros_like(gt_sorted)
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def lovasz_softmax(self, probas, labels):
        """Multi-class Lovász-Softmax loss"""
        if probas.ndim != 4:
            raise ValueError('probas should be [B, C, H, W]')
        
        B, C, H, W = probas.shape
        losses = []
        
        for prob, lab in zip(probas, labels):
            lab = lab.view(-1)
            prob = prob.view(C, -1).t()
            
            # Remove ignore pixels
            valid = lab != self.ignore_index
            if valid.sum() == 0:
                continue
                
            lab = lab[valid]
            prob = prob[valid]
            
            for c in range(C):
                # Class c foreground
                fg = (lab == c).float()
                if fg.sum() == 0:
                    continue
                    
                # Class c errors
                errors = (prob[:, c] - fg).abs()
                errors_sorted, perm = torch.sort(errors, descending=True)
                fg_sorted = fg[perm]
                grad = self.lovasz_grad(fg_sorted)
                losses.append(torch.dot(errors_sorted, grad))
        
        return torch.stack(losses).mean() if losses else torch.tensor(0., device=probas.device)
    
    def forward(self, logits, target):
        probas = F.softmax(logits, dim=1)
        return self.lovasz_softmax(probas, target)

class EdgeAwareLoss(nn.Module):
    """
    Edge-aware loss for sharp boundary segmentation
    """
    def __init__(self, edge_weight=2.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def get_edges(self, x):
        """Extract edges using Sobel operator"""
        # Convert to grayscale if multi-channel
        if x.dim() == 4 and x.size(1) > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges
    
    def forward(self, pred, target):
        # Get prediction probabilities
        pred_probs = F.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1, keepdim=True).float()
        
        # Get edges
        pred_edges = self.get_edges(pred_class)
        target_edges = self.get_edges(target.float().unsqueeze(1))
        
        # Edge loss
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        return self.edge_weight * edge_loss

class CombinedLoss(nn.Module):
    """
    Combined loss for optimal segmentation performance
    """
    def __init__(self, num_classes=6, weights=None):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        
        # Loss components
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=255)
        self.edge_loss = EdgeAwareLoss(edge_weight=1.0)
        
        # Loss weights (optimized for performance)
        self.weights = {
            'ce': 0.4,
            'dice': 0.3,
            'lovasz': 0.2,
            'edge': 0.1
        }
        
    def forward(self, pred, target):
        # Individual losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        edge = self.edge_loss(pred, target)
        
        # Combined loss
        total_loss = (
            self.weights['ce'] * ce +
            self.weights['dice'] * dice +
            self.weights['lovasz'] * lovasz +
            self.weights['edge'] * edge
        )
        
        return total_loss, {
            'ce_loss': ce.item(),
            'dice_loss': dice.item(),
            'lovasz_loss': lovasz.item(),
            'edge_loss': edge.item(),
            'total_loss': total_loss.item()
        }

class AdvancedTrainer:
    """
    Advanced trainer for GhanaSegNet targeting 30%+ mIoU
    """
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = CombinedLoss(num_classes=config['num_classes'])
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['initial_lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['max_lr'],
            total_steps=config['total_steps'],
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=10000
        )
        
        # Metrics
        self.iou_metric = IoUScore(num_classes=config['num_classes'])
        self.acc_metric = PixelAccuracy()
        
        # Training state
        self.best_miou = 0.0
        self.epoch = 0
        self.step = 0
        
        # Results tracking
        self.train_history = []
        self.val_history = []
        
        print(f"🚀 Advanced Trainer initialized")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🎯 Target: {config['target_miou']}% mIoU")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'iou': [], 'acc': []}
        
        print(f"\n📈 Epoch {self.epoch + 1} Training:")
        start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Loss computation
            loss, loss_dict = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            with torch.no_grad():
                iou = self.iou_metric(outputs, targets)
                acc = self.acc_metric(outputs, targets)
                
                epoch_losses.append(loss_dict)
                epoch_metrics['iou'].append(iou)
                epoch_metrics['acc'].append(acc)
            
            self.step += 1
            
            # Progress logging
            if batch_idx % self.config['log_interval'] == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx:3d}: Loss={loss:.4f}, IoU={iou:.3f}, Acc={acc:.3f}, LR={lr:.2e}")
                
                # Check for target achievement during training
                if iou * 100 >= self.config['target_miou']:
                    print(f"🎉 TARGET ACHIEVED! mIoU: {iou*100:.2f}% >= {self.config['target_miou']}%")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = np.mean([d['total_loss'] for d in epoch_losses])
        avg_iou = np.mean(epoch_metrics['iou'])
        avg_acc = np.mean(epoch_metrics['acc'])
        
        epoch_summary = {
            'epoch': self.epoch + 1,
            'train_loss': avg_loss,
            'train_iou': avg_iou,
            'train_acc': avg_acc,
            'lr': self.scheduler.get_last_lr()[0],
            'time': epoch_time
        }
        
        self.train_history.append(epoch_summary)
        
        print(f"📊 Train Summary: Loss={avg_loss:.4f}, mIoU={avg_iou*100:.2f}%, Acc={avg_acc*100:.2f}%, Time={epoch_time:.1f}s")
        
        return epoch_summary
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_losses = []
        val_metrics = {'iou': [], 'acc': []}
        
        print(f"🔍 Validation:")
        start_time = time.time()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)
                
                iou = self.iou_metric(outputs, targets)
                acc = self.acc_metric(outputs, targets)
                
                val_losses.append(loss_dict)
                val_metrics['iou'].append(iou)
                val_metrics['acc'].append(acc)
        
        # Validation summary
        val_time = time.time() - start_time
        avg_loss = np.mean([d['total_loss'] for d in val_losses])
        avg_iou = np.mean(val_metrics['iou'])
        avg_acc = np.mean(val_metrics['acc'])
        
        val_summary = {
            'epoch': self.epoch + 1,
            'val_loss': avg_loss,
            'val_iou': avg_iou,
            'val_acc': avg_acc,
            'time': val_time
        }
        
        self.val_history.append(val_summary)
        
        # Check for best model
        current_miou = avg_iou * 100
        if current_miou > self.best_miou:
            self.best_miou = current_miou
            self.save_checkpoint(is_best=True)
            improvement = "🏆 NEW BEST!"
        else:
            improvement = ""
        
        print(f"📋 Val Summary: Loss={avg_loss:.4f}, mIoU={current_miou:.2f}%, Acc={avg_acc*100:.2f}% {improvement}")
        
        # Target achievement check
        if current_miou >= self.config['target_miou']:
            print(f"🎯 TARGET ACHIEVED! Validation mIoU: {current_miou:.2f}% >= {self.config['target_miou']}%")
            print(f"🏁 Training can be stopped or continued for further improvement!")
        
        return val_summary
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'ghanasegnet_advanced_epoch_{self.epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'ghanasegnet_advanced_best.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting Advanced Training for {self.config['epochs']} epochs")
        print(f"🎯 Target: {self.config['target_miou']}% mIoU")
        print("-" * 80)
        
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Train
            train_summary = self.train_epoch()
            
            # Validate
            val_summary = self.validate()
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint()
            
            # Early stopping check
            if val_summary['val_iou'] * 100 >= self.config['target_miou'] and epoch >= self.config['min_epochs']:
                print(f"🎉 TARGET ACHIEVED! Stopping training at epoch {epoch + 1}")
                print(f"🏆 Final mIoU: {val_summary['val_iou']*100:.2f}%")
                break
            
            print("-" * 80)
        
        print(f"✅ Training completed!")
        print(f"🏆 Best mIoU achieved: {self.best_miou:.2f}%")
        
        return self.train_history, self.val_history

def main():
    """Main training function"""
    # Configuration
    config = {
        'num_classes': 6,
        'batch_size': 8,
        'epochs': 100,
        'min_epochs': 20,
        'initial_lr': 1e-4,
        'max_lr': 1e-3,
        'weight_decay': 1e-4,
        'target_miou': 30.0,  # 30% target
        'save_dir': './checkpoints/advanced',
        'log_interval': 10,
        'save_interval': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🔧 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create model
    model = create_ghanasegnet_advanced(num_classes=config['num_classes'])
    
    # Calculate total steps for scheduler
    # NOTE: Replace with actual data loader lengths
    train_steps_per_epoch = 100  # Replace with len(train_loader)
    config['total_steps'] = train_steps_per_epoch * config['epochs']
    
    # TODO: Load your actual data loaders here
    # train_loader = YourDataLoader(...)
    # val_loader = YourDataLoader(...)
    
    print("⚠️  Please replace the placeholder data loaders with your actual data!")
    print("📁 Expected data structure:")
    print("   - train_loader: DataLoader for training data")
    print("   - val_loader: DataLoader for validation data")
    print("   - Images: (B, C, H, W) tensors")
    print("   - Targets: (B, H, W) long tensors with class indices")
    
    # Uncomment when you have actual data loaders:
    """
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        config=config
    )
    
    # Start training
    train_history, val_history = trainer.train()
    
    print("🎉 Training completed successfully!")
    print(f"🏆 Best mIoU: {trainer.best_miou:.2f}%")
    """

if __name__ == "__main__":
    main()