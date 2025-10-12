"""
Advanced Dynamic Loss Weighting for Enhanced GhanaSegNet
Implements class balancing and adaptive loss weighting strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicClassBalancedLoss(nn.Module):
    """
    Dynamic class-balanced focal loss with adaptive weighting
    Adjusts class weights and focal parameters during training
    """
    def __init__(self, num_classes=6, alpha=None, gamma=2.0, adaptive_gamma=True):
        super(DynamicClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.adaptive_gamma = adaptive_gamma
        
        # Initialize class weights (will be updated dynamically)
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha)
        
        # Track class statistics for dynamic weighting
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_samples', torch.tensor(0.0))
        
        # Effective number of samples for class balancing
        self.beta = 0.9999
    
    def update_class_statistics(self, targets):
        """Update class occurrence statistics"""
        unique, counts = torch.unique(targets, return_counts=True)
        
        for cls, count in zip(unique, counts):
            if cls < self.num_classes:
                self.class_counts[cls] += count.float()
        
        self.total_samples += targets.numel()
    
    def compute_effective_numbers(self):
        """Compute effective number of samples for each class"""
        effective_nums = (1.0 - torch.pow(self.beta, self.class_counts)) / (1.0 - self.beta)
        effective_nums = torch.clamp(effective_nums, min=1.0)
        return effective_nums
    
    def compute_class_weights(self):
        """Compute dynamic class weights based on effective numbers"""
        effective_nums = self.compute_effective_numbers()
        weights = 1.0 / effective_nums
        weights = weights / weights.sum() * self.num_classes  # Normalize
        return weights
    
    def adaptive_gamma_schedule(self, epoch, max_epochs):
        """Adaptive gamma schedule: start high, decrease over time"""
        if not self.adaptive_gamma:
            return self.gamma
        
        # Start with higher gamma (more focus on hard examples)
        # Gradually decrease to focus on all examples
        min_gamma = 1.0
        max_gamma = 3.0
        progress = epoch / max_epochs
        gamma = max_gamma - (max_gamma - min_gamma) * progress
        return max(gamma, min_gamma)
    
    def forward(self, inputs, targets, epoch=0, max_epochs=100):
        """
        Args:
            inputs: Model predictions [B, C, H, W]
            targets: Ground truth labels [B, H, W]
            epoch: Current training epoch
            max_epochs: Total training epochs
        """
        # Ensure targets are correct shape
        if targets.dim() > 3:
            targets = targets.squeeze()
        if targets.dim() == 2:
            targets = targets.unsqueeze(0)
        
        # Update class statistics
        self.update_class_statistics(targets)
        
        # Compute dynamic weights
        class_weights = self.compute_class_weights().to(inputs.device)
        
        # Adaptive gamma
        current_gamma = self.adaptive_gamma_schedule(epoch, max_epochs)
        
        # Compute focal loss with dynamic weights
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class weights
        weight_t = class_weights[targets.long()]
        
        # Focal loss formulation
        focal_loss = weight_t * (1 - pt) ** current_gamma * ce_loss
        
        return focal_loss.mean()


class CurriculumLossScheduler:
    """
    Curriculum learning scheduler for loss components
    Gradually introduces more complex loss terms
    """
    def __init__(self, main_loss, boundary_loss, consistency_loss=None):
        self.main_loss = main_loss
        self.boundary_loss = boundary_loss
        self.consistency_loss = consistency_loss
        
        # Loss weights schedule
        self.main_weight_schedule = [1.0, 0.8, 0.6]      # Decrease main loss weight
        self.boundary_weight_schedule = [0.0, 0.3, 0.5]  # Increase boundary focus
        self.consistency_weight_schedule = [0.0, 0.1, 0.2] if consistency_loss else [0.0, 0.0, 0.0]
    
    def get_loss_weights(self, epoch, max_epochs):
        """Get current loss weights based on training progress"""
        # Divide training into 3 phases
        phase_length = max_epochs // 3
        
        if epoch < phase_length:
            phase = 0
        elif epoch < 2 * phase_length:
            phase = 1
        else:
            phase = 2
        
        return {
            'main_weight': self.main_weight_schedule[phase],
            'boundary_weight': self.boundary_weight_schedule[phase],
            'consistency_weight': self.consistency_weight_schedule[phase]
        }
    
    def compute_loss(self, inputs, targets, epoch, max_epochs, consistency_inputs=None):
        """Compute curriculum-weighted loss"""
        weights = self.get_loss_weights(epoch, max_epochs)
        
        # Main loss
        main_loss = self.main_loss(inputs, targets, epoch, max_epochs)
        total_loss = weights['main_weight'] * main_loss
        
        # Boundary loss
        if weights['boundary_weight'] > 0:
            boundary_loss = self.boundary_loss(inputs, targets)
            total_loss += weights['boundary_weight'] * boundary_loss
        
        # Consistency loss (if available)
        if self.consistency_loss and weights['consistency_weight'] > 0 and consistency_inputs is not None:
            consistency_loss = self.consistency_loss(inputs, consistency_inputs)
            total_loss += weights['consistency_weight'] * consistency_loss
        
        return total_loss


class AdaptiveLossBalancer:
    """
    Automatically balance multiple loss components based on their gradients
    """
    def __init__(self, loss_names, initial_weights=None, adaptation_rate=0.01):
        self.loss_names = loss_names
        self.adaptation_rate = adaptation_rate
        
        if initial_weights is None:
            self.weights = {name: 1.0 for name in loss_names}
        else:
            self.weights = dict(zip(loss_names, initial_weights))
        
        # Track gradient norms for adaptive weighting
        self.grad_norms = {name: [] for name in loss_names}
    
    def update_weights(self, losses, model):
        """Update loss weights based on gradient magnitudes"""
        # Compute gradients for each loss component
        grad_norms = {}
        
        for name, loss in losses.items():
            # Compute gradients
            grads = torch.autograd.grad(
                loss, model.parameters(), 
                retain_graph=True, create_graph=False
            )
            
            # Compute gradient norm
            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
            grad_norms[name] = grad_norm.item()
            self.grad_norms[name].append(grad_norm.item())
        
        # Adaptive weight update (inverse of gradient norm)
        total_grad_norm = sum(grad_norms.values())
        if total_grad_norm > 0:
            for name in self.loss_names:
                target_weight = total_grad_norm / (grad_norms[name] + 1e-8)
                # Smooth update
                self.weights[name] = (1 - self.adaptation_rate) * self.weights[name] + \
                                   self.adaptation_rate * target_weight
    
    def get_balanced_loss(self, losses):
        """Get weighted combination of losses"""
        total_loss = 0
        for name, loss in losses.items():
            total_loss += self.weights[name] * loss
        return total_loss


def create_advanced_loss_system(num_classes=6):
    """Create advanced loss system for Enhanced GhanaSegNet"""
    
    # Dynamic class-balanced focal loss
    main_loss = DynamicClassBalancedLoss(
        num_classes=num_classes,
        gamma=2.0,
        adaptive_gamma=True
    )
    
    # Import boundary loss (assuming it exists in losses.py)
    from utils.losses import AdvancedBoundaryLoss
    boundary_loss = AdvancedBoundaryLoss(boundary_weight=2.0)
    
    # Curriculum scheduler
    curriculum_scheduler = CurriculumLossScheduler(main_loss, boundary_loss)
    
    # Adaptive balancer
    adaptive_balancer = AdaptiveLossBalancer(['main', 'boundary'])
    
    return {
        'main_loss': main_loss,
        'boundary_loss': boundary_loss,
        'curriculum_scheduler': curriculum_scheduler,
        'adaptive_balancer': adaptive_balancer
    }