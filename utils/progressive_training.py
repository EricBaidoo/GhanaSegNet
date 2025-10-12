"""
Progressive Training Strategy for Enhanced GhanaSegNet
Implements multi-scale training to break through 24.4% mIoU plateau
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ProgressiveTrainingStrategy:
    """
    Progressive multi-scale training strategy for Enhanced GhanaSegNet
    - Start with 256x256 for stable learning
    - Progress to 320x320 for better details
    - Finish with 384x384 for maximum performance
    """
    def __init__(self, model, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.current_size = 256
        
        # Progressive schedule: epochs per resolution
        self.schedule = {
            256: 5,   # First 5 epochs: stable learning
            320: 6,   # Next 6 epochs: detail enhancement  
            384: 4    # Final 4 epochs: maximum resolution
        }
        
    def get_current_loader(self, batch_size, phase='train'):
        """Get data loader for current resolution"""
        dataset = self.train_dataset if phase == 'train' else self.val_dataset
        
        # Update dataset target size
        dataset.target_size = (self.current_size, self.current_size)
        
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=4,
            pin_memory=True,
            drop_last=(phase == 'train')
        )
    
    def should_increase_resolution(self, epoch):
        """Check if we should increase resolution"""
        if epoch <= 5 and self.current_size != 256:
            self.current_size = 256
            return True
        elif 5 < epoch <= 11 and self.current_size != 320:
            self.current_size = 320
            return True
        elif epoch > 11 and self.current_size != 384:
            self.current_size = 384
            return True
        return False
    
    def get_current_batch_size(self):
        """Adjust batch size based on resolution"""
        if self.current_size == 256:
            return 8
        elif self.current_size == 320:
            return 6
        else:  # 384
            return 4

# Add this to your enhanced_train_model function
def create_progressive_strategy(model, train_dataset, val_dataset):
    """Create progressive training strategy"""
    return ProgressiveTrainingStrategy(model, train_dataset, val_dataset)