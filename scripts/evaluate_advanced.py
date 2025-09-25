"""
Evaluation Script for GhanaSegNet Advanced
Comprehensive evaluation and comparison with baseline models

This script provides:
- Model loading and evaluation
- Performance comparison with baselines
- Detailed metrics analysis
- Visualization of results

Author: EricBaidoo
Target: 30%+ mIoU validation
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
import sys
sys.path.append('/workspace')
from models.ghanasegnet_advanced import create_ghanasegnet_advanced
from models.ghanasegnet import GhanaSegNet  # Original model
from utils.metrics import IoUScore, PixelAccuracy

class ModelEvaluator:
    """
    Comprehensive model evaluation for GhanaSegNet Advanced
    """
    def __init__(self, device='cuda', num_classes=6):
        self.device = device
        self.num_classes = num_classes
        
        # Metrics
        self.iou_metric = IoUScore(num_classes=num_classes)
        self.acc_metric = PixelAccuracy()
        
        print(f"🔍 Model Evaluator initialized")
        print(f"📊 Device: {device}")
        print(f"🎯 Classes: {num_classes}")
        
    def load_model(self, model_path, model_type='advanced'):
        """Load trained model"""
        print(f"📂 Loading {model_type} model from: {model_path}")
        
        # Create model
        if model_type == 'advanced':
            model = create_ghanasegnet_advanced(num_classes=self.num_classes)
        elif model_type == 'original':
            model = GhanaSegNet(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                best_miou = checkpoint.get('best_miou', 0.0)
                epoch = checkpoint.get('epoch', 0)
                print(f"✅ Loaded checkpoint from epoch {epoch}, best mIoU: {best_miou:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model weights")
        else:
            print(f"⚠️  Model path not found: {model_path}")
            print(f"📝 Using randomly initialized model for architecture testing")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_model(self, model, data_loader, model_name="Model"):
        """Evaluate a single model"""
        print(f"\n🧪 Evaluating {model_name}...")
        
        model.eval()
        all_ious = []
        all_accs = []
        class_ious = [[] for _ in range(self.num_classes)]
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(data_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate metrics
                batch_iou = self.iou_metric(outputs, targets)
                batch_acc = self.acc_metric(outputs, targets)
                
                all_ious.append(batch_iou)
                all_accs.append(batch_acc)
                
                # Per-class IoU
                pred_classes = torch.argmax(outputs, dim=1)
                for class_id in range(self.num_classes):
                    class_mask = (targets == class_id)
                    if class_mask.sum() > 0:  # Only if class is present
                        pred_mask = (pred_classes == class_id)
                        intersection = (class_mask & pred_mask).sum().float()
                        union = (class_mask | pred_mask).sum().float()
                        class_iou = intersection / (union + 1e-6)
                        class_ious[class_id].append(class_iou.item())
                
                total_samples += images.size(0)
                
                if batch_idx % 20 == 0:
                    print(f"  Processed {total_samples} samples...")
        
        # Calculate final metrics
        mean_iou = np.mean(all_ious) * 100
        mean_acc = np.mean(all_accs) * 100
        
        # Per-class results
        class_results = {}
        for class_id in range(self.num_classes):
            if class_ious[class_id]:
                class_results[f'class_{class_id}'] = np.mean(class_ious[class_id]) * 100
            else:
                class_results[f'class_{class_id}'] = 0.0
        
        results = {
            'model_name': model_name,
            'mean_iou': mean_iou,
            'mean_accuracy': mean_acc,
            'class_ious': class_results,
            'total_samples': total_samples
        }
        
        return results
    
    def compare_models(self, model_results):
        """Compare multiple model results"""
        print(f"\n📊 Model Comparison Results:")
        print("=" * 80)
        
        # Sort by mIoU
        sorted_results = sorted(model_results, key=lambda x: x['mean_iou'], reverse=True)
        
        for i, result in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            
            print(f"{rank} {result['model_name']}")
            print(f"   mIoU: {result['mean_iou']:.2f}%")
            print(f"   Accuracy: {result['mean_accuracy']:.2f}%")
            
            # Highlight if target achieved
            if result['mean_iou'] >= 30.0:
                print(f"   🎯 TARGET ACHIEVED! (≥30%)")
            
            print()
        
        # Performance comparison
        if len(sorted_results) >= 2:
            best = sorted_results[0]
            second = sorted_results[1]
            improvement = best['mean_iou'] - second['mean_iou']
            
            print(f"🚀 Best model improvement: +{improvement:.2f} percentage points")
            
            if best['mean_iou'] >= 30.0:
                print(f"🎉 SUCCESS! {best['model_name']} achieved 30%+ mIoU target!")
            else:
                needed = 30.0 - best['mean_iou']
                print(f"📈 Gap to 30% target: {needed:.2f} percentage points")
        
        return sorted_results
    
    def save_results(self, results, save_dir="./evaluation_results"):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.json")
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_result[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_result[key] = int(value)
                elif isinstance(value, dict):
                    serializable_result[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                              for k, v in value.items()}
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return results_file
    
    def create_comparison_plot(self, results, save_dir="./evaluation_results"):
        """Create visualization of model comparison"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract data for plotting
        model_names = [r['model_name'] for r in results]
        mious = [r['mean_iou'] for r in results]
        accuracies = [r['mean_accuracy'] for r in results]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # mIoU comparison
        bars1 = ax1.bar(model_names, mious, color=['#2E8B57' if x >= 30 else '#4682B4' for x in mious])
        ax1.set_title('Model mIoU Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('mIoU (%)', fontsize=12)
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='30% Target')
        ax1.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, mious):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Accuracy comparison
        bars2 = ax2.bar(model_names, accuracies, color='#FF7F50', alpha=0.8)
        ax2.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars2, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Comparison plot saved to: {plot_file}")
        
        return plot_file

def main():
    """Main evaluation function"""
    print("🚀 GhanaSegNet Advanced Evaluation")
    print("=" * 50)
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 6,
        'batch_size': 8
    }
    
    print(f"🔧 Configuration: {config}")
    
    # Create evaluator
    evaluator = ModelEvaluator(device=config['device'], num_classes=config['num_classes'])
    
    # Model paths (update these with your actual model paths)
    model_paths = {
        'GhanaSegNet_Advanced': './checkpoints/advanced/ghanasegnet_advanced_best.pth',
        'GhanaSegNet_Original': './checkpoints/ghanasegnet_best.pth',
        # Add more models as needed
    }
    
    print("⚠️  Please update model paths in the script with your actual model locations!")
    print("📁 Expected model files:")
    for name, path in model_paths.items():
        print(f"   {name}: {path}")
    
    # TODO: Load your actual test data loader
    # test_loader = YourTestDataLoader(...)
    
    print("\n⚠️  Please replace with your actual test data loader!")
    print("📊 Expected test data structure:")
    print("   - test_loader: DataLoader for test data")
    print("   - Images: (B, C, H, W) tensors")
    print("   - Targets: (B, H, W) long tensors")
    
    # Uncomment when you have actual data and models:
    """
    # Evaluate models
    results = []
    
    for model_name, model_path in model_paths.items():
        try:
            # Determine model type
            model_type = 'advanced' if 'advanced' in model_name.lower() else 'original'
            
            # Load and evaluate model
            model = evaluator.load_model(model_path, model_type=model_type)
            result = evaluator.evaluate_model(model, test_loader, model_name)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    if results:
        # Compare models
        sorted_results = evaluator.compare_models(results)
        
        # Save results
        evaluator.save_results(sorted_results)
        
        # Create visualization
        evaluator.create_comparison_plot(sorted_results)
        
        # Final summary
        best_model = sorted_results[0]
        print(f"\n🏆 FINAL RESULTS:")
        print(f"   Best Model: {best_model['model_name']}")
        print(f"   Best mIoU: {best_model['mean_iou']:.2f}%")
        
        if best_model['mean_iou'] >= 30.0:
            print(f"   🎉 30% TARGET ACHIEVED!")
        else:
            gap = 30.0 - best_model['mean_iou']
            print(f"   📈 Gap to target: {gap:.2f} percentage points")
    
    print("✅ Evaluation completed!")
    """

if __name__ == "__main__":
    main()